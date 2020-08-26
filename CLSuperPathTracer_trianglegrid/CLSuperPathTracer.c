//More complex path tracer in OpenCL based on https://fabiensanglard.net/rayTracing_back_of_business_card/
//Supports spheres, squares and triangles
//Grid acceleration structure for triangles
//Four materials (checkerboard texture, sky, diffusive, specular)

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>
#include <math.h>

#define CL_TARGET_OPENCL_VERSION 120
#define MAX 256
#define MAX_TRIANGLES 512
#define MAX_LIGHTS 5
#define MAX_NELS_PER_CELL 62 //Should be a power of two minus two for better alignment

#include "../ocl_boiler.h"
#include "../pamalign.h"

typedef struct{
	cl_float4 v0;
	cl_float4 v1;
	cl_float4 v2;
} cl_Triangle;

typedef struct{
	cl_float4 vmin;
	cl_float4 vmax;
} cl_Box;

typedef struct{
	cl_uint nels;
	cl_ushort elem_index[MAX_NELS_PER_CELL];
} cl_Cell;

int max(int x, int y){
	if(x > y) return x;
	return y;
}

int min(int x, int y){
	if(x < y) return x;
	return y;
}

cl_int4 convert_int4(cl_float4 x){
	cl_int4 value = { .x = (int)x.s0, .y = (int)x.s1, .z = (int)x.s2, .w = 0};
	return value;
}

cl_int clamp(cl_int v, cl_int min, cl_int max){
	if (v > max) return max;
	else if (v < min) return min;
	return v;
}

cl_int4 clampVec(cl_int4 v, cl_int4 min, cl_int4 max){
	cl_int4 value = { .x = clamp(v.s0, min.s0, max.s0), .y = clamp(v.s1, min.s1, max.s1), .z = clamp(v.s2, min.s2, max.s2), .w = 0};
	return value;
}

cl_float4 VectorSum(cl_float4 x, cl_float4 y){
	cl_float4 value = { .x = x.s0 + y.s0, .y = x.s1 + y.s1, .z = x.s2 + y.s2, .w = 0};
	return value;
}

cl_float4 VectorDifference(cl_float4 x, cl_float4 y){
	cl_float4 value = { .x = x.s0 - y.s0, .y = x.s1 - y.s1, .z = x.s2 - y.s2, .w = 0};
	return value;
}

cl_int4 VectorDifferenceInt(cl_int4 x, cl_int4 y){
	cl_int4 value = { .x = x.s0 - y.s0, .y = x.s1 - y.s1, .z = x.s2 - y.s2, .w = 0};
	return value;
}

cl_float4 VectorDivision(cl_float4 x, cl_float4 y){
	cl_float4 value = { .x = x.s0/y.s0, .y = x.s1/y.s1, .z = x.s2/y.s2, .w = 0};
	return value;
}

cl_float4 VectorDivisionFloatInt(cl_float4 x, cl_int4 y){
	cl_float4 value = { .x = x.s0/y.s0, .y = x.s1/y.s1, .z = x.s2/y.s2, .w = 0};
	return value;
}

cl_float4 ScalarTimesVector(float scalar, cl_float4 x){
	cl_float4 value = { .x = scalar * x.s0, .y = scalar * x.s1, .z = scalar * x.s2, .w = 0};
	return value;
}

//Defined as operator% in the simple CPU tracer
float ScalarProduct(cl_float4 x, cl_float4 y){
	return x.s0 * y.s0 + x.s1 * y.s1 + x.s2 * y.s2;
}

//Defined as operator^ in the simple CPU tracer
cl_float4 CrossProduct(cl_float4 x, cl_float4 y){
	cl_float4 value = { .x = x.s1 * y.s2 - x.s2 * y.s1, .y = x.s2 * y.s0 - x.s0 * y.s2, .z = x.s0 * y.s1 - x.s1 * y.s0, .w = 0};
	return value;
}

//Defined as operator! in the simple CPU tracer
cl_float4 Normalize(cl_float4 x){
	return ScalarTimesVector((1/sqrt(ScalarProduct(x, x))), x);
}

static inline uint64_t rdtsc(void)
{
	uint64_t val;
	uint32_t h, l;
    __asm__ __volatile__("rdtsc" : "=a" (l), "=d" (h));
        val = ((uint64_t)l) | (((uint64_t)h) << 32);
        return val;
}

//Method to retrieve spheres/squares information from file
int parseArrayFromFile(char * fileName, cl_int * arr){
	FILE * textFile;
	char str[MAX];
	int linectr = 0;
	textFile = fopen(fileName, "r");
	do{
		fgets(str, MAX, textFile);
		arr[linectr] = atoi(str);
		linectr++;
	}while(!feof(textFile) && linectr < 9);
	fclose(textFile);
	return 1;
}

//Method to retrieve vertices from triangles.txt
//Also computes the min and max positions for the bounding box that contains all the triangles
int parseTrianglesFromFile(char * fileName, cl_Triangle * arr, cl_Box * trianglesBox){
	FILE * textFile;
	char x[MAX], y[MAX], z[MAX];
	int curr_triangle = 0;
	float curr_x, curr_y, curr_z;
	cl_float4 curr_max = { .x = CL_FLT_MIN, .y = CL_FLT_MIN, .z = CL_FLT_MIN, .w = 0};
	cl_float4 curr_min = { .x = CL_FLT_MAX, .y = CL_FLT_MAX, .z = CL_FLT_MAX, .w = 0};
	textFile = fopen(fileName, "r");
	while(!feof(textFile) && curr_triangle < MAX_TRIANGLES){
		fgets(x, MAX, textFile);
		fgets(y, MAX, textFile);
		fgets(z, MAX, textFile);
		curr_x = atof(x);
		if (curr_x < curr_min.x) curr_min.x = curr_x;
		if (curr_x > curr_max.x) curr_max.x = curr_x;
		arr[curr_triangle].v0.x = curr_x;
		curr_y = atof(y);
		if (curr_y < curr_min.y) curr_min.y = curr_y;
		if (curr_y > curr_max.y) curr_max.y = curr_y;
		arr[curr_triangle].v0.y = curr_y;
		curr_z = atof(z);
		if (curr_z < curr_min.z) curr_min.z = curr_z;
		if (curr_z > curr_max.z) curr_max.z = curr_z;
		arr[curr_triangle].v0.z = curr_z;
		arr[curr_triangle].v0.w = 0.0f;

		fgets(x, MAX, textFile);	//read END_VERTEX and ignore

		fgets(x, MAX, textFile);
		fgets(y, MAX, textFile);
		fgets(z, MAX, textFile);
		curr_x = atof(x);
		if (curr_x < curr_min.x) curr_min.x = curr_x;
		if (curr_x > curr_max.x) curr_max.x = curr_x;
		arr[curr_triangle].v1.x = curr_x;
		curr_y = atof(y);
		if (curr_y < curr_min.y) curr_min.y = curr_y;
		if (curr_y > curr_max.y) curr_max.y = curr_y;
		arr[curr_triangle].v1.y = curr_y;
		curr_z = atof(z);
		if (curr_z < curr_min.z) curr_min.z = curr_z;
		if (curr_z > curr_max.z) curr_max.z = curr_z;
		arr[curr_triangle].v1.z = curr_z;
		arr[curr_triangle].v1.w = 0.0f;

		fgets(x, MAX, textFile);	//read END_VERTEX and ignore

		fgets(x, MAX, textFile);
		fgets(y, MAX, textFile);
		fgets(z, MAX, textFile);
		curr_x = atof(x);
		if (curr_x < curr_min.x) curr_min.x = curr_x;
		if (curr_x > curr_max.x) curr_max.x = curr_x;
		arr[curr_triangle].v2.x = curr_x;
		curr_y = atof(y);
		if (curr_y < curr_min.y) curr_min.y = curr_y;
		if (curr_y > curr_max.y) curr_max.y = curr_y;
		arr[curr_triangle].v2.y = curr_y;
		curr_z = atof(z);
		if (curr_z < curr_min.z) curr_min.z = curr_z;
		if (curr_z > curr_max.z) curr_max.z = curr_z;
		arr[curr_triangle].v2.z = curr_z;
		arr[curr_triangle].v2.w = 0.0f;

		fgets(x, MAX, textFile);	//read END_VERTEX and ignore
		fgets(x, MAX, textFile);	//read END_TRIANGLE and ignore

		curr_triangle++;
	}
	fclose(textFile);
	trianglesBox->vmax = curr_max;
	trianglesBox->vmin = curr_min;
	return curr_triangle;
}

//Method to retrieve point lights from lights.txt
int parseLightsFromFile(char * fileName, cl_float4 * arr){
	FILE * textFile;
	char x[MAX], y[MAX], z[MAX], w[MAX];
	int curr_light = 0;
	textFile = fopen(fileName, "r");
	while(!feof(textFile) && curr_light < MAX_LIGHTS){
		fgets(x, MAX, textFile);
		fgets(y, MAX, textFile);
		fgets(z, MAX, textFile);
		fgets(w, MAX, textFile);
		arr[curr_light].x = atof(x);
		arr[curr_light].y = atof(y);
		arr[curr_light].z = atof(z);
		arr[curr_light].w = atof(w);
		printf("Light %d: %f %f %f %f\n", curr_light, arr[curr_light].x, arr[curr_light].y, arr[curr_light].z, arr[curr_light].w);
		curr_light++;
	}
	fclose(textFile);
	return curr_light;
}

void initTrianglesGrid_host(cl_Cell * TrianglesGrid, cl_Triangle * Triangles, cl_int4 grid_res, cl_float4 cell_size, cl_Box trianglesBox, cl_int ntriangles){
	cl_int4 unitVec = { .x = 1, .y = 1, .z = 1, .w = 0};
	cl_int4 zeroVec = { .x = 0, .y = 0, .z = 0, .w = 0};
	for(int curr_triangle=0; curr_triangle < ntriangles; ++curr_triangle){
		const cl_Triangle t = Triangles[curr_triangle];
		//Compute triangle bounding box
		cl_float4 fmin = { .x = CL_FLT_MAX, .y = CL_FLT_MAX, .z = CL_FLT_MAX, .w = 0};
		cl_float4 fmax = { .x = CL_FLT_MIN, .y = CL_FLT_MIN, .z = CL_FLT_MIN, .w = 0};
		for (int k = 0; k < 3; ++k){
			if (t.v0.s[k] < fmin.s[k]) fmin.s[k] = t.v0.s[k];
			if (t.v1.s[k] < fmin.s[k]) fmin.s[k] = t.v1.s[k];
			if (t.v2.s[k] < fmin.s[k]) fmin.s[k] = t.v2.s[k];

			if (t.v0.s[k] > fmax.s[k]) fmax.s[k] = t.v0.s[k];
			if (t.v1.s[k] > fmax.s[k]) fmax.s[k] = t.v1.s[k];
			if (t.v2.s[k] > fmax.s[k]) fmax.s[k] = t.v2.s[k];
		}
		//Convert to cell coordinates
		fmin = VectorDivision(VectorDifference(fmin, trianglesBox.vmin), cell_size);
		fmax = VectorDivision(VectorDifference(fmax, trianglesBox.vmin), cell_size);
		const cl_int4 min = clampVec(convert_int4(fmin), zeroVec, VectorDifferenceInt(grid_res, unitVec));
		const cl_int4 max = clampVec(convert_int4(fmax), zeroVec, VectorDifferenceInt(grid_res, unitVec));
		for(int z = min.z; z <= max.z; ++z){
			for(int y = min.y; y <= max.y; ++y){
				for(int x = min.x; x <= max.x; ++x){
					const int index = z*grid_res.x*grid_res.y + y*grid_res.x + x;
					if (TrianglesGrid[index].nels == MAX_NELS_PER_CELL) continue;
					TrianglesGrid[index].elem_index[TrianglesGrid[index].nels++] = curr_triangle;
				}
			}
		}
	}
}

void printTrianglesGrid_host(cl_Cell * TrianglesGrid, cl_int4 grid_res){
	int nels_count = 0;
	int max_nels = 0;
	for(int k=0; k<grid_res.x*grid_res.y*grid_res.z; ++k){
		for(int i=0; i<TrianglesGrid[k].nels; ++i){
			printf("Cell %d, triangle index %hu, nels %d\n", k, TrianglesGrid[k].elem_index[i], TrianglesGrid[k].nels);
		}
		nels_count += TrianglesGrid[k].nels;
		if (TrianglesGrid[k].nels > max_nels) max_nels = TrianglesGrid[k].nels;
	}
	printf("Total nels in grid (with duplicates): %d\nMax nels: %d\n", nels_count, max_nels);
}

cl_event initTrianglesGrid_device(cl_kernel initTrianglesGrid_k, cl_command_queue que, cl_mem d_TrianglesGrid, cl_mem d_Triangles, cl_float4 trianglesBoxMin, cl_int4 grid_res, cl_float4 cell_size, cl_int ntriangles){

	const size_t gws[] = { ntriangles };
	
	cl_event initTrianglesGrid_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(initTrianglesGrid_k, i++, sizeof(d_TrianglesGrid), &d_TrianglesGrid);
	ocl_check(err, "set initTrianglesGrid arg %d", i-1);
	err = clSetKernelArg(initTrianglesGrid_k, i++, sizeof(d_Triangles), &d_Triangles);
	ocl_check(err, "set initTrianglesGrid arg %d", i-1);
	err = clSetKernelArg(initTrianglesGrid_k, i++, sizeof(trianglesBoxMin), &trianglesBoxMin);
	ocl_check(err, "set initTrianglesGrid arg %d", i-1);
	err = clSetKernelArg(initTrianglesGrid_k, i++, sizeof(grid_res), &grid_res);
	ocl_check(err, "set initTrianglesGrid arg %d", i-1);
	err = clSetKernelArg(initTrianglesGrid_k, i++, sizeof(cell_size), &cell_size);
	ocl_check(err, "set initTrianglesGrid arg %d", i-1);

	err = clEnqueueNDRangeKernel(que, initTrianglesGrid_k, 1, NULL, gws, NULL,
		0, NULL, &initTrianglesGrid_evt);
	ocl_check(err, "enqueue initTrianglesGrid");

	return initTrianglesGrid_evt;	

}

cl_event printTrianglesGrid(cl_kernel printTrianglesGrid_k, cl_command_queue que, cl_mem d_TrianglesGrid, cl_int4 grid_res, cl_event initTrianglesGrid_evt){
	const size_t gws[] = { grid_res.x*grid_res.y*grid_res.z };
	cl_event printTrianglesGrid_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(printTrianglesGrid_k, i++, sizeof(d_TrianglesGrid), &d_TrianglesGrid);
	ocl_check(err, "set printTrianglesGrid arg %d", i-1);

	err = clEnqueueNDRangeKernel(que, printTrianglesGrid_k, 1, NULL, gws, NULL,
		1, &initTrianglesGrid_evt, &printTrianglesGrid_evt);
	ocl_check(err, "enqueue printTrianglesGrid");

	return printTrianglesGrid_evt;
}

//Setting up the kernel to render the image
cl_event pathTracer(cl_kernel pathtracer_k, cl_command_queue que, cl_mem d_render, 
	cl_mem d_Spheres, cl_mem d_Squares, cl_mem d_Triangles, cl_int ntriangles,
	cl_Box trianglesBox, cl_mem d_TriangleGrid, cl_int4 grid_res, cl_float4 cell_size,
	cl_mem d_scenelights, cl_int nlights,
	cl_uint4 seeds, cl_float4 cam_forward, cl_float4 cam_up, cl_float4 cam_right, 
	cl_float4 eye_offset, cl_int renderWidth, cl_int renderHeight, cl_event TrianglesGrid_evt){

	const size_t gws[] = { renderWidth, renderHeight };

	cl_event pathtracer_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(pathtracer_k, i++, sizeof(d_render), &d_render);
	ocl_check(err, "set path tracer arg %d", i-1);
	err = clSetKernelArg(pathtracer_k, i++, sizeof(d_Spheres), &d_Spheres);
	ocl_check(err, "set path tracer arg %d", i-1);
	err = clSetKernelArg(pathtracer_k, i++, sizeof(d_Squares), &d_Squares);
	ocl_check(err, "set path tracer arg %d", i-1);
	err = clSetKernelArg(pathtracer_k, i++, sizeof(d_Triangles), &d_Triangles);
	ocl_check(err, "set path tracer arg %d", i-1);
	err = clSetKernelArg(pathtracer_k, i++, sizeof(ntriangles), &ntriangles);
	ocl_check(err, "set path tracer arg %d", i-1);
	err = clSetKernelArg(pathtracer_k, i++, sizeof(trianglesBox), &trianglesBox);
	ocl_check(err, "set path tracer arg %d", i-1);
	err = clSetKernelArg(pathtracer_k, i++, sizeof(d_TriangleGrid), &d_TriangleGrid);
	ocl_check(err, "set path tracer arg %d", i-1);
	err = clSetKernelArg(pathtracer_k, i++, sizeof(grid_res), &grid_res);
	ocl_check(err, "set path tracer arg %d", i-1);
	err = clSetKernelArg(pathtracer_k, i++, sizeof(cell_size), &cell_size);
	ocl_check(err, "set path tracer arg %d", i-1);
	err = clSetKernelArg(pathtracer_k, i++, sizeof(d_scenelights), &d_scenelights);
	ocl_check(err, "set path tracer arg %d", i-1);
	err = clSetKernelArg(pathtracer_k, i++, sizeof(nlights), &nlights);
	ocl_check(err, "set path tracer arg %d", i-1);
	err = clSetKernelArg(pathtracer_k, i++, sizeof(cam_forward), &cam_forward);
	ocl_check(err, "set path tracer arg %d", i-1);
	err = clSetKernelArg(pathtracer_k, i++, sizeof(cam_up), &cam_up);
	ocl_check(err, "set path tracer arg %d", i-1);
	err = clSetKernelArg(pathtracer_k, i++, sizeof(cam_right), &cam_right);
	ocl_check(err, "set path tracer arg %d", i-1);
	err = clSetKernelArg(pathtracer_k, i++, sizeof(eye_offset), &eye_offset);
	ocl_check(err, "set path tracer arg %d", i-1);
	err = clSetKernelArg(pathtracer_k, i++, sizeof(seeds), &seeds);
	ocl_check(err, "set path tracer arg %d", i-1);
	err = clSetKernelArg(pathtracer_k, i++, sizeof(cl_int)*9 , NULL);	//lSpheres
	ocl_check(err, "set path tracer arg %d", i-1);
	err = clSetKernelArg(pathtracer_k, i++, sizeof(cl_int)*9 , NULL);	//lSquares
	ocl_check(err, "set path tracer arg %d", i-1);
	err = clSetKernelArg(pathtracer_k, i++, sizeof(cl_float4)*nlights , NULL);	//lScenelights
	ocl_check(err, "set path tracer arg %d", i-1);

	err = clEnqueueNDRangeKernel(que, pathtracer_k, 2, NULL, gws, NULL,
		1, &TrianglesGrid_evt, &pathtracer_evt);
	ocl_check(err, "enqueue path tracer");

	return pathtracer_evt;	
}

int main(int argc, char* argv[]){

	int img_width = 512, img_height = 512;
	float CELL_SIZE_MODIFIER = 3.0f;
	printf("Usage: %s [img_width] [img_height] [CELL_SIZE_MODIFIER]\nLoads data from triangles.txt, lights.txt, spheres.txt and squares.txt\n", argv[0]);

	if(argc > 1){
		img_width = atoi(argv[1]);
	}
	if (argc > 2){
		img_height = atoi(argv[2]);
	}

	if(argc > 3){
		CELL_SIZE_MODIFIER = atof(argv[3]);
	}

	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("pathtracer.ocl", ctx, d);
	cl_int err;

	cl_kernel initTrianglesGrid_k = clCreateKernel(prog, "initTrianglesGrid", &err);
	ocl_check(err, "create kernel initTrianglesGrid_k");

	cl_kernel printTrianglesGrid_k = clCreateKernel(prog, "printTrianglesGrid", &err);
	ocl_check(err, "create kernel printTrianglesGrid_k");

	cl_kernel pathtracer_k = clCreateKernel(prog, "pathTracer", &err);
	ocl_check(err, "create kernel pathtracer_k");
	
	//seeds for the edited MWC64X
	cl_uint4 seeds = {.x = time(0) & 134217727, .y = (getpid() * getpid() * getpid()) & 134217727, .z = (clock()*clock()) & 134217727, .w = rdtsc() & 134217727};

	printf("Seeds: %d, %d, %d, %d\n", seeds.x, seeds.y, seeds.z, seeds.w);

	size_t lws_max;
	err = clGetKernelWorkGroupInfo(pathtracer_k, d, CL_KERNEL_WORK_GROUP_SIZE, 
		sizeof(lws_max), &lws_max, NULL);
	ocl_check(err, "Max lws for pathtracer");

	const char *imageName = "result.ppm";
	struct imgInfo resultInfo;
	resultInfo.channels = 4;
	resultInfo.depth = 8;
	resultInfo.maxval = 0xff;
	resultInfo.width = img_width;
	resultInfo.height = img_height;	
	resultInfo.data_size = resultInfo.width*resultInfo.height*resultInfo.channels;
	resultInfo.data = malloc(resultInfo.data_size);
	printf("Processing image %dx%d with data size %ld bytes\n", resultInfo.width, resultInfo.height, resultInfo.data_size);

	cl_mem d_render = clCreateBuffer(ctx,
		CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
		resultInfo.data_size, NULL,
		&err);
	ocl_check(err, "create buffer d_render");
	
	cl_float4 zVect = { .x = 0, .y = 0, .z = -1, .w = 0 };

	cl_float4 cam_forward = { .x = -6, .y = -16, .z = 0, .w = 0 };
	cam_forward = Normalize(cam_forward);
	cl_float4 cam_up = ScalarTimesVector(0.002, Normalize(CrossProduct(zVect, cam_forward)));
	cl_float4 cam_right = ScalarTimesVector(0.002, Normalize(CrossProduct(cam_forward, cam_up)));

	cl_float4 eye_offset = VectorSum(ScalarTimesVector((float)(-256), VectorSum(cam_up, cam_right)), cam_forward);
	
	/*
	cl_float4 cam_up = { .x = 0.001873f, .y = -0.000702f, .z = 0.0f, .w = 0 };
	cl_float4 cam_right = { .x = 0.0f, .y = 0.0f, .z = 0.002f, .w = 0 };
	cl_float4 eye_offset = { .x = -0.830524f, .y = -0.756554f, .z = -0.512f, .w = 0 };
	*/

	printf("Cam values:\nCam_forward %f %f %f\nCam_up %f %f %f\nCam_right %f %f %f\neye_offset %f %f %f\n", cam_forward.x, cam_forward.y, cam_forward.z, cam_up.x, cam_up.y, cam_up.z, cam_right.x, cam_right.y, cam_right.z, eye_offset.x, eye_offset.y, eye_offset.z);

	//Point lights coordinates and intensity
	cl_float4 * scenelights = malloc(sizeof(cl_float4)*MAX_LIGHTS);

	//Geometries
	cl_int * Spheres = malloc(sizeof(cl_int)*9);
	cl_int * Squares = malloc(sizeof(cl_int)*9);
	cl_Triangle * Triangles = malloc(sizeof(cl_float4)*3*MAX_TRIANGLES);

	parseArrayFromFile("spheres.txt", Spheres);
	parseArrayFromFile("squares.txt", Squares);
	
	cl_Box trianglesBox;
	cl_int ntriangles = parseTrianglesFromFile("triangles.txt", Triangles, &trianglesBox);
	printf("Triangles bounding box values:\nvmax: %f %f %f, vmin: %f %f %f\n", trianglesBox.vmax.x, trianglesBox.vmax.y, trianglesBox.vmax.z, trianglesBox.vmin.x, trianglesBox.vmin.y, trianglesBox.vmin.z);

	//Compute grid values
	cl_float4 grid_size = VectorDifference(trianglesBox.vmax, trianglesBox.vmin);
	float cubeRoot = cbrt(CELL_SIZE_MODIFIER*ntriangles/(grid_size.s0 * grid_size.s1 * grid_size.s2));
	cl_int4 grid_res;
	for (int i=0; i<3; ++i){
		grid_res.s[i] = (int)(floor(grid_size.s[i] * cubeRoot));
		grid_res.s[i] = max(1, min(grid_res.s[i], 128));
	}
	cl_float4 cell_size = VectorDivisionFloatInt(grid_size, grid_res);
	size_t grid_memsize = sizeof(cl_Cell)*grid_res.s0*grid_res.s1*grid_res.s2;
	cl_Cell * TrianglesGrid = calloc(1, grid_memsize);
	printf("Triangles grid size: %d x %d x %d\n", grid_res.x, grid_res.y, grid_res.z);

	cl_int nlights = parseLightsFromFile("lights.txt", scenelights);

	printf("Number of triangles: %d\n", ntriangles);
	printf("Number of lights: %d\n", nlights);

	cl_mem d_Spheres = clCreateBuffer(ctx,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_int)*9, Spheres,
		&err);
	ocl_check(err, "create buffer d_Spheres");

	cl_mem d_Squares = clCreateBuffer(ctx,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_int)*9, Squares,
		&err);
	ocl_check(err, "create buffer d_Squares");

	cl_mem d_Triangles = clCreateBuffer(ctx,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_float4)*3*ntriangles, Triangles,
		&err);
	ocl_check(err, "create buffer d_Triangles");

	cl_mem d_TrianglesGrid = clCreateBuffer(ctx,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		grid_memsize, TrianglesGrid,
		&err);
	ocl_check(err, "create buffer d_TrianglesGrid");

	cl_mem d_scenelights = clCreateBuffer(ctx,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_float4)*nlights, scenelights,
		&err);
	ocl_check(err, "create buffer d_scenelights");
	//clock_t start_initTrianglesGrid, end_initTrianglesGrid;
  	//start_initTrianglesGrid = clock();
	//initTrianglesGrid(TrianglesGrid, Triangles, grid_res, cell_size, trianglesBox, ntriangles);
	//end_initTrianglesGrid = clock();
	//printTrianglesGrid_host(TrianglesGrid, grid_res);

	cl_event initTrianglesGrid_evt = initTrianglesGrid_device(initTrianglesGrid_k, que, d_TrianglesGrid, d_Triangles, trianglesBox.vmin, grid_res, cell_size, ntriangles);
	cl_event printTrianglesGrid_evt = printTrianglesGrid(printTrianglesGrid_k, que, d_TrianglesGrid, grid_res, initTrianglesGrid_evt);

	cl_event pathtracer_evt = pathTracer(pathtracer_k, que, d_render, 
	d_Spheres, d_Squares, d_Triangles, ntriangles, trianglesBox,
	d_TrianglesGrid, grid_res, cell_size, d_scenelights, nlights, seeds, 
	cam_forward, cam_up, cam_right, eye_offset, 
	resultInfo.width, resultInfo.height, printTrianglesGrid_evt);

	cl_event getRender_evt;
	
	resultInfo.data = clEnqueueMapBuffer(que, d_render, CL_TRUE,
		CL_MAP_READ,
		0, resultInfo.data_size,
		1, &pathtracer_evt, &getRender_evt, &err);
	ocl_check(err, "enqueue map d_render");

	err = save_pam(imageName, &resultInfo);
	if (err != 0) {
		fprintf(stderr, "error writing %s\n", imageName);
		exit(1);
	}
	else printf("\nSuccessfully created render image %s in the current directory\n\n", imageName);

	double runtime_initTrianglesGrid_ms = runtime_ms(initTrianglesGrid_evt);
	//double runtime_initTrianglesGrid_ms = (end_initTrianglesGrid - start_initTrianglesGrid)*1.0e3/CLOCKS_PER_SEC;
	double runtime_pathtracer_ms = runtime_ms(pathtracer_evt);
	double runtime_getRender_ms = runtime_ms(getRender_evt);
	double total_time_ms = runtime_pathtracer_ms + runtime_getRender_ms;

	double pathtracer_bw_gbs = resultInfo.data_size/1.0e6/runtime_pathtracer_ms;
	double initTrianglesGrid_bw_gbs = grid_memsize/1.0e6/runtime_initTrianglesGrid_ms;
	double getRender_bw_gbs = resultInfo.data_size/1.0e6/runtime_getRender_ms;

	printf("init triangles grid : %d cells in %gms: %g GB/s\n",
		grid_res.x*grid_res.y*grid_res.z, runtime_initTrianglesGrid_ms, initTrianglesGrid_bw_gbs);
	printf("rendering : %d pixels in %gms: %g GB/s\n",
		img_width*img_height, runtime_pathtracer_ms, pathtracer_bw_gbs);
	printf("read render data : %ld uchar in %gms: %g GB/s\n",
		resultInfo.data_size, runtime_getRender_ms, getRender_bw_gbs);
	printf("\nTotal time: %g ms.\n", total_time_ms);

	err = clEnqueueUnmapMemObject(que, d_render, resultInfo.data, 0, NULL, NULL);
	ocl_check(err, "unmap render");
	clReleaseMemObject(d_render);

	free(Spheres);
	free(Squares);
	free(Triangles);
	free(TrianglesGrid);
	free(scenelights);

	clReleaseKernel(pathtracer_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);
}