//More complex path tracer in OpenCL based on https://fabiensanglard.net/rayTracing_back_of_business_card/
//Supports spheres, planes and triangles
//Four materials (checkerboard texture, sky, diffusive, specular)
//No depth of field

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

#include "../ocl_boiler.h"
#include "../pamalign.h"

typedef struct{
	cl_float4 v0;
	cl_float4 v1;
	cl_float4 v2;
} cl_Triangle;

cl_float4 VectorSum(cl_float4 x, cl_float4 y){
	cl_float4 value = { .x = x.s0 + y.s0, .y = x.s1 + y.s1, .z = x.s2 + y.s2, .w = 0};
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

//Method to retrieve spheres/planes information from file
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
int parseTrianglesFromFile(char * fileName, cl_Triangle * arr){
	FILE * textFile;
	char x[MAX], y[MAX], z[MAX];
	int curr_triangle = 0;
	textFile = fopen(fileName, "r");
	while(!feof(textFile) && curr_triangle < MAX_TRIANGLES){
		fgets(x, MAX, textFile);
		fgets(y, MAX, textFile);
		fgets(z, MAX, textFile);
		arr[curr_triangle].v0.x = atof(x);
		arr[curr_triangle].v0.y = atof(y);
		arr[curr_triangle].v0.z = atof(z);
		arr[curr_triangle].v0.w = 0.0f;

		fgets(x, MAX, textFile);	//read END_VERTEX and ignore

		fgets(x, MAX, textFile);
		fgets(y, MAX, textFile);
		fgets(z, MAX, textFile);
		arr[curr_triangle].v1.x = atof(x);
		arr[curr_triangle].v1.y = atof(y);
		arr[curr_triangle].v1.z = atof(z);
		arr[curr_triangle].v1.w = 0.0f;

		fgets(x, MAX, textFile);	//read END_VERTEX and ignore

		fgets(x, MAX, textFile);
		fgets(y, MAX, textFile);
		fgets(z, MAX, textFile);
		arr[curr_triangle].v2.x = atof(x);
		arr[curr_triangle].v2.y = atof(y);
		arr[curr_triangle].v2.z = atof(z);
		arr[curr_triangle].v2.w = 0.0f;

		fgets(x, MAX, textFile);	//read END_VERTEX and ignore
		fgets(x, MAX, textFile);	//read END_TRIANGLE and ignore

		curr_triangle++;
	}
	fclose(textFile);
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

//Setting up the kernel to render the image
cl_event pathTracer(cl_kernel pathtracer_k, cl_command_queue que, cl_mem d_temprender,
	cl_mem d_Spheres, cl_mem d_Planes, cl_mem d_Triangles, cl_int ntriangles, 
	cl_mem d_scenelights, cl_int nlights,
	cl_uint4 seeds, cl_float4 cam_forward, cl_float4 cam_up, cl_float4 cam_right, 
	cl_float4 eye_offset, cl_int renderWidth, cl_int renderHeight){

	const size_t gws[] = { renderWidth*8, renderHeight*8 };

	cl_event pathtracer_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(pathtracer_k, i++, sizeof(d_temprender), &d_temprender);
	ocl_check(err, "set path tracer arg %d", i-1);
	err = clSetKernelArg(pathtracer_k, i++, sizeof(d_Spheres), &d_Spheres);
	ocl_check(err, "set path tracer arg %d", i-1);
	err = clSetKernelArg(pathtracer_k, i++, sizeof(d_Planes), &d_Planes);
	ocl_check(err, "set path tracer arg %d", i-1);
	err = clSetKernelArg(pathtracer_k, i++, sizeof(d_Triangles), &d_Triangles);
	ocl_check(err, "set path tracer arg %d", i-1);
	err = clSetKernelArg(pathtracer_k, i++, sizeof(ntriangles), &ntriangles);
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
	err = clSetKernelArg(pathtracer_k, i++, sizeof(cl_int)*9 , NULL);	//lPlanes
	ocl_check(err, "set path tracer arg %d", i-1);
	err = clSetKernelArg(pathtracer_k, i++, sizeof(cl_Triangle)*ntriangles , NULL);	//lTriangles
	ocl_check(err, "set path tracer arg %d", i-1);
	err = clSetKernelArg(pathtracer_k, i++, sizeof(cl_float4)*nlights , NULL);	//lScenelights
	ocl_check(err, "set path tracer arg %d", i-1);

	err = clEnqueueNDRangeKernel(que, pathtracer_k, 2, NULL, gws, NULL,
		0, NULL, &pathtracer_evt);
	ocl_check(err, "enqueue path tracer");

	return pathtracer_evt;	
}

cl_event reduceimg(cl_kernel reduceimg_k, cl_command_queue que, cl_mem d_temprender, cl_mem d_render, int plotWidth, int plotHeight, cl_event pathtracer_evt){

	const size_t lws[] = { 8, 8 };
	const size_t gws[] = { plotWidth*lws[0], plotHeight*lws[1] };

	cl_event reduceimg_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(reduceimg_k, i++, sizeof(d_temprender), &d_temprender);
	ocl_check(err, "set reduceimg arg %d ", i-1);
	err = clSetKernelArg(reduceimg_k, i++, sizeof(d_render), &d_render);
	ocl_check(err, "set reduceimg arg %d ", i-1);
	err = clSetKernelArg(reduceimg_k, i++, sizeof(cl_float4)*lws[0]*lws[1], NULL);
	ocl_check(err, "set reduceimg arg %d ", i-1);

	err = clEnqueueNDRangeKernel(que, reduceimg_k, 2, NULL, gws, lws,
		1, &pathtracer_evt, &reduceimg_evt);
	ocl_check(err, "enqueue reduceimg");

	return reduceimg_evt;	
}

int main(int argc, char* argv[]){

	int img_width = 512, img_height = 512;
	const int samplesPerPixel = 64;
	printf("Usage: %s [img_width] [img_height]\nLoads data from triangles.txt, lights.txt, spheres.txt and planes.txt", argv[0]);

	if(argc > 1){
		img_width = atoi(argv[1]);
	}
	if (argc > 2){
		img_height = atoi(argv[2]);
	}

	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("pathtracer.ocl", ctx, d);
	cl_int err;

	cl_kernel pathtracer_k = clCreateKernel(prog, "pathTracer", &err);
	ocl_check(err, "create kernel pathtracer_k");
	cl_kernel reduceimg_k = clCreateKernel(prog, "reduce4img_lmem", &err);
	ocl_check(err, "create kernel reduceimg_k");
	
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

	cl_mem d_temprender = clCreateBuffer(ctx,	//Temp render with 64 float4 for each pixel
		CL_MEM_READ_WRITE,
		resultInfo.data_size*samplesPerPixel*sizeof(float), NULL,
		&err);
	ocl_check(err, "create buffer d_temprender");

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

	printf("Cam values:\nCam_forward %f %f %f\nCam_up %f %f %f\nCam_right %f %f %f\n eye_offset %f %f %f\n", cam_forward.x, cam_forward.y, cam_forward.z, cam_up.x, cam_up.y, cam_up.z, cam_right.x, cam_right.y, cam_right.z, eye_offset.x, eye_offset.y, eye_offset.z);

	//Point lights coordinates and intensity
	cl_float4 * scenelights = malloc(sizeof(cl_float4)*MAX_LIGHTS);

	//Geometries
	cl_int * Spheres = malloc(sizeof(cl_int)*9);
	cl_int * Planes = malloc(sizeof(cl_int)*9);
	cl_Triangle * Triangles = malloc(sizeof(cl_float4)*3*MAX_TRIANGLES);

	parseArrayFromFile("spheres.txt", Spheres);
	parseArrayFromFile("planes.txt", Planes);
	
	cl_int ntriangles = parseTrianglesFromFile("triangles.txt", Triangles);
	if(ntriangles > lws_max){
		printf("Too many triangles for local memory: reducing from %d to to %ld\n", ntriangles, lws_max);
		ntriangles = lws_max;
	}

	cl_int nlights = parseLightsFromFile("lights.txt", scenelights);

	printf("Number of triangles: %d\n", ntriangles);
	printf("Number of lights: %d\n", nlights);

	cl_mem d_Spheres = clCreateBuffer(ctx,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_int)*9, Spheres,
		&err);
	ocl_check(err, "create buffer d_Spheres");

	cl_mem d_Planes = clCreateBuffer(ctx,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_int)*9, Planes,
		&err);
	ocl_check(err, "create buffer d_Planes");

	cl_mem d_Triangles = clCreateBuffer(ctx,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_float4)*3*ntriangles, Triangles,
		&err);
	ocl_check(err, "create buffer d_Triangles");

	cl_mem d_scenelights = clCreateBuffer(ctx,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_float4)*nlights, scenelights,
		&err);
	ocl_check(err, "create buffer d_scenelights");

	cl_event pathtracer_evt = pathTracer(pathtracer_k, que, d_temprender, 
	d_Spheres, d_Planes, d_Triangles, ntriangles, 
	d_scenelights, nlights, seeds, 
	cam_forward, cam_up, cam_right, eye_offset, 
	resultInfo.width, resultInfo.height);

	cl_event reduceimg_evt = reduceimg(reduceimg_k, que, d_temprender, 
	d_render, resultInfo.width, resultInfo.height, pathtracer_evt);

	cl_event getRender_evt;
	
	resultInfo.data = clEnqueueMapBuffer(que, d_render, CL_TRUE,
		CL_MAP_READ,
		0, resultInfo.data_size,
		1, &reduceimg_evt, &getRender_evt, &err);
	ocl_check(err, "enqueue map d_render");

	err = save_pam(imageName, &resultInfo);
	if (err != 0) {
		fprintf(stderr, "error writing %s\n", imageName);
		exit(1);
	}
	else printf("\nSuccessfully created render image %s in the current directory\n\n", imageName);

	double runtime_pathtracer_ms = runtime_ms(pathtracer_evt);
	double runtime_reduceimg_ms = runtime_ms(reduceimg_evt);
	double runtime_getRender_ms = runtime_ms(getRender_evt);
	double total_time_ms = runtime_pathtracer_ms + runtime_reduceimg_ms + runtime_getRender_ms;

	double pathtracer_bw_gbs = resultInfo.data_size*samplesPerPixel*sizeof(float)/1.0e6/runtime_pathtracer_ms;
	double reduceimg_bw_gbs = resultInfo.data_size/1.0e6/runtime_pathtracer_ms;
	double getRender_bw_gbs = resultInfo.data_size/1.0e6/runtime_getRender_ms;

	printf("rendering : %d pixels (with %d samples) in %gms: %g GB/s\n",
		img_width*img_height, samplesPerPixel, runtime_pathtracer_ms, pathtracer_bw_gbs);
	printf("reduce img samples : %d pixels (with %d samples) in %gms: %g GB/s\n",
		img_width*img_height, samplesPerPixel, runtime_reduceimg_ms, reduceimg_bw_gbs);
	printf("read render data : %ld uchar in %gms: %g GB/s\n",
		resultInfo.data_size, runtime_getRender_ms, getRender_bw_gbs);
	printf("\nTotal time: %g ms.\n", total_time_ms);

	err = clEnqueueUnmapMemObject(que, d_render, resultInfo.data, 0, NULL, NULL);
	ocl_check(err, "unmap render");
	clReleaseMemObject(d_render);

	free(Spheres);
	free(Planes);
	free(Triangles);
	free(scenelights);

	clReleaseKernel(pathtracer_k);
	clReleaseKernel(reduceimg_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);
}