//Simple path tracer in OpenCL based on https://fabiensanglard.net/rayTracing_back_of_business_card/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>
#include <math.h>

#define CL_TARGET_OPENCL_VERSION 120
#define MAX 256

#include "../ocl_boiler.h"
#include "../pamalign.h"

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
cl_float4 NotOperator(cl_float4 x){
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

cl_event imginit(cl_kernel imginit_k, cl_command_queue que, cl_mem d_render, int plotWidth, int plotHeight){

	const size_t gws[] = { plotWidth, plotHeight };

	cl_event imginit_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(imginit_k, i++, sizeof(d_render), &d_render);
	ocl_check(err, "set imginit arg %d ", i-1);

	err = clEnqueueNDRangeKernel(que, imginit_k, 2, NULL, gws, NULL,
		0, NULL, &imginit_evt);
	ocl_check(err, "enqueue imginit");

	return imginit_evt;	
}

//Setting up the kernel to render the image
cl_event pathTracer(cl_kernel spt_k, cl_command_queue que, cl_mem d_render, cl_mem d_G, 
	cl_uint4 seeds, cl_float4 cam_forward, cl_float4 cam_up, cl_float4 cam_right, cl_float4 eye_offset, 
	cl_int plotWidth, cl_int plotHeight){

	const size_t gws[] = { plotWidth, plotHeight };

	cl_event spt_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(spt_k, i++, sizeof(d_render), &d_render);
	ocl_check(err, "set path tracer arg %d", i-1);
	err = clSetKernelArg(spt_k, i++, sizeof(d_G), &d_G);
	ocl_check(err, "set path tracer arg %d", i-1);
	err = clSetKernelArg(spt_k, i++, sizeof(cam_forward), &cam_forward);
	ocl_check(err, "set path tracer arg %d", i-1);
	err = clSetKernelArg(spt_k, i++, sizeof(cam_up), &cam_up);
	ocl_check(err, "set path tracer arg %d", i-1);
	err = clSetKernelArg(spt_k, i++, sizeof(cam_right), &cam_right);
	ocl_check(err, "set path tracer arg %d", i-1);
	err = clSetKernelArg(spt_k, i++, sizeof(eye_offset), &eye_offset);
	ocl_check(err, "set path tracer arg %d", i-1);
	err = clSetKernelArg(spt_k, i++, sizeof(seeds), &seeds);
	ocl_check(err, "set path tracer arg %d", i-1);

	err = clEnqueueNDRangeKernel(que, spt_k, 2, NULL, gws, NULL,
		0, NULL, &spt_evt);
	ocl_check(err, "enqueue path tracer");

	return spt_evt;	
}

int main(int argc, char* argv[]){

	int img_width = 256, img_height = 256;

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
	cl_program prog = create_program("spt.ocl", ctx, d);
	cl_int err;

	cl_kernel imginit_k = clCreateKernel(prog, "imginit_buf", &err);
	ocl_check(err, "create kernel imginit");

	cl_kernel spt_k = clCreateKernel(prog, "pathTracer", &err);
	ocl_check(err, "create kernel spt_k");
	
	//seeds for the edited MWC64X
	cl_uint4 seeds = {.x = time(0) & 134217727, .y = (getpid() * getpid() * getpid()) & 134217727, .z = (clock()*clock()) & 134217727, .w = rdtsc() & 134217727};

	printf("Seeds: %d, %d, %d, %d\n", seeds.x, seeds.y, seeds.z, seeds.w);

	size_t lws_max;
	err = clGetKernelWorkGroupInfo(spt_k, d, CL_KERNEL_WORK_GROUP_SIZE, 
		sizeof(lws_max), &lws_max, NULL);
	ocl_check(err, "Max lws for spt");
	size_t gws_max = 131072;

	const char *imageName = "result.ppm";
	struct imgInfo resultInfo;
	resultInfo.channels = 4;
	resultInfo.depth = 8;
	resultInfo.maxval = 0xff;
	resultInfo.width = img_width;
	resultInfo.height = img_height;	
	resultInfo.data_size = resultInfo.width*resultInfo.height*resultInfo.channels;
	resultInfo.data = malloc(resultInfo.data_size);
	//createBlankImage((uchar*)resultInfo.data, resultInfo.data_size);
	printf("Processing image %dx%d with data size %ld bytes\n", resultInfo.width, resultInfo.height, resultInfo.data_size);

	cl_mem d_render = clCreateBuffer(ctx,
		CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
		resultInfo.data_size, NULL,
		&err);
	ocl_check(err, "create buffer d_render");

	cl_event initRender_evt = imginit(imginit_k, que, d_render, resultInfo.width, resultInfo.height);

	cl_float4 zVect = { .x = 0, .y = 0, .z = -1, .w = 0 };

	cl_float4 cam_forward = { .x = -6, .y = -16, .z = 0, .w = 0 };
	cam_forward = NotOperator(cam_forward);
	cl_float4 cam_up = ScalarTimesVector(0.002, NotOperator(CrossProduct(zVect, cam_forward)));
	cl_float4 cam_right = ScalarTimesVector(0.002, NotOperator(CrossProduct(cam_forward, cam_up)));

	cl_float4 eye_offset = VectorSum(ScalarTimesVector((float)(-256), VectorSum(cam_up, cam_right)), cam_forward);

	/*
	cl_float4 cam_up = { .x = 0.001873f, .y = -0.000702f, .z = 0.0f, .w = 0 };
	cl_float4 cam_right = { .x = 0.0f, .y = 0.0f, .z = 0.002f, .w = 0 };
	cl_float4 eye_offset = { .x = -0.830524f, .y = -0.756554f, .z = -0.512f, .w = 0 };
	*/

	printf("Cam values:\nCam_forward %f %f %f\nCam_up %f %f %f\nCam_right %f %f %f\n eye_offset %f %f %f\n", cam_forward.x, cam_forward.y, cam_forward.z, cam_up.x, cam_up.y, cam_up.z, cam_right.x, cam_right.y, cam_right.z, eye_offset.x, eye_offset.y, eye_offset.z);

	//Spheres
	cl_int G[9] = {247570, 280596, 280600, 249748, 18578, 18577, 231184, 16, 16};
	//cl_int G[9] = {2048, 0, 0, 0, 0, 0, 0, 0, 0};

	cl_mem d_G = clCreateBuffer(ctx,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(G), G,
		&err);
	ocl_check(err, "create buffer d_G");

	cl_event spt_evt = pathTracer(spt_k, que, d_render, d_G, seeds, cam_forward, cam_up, cam_right, eye_offset, resultInfo.width, resultInfo.height);

	cl_event getRender_evt;

	resultInfo.data = clEnqueueMapBuffer(que, d_render, CL_TRUE,
		CL_MAP_READ,
		0, resultInfo.data_size,
		1, &spt_evt, &getRender_evt, &err);
	ocl_check(err, "enqueue map d_render");

	err = save_pam(imageName, &resultInfo);
	if (err != 0) {
		fprintf(stderr, "error writing %s\n", imageName);
		exit(1);
	}
	else printf("\nSuccessfully created render image %s in the current directory\n\n", imageName);

	double runtime_initRender_ms = runtime_ms(initRender_evt);
	double runtime_spt_ms = runtime_ms(spt_evt);
	double runtime_getRender_ms = runtime_ms(getRender_evt);
	double total_time_ms = runtime_initRender_ms + runtime_spt_ms + runtime_getRender_ms;

	double initRender_bw_gbs = resultInfo.data_size/1.0e6/runtime_initRender_ms;
	double getRender_bw_gbs = resultInfo.data_size/1.0e6/runtime_getRender_ms;
	double spt_bw_gbs = resultInfo.data_size/1.0e6/runtime_spt_ms;

	printf("init image: %ld uchar in %gms: %g GB/s\n", resultInfo.data_size, runtime_initRender_ms, initRender_bw_gbs);
	printf("rendering : %d pixels in %gms: %g GB/s\n",
		img_width*img_height, runtime_spt_ms, spt_bw_gbs);
	printf("read render data : %ld uchar in %gms: %g GB/s\n",
		resultInfo.data_size, runtime_getRender_ms, getRender_bw_gbs);
	printf("\nTotal time: %g ms.\n", total_time_ms);
	//printf("Total:%g;%g\n", intervalSize, runtime_max_ms);

	err = clEnqueueUnmapMemObject(que, d_render, resultInfo.data, 0, NULL, NULL);
	ocl_check(err, "unmap render");
	clReleaseMemObject(d_render);

	clReleaseKernel(imginit_k);
	clReleaseKernel(spt_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);
}