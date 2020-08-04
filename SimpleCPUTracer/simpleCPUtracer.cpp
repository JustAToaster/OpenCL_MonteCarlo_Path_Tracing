//Source: https://fabiensanglard.net/rayTracing_back_of_business_card/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../pamalign.h"

struct v {
  float x, y, z;
  v operator+(v r) { return v(x + r.x, y + r.y, z + r.z); }
  v operator*(float r) { return v(x * r, y * r, z * r); }
  float operator%(v r) { return x * r.x + y * r.y + z * r.z; }
  v() {}
  v operator^(v r) {
    return v(y * r.z - z * r.y, z * r.x - x * r.z, x * r.y - y * r.x);
  }
  v(float a, float b, float c) {
    x = a;
    y = b;
    z = c;
  }
  v operator!() { return *this * (1 / sqrt(*this % *this)); }
};

int G[] = {247570, 280596, 280600, 249748, 18578, 18577, 231184, 16, 16};
/*
 
   16                    1    
   16                    1    
   231184   111    111   1    
   18577       1  1   1  1   1
   18578       1  1   1  1  1 
   249748   1111  11111  1 1  
   280600  1   1  1      11   
   280596  1   1  1      1 1  
   247570   1111   111   1  1 
 
   */

float R() {
    return (float)rand() / RAND_MAX;
}

//The intersection test for line [o,v].
// Return 2 if a hit was found (and also return distance t and bouncing ray n).
// Return 0 if no hit was found but ray goes upward
// Return 1 if no hit was found but ray goes downward
int TraceRay(v origin, v direction, float &t, v &normal) {
  t = 1e9;
  int m = 0;
  float p = -origin.z / direction.z;
  //printf("p1: %f\n", p);
  if (.01 < p) {
      t = p;
      normal = v(0, 0, 1);
      m = 1;
  }

  for (int k = 19; k--;)
    for (int j = 9; j--;)
      if (G[j] & 1 << k) {
        v p = origin + v(-k, 0, -j - 4);
        float b = p % direction;
        float c = p % p - 1;
        float q = b * b - c;
        //printf("b: %f\n", b);

          //Does the ray hit the sphere ?
        if (q > 0) {
          float s = -b - sqrt(q);
            //It does, compute the distance camera-sphere
          if (s < t && s > .01) {
              t = s;
              normal = !(p + direction * t);
              m = 2;
          }
        }
      }
  return m;
}
v Sample(v origin, v direction) {
  float t;
  v normal;
  int match = TraceRay(origin, direction, t, normal);
  if (!match) {
      //No sphere found and the ray goes upward: Generate a sky color
      return v(.7, .6, 1) * pow(1 - direction.z, 4);
  }

  //A sphere was maybe hit.
  v intersection = origin + direction * t;
  //printf("direction: %f %f %f\n", direction.x, direction.y, direction.z);
  v light_dir = !(v(9 + R(), 9 + R(), 16) + intersection * -1);
  //printf("lightdir: %f %f %f\n", light_dir.x, light_dir.y, light_dir.z);
  v half_vec = direction + normal * (normal % direction * -2);

  //Calculated the lambertian factor
  float lamb_f = light_dir % normal;

    //Calculate illumination factor (lambertian coefficient > 0 or in shadow)?
  if (lamb_f < 0 || TraceRay(intersection, light_dir, t, normal)) {
      lamb_f = 0;
  }

  float color = pow(light_dir % half_vec * (lamb_f > 0), 99);

  if (match & 1) {
    //No sphere was hit and the ray was going downward: Generate a floor color
    intersection = intersection * .2;
    return ((int)(ceil(intersection.x) + ceil(intersection.y)) & 1 ? v(3, 1, 1) : v(3, 3, 3)) *
           (lamb_f * .2 + .1);
  }

  //m == 2 A sphere was hit. Cast an ray bouncing from the sphere surface.
  //Attenuate color by 50% since it is bouncing (* .5)
  return v(color, color, color) + Sample(intersection, half_vec) * .5;
}

void WriteColor(uchar * imgData, int x, int y, int width, int height, v color){
	int index = 4*(y*width+x);
  imgData[index] = (uchar)color.x;
  imgData[index+1] = (uchar)color.y;
  imgData[index+2] = (uchar)color.z;
  imgData[index+3] = 255;
}

void createBlankImage(uchar * imgData, size_t n){
	for(int i=0; i<n; i+=4){
		imgData[i]=255;
		imgData[i+1]=255;
		imgData[i+2]=255;
		imgData[i+3]=255;
	}
}

int main(int argc, char* argv[]) {
  int width = 256, height = 256;
  if(argc > 1){
    width = atoi(argv[1]);
  }
  if (argc > 2){
    height = atoi(argv[2]);
  }

  const char * imageName = "resultCPU.ppm";
  struct imgInfo resultInfo;
	resultInfo.channels = 4;
	resultInfo.depth = 8;
	resultInfo.maxval = 0xff;
	resultInfo.width = width;
	resultInfo.height = height;	
	resultInfo.data_size = resultInfo.width*resultInfo.height*resultInfo.channels;
	resultInfo.data = malloc(resultInfo.data_size);
  createBlankImage((uchar*)resultInfo.data, resultInfo.data_size);
	printf("Processing image %dx%d with data size %ld bytes\n", resultInfo.width, resultInfo.height, resultInfo.data_size);

  v cam_forward = !v(-6, -16, 0);
  v cam_up = !(v(0, 0, 1) ^ cam_forward) * .002;
  v cam_right = !(cam_forward ^ cam_up) * .002, eye_offset = (cam_up + cam_right) * -256 + cam_forward;

  printf("Cam values:\nCam_forward %f %f %f\nCam_up %f %f %f\nCam_right %f %f %f\n eye_offset %f %f %f\n", cam_forward.x, cam_forward.y, cam_forward.z, cam_up.x, cam_up.y, cam_up.z, cam_right.x, cam_right.y, cam_right.z, eye_offset.x, eye_offset.y, eye_offset.z);

  clock_t start_render, end_render;

  start_render = clock();
  for (int y = height; y--;)
    for (int x = width; x--;) {
      v color(13, 13, 13);
      for (int r = 64; r--;) {
        v delta = cam_up * (R() - .5) * 99 + cam_right * (R() - .5) * 99;
        //printf("%d %d delta: %f\n", x, y, delta.x);
        color = Sample(v(17, 16, 8) + delta, !(delta * -1 + (cam_up * (R() + x) + cam_right * (y + R()) + eye_offset) * 16)) * 3.5 + color;
      }
      //printf("Pixel %d %d, color %f %f %f\n", x, y, color.x, color.y, color.z);
      WriteColor((uchar*)resultInfo.data, width-x, height-y, width, height, color);
    }
  end_render = clock();
  int err = save_pam(imageName, &resultInfo);
	if (err != 0) {
		fprintf(stderr, "error writing %s\n", imageName);
		exit(1);
	}
	else printf("Successfully created render image %s in the current directory\n", imageName);

  double runtime_rendering_ms = (end_render - start_render)*1.0e3/CLOCKS_PER_SEC;
  double rendering_bw_gbs = width*height*sizeof(float)/1.0e6/runtime_rendering_ms;

  printf("rendering (host) : %d float in %gms: %g GB/s\n",
		width*height, runtime_rendering_ms, rendering_bw_gbs);

  return EXIT_SUCCESS;
}