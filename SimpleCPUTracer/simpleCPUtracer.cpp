//Source: https://fabiensanglard.net/rayTracing_back_of_business_card/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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

float R() {
    return (float)rand() / RAND_MAX;
}

//The intersection test for line [o,v].
// Return 2 if a hit was found (and also return distance t and bouncing ray n).
// Return 0 if no hit was found but ray goes upward
// Return 1 if no hit was found but ray goes downward
int TraceRay(v origin, v destination, float &t, v &normal) {
  t = 1e9;
  int m = 0;
  float p = -origin.z / destination.z;
  if (.01 < p) {
      t = p;
      normal = v(0, 0, 1);
      m = 1;
  }

  for (int k = 19; k--;)
    for (int j = 9; j--;)
      if (G[j] & 1 << k) {
        v p = origin + v(-k, 0, -j - 4);
        float b = p % destination;
        float c = p % p - 1;
        float q = b * b - c;

          //Does the ray hit the sphere ?
        if (q > 0) {
          float s = -b - sqrt(q);
            //It does, compute the distance camera-sphere
          if (s < t && s > .01) {
              t = s;
              normal = !(p + destination * t);
              m = 2;
          }
        }
      }
  return m;
}
v Sample(v origin, v destination) {
  float t;
  v normal;
  int match = TraceRay(origin, destination, t, normal);
  if (!match) {
      //No sphere found and the ray goes upward: Generate a sky color
      return v(.7, .6, 1) * pow(1 - destination.z, 4);
  }

  //A sphere was maybe hit.
  v intersection = origin + destination * t;
  v light_dir = !(v(9 + R(), 9 + R(), 16) + intersection * -1);
  v half_vec = destination + normal * (normal % destination * -2);

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
int main() {
  printf("P6 512 512 255 ");

  v cam_forward = !v(-6, -16, 0);
  v cam_up = !(v(0, 0, 1) ^ cam_forward) * .002;
  v cam_right = !(cam_forward ^ cam_up) * .002, c = (cam_up + cam_right) * -256 + cam_forward;

  for (int y = 512; y--;)
    for (int x = 512; x--;) {
      v color(13, 13, 13);
      for (int r = 64; r--;) {
        v delta = cam_up * (R() - .5) * 99 + cam_right * (R() - .5) * 99;
        color = Sample(v(17, 16, 8) + delta, !(delta * -1 + (cam_up * (R() + x) + cam_right * (y + R()) + c) * 16)) * 3.5 + color;
      }
      printf("%c%c%c", (int)color.x, (int)color.y, (int)color.z);
    }
  return EXIT_SUCCESS;
}