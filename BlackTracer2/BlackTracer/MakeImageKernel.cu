#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "kernel.cuh"
#include <iostream>
#include <stdint.h>
#include <iomanip>
#include <algorithm>
#include <chrono>

using namespace std;

/* ------------- DEFINITIONS & DECLARATIONS --------------*/
#pragma region define
#define L(x,y) __launch_bounds__(x,y)

#define TILE_W 8
#define TILE_H 8
#define RASTER_W (TILE_W + 1)
#define SHARE (TILE_W*TILE_H+TILE_W+TILE_H+1)*2
#define PI2 6.283185307179586476f
#define PI 3.141592653589793238f
#define INVSQRTPI 0.318310f
#define SQ0422 2.768166f
#define ESQ0422 0.0587498f
#define inv22 4.5454545f

#define ij (i*M+j)
#define N1Half (N/2+1)
#define N1 (N+1)
#define M1 (M+1)
#define cam_speed cam[0]
#define cam_alpha cam[1]
#define cam_w cam[2]
#define cam_wbar cam[3]

texture<float4, 2, cudaReadModeElementType> texRef;
__constant__ int treeConst[255];
__constant__ const int tempToRGB[1173] = { 255, 56, 0, 255, 71, 0, 255, 83, 0, 255, 93, 0, 255, 101, 0, 255, 109, 0, 255, 115, 0, 255, 121, 0, 
									   255, 126, 0, 255, 131, 0, 255, 137, 18, 255, 142, 33, 255, 147, 44, 255, 152, 54, 255, 157, 63, 255, 161, 72, 
									   255, 165, 79, 255, 169, 87, 255, 173, 94, 255, 177, 101, 255, 180, 107, 255, 184, 114, 255, 187, 120, 
									   255, 190, 126, 255, 193, 132, 255, 196, 137, 255, 199, 143, 255, 201, 148, 255, 204, 153, 255, 206, 159,  
									   255, 209, 163, 255, 211, 168, 255, 213, 173, 255, 215, 177, 255, 217, 182, 255, 219, 186, 255, 221, 190,
									   255, 223, 194, 255, 225, 198, 255, 227, 202, 255, 228, 206, 255, 230, 210, 255, 232, 213, 255, 233, 217,
									   255, 235, 220, 255, 236, 224, 255, 238, 227, 255, 239, 230, 255, 240, 233, 255, 242, 236, 255, 243, 239,
									   255, 244, 242, 255, 245, 245, 255, 246, 248, 255, 248, 251, 255, 249, 253, 254, 249, 255, 252, 247, 255,
									   249, 246, 255, 247, 245, 255, 245, 243, 255, 243, 242, 255, 240, 241, 255, 239, 240, 255, 237, 239, 255, 
									   235, 238, 255, 233, 237, 255, 231, 236, 255, 230, 235, 255, 228, 234, 255, 227, 233, 255, 225, 232, 255, 
									   224, 231, 255, 222, 230, 255, 221, 230, 255, 220, 229, 255, 218, 228, 255, 217, 227, 255, 216, 227, 255, 
									   215, 226, 255, 214, 225, 255, 212, 225, 255, 211, 224, 255, 210, 223, 255, 209, 223, 255, 208, 222, 255, 
									   207, 221, 255, 207, 221, 255, 206, 220, 255, 205, 220, 255, 204, 219, 255, 203, 219, 255, 202, 218, 255, 
									   201, 218, 255, 201, 217, 255, 200, 217, 255, 199, 216, 255, 199, 216, 255, 198, 216, 255, 197, 215, 255, 
									   196, 215, 255, 196, 214, 255, 195, 214, 255, 195, 214, 255, 194, 213, 255, 193, 213, 255, 193, 212, 255, 
									   192, 212, 255, 192, 212, 255, 191, 211, 255, 191, 211, 255, 190, 211, 255, 190, 210, 255, 189, 210, 255, 
									   189, 210, 255, 188, 210, 255, 188, 209, 255, 187, 209, 255, 187, 209, 255, 186, 208, 255, 186, 208, 255, 
									   185, 208, 255, 185, 208, 255, 185, 207, 255, 184, 207, 255, 184, 207, 255, 183, 207, 255, 183, 206, 255, 
									   183, 206, 255, 182, 206, 255, 182, 206, 255, 182, 205, 255, 181, 205, 255, 181, 205, 255, 181, 205, 255, 
									   180, 205, 255, 180, 204, 255, 180, 204, 255, 179, 204, 255, 179, 204, 255, 179, 204, 255, 178, 203, 255,
									   178, 203, 255, 178, 203, 255, 178, 203, 255, 177, 203, 255, 177, 202, 255, 177, 202, 255, 177, 202, 255,
									   176, 202, 255, 176, 202, 255, 176, 202, 255, 175, 201, 255, 175, 201, 255, 175, 201, 255, 175, 201, 255, 
									   175, 201, 255, 174, 201, 255, 174, 201, 255, 174, 200, 255, 174, 200, 255, 173, 200, 255, 173, 200, 255,
									   173, 200, 255, 173, 200, 255, 173, 200, 255, 172, 199, 255, 172, 199, 255, 172, 199, 255, 172, 199, 255,
									   172, 199, 255, 171, 199, 255, 171, 199, 255, 171, 199, 255, 171, 198, 255, 171, 198, 255, 170, 198, 255,
									   170, 198, 255, 170, 198, 255, 170, 198, 255, 170, 198, 255, 170, 198, 255, 169, 198, 255, 169, 197, 255,
									   169, 197, 255, 169, 197, 255, 169, 197, 255, 169, 197, 255, 169, 197, 255, 168, 197, 255, 168, 197, 255,
									   168, 197, 255, 168, 197, 255, 168, 196, 255, 168, 196, 255, 168, 196, 255, 167, 196, 255, 167, 196, 255,
									   167, 196, 255, 167, 196, 255, 167, 196, 255, 167, 196, 255, 167, 196, 255, 166, 196, 255, 166, 195, 255,
									   166, 195, 255, 166, 195, 255, 166, 195, 255, 166, 195, 255, 166, 195, 255, 166, 195, 255, 165, 195, 255,
									   165, 195, 255, 165, 195, 255, 165, 195, 255, 165, 195, 255, 165, 195, 255, 165, 194, 255, 165, 194, 255,
									   165, 194, 255, 164, 194, 255, 164, 194, 255, 164, 194, 255, 164, 194, 255, 164, 194, 255, 164, 194, 255,
									   164, 194, 255, 164, 194, 255, 164, 194, 255, 164, 194, 255, 163, 194, 255, 163, 194, 255, 163, 193, 255,
									   163, 193, 255, 163, 193, 255, 163, 193, 255, 163, 193, 255, 163, 193, 255, 163, 193, 255, 163, 193, 255,
									   163, 193, 255, 162, 193, 255, 162, 193, 255, 162, 193, 255, 162, 193, 255, 162, 193, 255, 162, 193, 255,
									   162, 193, 255, 162, 193, 255, 162, 192, 255, 162, 192, 255, 162, 192, 255, 162, 192, 255, 162, 192, 255,
									   161, 192, 255, 161, 192, 255, 161, 192, 255, 161, 192, 255, 161, 192, 255, 161, 192, 255, 161, 192, 255,
									   161, 192, 255, 161, 192, 255, 161, 192, 255, 161, 192, 255, 161, 192, 255, 161, 192, 255, 161, 192, 255,
									   160, 192, 255, 160, 192, 255, 160, 191, 255, 160, 191, 255, 160, 191, 255, 160, 191, 255, 160, 191, 255,
									   160, 191, 255, 160, 191, 255, 160, 191, 255, 160, 191, 255, 160, 191, 255, 160, 191, 255, 160, 191, 255,
									   160, 191, 255, 159, 191, 255, 159, 191, 255, 159, 191, 255, 159, 191, 255, 159, 191, 255, 159, 191, 255,
									   159, 191, 255, 159, 191, 255, 159, 191, 255, 159, 191, 255, 159, 191, 255, 159, 190, 255, 159, 190, 255,
									   159, 190, 255, 159, 190, 255, 159, 190, 255, 159, 190, 255, 159, 190, 255, 158, 190, 255, 158, 190, 255,
									   158, 190, 255, 158, 190, 255, 158, 190, 255, 158, 190, 255, 158, 190, 255, 158, 190, 255, 158, 190, 255,
									   158, 190, 255, 158, 190, 255, 158, 190, 255, 158, 190, 255, 158, 190, 255, 158, 190, 255, 158, 190, 255,
									   158, 190, 255, 158, 190, 255, 158, 190, 255, 158, 190, 255, 158, 190, 255, 157, 190, 255, 157, 190, 255,
									   157, 189, 255, 157, 189, 255, 157, 189, 255, 157, 189, 255, 157, 189, 255, 157, 189, 255, 157, 189, 255,
									   157, 189, 255, 157, 189, 255, 157, 189, 255, 157, 189, 255, 157, 189, 255, 157, 189, 255, 157, 189, 255, 
									   157, 189, 255, 157, 189, 255, 157, 189, 255, 157, 189, 255, 157, 189, 255, 157, 189, 255, 157, 189, 255, 
									   157, 189, 255, 156, 189, 255, 156, 189, 255, 156, 189, 255, 156, 189, 255, 156, 189, 255, 156, 189, 255,
									   156, 189, 255, 156, 189, 255, 156, 189, 255, 156, 189, 255, 156, 189, 255, 156, 189, 255, 156, 189, 255, 
									   156, 189, 255, 156, 189, 255, 156, 189, 255, 156, 188, 255, 156, 188, 255, 156, 188, 255, 156, 188, 255, 
									   156, 188, 255, 156, 188, 255, 156, 188, 255, 156, 188, 255, 156, 188, 255, 156, 188, 255, 156, 188, 255, 
									   156, 188, 255, 155, 188, 255, 155, 188, 255, 155, 188, 255, 155, 188, 255, 155, 188, 255, 155, 188, 255, 
									   155, 188, 255, 155, 188, 255, 155, 188, 255, 155, 188, 255, 155, 188, 255, 155, 188, 255, 155, 188, 255, 
									   155, 188, 255, 155, 188, 255, 155, 188, 255, 155, 188, 255 };

//__device__ float maxDistSq = 0.f;
//__device__ int filterW = 0;

extern int makeImage(float *out, const float2 *thphi, const int *pi, const float *ver, const float *hor,
	const float *stars, const int *starTree, const int starSize, const float *camParam, const float *magnitude, const int treeLevel,
	const bool symmetry, const int M, const int N, const int step, const cv::Mat csImage);

cudaError_t cudaPrep(float *out, const float2 *thphi, const int *pi, const float *ver, const float *hor,
	const float *stars, const int *starTree, const int starSize, const float *camParam, const float *magnitude, const int treeLevel,
	const bool symmetry, const int M, const int N, const int step, const cv::Mat csImage);

#pragma endregion

/// <summary>
/// Checks if the cross product between two vectors a and b is positive.
/// </summary>
/// <param name="t_a, p_a">Theta and phi of the a vector.</param>
/// <param name="t_b, p_b">Theta of the b vector.</param>
/// <param name="starTheta, starPhi">The star theta and phi.</param>
/// <param name="sgn">The winding order of the polygon + for CW, - for CCW.</param>
/// <returns></returns>
__device__ bool checkCrossProduct(float t_a, float t_b, float p_a, float p_b,
	float starTheta, float starPhi, int sgn) {
	float c1t = (float)sgn * (t_a - t_b);
	float c1p = (float)sgn * (p_a - p_b);
	float c2t = sgn ? starTheta - t_b : starTheta - t_a;
	float c2p = sgn ? starPhi - p_b : starPhi - p_a;
	return (c1t * c2p - c2t * c1p) > 0;
}

/// <summary>
/// Interpolates the corners of a projected pixel on the celestial sky to find the position
/// of a star in the (normal, unprojected) pixel in the output image.
/// </summary>
/// <param name="t0 - t4">The theta values of the projected pixel.</param>
/// <param name="p0 - p4">The phi values of the projected pixel.</param>
/// <param name="start, starp">The star theta and phi.</param>
/// <param name="sgn">The winding order of the polygon + for CW, - for CCW.</param>
/// <returns></returns>
__device__ void interpolate(float t0, float t1, float t2, float t3, float p0, float p1, float p2, float p3,
	float &start, float &starp, int sgn) {
	float error = 0.00001f;

	float midT = (t0 + t1 + t2 + t3) * .25f;
	float midP = (p0 + p1 + p2 + p3) * .25f;

	float starInPixY = 0.5f;
	float starInPixX = 0.5f;

	float perc = 0.5f;
	int count = 0;
	#pragma unroll
	for (int q = 0; q < 10; q++) {
		if ((fabs(start - midT) < error) && (fabs(starp - midP) < error)) break;

		float half01T = (t0 + t1) * .5f;
		float half23T = (t2 + t3) * .5f;
		float half12T = (t2 + t1) * .5f;
		float half03T = (t0 + t3) * .5f;
		float half01P = (p0 + p1) * .5f;
		float half23P = (p2 + p3) * .5f;
		float half12P = (p2 + p1) * .5f;
		float half03P = (p0 + p3) * .5f;

		float line01to23T = half23T - half01T;
		float line03to12T = half12T - half03T;
		float line01to23P = half23P - half01P;
		float line03to12P = half12P - half03P;

		float line01toStarT = start - half01T;
		float line03toStarT = start - half03T;
		float line01toStarP = starp - half01P;
		float line03toStarP = starp - half03P;

		int a = (((line03to12T * line03toStarP) - (line03toStarT * line03to12P)) > 0.f) ? 1 : -1;
		int b = (((line01to23T * line01toStarP) - (line01toStarT * line01to23P)) > 0.f) ? 1 : -1;

		perc *= 0.5f;

		if (sgn*a > 0) {
			if (sgn*b > 0) {
				t2 = half12T;
				t0 = half01T;
				t3 = midT;
				p2 = half12P;
				p0 = half01P;
				p3 = midP;
				starInPixX -= perc;
				starInPixY -= perc;
			}
			else {
				t2 = midT;
				t1 = half01T;
				t3 = half03T;
				p2 = midP;
				p1 = half01P;
				p3 = half03P;
				starInPixX -= perc;
				starInPixY += perc;
			}
		}
		else {
			if (sgn*b > 0) {
				t1 = half12T;
				t3 = half23T;
				t0 = midT;
				p1 = half12P;
				p3 = half23P;
				p0 = midP;
				starInPixX += perc;
				starInPixY -= perc;
			}
			else {
				t0 = half03T;
				t1 = midT;
				t2 = half23T;
				p0 = half03P;
				p1 = midP;
				p2 = half23P;
				starInPixX += perc;
				starInPixY += perc;
			}
		}
		midT = (t0 + t1 + t2 + t3) * .25f;
		midP = (p0 + p1 + p2 + p3) * .25f;
	}
	start = starInPixX;
	starp = starInPixY;
}

/// <summary>
/// Returns if a (star) location lies within the boundaries of the provided polygon.
/// </summary>
/// <param name="t, p">The theta and phi values of the polygon corners.</param>
/// <param name="start, starp">The star theta and phi.</param>
/// <param name="sgn">The winding order of the polygon + for CW, - for CCW.</param>
/// <returns></returns>
__device__ bool starInPolygon(const float *t, const float *p, float start, float starp, int sgn) {
	return (checkCrossProduct(t[0], t[1], p[0], p[1], start, starp, sgn)) &&
		   (checkCrossProduct(t[1], t[2], p[1], p[2], start, starp, sgn)) &&
		   (checkCrossProduct(t[2], t[3], p[2], p[3], start, starp, sgn)) &&
		   (checkCrossProduct(t[3], t[0], p[3], p[0], start, starp, sgn));
}

/// <summary>
/// Checks and corrects phi values for 2-pi crossings.
/// </summary>
/// <param name="p">The phi values to check.</param>
/// <param name="factor">The factor to check if a point is close to the border.</param>
/// <returns></returns>
__device__ bool piCheck(float *p, float factor) {
	float factor1 = PI2*(1.f - factor);
	bool check = false;
	#pragma unroll
	for (int q = 0; q < 4; q++) {
		if (p[q] > factor1) {
			check = true;
			break;
		}
	}
	if (!check) return false;
	check = false;
	float factor2 = PI2 * factor;
	#pragma unroll
	for (int q = 0; q < 4; q++) {
		if (p[q] < factor2) {
			p[q] += PI2;
			check = true;
		}
	}
	return check;
}

/// <summary>
/// Computes the euclidean distance between two points a and b.
/// </summary>
__device__ float distSq(float t_a, float t_b, float p_a, float p_b) {
	return (t_a - t_b)*(t_a - t_b) + (p_a - p_b)*(p_a - p_b);
}

/// <summary>
/// Computes a semigaussian for the specified distance value.
/// </summary>
/// <param name="dist">The distance value.</param>
__device__ float gaussian(float dist) {
	return expf(-2.f*dist) * INVSQRTPI;
}

/// <summary>
/// Calculates the redshift for the specified theta-phi on the camera sky.
/// </summary>
/// <param name="theta">The theta of the position on the camera sky.</param>
/// <param name="phi">The phi of the position on the camera sky.</param>
/// <param name="cam">The camera parameters.</param>
/// <returns></returns>
__device__ float redshift(float theta, float phi, const float *cam) {
	float yCam = sin(theta) * sin(phi);
	float part = (1. - cam_speed*yCam);
	float betaPart = sqrt(1 - cam_speed * cam_speed) / part;

	float yFido = (-yCam + cam_speed) / part;
	float eF = 1. / (cam_alpha + cam_w * cam_wbar * yFido);
	float b = eF * cam_wbar * yFido;

	return 1. / (betaPart * (1 - b*cam_w) / cam_alpha);
}

/// <summary>
/// Searches where in the star tree the bounding box of the projected pixel falls.
/// </summary>
/// <param name="tree">The binary star tree.</param>
/// <param name="thphiPixMin">The minimum theta and phi values of the bounding box.</param>
/// <param name="thphiPixMax">The maximum theta and phi values of the bounding box.</param>
/// <returns></returns>
__device__ int searchTree(const int *tree, const float *thphiPixMin, const float *thphiPixMax, const int treeLevel) {
	float nodeStart[2] = { 0.f, 0.f };
	float nodeSize[2] = { PI, PI2 };
	int node = 0;

	for (int level = 1; level <= treeLevel; level++) {
		int star_n = tree[node];
		if (node != 0 && ((node + 1) & node) != 0) {
			star_n -= tree[node - 1];
		}
		if (star_n == 0) break;
		int tp = level & 1;
		nodeSize[tp] = nodeSize[tp] * .5f;

		float check = nodeStart[tp] + nodeSize[tp];
		bool lu = thphiPixMin[tp] < check;
		bool rd = thphiPixMax[tp] >= check;

		if (lu && rd) {
			break;
		}
		else if (lu) node = node * 2 + 1;
		else if (rd) {
			node = node * 2 + 2;
			nodeStart[tp] = nodeStart[tp] + nodeSize[tp];
		}
	}
	return node;
}

__global__ void checkStar(float3 *starLight, const float *stars, const float *camParam, const float *magnitude, 
							const int starInd, const int num, const int sgn, const float2 size, const float2 offset, 
							const int step, const float frac, const int pixnum, const float* t, const float *p) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int filterW = step * 2 + 1;
	float maxDistSq = (step + .5f)*(step + .5f);
	//atomicAdd(&starLight[pixnum].z, 1.f);
	if (i < num) {
		float start = stars[2 * i + starInd];//[q].x;
		float starp = stars[2 * i + 1 + starInd];//[q].y;
		if (starInPolygon(t, p, start, starp, sgn)) {
			interpolate(t[0], t[1], t[2], t[3], p[0], p[1], p[2], p[3], start, starp, sgn);
	
			float thetaCam = offset.x + size.x * start;
			float phiCam = offset.y + size.y * starp;
	
			float redshft = redshift(thetaCam, phiCam, camParam);
			float part = -0.4f * magnitude[i * 2 + starInd] - 1.6f * log10f(redshft);
			float temp = 46.f / redshft * ((1.f / ((0.92f * magnitude[2 * i + 1 + starInd]) + 1.7)) + (1.f / ((0.92f * magnitude[2 * i + 1 + starInd]) + 0.62f))) - 10.f;
			int index = max(0, min((int)floorf(temp), 1170));
			float3 rgb = { tempToRGB[3 * index] * tempToRGB[3 * index], tempToRGB[3 * index + 1] * tempToRGB[3 * index + 1], tempToRGB[3 * index + 2] * tempToRGB[3 * index + 2] };
			float H, S, P;
	
			for (int u = 0; u <= 2 * step; u++) {
				for (int v = 0; v <= 2 * step; v++) {
					float dist = distSq(-step + u + .5f, start, -step + v + .5f, starp);
					if (dist > maxDistSq) continue;
					else {
						float appMag = part + log10f(frac*gaussian(dist));
						float brightness = exp10f(appMag);
						atomicAdd(&starLight[filterW*filterW * pixnum + filterW * u + v].x, brightness*rgb.x);
						atomicAdd(&starLight[filterW*filterW * pixnum + filterW * u + v].y, brightness*rgb.y);
						atomicAdd(&starLight[filterW*filterW * pixnum + filterW * u + v].z, brightness*rgb.z);
						//starLight[filterW*filterW * pixnum + filterW * u + v].x += brightness*rgb.x;// *rgb.x*255.f*255.f;
						//starLight[filterW*filterW * pixnum + filterW * u + v].y += brightness*rgb.y;// *rgb.y*255.f*255.f;
						//starLight[filterW*filterW * pixnum + filterW * u + v].z += brightness*rgb.z;// *rgb.z*255.f*255.f;
					}
				}
			}
		}
	}
}

__global__ void makeImageKernel(float3 *starLight, const float2 *thphi, const int *pi, const float *hor, const float *ver,
	const float *stars, const int *tree, const int starSize, const float *camParam, const float *magnitude, const int treeLevel,
	const bool symmetry, const int M, const int N, const int step, float offset, float *ts, float *ps, int startN, int num, int sgn, 
	float2 size, float2 offsetpix, float frac, int i_j) {

	/*----- SHARED MEMORY INITIALIZATION -----*/
	#pragma region shared_mem
	//__shared__ int tree[TREESIZE];
	//int lidx = threadIdx.x *blockDim.y + threadIdx.y;
	//while (lidx < TREESIZE) {
	//	tree[lidx] = _tree[lidx];
	//	lidx += blockDim.y * blockDim.x;
	//}
	//
	//__shared__ float s_tp[SHARE];
	//int idx = threadIdx.x * blockDim.y + threadIdx.y;
	//int blockStart = (blockIdx.x * blockDim.x) * (M + 1) + (blockIdx.y * blockDim.y);
	//float raster = (float)RASTER_W;
	//while (idx < SHARE / 2) {
	//	float idxf = idx + 0.000001;
	//	int mod = (int) fmod(idxf, raster);
	//	int ind = 2 * (blockStart + mod + (M + 1) * (int)(idxf / raster));
	//	s_tp[2 * idx] = thphi[ind];
	//	s_tp[2 * idx + 1] = thphi[ind + 1];
	//	idx += blockDim.y * blockDim.x;
	//}

	//__syncthreads();
	#pragma endregion

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	// Only compute if pixel is not black hole.
	int pibh = (symmetry && i >= N / 2) ? pi[(N - 1 - i)*M + j] : pi[ij];
	if (pibh >= 0) {

		// Set values for projected pixel corners & update phi values in case of 2pi crossing.
		#pragma region Retrieve pixel corners
		float t[4], p[4];
		if (symmetry && i >= N / 2) {
			int ix = N - 1 - i;
			int ind = ix * M1 + j;
			p[0] = thphi[ind].y;
			p[1] = thphi[ind + M1].y;
			p[2] = thphi[ind + M1 + 1].y;
			p[3] = thphi[ind + 1].y;
			t[0] = PI - thphi[ind].x;
			t[1] = PI - thphi[ind + M1].x;
			t[2] = PI - thphi[ind + M1 + 1].x;
			t[3] = PI - thphi[ind + 1].x;
		}
		else {
			int ind = i * M1 + j;
			t[0] = thphi[ind + M1].x;
			t[1] = thphi[ind].x;
			t[2] = thphi[ind + 1].x;
			t[3] = thphi[ind + M1 + 1].x;
			p[0] = thphi[ind + M1].y;
			p[1] = thphi[ind].y;
			p[2] = thphi[ind + 1].y;
			p[3] = thphi[ind + M1 + 1].y;
		}

		int crossed2pi = 0;
		#pragma unroll
		for (int q = 0; q < 4; q++) {
			p[q] += offset;
			if (p[q] > PI2) {
				p[q] = fmodf(p[q], PI2);
				crossed2pi++;
			}
		}
		// Check and correct for 2pi crossings.
		bool picheck = false;
		if (pibh > 0 || (crossed2pi > 0 && crossed2pi < 4)) {
			picheck = piCheck(p, .2f);
		}

		//int idx2 = 2 * (threadIdx.x * RASTER_W + threadIdx.y);
		//int R2 = RASTER_W * 2;
		//float t[4] = { s_tp[idx2 + R2],		s_tp[idx2],		s_tp[idx2 + 2],	s_tp[idx2 + R2 + 2] };
		//float p[4] = { s_tp[idx2 + R2 + 1], s_tp[idx2 + 1], s_tp[idx2 + 3], s_tp[idx2 + R2 + 3] };

		#pragma endregion

		// Calculate orientation and size of projected polygon (positive -> CW, negative -> CCW)
		float orient = (t[1] - t[0]) * (p[1] + p[0]) + (t[2] - t[1]) * (p[2] + p[1]) +
					   (t[3] - t[2]) * (p[3] + p[2]) + (t[0] - t[3]) * (p[0] + p[3]);
		sgn = orient < 0 ? -1 : 1;

		// Search where in the star tree the bounding box of the projected pixel falls.
		const float thphiPixMax[2] = { max(max(t[0], t[1]), max(t[2], t[3])),
									   max(max(p[0], p[1]), max(p[2], p[3])) };
		const float thphiPixMin[2] = { min(min(t[0], t[1]), min(t[2], t[3])),
									   min(min(p[0], p[1]), min(p[2], p[3])) };

		//float pixImSize = PI / float(imsize.x);
		//int pixMax = int(thphiPixMax[1] / pixImSize);
		//int pixMin = int(thphiPixMin[1] / pixImSize);
		//int pixNum = pixMax - pixMin + 1;
		//int2 minmax[2024];

		float pixVertSize = ver[i + 1] - ver[i];
		float pixHorSize = hor[i + 1] - hor[i];
		float pixSize = pixVertSize * pixHorSize;
		float pixposT = i + .5f;
		float pixposP = j + .5f;
		frac = pixSize / (fabs(orient) * .5f);
		int filterW = step * 2 + 1;
		float maxDistSq = (step + .5f)*(step + .5f);

		#pragma region picheck
		if (picheck) {
			for (int q = 0; q < starSize; q++) {
				float start = stars[2 * q];//[q].x;
				float starp = stars[2 * q + 1];//[q].y;
				bool starInPoly = starInPolygon(t, p, start, starp, sgn);
				if (!starInPoly && starp < PI2 * .2f) {
					starp += PI2;
					starInPoly = starInPolygon(t, p, start, starp, sgn);
				}
				if (starInPoly) {
					interpolate(t[0], t[1], t[2], t[3], p[0], p[1], p[2], p[3], start, starp, sgn);

					float thetaCam = ver[i] + pixVertSize * start;
					float phiCam = hor[j] + pixHorSize * starp;
					float redshft = redshift(thetaCam, phiCam, camParam);
					float part = magnitude[q * 2] + 4.f * log10f(redshft);
					float temp = 46.f / redshft * ((1.f / ((0.92f * magnitude[2 * q + 1]) + 1.7f)) + (1.f / ((0.92f * magnitude[2 * q + 1]) + 0.62f))) - 10.f;
					int index = max(0, min((int)floorf(temp), 1170));
					float3 rgb = { tempToRGB[3 * index] * tempToRGB[3 * index], tempToRGB[3 * index + 1] * tempToRGB[3 * index + 1], tempToRGB[3 * index + 2] * tempToRGB[3 * index + 2] };

					for (int u = 0; u <= 2 * step; u++) {
						for (int v = 0; v <= 2 * step; v++) {
							float dist = distSq(-step + u + .5f, start, -step + v + .5f, starp);
							if (dist > maxDistSq) continue;
							else {
								float appMag = part - 2.5f*log10f(frac*gaussian(dist));
								float brightness = exp10f(-appMag*0.4f);
								starLight[filterW*filterW * ij + filterW * u + v].x += brightness*rgb.x;
								starLight[filterW*filterW * ij + filterW * u + v].y += brightness*rgb.y;
								starLight[filterW*filterW * ij + filterW * u + v].z += brightness*rgb.z;
							}
						}
					}
				}
			}
		}
		#pragma endregion
		else {
			int node = searchTree(tree, thphiPixMin, thphiPixMax, treeLevel);
			// Check set of stars in tree
			int startN = 0;
			if (node != 0 && ((node + 1) & node) != 0) {
				startN = tree[node - 1];
			}
			num = tree[node] - startN;
			//if (num > 10000) {
			//	int threadsPerBlock = 64;
			//	int numBlocks = ((num - 1) / threadsPerBlock + 1);
			//	size = { pixVertSize, pixHorSize };
			//	offsetpix = { pixposT, pixposP };
			//	i_j = ij;
			//	#pragma unroll
			//	for (int q = 0; q < 4; q++) {
			//		ts[q] = t[q];
			//		ps[q] = p[q];
			//	}
			//	checkStar << <numBlocks, threadsPerBlock >> >(starLight, stars, camParam, magnitude, startN, num, sgn, size, offsetpix, step, frac, i_j, ts, ps);
			//}
			//else {
				for (int q = startN; q < tree[node]; q++) {
					float start = stars[2 * q];//[q].x;
					float starp = stars[2 * q + 1];//[q].y;
					if (starInPolygon(t, p, start, starp, sgn)) {
						interpolate(t[0], t[1], t[2], t[3], p[0], p[1], p[2], p[3], start, starp, sgn);

						float thetaCam = ver[i] + pixVertSize * start;
						float phiCam = hor[j] + pixHorSize * starp;

						float redshft = redshift(thetaCam, phiCam, camParam);
						float part = -0.4f * magnitude[q * 2] - 1.6f * log10f(redshft);
						float temp = 46.f / redshft * ((1.f / ((0.92f * magnitude[2 * q + 1]) + 1.7f)) + (1.f / ((0.92f * magnitude[2 * q + 1]) + 0.62f))) - 10.f;
						int index = max(0, min((int)floorf(temp), 1170));
						float3 rgb = { tempToRGB[3 * index] * tempToRGB[3 * index], tempToRGB[3 * index + 1] * tempToRGB[3 * index + 1], tempToRGB[3 * index + 2] * tempToRGB[3 * index + 2] };

						for (int u = 0; u <= 2 * step; u++) {
							for (int v = 0; v <= 2 * step; v++) {
								float dist = distSq(-step + u + .5f, start, -step + v + .5f, starp);
								if (dist > maxDistSq) continue;
								else {
									float appMag = part + log10f(frac*gaussian(dist));
									float brightness = exp10f(appMag);

									starLight[filterW*filterW * ij + filterW * u + v].x += brightness*rgb.x;
									starLight[filterW*filterW * ij + filterW * u + v].y += brightness*rgb.y;
									starLight[filterW*filterW * ij + filterW * u + v].z += brightness*rgb.z;
								}
							}
						}
					}
				}
			//}
		}
	}
}

/// <summary>
/// <returns></returns>
__global__ void makeImageFromImageKernel(const float2 *thphi, uchar4 *out, const int *pi, const int2 imsize, const float4 mean,
										 const bool symmetry, const int M, const int N, float offset, float4* sumTable) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	float4 color = { 0.f, 0.f, 0.f, 0.f };

	// Only compute if pixel is not black hole.
	int pibh = (symmetry && i >= N / 2) ? pi[(N - 1 - i)*M + j] : pi[ij];
	if (pibh >= 0) {
		// Set values for projected pixel corners & update phi values in case of 2pi crossing.
		#pragma region Retrieve pixel corners
		float t[4];
		float p[4];
	
		if (symmetry && i >= N / 2) {
			int ix = N - 1 - i;
			int ind = ix * M1 + j;
			p[0] = thphi[ind].y;
			p[1] = thphi[ind + M1].y;
			p[2] = thphi[ind + M1 + 1].y;
			p[3] = thphi[ind + 1].y;
			t[0] = PI - thphi[ind].x;
			t[1] = PI - thphi[ind + M1].x;
			t[2] = PI - thphi[ind + M1 + 1].x;
			t[3] = PI - thphi[ind + 1].x;
		}
		else {
			int ind = i * M1 + j;
			t[0] = thphi[ind + M1].x;
			t[1] = thphi[ind].x;
			t[2] = thphi[ind + 1].x;
			t[3] = thphi[ind + M1 + 1].x;
			p[0] = thphi[ind + M1].y;
			p[1] = thphi[ind].y;
			p[2] = thphi[ind + 1].y;
			p[3] = thphi[ind + M1 + 1].y;
		}

		int crossed2pi = 0;
		#pragma unroll
		for (int q = 0; q < 4; q++) {
			p[q] += offset;
			if (p[q] > PI2) {
				p[q] = fmodf(p[q], PI2);
				crossed2pi++;
			}
		}
		// Check and correct for 2pi crossings.
		bool picheck = false;// piCheck(&p[0], .25f);//false;
		//if (pibh > 0 || (crossed2pi > 0 && crossed2pi < 4)) {
			picheck = piCheck(&p[0], .2f);
		//}
		#pragma endregion
	
		float pixSize = PI / float(imsize.x);
		float phMax = max(max(p[0], p[1]), max(p[2], p[3]));
		float phMin = min(min(p[0], p[1]), min(p[2], p[3]));
		int pixMax = int(phMax / pixSize);
		int pixMin = int(phMin / pixSize);
		int pixNum = pixMax - pixMin + 1;
		int2 minmax[1000];

		if (pixNum < 1000) {
			for (int q = 0; q < pixNum; q++) {
				minmax[q] = { imsize.x + 1, -100 };
			}
			for (int q = 0; q < 4; q++) {
				float ApixP = p[q] / pixSize;
				float BpixP = p[(q + 1) % 4] / pixSize;
				float ApixT = t[q] / pixSize;

				int ApixTi = int(ApixT);
				int ApixP_local = int(ApixP) - pixMin;
				int BpixP_local = int(BpixP) - pixMin;
				int pixSepP = BpixP_local - ApixP_local;

				if (ApixTi > minmax[ApixP_local].y) minmax[ApixP_local].y = ApixTi;
				if (ApixTi < minmax[ApixP_local].x) minmax[ApixP_local].x = ApixTi;

				if (pixSepP*pixSepP > 1) {
					int sgn = pixSepP < 0 ? -1 : 1;
					float BpixT = t[(q + 1) % 4] / pixSize;
					int BpixTi = int(BpixT);

					int pixSepT = abs(ApixTi - BpixTi);
					double slope = float(sgn)*(t[(q + 1) % 4] - t[q]) / (p[(q + 1) % 4] - p[q]);

					int phiSteps = 0;
					int thetaSteps = 0;

					float AposInPixP = fmodf(ApixP, 1.f);
					if (sgn > 0) AposInPixP = 1.f - AposInPixP;
					float AposInPixT = fmodf(ApixT, 1.f);
					while (phiSteps != pixSepP) {
						float alpha = AposInPixP * slope + AposInPixT;
						AposInPixT = alpha;
						if (alpha < 0.f || alpha > 1.f) {
							thetaSteps += (int)floorf(alpha);
							if (ApixTi + thetaSteps > minmax[ApixP_local + phiSteps].y)
								minmax[ApixP_local + phiSteps].y = ApixTi + thetaSteps;
							if (ApixTi + thetaSteps < minmax[ApixP_local + phiSteps].x)
								minmax[ApixP_local + phiSteps].x = ApixTi + thetaSteps;
							AposInPixT -= floorf(alpha);
						}
						phiSteps += sgn;
						if (ApixTi + thetaSteps > minmax[ApixP_local + phiSteps].y)
							minmax[ApixP_local + phiSteps].y = ApixTi + thetaSteps;
						if (ApixTi + thetaSteps < minmax[ApixP_local + phiSteps].x)
							minmax[ApixP_local + phiSteps].x = ApixTi + thetaSteps;

						AposInPixP = 1.f;
					}
				}
			}
			uint pixcount = 0;
			for (int q = 0; q < pixNum; q++) {
				int min = minmax[q].x;
				int max = minmax[q].y;
				pixcount += (max - min + 1);
				int index = max*imsize.y + (pixMin + q) % imsize.y;
				float4 maxColor = sumTable[index];
				index = (min - 1)*imsize.y + (pixMin + q) % imsize.y;
				float4 minColor;
				if (index > 0) minColor = sumTable[index];
				else minColor = { 0.f, 0.f, 0.f, 0.f };
				color.x += maxColor.x - minColor.x;
				color.y += maxColor.y - minColor.y;
				color.z += maxColor.z - minColor.z;

			}
			color.x = 100.f*sqrtf(color.x / float(pixcount));
			color.y = 100.f*sqrtf(color.y / float(pixcount));
			color.z = 100.f*sqrtf(color.z / float(pixcount));

		}
		else {
			color = mean;
		}
	}
	out[ij] = { min(255, int(color.x)), min(255, int(color.y)), min(255, int(color.z)), 255 };
}

__global__ void sumStarLight(float3 *starLight, uchar4 *out, int *bh, int step, int M, int N, int filterW, bool symmetry) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	//out[ij] = { 0, 0, 0 };
	float3 brightness = { 0.f, 0.f, 0.f };
	int start = max(0, step - i);
	int stop = min(2*step, step + N - i - 1);
	for (int u = start; u <= stop; u++) {
		for (int v = 0; v <= 2 * step; v++) {
			brightness.x += starLight[filterW*filterW*((i + u - step)*M + ((j + v - step + M)%M)) + filterW*filterW - (filterW * u + v + 1)].x;
			brightness.y += starLight[filterW*filterW*((i + u - step)*M + ((j + v - step + M)%M)) + filterW*filterW - (filterW * u + v + 1)].y;
			brightness.z += starLight[filterW*filterW*((i + u - step)*M + ((j + v - step + M)%M)) + filterW*filterW - (filterW * u + v + 1)].z;

		}
	}
	out[ij] = { min(255, 20*(int)sqrtf(brightness.x)), min(255, 20*(int)sqrtf(brightness.y)), min(255, 20*(int)sqrtf(brightness.z)) , 255};
}

int makeImage(float *out, const float2 *thphi, const int *pi, const float *ver, const float *hor,
			const float *stars, const int *starTree, const int starSize, const float *camParam, const float *mag, const int treeLevel,
			const bool symmetry, const int M, const int N, const int step, const cv::Mat csImage) {
	cudaError_t cudaStatus = cudaPrep(out, thphi, pi, ver, hor, stars, starTree, starSize, camParam, mag, treeLevel, symmetry, M, N, step, csImage);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}


cudaError_t cudaPrep(float *out, const float2 *thphi, const int *pi, const float *ver, const float *hor,
					const float *stars, const int *tree, const int starSize, const float *camParam, const float *mag, const int treeLevel,
					const bool symmetry, const int M, const int N, const int step, cv::Mat celestImg) {
	#pragma region Set variables
	// Device pointer variables
	float2 *dev_thphi = 0;
	float *dev_st = 0;
	float *dev_cam = 0;
	float *dev_mag = 0;
	uchar4 *dev_img = 0;
	float *dev_out = 0;
	float *dev_hor = 0;
	float *dev_ver = 0;
	int *dev_tree = 0;
	int *dev_pi = 0;
	float4 *dev_sumTable = 0;
	float3 *dev_temp = 0;
	cudaArray* dev_csImg;
	float *dev_t = 0;
	float *dev_p = 0;

	// Image and frame parameters
	int movielength = 1;
	vector<uchar4> image(N*M);
	vector<int> compressionParams;
	compressionParams.push_back(cv::IMWRITE_PNG_COMPRESSION);
	compressionParams.push_back(0);
	cv::Mat summedTable = cv::Mat::zeros(celestImg.size(), cv::DataType<cv::Vec4f>::type);

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	// Size parameters for malloc and memcopy
	int filterW = step * 2 + 1;
	int bhpiSize = symmetry ? M*N / 2 : M*N;
	int rastSize = symmetry ? M1*N1Half : M1*N1;
	int treeSize = (1 << (treeLevel + 1)) - 1;
	int2 imsize = { celestImg.rows, celestImg.cols };

	float4 mean = { 0.f, 0.f, 0.f, 0.f };
	#pragma omp parallel for
	for (int q = 0; q < celestImg.cols; q++) {
		cv::Vec4f prev = { 0.f, 0.f, 0.f, 0.f };
		for (int p = 0; p < celestImg.rows; p++) {
			uchar4 pix = { celestImg.at<uchar3>(p, q).x, celestImg.at<uchar3>(p, q).y, celestImg.at<uchar3>(p, q).z, 255 };// csImage.at<uchar4>(p, q);
			prev.val[0] += pix.x*pix.x*0.0001f;
			prev.val[1] += pix.y*pix.y*0.0001f;
			prev.val[2] += pix.z*pix.z*0.0001f;
			summedTable.at<cv::Vec4f>(p, q) = prev;
		}
	}
	for (int q = 0; q < celestImg.cols; q++) {
		cv::Vec4f pixf = summedTable.at<cv::Vec4f>(imsize.x - 1, q);
		mean.x += pixf.val[0], mean.y += pixf.val[1], mean.z += pixf.val[2];
	}
	mean.x = 100.f*sqrtf(mean.x / float(celestImg.total()));
	mean.y = 100.f*sqrtf(mean.y / float(celestImg.total()));
	mean.z = 100.f*sqrtf(mean.z / float(celestImg.total()));
	
	float* sumTableData = (float*) summedTable.data;
	//cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc< uchar4 >();
	//cudaChannelFormatDesc channelDescSum = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	#pragma endregion

	#pragma region cudaMalloc
	cudaStatus = cudaMalloc((void**)&dev_t, 4 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! t");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_p, 4 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! p");
		goto Error;
	}

	//cudaStatus = cudaMallocArray(&dev_csImg, &channelDescSum, csImage.cols, csImage.rows);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMalloc failed! csimg");
	//	goto Error;
	//}

	cudaStatus = cudaMalloc((void**)&dev_sumTable, imsize.x*imsize.y * sizeof(float4));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! sumtable");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_pi, bhpiSize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! bhpi");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_tree, treeSize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! tree");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_img, N*M* sizeof(uchar4));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! img");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_out, N*M* sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! out");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_temp, M * N * filterW*filterW * sizeof(float3));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! temp");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_thphi, rastSize * sizeof(float2));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! thphi");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_st, starSize * 2 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! stars");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_mag, starSize * 2 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! mag");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_cam, 4 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! cam");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_ver, N1 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! ver");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_hor, M1 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! hor");
		goto Error;
	}
	#pragma endregion

	#pragma region cudaMemcopy Host to Device
	float p[4] = { 0.f, 0.f, 0.f, 0.f };
	float t[4] = { 0.f, 0.f, 0.f, 0.f };
	cudaStatus = cudaMemcpy(dev_p, p, 4 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! bhpi");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_t, t, 4 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! bhpi");
		goto Error;
	}

	//cudaStatus = cudaMemcpyToArray(dev_csImg, 0, 0, (float4*)sumTableData, sizeof(float4)*csImage.cols*csImage.rows, cudaMemcpyHostToDevice);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMemcpy failed! csimg");
	//	goto Error;
	//}

	cudaStatus = cudaMemcpy(dev_sumTable, (float4*)sumTableData, imsize.x*imsize.y * sizeof(float4), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! bhpi");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_pi, pi, bhpiSize * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! bhpi");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_tree, tree, treeSize * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! tree");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_thphi, thphi, rastSize * sizeof(float2), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! thphi ");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_st, stars, starSize * 2 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! stars ");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_mag, mag, starSize * 2 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! mag ");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_cam, camParam, 4 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! cam ");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_ver, ver, N1 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! ver ");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_hor, hor, M1 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! hor ");
		goto Error;
	}
	#pragma endregion

	#pragma region texbinding
	//texRef.addressMode[0] = cudaAddressModeWrap;
	//texRef.addressMode[1] = cudaAddressModeWrap;
	//texRef.filterMode = cudaFilterModePoint;
	//texRef.normalized = false;
	//cudaStatus = cudaBindTextureToArray(texRef, dev_csImg, channelDescSum);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaBindTextureToArray failed!");
	//	goto Error;
	//}
	#pragma endregion

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0.f;

	dim3 threadsPerBlock(TILE_H, TILE_W);
	dim3 numBlocks((N - 1) / threadsPerBlock.x + 1, (M-1) / threadsPerBlock.y + 1);

	for (int q = 0; q < movielength; q++) {
		//vector<float> temp(M * N * filterW*filterW);
		//cudaStatus = cudaMemcpy(dev_temp, &temp[0], M * N * filterW*filterW * sizeof(float), cudaMemcpyHostToDevice);
		//if (cudaStatus != cudaSuccess) {
		//	fprintf(stderr, "cudaMemcpy failed!");
		//	goto Error;
		//}
		cudaEventRecord(start);
		float offset = 1.f*q / 4096.f;
		float2 size = { 0.f, 0.f };
		float2 offsetsize = { 0.f, 0.f };
		makeImageKernel << <numBlocks, threadsPerBlock >> >(dev_temp, dev_thphi, dev_pi, dev_hor, dev_ver,
															dev_st, dev_tree, starSize, dev_cam, dev_mag, treeLevel,
															symmetry, M, N, step, offset, dev_t, dev_p, 0, 0, 0,
															size, offsetsize, 0.f, 0);
		
		sumStarLight << <numBlocks, threadsPerBlock >> >(dev_temp, dev_img, dev_pi, step, M, N, filterW, symmetry);

		//makeImageFromImageKernel << <numBlocks, threadsPerBlock >> >(dev_thphi, dev_img, dev_pi, imsize, mean, symmetry, M, N, offset, dev_sumTable);

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		cout << "time to launch kernel no " << q << ": " << milliseconds << endl;

		#pragma region Check kernel errors
		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching makeImageKernel!\n", cudaStatus);
			goto Error;
		}
		#pragma endregion

		#pragma region cudaMemcopy Device to Host
		auto start_time = std::chrono::high_resolution_clock::now();

		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(&image[0], dev_img, N * M *  sizeof(uchar4), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed! out ");
			goto Error;
		}
		auto end_time = std::chrono::high_resolution_clock::now();
		cout << " time memcpy in microseconds: " << std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() << endl;
		stringstream ss2;
		ss2 << "bhmovie_" << N << "by" << M << "_" << starSize << "star_frame" << q << ".png";
		string imgname = ss2.str();
		cv::Mat img = cv::Mat(N, M, CV_8UC4, (void*)&image[0]);
		cv::Mat out;
		cv::imwrite(imgname, img, compressionParams);
		#pragma endregion
	}
	#pragma region Error-End
	Error:
		//cudaUnbindTexture(texRef);
		//cudaFreeArray(dev_csImg);
		cudaFree(dev_thphi);
		cudaFree(dev_st);
		cudaFree(dev_tree);
		cudaFree(dev_pi);
		cudaFree(dev_out);
		cudaFree(dev_cam);
		cudaFree(dev_mag);
		cudaFree(dev_hor);
		cudaFree(dev_ver);
		cudaFree(dev_temp);
		cudaFree(dev_img);
		cudaDeviceReset();
		return cudaStatus;
	#pragma endregion
}