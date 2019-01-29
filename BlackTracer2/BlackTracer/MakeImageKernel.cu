#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "kernel.cuh"
#include <iostream>
#include <stdint.h>
#include <iomanip>
#include <algorithm>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>
#include <chrono>

using namespace std;

/* ------------- DEFINITIONS & DECLARATIONS --------------*/
#pragma region define
#define TILE_W 4
#define TILE_H 4
#define PI2 6.283185307179586476f
#define PI 3.141592653589793238f
#define SQRT2PI 2.506628274631f
#define INVSQRTPI 0.318310f

#define ij (i*M+j)
#define N1Half (N/2+1)
#define N1 (N+1)
#define M1 (M+1)
#define cam_speed cam[0]
#define cam_alpha cam[1]
#define cam_w cam[2]
#define cam_wbar cam[3]

#pragma region temprgbArray
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
#pragma endregion

extern void makeImage(float *out, const float2 *thphi, const int *pi, const float *ver, const float *hor,
	const float *stars, const int *starTree, const int starSize, const float *camParam, const float *magnitude, const int treeLevel,
	const bool symmetry, const int M, const int N, const int step, const cv::Mat csImage);

void cudaPrep(float *out, const float2 *thphi, const int *pi, const float *ver, const float *hor,
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
__device__ float gaussian(float dist, int step) {
	float sigma = 1.f/2.5f*powf(2, step-1);
	return expf(-.5f*dist/(sigma*sigma)) * 1.f /(sigma*SQRT2PI);
}

/// <summary>
/// Calculates the redshift (1+z) for the specified theta-phi on the camera sky.
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

#define  Pr  .299f
#define  Pg  .587f
#define  Pb  .114f
/// <summary>
/// public domain function by Darel Rex Finley, 2006
///  This function expects the passed-in values to be on a scale
///  of 0 to 1, and uses that same scale for the return values.
///  See description/examples at alienryderflex.com/hsp.html
/// </summary>
__device__ void RGBtoHSP(float  R, float  G, float  B, float &H, float &S, float &P) {
	//  Calculate the Perceived brightness.
	P = sqrtf(R*R*Pr + G*G*Pg + B*B*Pb);

	//  Calculate the Hue and Saturation.  (This part works
	//  the same way as in the HSV/B and HSL systems???.)
	if (R == G && R == B) {
		H = 0.f; S = 0.f; return;
	}
	if (R >= G && R >= B) {   //  R is largest
		if (B >= G) {
			H = 6.f / 6.f - 1.f / 6.f*(B - G) / (R - G);
			S = 1.f - G / R;
		}
		else {
			H = 0.f / 6.f + 1.f / 6.f*(G - B) / (R - B);
			S = 1.f - B / R;
		}
	}
	else if (G >= R && G >= B) {   //  G is largest
		if (R >= B) {
			H = 2.f / 6.f - 1.f / 6.f*(R - B) / (G - B);
			S = 1.f - B / G;
		}
		else {
			H = 2.f / 6.f + 1.f / 6.f*(B - R) / (G - R);
			S = 1.f - R / G;
		}
	}
	else {   //  B is largest
		if (G >= R) {
			H = 4.f / 6.f - 1.f / 6.f*(G - R) / (B - R);
			S = 1.f - R / B;
		}
		else {
			H = 4.f / 6.f + 1.f / 6.f*(R - G) / (B - G);
			S = 1.f - G / B;
		}
	}
}

//  public domain function by Darel Rex Finley, 2006
//
//  This function expects the passed-in values to be on a scale
//  of 0 to 1, and uses that same scale for the return values.
//
//  Note that some combinations of HSP, even if in the scale
//  0-1, may return RGB values that exceed a value of 1.  For
//  example, if you pass in the HSP color 0,1,1, the result
//  will be the RGB color 2.037,0,0.
//
//  See description/examples at alienryderflex.com/hsp.html
__device__ void HSPtoRGB(float  H, float  S, float  P, float &R, float &G, float &B) {
	float part, minOverMax = 1.f - S;
	if (minOverMax>0.f) {
		if (H<1.f / 6.f) {   //  R>G>B
			H = 6.f*H;
			part = 1.f + H*(1.f / minOverMax - 1.f);
			B = P / sqrtf(Pr / minOverMax / minOverMax + Pg*part*part + Pb);
			R = B / minOverMax;
			G = B + H*(R - B);
		}
		else if (H<2.f / 6.f) {   //  G>R>B
			H = 6.f*(-H + 2.f / 6.f);
			part = 1.f + H*(1.f / minOverMax - 1.f);
			B = P / sqrtf(Pg / minOverMax / minOverMax + Pr*part*part + Pb);
			G = B / minOverMax;
			R = B + H*(G - B);
		}
		else if (H<3.f / 6.f) {   //  G>B>R
			H = 6.f*(H - 2.f / 6.f);
			part = 1.f + H*(1.f / minOverMax - 1.f);
			R = P / sqrtf(Pg / minOverMax / minOverMax + Pb*part*part + Pr);
			G = R / minOverMax;
			B = R + H*(G - R);
		}
		else if (H<4.f / 6.f) {   //  B>G>R
			H = 6.f*(-H + 4.f / 6.f);
			part = 1.f + H*(1.f / minOverMax - 1.f);
			R = P / sqrtf(Pb / minOverMax / minOverMax + Pg*part*part + Pr);
			B = R / minOverMax;
			G = R + H*(B - R);
		}
		else if (H<5.f / 6.f) {   //  B>R>G
			H = 6.f*(H - 4.f / 6.f);
			part = 1.f + H*(1.f / minOverMax - 1.f);
			G = P / sqrtf(Pb / minOverMax / minOverMax + Pr*part*part + Pg);
			B = G / minOverMax;
			R = G + H*(B - G);
		}
		else {   //  R>B>G
			H = 6.f*(-H + 6.f / 6.f);
			part = 1.f + H*(1.f / minOverMax - 1.f);
			G = P / sqrtf(Pr / minOverMax / minOverMax + Pb*part*part + Pg);
			R = G / minOverMax;
			B = G + H*(R - G);
		}
	}
	else {
		if (H<1.f / 6.f) {   //  R>G>B
			H = 6.f*(H);
			R = sqrtf(P*P / (Pr + Pg*H*H));
			G = R*H;
			B = 0.f;
		}
		else if (H<2.f / 6.f) {   //  G>R>B
			H = 6.f*(-H + 2.f / 6.f);
			G = sqrtf(P*P / (Pg + Pr*H*H));
			R = G*H;
			B = 0.f;
		}
		else if (H<.5f) {   //  G>B>R
			H = 6.f*(H - 2.f / 6.f);
			G = sqrtf(P*P / (Pg + Pb*H*H));
			B = G*H;
			R = 0.f;
		}
		else if (H<4.f / 6.f) {   //  B>G>R
			H = 6.f*(-H + 4.f / 6.f);
			B = sqrtf(P*P / (Pb + Pg*H*H));
			G = B*H;
			R = 0.f;
		}
		else if (H<5.f / 6.f) {   //  B>R>G
			H = 6.f*(H - 4.f / 6.f);
			B = sqrtf(P*P / (Pb + Pr*H*H));
			R = B*H;
			G = 0.f;
		}
		else {   //  R>B>G
			H = 6.f*(-H + 1.f);
			R = sqrtf(P*P / (Pr + Pb*H*H));
			B = R*H;
			G = 0.f;
		}
	}
}

/**
* Converts an RGB color value to HSL. Conversion formula
* adapted from http://en.wikipedia.org/wiki/HSL_color_space.
*
* @param   {number}  r       The red color value
* @param   {number}  g       The green color value
* @param   {number}  b       The blue color value
* @return  {Array}           The HSL representation
*/
__device__ void rgbToHsl(float  r, float  g, float  b, float &h, float &s, float &l) {
	float maxv = max(max(r, g), b);
	float minv = min(min(r, g), b);
	h, s, l = (maxv + minv) / 2.f;

	if (maxv == minv) {
		h = s = 0.f; // achromatic
	}
	else {
		float d = maxv - minv;
		s = l > 0.5f ? d / (2 - maxv - minv) : d / (maxv + minv);
		if (maxv == r) h = (g - b) / d + (g < b ? 6.f : 0.f);
		else if (maxv == g) h = (b - r) / d + 2.f;
		else h = (r - g) / d + 4.f;
		h /= 6.f;
	}
}
__device__ float hue2rgb(float p, float q, float t) {
	if (t < 0.f) t += 1.f;
	if (t > 1.f) t -= 1.f;
	if (t < 1.f / 6.f) return p + (q - p) * 6.f * t;
	if (t < 1.f / 2.f) return q;
	if (t < 2.f / 3.f) return p + (q - p) * (2.f / 3.f - t) * 6;
	return p;
}
/**
* Converts an HSL color value to RGB. Conversion formula
* adapted from http://en.wikipedia.org/wiki/HSL_color_space.
*
* @param   {number}  h       The hue
* @param   {number}  s       The saturation
* @param   {number}  l       The lightness
* @return  {Array}           The RGB representation
*/
__device__ void hslToRgb(float h, float s, float l, float &r, float &g, float &b) {
	if (s == 0) {
		r = g = b = l; // achromatic
	}
	else {
		float q = l < 0.5f ? l * (1.f + s) : l + s - l * s;
		float p = 2.f * l - q;
		r = hue2rgb(p, q, h + 1.f / 3.f);
		g = hue2rgb(p, q, h);
		b = hue2rgb(p, q, h - 1.f / 3.f);
	}
}

__device__ float calcArea(float t[4], float p[4]) {
	float x[4], y[4], z[4];
	#pragma unroll
	for (int q = 0; q < 4; q++) {
		float sint = sinf(t[q]);
		x[q] = sint * cosf(p[q]);
		y[q] = sint * sinf(p[q]);
		z[q] = cosf(t[q]);
	}
	float dotpr1 = 1.f;
	dotpr1 += x[0] * x[2] + y[0] * y[2] + z[0] * z[2];
	float dotpr2 = dotpr1;
	dotpr1 += x[2] * x[1] + y[2] * y[1] + z[2] * z[1];
	dotpr1 += x[0] * x[1] + y[0] * y[1] + z[0] * z[1];
	dotpr2 += x[0] * x[3] + y[0] * y[3] + z[0] * z[3];
	dotpr2 += x[2] * x[3] + y[2] * y[3] + z[2] * z[3];
	float triprod1 = fabsf(x[0] * (y[1] * z[2] - y[2] * z[1]) -
						   y[0] * (x[1] * z[2] - x[2] * z[1]) + 
						   z[0] * (x[1] * y[2] - x[2] * y[1]));
	float triprod2 = fabsf(x[0] * (y[2] * z[3] - y[3] * z[2]) -
						   y[0] * (x[2] * z[3] - x[3] * z[2]) +
						   z[0] * (x[2] * y[3] - x[3] * y[2]));
	float area = 2.f*(atanf(triprod1 / dotpr1) + atanf(triprod2 / dotpr2));
	return area;
}

__device__ void searchTree(const int *tree, const float *thphiPixMin, const float *thphiPixMax, const int treeLevel, int *searchNrs, int startNr, int &pos, int picheck) {
	float nodeStart[2] = { 0.f, 0.f +picheck*PI};
	float nodeSize[2] = { PI, PI2 };
	int node = 0;
	uint bitMask = powf(2, treeLevel);
	int level = 0;
	int lvl = 0;
	while (bitMask != 0) {
		bitMask &= ~(1UL << (treeLevel - level));

		for (lvl = level + 1; lvl <= treeLevel; lvl++) {
			int star_n = tree[node];
			if (node != 0 && ((node + 1) & node) != 0) {
				star_n -= tree[node - 1];
			}
			int tp = lvl & 1;

			float x_overlap = max(0.f, min(thphiPixMax[0], nodeStart[0] + nodeSize[0]) -max(thphiPixMin[0], nodeStart[0]));
			float y_overlap = max(0.f, min(thphiPixMax[1], nodeStart[1] + nodeSize[1]) - max(thphiPixMin[1], nodeStart[1]));
			float overlapArea = x_overlap * y_overlap;
			bool size = overlapArea / (nodeSize[0] * nodeSize[1]) > 0.8f;
			nodeSize[tp] = nodeSize[tp] * .5f;
			if (star_n == 0) {
				node = node * 2 + 1; break;
			}

			float check = nodeStart[tp] + nodeSize[tp];
			bool lu = thphiPixMin[tp] < check;
			bool rd = thphiPixMax[tp] >= check;
			if (lvl == 1 && picheck) {
				bool tmp = lu;
				lu = rd;
				rd = tmp;
			}
			if (lvl == treeLevel || (rd && lu && size)) {
				if (rd) {
					searchNrs[startNr + pos] = node * 2 + 2;
					pos++;
				}
				if (lu) {
					searchNrs[startNr + pos] = node * 2 + 1;
					pos++;
				}
				node = node * 2 + 1;
				break;
			}
			else {
				node = node * 2 + 1;
				if (rd) bitMask |= 1UL << (treeLevel - lvl);
				if (!lu) break;
				else if (lvl == 1 && picheck) nodeStart[1] += nodeSize[1];
			}
		}
		level = treeLevel - __ffs(bitMask) + 1;
		if (level >= 0) {
			int diff = lvl - level;
			for (int i = 0; i < diff; i++) {
				int tp = (lvl - i) & 1;
				if (!(node & 1)) nodeStart[tp] -= nodeSize[tp];
				nodeSize[tp] = nodeSize[tp] * 2.f;
				node = (node - 1) / 2;
			}
			node++;
			int tp = level & 1;
			if (picheck && level == 1) nodeStart[tp] -= nodeSize[tp];
			else nodeStart[tp] += nodeSize[tp];
		}
	}
}

__global__ void makeImageKernel(float3 *starLight, const float2 *thphi, const int *pi, const float *hor, const float *ver,
	const float *stars, const int *tree, const int starSize, const float *camParam, const float *magnitude, const int treeLevel,
	const bool symmetry, const int M, const int N, const int step, float offset, int *search, int searchNr) {

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	// Only compute if pixel is not black hole.
	int pibh = (symmetry && i >= N / 2) ? pi[(N - 1 - i)*M + j] : pi[ij];
	if (pibh >= 0) {

		// Set starlight array to zero
		int filterW = step * 2 + 1;
		for (int u = 0; u <= 2 * step; u++) {
			for (int v = 0; v <= 2 * step; v++) {
				starLight[filterW*filterW * ij + filterW * u + v] = { 0.f, 0.f, 0.f };
			}
		}

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
		//if (pibh > 0 || (crossed2pi > 0 && crossed2pi < 4)) {
			picheck = piCheck(p, .2f);
		//}

		#pragma endregion
		
		// Search where in the star tree the bounding box of the projected pixel falls.
		const float thphiPixMax[2] = { max(max(t[0], t[1]), max(t[2], t[3])),
			max(max(p[0], p[1]), max(p[2], p[3])) };
		const float thphiPixMin[2] = { min(min(t[0], t[1]), min(t[2], t[3])),
			min(min(p[0], p[1]), min(p[2], p[3])) };
		int pos = 0;
		int startnr = searchNr*(ij);
		searchTree(tree, thphiPixMin, thphiPixMax, treeLevel, search, startnr, pos, 0);

		if (pos == 0) return;

		// Calculate orientation and size of projected polygon (positive -> CW, negative -> CCW)
		float orient = (t[1] - t[0]) * (p[1] + p[0]) + (t[2] - t[1]) * (p[2] + p[1]) +
			(t[3] - t[2]) * (p[3] + p[2]) + (t[0] - t[3]) * (p[0] + p[3]);
		int sgn = orient < 0 ? -1 : 1;

		float pixVertSize = - ver[i] + ver[i + 1];
		float pixHorSize = - hor[j] + hor[j + 1];
		float solidAngle = calcArea(t, p);
		float ver4[4] = { ver[i], ver[i + 1], ver[i + 1], ver[i] };
		float hor4[4] = { hor[i], hor[i], hor[i + 1], hor[i + 1] };
		float pixArea = calcArea(ver4, hor4);
		float frac = pixArea / solidAngle;
		float maxDistSq = (step + .5f)*(step + .5f);
		float thetaCam = ver[i] + pixVertSize * .5f;
		float phiCam = hor[j] + pixHorSize * .5f;
		float redshft = redshift(thetaCam, phiCam, camParam);
		float red = 4.f * log10f(redshft);

		for (int s = 0; s < pos; s++) {
			int node = search[startnr + s];
			int startN = 0;
			if (node != 0 && ((node + 1) & node) != 0) {
				startN = tree[node - 1];
			}
			for (int q = startN; q < tree[node]; q++) {
				float start = stars[2*q];
				float starp = stars[2*q+1];
				bool starInPoly = starInPolygon(t, p, start, starp, sgn);
				if (picheck && !starInPoly && starp < PI2 * .2f) {
					starp += PI2;
					starInPoly = starInPolygon(t, p, start, starp, sgn);
				}
				if (starInPoly) {
					interpolate(t[0], t[1], t[2], t[3], p[0], p[1], p[2], p[3], start, starp, sgn);

					float part = magnitude[2 * q] + red;
					float temp = 46.f / redshft * ((1.f / ((0.92f * magnitude[2*q+1]) + 1.7f)) + 
												   (1.f / ((0.92f * magnitude[2*q+1]) + 0.62f))) - 10.f;
					int index = max(0, min((int)floorf(temp), 1170));
					float3 rgb = { tempToRGB[3 * index] * tempToRGB[3 * index],
								   tempToRGB[3 * index + 1] * tempToRGB[3 * index + 1],
								   tempToRGB[3 * index + 2] * tempToRGB[3 * index + 2] };

					for (int u = 0; u <= 2 * step; u++) {
						for (int v = 0; v <= 2 * step; v++) {
							float dist = distSq(-step + u + .5f, start, -step + v + .5f, starp);
							if (dist > maxDistSq) continue;
							else {
								float appMag = part - 2.5f * log10f(frac * gaussian(dist, step));
								float brightness = 15.f * 15.f * exp10f(-.4f * appMag);

								starLight[filterW*filterW * ij + filterW * u + v].x += brightness*rgb.z;
								starLight[filterW*filterW * ij + filterW * u + v].y += brightness*rgb.y;
								starLight[filterW*filterW * ij + filterW * u + v].z += brightness*rgb.x;
							}
						}
					}
				}
			}
		}
	}
}

__global__ void makeImageFromImageKernel(const float2 *thphi, uchar4 *out, const int *pi, const int2 imsize,
										const bool symmetry, const int M, const int N, float offset, float4* sumTable, 
										const float *hor, const float *ver, const float *camParam, 
										int minmaxSize, int2 *minmaxx) {

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
			t[0] = PI - thphi[ind].x;
			p[0] = thphi[ind].y;
			t[3] = PI - thphi[ind + 1].x;
			p[3] = thphi[ind + 1].y;
			t[1] = PI - thphi[ind + M1].x;
			p[1] = thphi[ind + M1].y;
			t[2] = PI - thphi[ind + M1 + 1].x;
			p[2] = thphi[ind + M1 + 1].y;
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

		#pragma unroll
		for (int q = 0; q < 4; q++) {
			p[q] += offset;
			if (p[q] > PI2) {
				p[q] = fmodf(p[q], PI2);
			}
		}
		// Check and correct for 2pi crossings.
		bool picheck = false;
		picheck = piCheck(&p[0], .2f);
		#pragma endregion
	
		float pixSize = PI / float(imsize.x);
		float phMax = max(max(p[0], p[1]), max(p[2], p[3]));
		float phMin = min(min(p[0], p[1]), min(p[2], p[3]));
		int pixMax = int(phMax / pixSize);
		int pixMin = int(phMin / pixSize);
		int pixNum = pixMax - pixMin + 1;
		uint pixcount = 0;
		int maxsize = minmaxSize;

		if (imsize.y > 2000) {
			minmaxSize = 0;
			maxsize = 1000;
		}
 
		if (pixNum < maxsize) {
			int2 minmax[1000];

			int minmaxPos = ij*minmaxSize;
			for (int q = 0; q < pixNum; q++) {
				minmax[ij*minmaxSize + q] = { imsize.x + 1, -100 };
			}
			for (int q = 0; q < 4; q++) {
				float ApixP = p[q] / pixSize;
				float BpixP = p[(q + 1) % 4] / pixSize;
				float ApixT = t[q] / pixSize;

				int ApixTi = int(ApixT);
				int ApixP_local = int(ApixP) - pixMin;
				int BpixP_local = int(BpixP) - pixMin;
				int pixSepP = BpixP_local - ApixP_local;

				if (ApixTi > minmax[minmaxPos + ApixP_local].y) minmax[minmaxPos + ApixP_local].y = ApixTi;
				if (ApixTi < minmax[minmaxPos + ApixP_local].x) minmax[minmaxPos + ApixP_local].x = ApixTi;

				if (pixSepP > 0) {
					int sgn = pixSepP < 0 ? -1 : 1;
					float BpixT = t[(q + 1) % 4] / pixSize;
					int BpixTi = int(BpixT);

					int pixSepT = abs(ApixTi - BpixTi);
					float slope = float(sgn)*(t[(q + 1) % 4] - t[q]) / (p[(q + 1) % 4] - p[q]);

					int phiSteps = 0;
					int thetaSteps = 0;

					float AposInPixP = ApixP - float((int)ApixP);
					if (sgn > 0) AposInPixP = 1.f - AposInPixP;
					float AposInPixT = ApixT - (float)ApixTi;
					while (phiSteps < pixSepP) {
						float alpha = AposInPixP * slope + AposInPixT;
						AposInPixT = alpha;
						int pixT, pixPpos;
						if (alpha < 0.f || alpha > 1.f) {
							thetaSteps += (int)floorf(alpha);
							pixT = ApixTi + thetaSteps;
							pixPpos = minmaxPos + ApixP_local + phiSteps;
							if (pixT > minmax[pixPpos].y) minmax[pixPpos].y = pixT;
							if (pixT < minmax[pixPpos].x) minmax[pixPpos].x = pixT;
							AposInPixT -= floorf(alpha);
						}
						phiSteps += sgn;
						pixT = ApixTi + thetaSteps;
						pixPpos = minmaxPos + ApixP_local + phiSteps;
						if (pixT > minmax[pixPpos].y) minmax[pixPpos].y = pixT;
						if (pixT < minmax[pixPpos].x) minmax[pixPpos].x = pixT;
						AposInPixP = 1.f;
					}
				}
			}
			for (int q = 0; q < pixNum; q++) {
				int min = minmax[minmaxPos + q].x;
				int max = minmax[minmaxPos + q].y;
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
		}
		else {
			float thMax = max(max(t[0], t[1]), max(t[2], t[3]));
			float thMin = min(min(t[0], t[1]), min(t[2], t[3]));
			int thMaxPix = int(thMax / pixSize);
			int thMinPix = int(thMin / pixSize);
			pixcount = pixNum * (thMaxPix - thMinPix);
			thMaxPix *= imsize.y;
			thMinPix *= imsize.y;
			for (int q = 0; q < pixNum; q++) {
				float4 maxColor = sumTable[thMaxPix + (pixMin + q) % imsize.y];
				float4 minColor = sumTable[thMinPix + (pixMin + q) % imsize.y];
				color.x += maxColor.x - minColor.x;
				color.y += maxColor.y - minColor.y;
				color.z += maxColor.z - minColor.z;
			}
		}
		color.x = min(255.f, 100.f*sqrtf(color.x / float(pixcount)));
		color.y = min(255.f, 100.f*sqrtf(color.y / float(pixcount)));
		color.z = min(255.f, 100.f*sqrtf(color.z / float(pixcount)));

		float H, S, P;
		float pixVertSize = -ver[i] + ver[i + 1];
		float pixHorSize = -hor[j] + hor[j + 1];
		float solidAngle = calcArea(t, p);
		float ver4[4] = { ver[i], ver[i + 1], ver[i + 1], ver[i] };
		float hor4[4] = { hor[i], hor[i], hor[i + 1], hor[i + 1] };
		float pixArea = calcArea(ver4, hor4);
		float redshft = redshift(ver[i] + .5f*pixVertSize, hor[j] + .5f*pixHorSize, camParam);
		RGBtoHSP(color.z/255.f, color.y/255.f, color.x/255.f, H, S, P);
		P *= (pixArea / solidAngle);
		P = redshft < 1.f ? P * 1.f / redshft : powf(P, redshft);
		HSPtoRGB(H, S, min(1.f,P), color.z, color.y, color.x);
	}
	out[ij] = { min(255, int(color.x * 255)), min(255, int(color.y * 255)), min(255, int(color.z * 255)), 255 };
}

__global__ void sumStarLight(float3 *starLight, uchar4 *out, int *bh, int step, int M, int N, int filterW, bool symmetry) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
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

	// check whether the pixel itself (halfway in the list) has a high value (higher than threshold) If yes, mark it.
	// In a second pass do the convolution with the pattern over all the pixels that got marked.

	out[ij] = { min(255, (int)sqrtf(brightness.x)), min(255, (int)sqrtf(brightness.y)), min(255, (int)sqrtf(brightness.z)), 255 };
}



#pragma region glutzooi

// Device pointer variables
float2 *dev_thphi = 0;
float *dev_st = 0;
float *dev_cam = 0;
float *dev_mag = 0;
uchar4 *dev_img = 0;
float *dev_hor = 0;
float *dev_ver = 0;
int *dev_tree = 0;
int *dev_pi = 0;
float4 *dev_sumTable = 0;
float3 *dev_temp = 0;
int *dev_search = 0;
int2 *dev_minmax = 0;

// Other kernel variables
int dev_M, dev_N , dev_minmaxnr;
bool dev_sym;
float offset = 0.f;
int2 dev_imsize;

uchar4 *d_out = 0;
// texture and pixel objects
GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint tex = 0;     // OpenGL texture object
struct cudaGraphicsResource *cuda_pbo_resource;

void render() {
	dim3 threadsPerBlock(TILE_H, TILE_W);
	dim3 numBlocks((dev_N - 1) / threadsPerBlock.x + 1, (dev_M - 1) / threadsPerBlock.y + 1);
	offset += PI2 / (4.f*dev_M);
	makeImageFromImageKernel <<< numBlocks, threadsPerBlock >>>(dev_thphi, d_out, dev_pi, dev_imsize, dev_sym,
																dev_M, dev_N, offset, dev_sumTable, dev_hor, dev_ver, dev_cam,
																dev_minmaxnr, dev_minmax);
	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		glutExit();
	}
}

void drawTexture() {
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, dev_M, dev_N, 0, GL_RGBA,
		GL_UNSIGNED_BYTE, NULL);
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(0, 0);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(0, dev_N);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(dev_M, dev_N);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(dev_M, 0);
	glEnd();
	glDisable(GL_TEXTURE_2D);

}

void display() {
	render();
	drawTexture();
	glutSwapBuffers();
	glutPostRedisplay();
}

void initGLUT(int *argc, char **argv) {
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(dev_M, dev_N);
	glutCreateWindow("whooptiedoe");
	glewInit();
}

void initPixelBuffer() {
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, 4*dev_M*dev_N*sizeof(GLubyte), 0, GL_STREAM_DRAW);
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);

	cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL,	cuda_pbo_resource);
	cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

cudaError_t cleanup() {
	cudaFree(dev_search);
	cudaFree(dev_minmax);
	cudaFree(dev_thphi);
	cudaFree(dev_st);
	cudaFree(dev_tree);
	cudaFree(dev_pi);
	cudaFree(dev_cam);
	cudaFree(dev_mag);
	cudaFree(dev_hor);
	cudaFree(dev_ver);
	cudaFree(dev_temp);
	cudaFree(dev_img);
	cudaError_t cudaStatus = cudaDeviceReset();
	return cudaStatus;
}

void exitfunc() {
	if (pbo) {
		cudaGraphicsUnregisterResource(cuda_pbo_resource);
		glDeleteBuffers(1, &pbo);
		glDeleteTextures(1, &tex);
	}
}

#pragma endregion


void makeImage(float *out, const float2 *thphi, const int *pi, const float *ver, const float *hor,
			const float *stars, const int *starTree, const int starSize, const float *camParam, const float *mag, const int treeLevel,
			const bool symmetry, const int M, const int N, const int step, const cv::Mat csImage) {
	cudaPrep(out, thphi, pi, ver, hor, stars, starTree, starSize, camParam, mag, treeLevel, symmetry, M, N, step, csImage);

	int foo = 1;
	char * bar[1] = { " " };
	initGLUT(&foo, bar);
	gluOrtho2D(0, M, N, 0);
	glutDisplayFunc(display);
	initPixelBuffer();
	glutMainLoop();
	atexit(exitfunc);

	cudaError_t cudaStatus = cleanup();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}
}

void checkCudaStatus(cudaError_t cudaStatus, const char* message) {
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, message);
		printf("\n");
		cleanup();
	}
}

void cudaPrep(float *out, const float2 *thphi, const int *pi, const float *ver, const float *hor,
					const float *stars, const int *tree, const int starSize, const float *camParam, const float *mag, const int treeLevel,
					const bool symmetry, const int M, const int N, const int step, cv::Mat celestImg) {
	#pragma region Set variables
	// Device pointer variables
	//float2 *dev_thphi = 0;
	//float *dev_st = 0;
	//float *dev_cam = 0;
	//float *dev_mag = 0;
	//uchar4 *dev_img = 0;
	//float *dev_hor = 0;
	//float *dev_ver = 0;
	//int *dev_tree = 0;
	//int *dev_pi = 0;
	//float4 *dev_sumTable = 0;
	//float3 *dev_temp = 0;
	//int *dev_search = 0;
	//int2 *dev_minmax = 0;
	//int dev_M, dev_N;
	dev_M = M;
	dev_N = N;
	dev_sym = symmetry;

	// Image and frame parameters
	bool star = false;
	int movielength = 5;
	if (!star) movielength = 1;
	vector<uchar4> image(N*M);
	vector<int> compressionParams;
	compressionParams.push_back(cv::IMWRITE_PNG_COMPRESSION);
	compressionParams.push_back(0);
	cv::Mat summedTable = cv::Mat::zeros(celestImg.size(), cv::DataType<cv::Vec4f>::type);

	// Choose which GPU to run on, change this on a multi-GPU system.
	checkCudaStatus(cudaSetDevice(0), "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");

	// Size parameters for malloc and memcopy
	int filterW = step * 2 + 1;
	int bhpiSize = symmetry ? M*N / 2 : M*N;
	int rastSize = symmetry ? M1*N1Half : M1*N1;
	int treeSize = (1 << (treeLevel + 1)) - 1;
	int searchNr = (int)powf(2, treeLevel / 3 * 2);
	int2 imsize = { celestImg.rows, celestImg.cols };
	dev_minmaxnr = (int)imsize.y / 5.f;
	dev_imsize = imsize;

	#pragma omp parallel for
	for (int q = 0; q < celestImg.cols; q++) {
		cv::Vec4f prev = { 0.f, 0.f, 0.f, 0.f };
		for (int p = 0; p < celestImg.rows; p++) {
			uchar4 pix = { celestImg.at<uchar3>(p, q).x, 
							celestImg.at<uchar3>(p, q).y, 
							celestImg.at<uchar3>(p, q).z, 255 };
			prev.val[0] += pix.x*pix.x*0.0001f;
			prev.val[1] += pix.y*pix.y*0.0001f;
			prev.val[2] += pix.z*pix.z*0.0001f;
			summedTable.at<cv::Vec4f>(p, q) = prev;
		}
	}
	
	float* sumTableData = (float*) summedTable.data;
	#pragma endregion

	#pragma region cudaMalloc
	checkCudaStatus(cudaMalloc((void**)&dev_sumTable, imsize.x*imsize.y * sizeof(float4)),		"cudaMalloc failed! sumtable");
	checkCudaStatus(cudaMalloc((void**)&dev_pi, bhpiSize * sizeof(int)),						"cudaMalloc failed! bhpi");
	checkCudaStatus(cudaMalloc((void**)&dev_tree, treeSize * sizeof(int)),						"cudaMalloc failed! tree");
	checkCudaStatus(cudaMalloc((void**)&dev_search, searchNr * M * N * sizeof(int)),			"cudaMalloc failed! search");
	if (imsize.y < 2000)
		checkCudaStatus(cudaMalloc((void**)&dev_minmax, dev_minmaxnr * M * N * sizeof(int2)),	"cudaMalloc failed! minmax");
	checkCudaStatus(cudaMalloc((void**)&dev_img, N * M * sizeof(uchar4)),						"cudaMalloc failed! img");
	checkCudaStatus(cudaMalloc((void**)&dev_temp, M * N * filterW*filterW * sizeof(float3)),	"cudaMalloc failed! temp");
	checkCudaStatus(cudaMalloc((void**)&dev_thphi, rastSize * sizeof(float2)),					"cudaMalloc failed! thphi");
	checkCudaStatus(cudaMalloc((void**)&dev_st, starSize * 2 * sizeof(float)),					"cudaMalloc failed! stars");
	checkCudaStatus(cudaMalloc((void**)&dev_mag, starSize * 2 * sizeof(float)),					"cudaMalloc failed! mag");
	checkCudaStatus(cudaMalloc((void**)&dev_cam, 4 * sizeof(float)),							"cudaMalloc failed! cam");
	checkCudaStatus(cudaMalloc((void**)&dev_ver, N1 * sizeof(float)),							"cudaMalloc failed! ver");
	checkCudaStatus(cudaMalloc((void**)&dev_hor, M1 * sizeof(float)),							"cudaMalloc failed! hor");
	#pragma endregion

	#pragma region cudaMemcopy Host to Device

	checkCudaStatus(cudaMemcpy(dev_sumTable, (float4*)sumTableData, imsize.x*imsize.y * sizeof(float4), cudaMemcpyHostToDevice),	"cudaMemcpy failed! sumtable");
	checkCudaStatus(cudaMemcpy(dev_pi, pi, bhpiSize * sizeof(int), cudaMemcpyHostToDevice),											"cudaMemcpy failed! bhpi");
	checkCudaStatus(cudaMemcpy(dev_tree, tree, treeSize * sizeof(int), cudaMemcpyHostToDevice),										"cudaMemcpy failed! tree");
	checkCudaStatus(cudaMemcpy(dev_thphi, thphi, rastSize * sizeof(float2), cudaMemcpyHostToDevice),								"cudaMemcpy failed! thphi ");
	checkCudaStatus(cudaMemcpy(dev_st, stars, starSize * 2 * sizeof(float), cudaMemcpyHostToDevice), 								"cudaMemcpy failed! stars ");
	checkCudaStatus(cudaMemcpy(dev_mag, mag, starSize * 2 * sizeof(float), cudaMemcpyHostToDevice), 								"cudaMemcpy failed! mag ");
	checkCudaStatus(cudaMemcpy(dev_cam, camParam, 4 * sizeof(float), cudaMemcpyHostToDevice), 										"cudaMemcpy failed! cam ");
	checkCudaStatus(cudaMemcpy(dev_ver, ver, N1 * sizeof(float), cudaMemcpyHostToDevice), 											"cudaMemcpy failed! ver ");
	checkCudaStatus(cudaMemcpy(dev_hor, hor, M1 * sizeof(float), cudaMemcpyHostToDevice),											"cudaMemcpy failed! hor ");

	#pragma endregion

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0.f;
	
	dim3 threadsPerBlock(TILE_H, TILE_W);
	dim3 numBlocks((N - 1) / threadsPerBlock.x + 1, (M-1) / threadsPerBlock.y + 1);
	
	for (int q = 0; q < movielength; q++) {

		cudaEventRecord(start);
		float offset = PI2*q / (2.f * M);
	
		if (star) {
			makeImageKernel <<<numBlocks, threadsPerBlock>>>(dev_temp, dev_thphi, dev_pi, dev_hor, dev_ver,
																dev_st, dev_tree, starSize, dev_cam, dev_mag, treeLevel,
																symmetry, M, N, step, offset, dev_search, searchNr);
			
			sumStarLight <<<numBlocks, threadsPerBlock>>>(dev_temp, dev_img, dev_pi, step, M, N, filterW, symmetry);
		}
		else {
			makeImageFromImageKernel <<<numBlocks, threadsPerBlock>>>(dev_thphi, dev_img, dev_pi, imsize, symmetry, 
																		 M, N, offset, dev_sumTable, dev_hor, dev_ver, dev_cam,
																		 dev_minmaxnr, dev_minmax);
		}
	
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		cout << "time to launch kernel no " << q << ": " << milliseconds << endl;
	
		#pragma region Check kernel errors
		// Check for any errors launching the kernel
		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			cleanup();
		}
		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching makeImageKernel!\n", cudaStatus);
			cleanup();
		}
		#pragma endregion
	
		#pragma region cudaMemcopy Device to Host
		auto start_time = std::chrono::high_resolution_clock::now();

		// Copy output vector from GPU buffer to host memory.
		checkCudaStatus(cudaMemcpy(&image[0], dev_img, N * M *  sizeof(uchar4), cudaMemcpyDeviceToHost), "cudaMemcpy failed! Dev to Host");
		auto end_time = std::chrono::high_resolution_clock::now();
		cout << " time memcpy in microseconds: " << std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() << endl;
		stringstream ss2;
		star ? ss2 << "bh_" << N << "by" << M << "_" << starSize << "_stars_" << q << ".png" :
			ss2 << "bh_" << N << "by" << M << "_" << celestImg.total() << "_imagex.png";

		string imgname = ss2.str();
		cv::Mat img = cv::Mat(N, M, CV_8UC4, (void*)&image[0]);
		cv::Mat out;
		cv::imwrite(imgname, img, compressionParams);

		#pragma endregion
	}
}
