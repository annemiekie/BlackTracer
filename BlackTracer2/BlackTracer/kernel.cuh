#pragma once

#define HOST __host__
#define DEVICE __device__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Const.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N1 (N+1)
#define M1 (M+1)
#define cam_speed cam[0]
#define cam_alpha cam[1]
#define cam_w cam[2]
#define cam_wbar cam[3]
#define cam_br cam[4]
#define cam_btheta cam[5]
#define cam_bphi cam[6]
#define SQRT2PI 2.506628274631f
#define PI2 6.283185307179586476f
#define PI 3.141592653589793238f

extern void makeImage(const float *stars, const int *starTree, 
	const int starSize, const float *camParam, const float *magnitude, const int treeLevel, 
	const int M, const int N, const int step, const cv::Mat csImage, const int G, const float gridStart,
	const float gridStep, const float2 *hit, 
	const float2 *viewthing, const float viewAngle, const int GM, const int GN, const float2 *grid, const int gridlvl);

void cudaPrep(const float *stars, const int *starTree, 
	const int starSize, const float *camParam, const float *magnitude, const int treeLevel, 
	const int M, const int N, const int step, const cv::Mat csImage, const int G, const float gridStart, 
	const float gridStep, const float2 *hit,
	const float2 *viewthing, const float viewAngle, const int GM, const int GN, const float2 *grid, const int gridlvl);

__device__ float2 interpolateHermite(const int i, const int j, int gap, const int GM, const int GN, const float percDown, const float percRight,
	const int g, float2 *cornersCel, const float2 *grid);

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

/// <summary>
/// Interpolates the corners of a projected pixel on the celestial sky to find the position
/// of a star in the (normal, unprojected) pixel in the output image.
/// </summary>
/// <param name="t0 - t4">The theta values of the projected pixel.</param>
/// <param name="p0 - p4">The phi values of the projected pixel.</param>
/// <param name="start, starp">The star theta and phi.</param>
/// <param name="sgn">The winding order of the polygon + for CW, - for CCW.</param>
/// <returns></returns>
inline DEVICE void interpolate(float t0, float t1, float t2, float t3, float p0, float p1, float p2, float p3,
	float &start, float &starp, int sgn, int i, int j) {
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
	start = starInPixY;
	starp = starInPixX;
}

inline DEVICE void bv2rgb(float &r, float &g, float &b, float bv)    // RGB <0,1> <- BV <-0.4,+2.0> [-]
{
	float t;  r = 0.0; g = 0.0; b = 0.0; if (bv<-0.4) bv = -0.4; if (bv> 2.0) bv = 2.0;
	if ((bv >= -0.40) && (bv<0.00)) {
		t = (bv + 0.40) / (0.00 + 0.40); r = 0.61 + (0.11*t) + (0.1*t*t);
	}
	else if ((bv >= 0.00) && (bv<0.40)) {
		t = (bv - 0.00) / (0.40 - 0.00); r = 0.83 + (0.17*t);
	}
	else if ((bv >= 0.40) && (bv<2.10)) {
		t = (bv - 0.40) / (2.10 - 0.40); r = 1.00;
	}
	if ((bv >= -0.40) && (bv<0.00)) {
		t = (bv + 0.40) / (0.00 + 0.40); g = 0.70 + (0.07*t) + (0.1*t*t);
	}
	else if ((bv >= 0.00) && (bv<0.40)) {
		t = (bv - 0.00) / (0.40 - 0.00); g = 0.87 + (0.11*t);
	}
	else if ((bv >= 0.40) && (bv<1.60)) {
		t = (bv - 0.40) / (1.60 - 0.40); g = 0.98 - (0.16*t);
	}
	else if ((bv >= 1.60) && (bv<2.00)) {
		t = (bv - 1.60) / (2.00 - 1.60); g = 0.82 - (0.5*t*t);
	}
	if ((bv >= -0.40) && (bv<0.40)) {
		t = (bv + 0.40) / (0.40 + 0.40); b = 1.00;
	}
	else if ((bv >= 0.40) && (bv<1.50)) {
		t = (bv - 0.40) / (1.50 - 0.40); b = 1.00 - (0.47*t) + (0.1*t*t);
	}
	else if ((bv >= 1.50) && (bv<1.94)) {
		t = (bv - 1.50) / (1.94 - 1.50); b = 0.63 - (0.6*t*t);
	}
}

/// <summary>
/// Checks if the cross product between two vectors a and b is positive.
/// </summary>
/// <param name="t_a, p_a">Theta and phi of the a vector.</param>
/// <param name="t_b, p_b">Theta of the b vector.</param>
/// <param name="starTheta, starPhi">The star theta and phi.</param>
/// <param name="sgn">The winding order of the polygon + for CW, - for CCW.</param>
/// <returns></returns>
inline DEVICE bool checkCrossProduct(float t_a, float t_b, float p_a, float p_b,
	float starTheta, float starPhi, int sgn) {
	float c1t = (float)sgn * (t_a - t_b);
	float c1p = (float)sgn * (p_a - p_b);
	float c2t = sgn ? starTheta - t_b : starTheta - t_a;
	float c2p = sgn ? starPhi - p_b : starPhi - p_a;
	return (c1t * c2p - c2t * c1p) > 0;
}

/// <summary>
/// Returns if a (star) location lies within the boundaries of the provided polygon.
/// </summary>
/// <param name="t, p">The theta and phi values of the polygon corners.</param>
/// <param name="start, starp">The star theta and phi.</param>
/// <param name="sgn">The winding order of the polygon + for CW, - for CCW.</param>
/// <returns></returns>
inline DEVICE bool starInPolygon(const float *t, const float *p, float start, float starp, int sgn) {
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
inline DEVICE bool piCheck(float *p, float factor) {
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
/// Checks and corrects phi values for 2-pi crossings.
/// </summary>
/// <param name="p">The phi values to check.</param>
/// <param name="factor">The factor to check if a point is close to the border.</param>
/// <returns></returns>
inline DEVICE bool piCheckTot(float2 *tp, float factor, int size) {
	float factor1 = PI2*(1.f - factor);
	bool check = false;
	for (int q = 0; q < size; q++) {
		if (tp[q].y > factor1) {
			check = true;
			break;
		}
	}
	if (!check) return false;
	check = false;
	float factor2 = PI2 * factor;
	for (int q = 0; q < size; q++) {
		if (tp[q].y < factor2) {
			tp[q].y += PI2;
			check = true;
		}
	}
	return check;
}

/// <summary>
/// Computes the euclidean distance between two points a and b.
/// </summary>
inline DEVICE float distSq(float t_a, float t_b, float p_a, float p_b) {
	return (t_a - t_b)*(t_a - t_b) + (p_a - p_b)*(p_a - p_b);
}

/// <summary>
/// Computes a semigaussian for the specified distance value.
/// </summary>
/// <param name="dist">The distance value.</param>
inline DEVICE float gaussian(float dist, int step) {
	float sigma = 1.f / 2.5f*powf(2, step - 1);
	return expf(-.5f*dist / (sigma*sigma)) * 1.f / (sigma*SQRT2PI);
}


/// <summary>
/// Calculates the redshift (1+z) for the specified theta-phi on the camera sky.
/// </summary>
/// <param name="theta">The theta of the position on the camera sky.</param>
/// <param name="phi">The phi of the position on the camera sky.</param>
/// <param name="cam">The camera parameters.</param>
/// <returns></returns>
inline __device__ float redshift(float theta, float phi, const float *cam) {
	float xCam = sinf(theta)*cosf(phi);
	float zCam = cosf(theta);
	float yCam = sinf(theta) * sinf(phi);
	float part = (1.f - cam_speed*yCam);
	float betaPart = sqrtf(1.f - cam_speed * cam_speed) / part;

	float xFido = -sqrtf(1.f - cam_speed*cam_speed) * xCam / part;
	float zFido = -sqrtf(1.f - cam_speed*cam_speed) * zCam / part;
	float yFido = (-yCam + cam_speed) / part;
	float k = sqrtf(1.f - cam_btheta*cam_btheta);
	float phiFido = -xFido * cam_br / k + cam_bphi*yFido + cam_bphi*cam_btheta / k*zFido;

	float eF = 1.f / (cam_alpha + cam_w * cam_wbar * phiFido);
	float b = eF * cam_wbar * phiFido;

	return 1.f / (betaPart * (1.f - b*cam_w) / cam_alpha);
}
//inline DEVICE float redshift(float theta, float phi, const float *cam) {
//
//	float yCam = sin(theta) * sin(phi);
//	float part = (1. - cam_speed*yCam);
//	float betaPart = sqrt(1 - cam_speed * cam_speed) / part;
//
//	float yFido = (-yCam + cam_speed) / part;
//	float eF = 1. / (cam_alpha + cam_w * cam_wbar * yFido);
//	float b = eF * cam_wbar * yFido;
//
//	return 1. / (betaPart * (1 - b*cam_w) / cam_alpha);
//}

#define  Pr  .299f
#define  Pg  .587f
#define  Pb  .114f
/// <summary>
/// public domain function by Darel Rex Finley, 2006
///  This function expects the passed-in values to be on a scale
///  of 0 to 1, and uses that same scale for the return values.
///  See description/examples at alienryderflex.com/hsp.html
/// </summary>
inline DEVICE void RGBtoHSP(float  R, float  G, float  B, float &H, float &S, float &P) {
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
inline DEVICE void HSPtoRGB(float  H, float  S, float  P, float &R, float &G, float &B) {
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
inline DEVICE void rgbToHsl(float  r, float  g, float  b, float &h, float &s, float &l) {
	float maxv = max(max(r, g), b);
	float minv = min(min(r, g), b);
	h = (maxv + minv) / 2.f;
	s = h;
	l = h;

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

inline DEVICE float hue2rgb(float p, float q, float t) {
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
inline DEVICE void hslToRgb(float h, float s, float l, float &r, float &g, float &b) {
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

inline DEVICE HOST float calcArea(float t[4], float p[4]) {
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

inline DEVICE HOST float calcAreax(float t[3], float p[3]) {
	float xi[3], yi[3], zi[3];

#pragma unroll
	for (int q = 0; q < 3; q++) {
		float sint = sinf(t[q]);
		xi[q] = sint * cosf(p[q]);
		yi[q] = sint * sinf(p[q]);
		zi[q] = cosf(t[q]);
	}
	float dot01 = xi[0] * xi[1] + yi[0] * yi[1] + zi[0] * zi[1];
	float dot02 = xi[0] * xi[2] + yi[0] * yi[2] + zi[0] * zi[2];
	float dot12 = xi[2] * xi[1] + yi[2] * yi[1] + zi[2] * zi[1];
	float x[3] = { xi[0], xi[1], xi[2] };
	float y[3] = { yi[0], yi[1], yi[2] };
	float z[3] = { zi[0], zi[1], zi[2] };

	if (dot01 < dot02 && dot01 < dot12) {
		x[0] = xi[2]; x[1] = xi[0]; x[2] = xi[1];
		y[0] = yi[2]; y[1] = yi[0]; y[2] = yi[1];
		z[0] = zi[2]; z[1] = zi[0]; z[2] = zi[1];
	}
	else if (dot02 < dot12 && dot02 < dot01) {
		x[0] = xi[1]; x[1] = xi[2]; x[2] = xi[0];
		y[0] = yi[1]; y[1] = yi[2]; y[2] = yi[0];
		z[0] = zi[1]; z[1] = zi[2]; z[2] = zi[0];
	}

	float dotpr1 = 1.f;
	dotpr1 += x[0] * x[2] + y[0] * y[2] + z[0] * z[2];
	dotpr1 += x[2] * x[1] + y[2] * y[1] + z[2] * z[1];
	dotpr1 += x[0] * x[1] + y[0] * y[1] + z[0] * z[1];
	float triprod1 = fabsf(x[0] * (y[1] * z[2] - y[2] * z[1]) -
		y[0] * (x[1] * z[2] - x[2] * z[1]) +
		z[0] * (x[1] * y[2] - x[2] * y[1]));
	float area = 2.f*(atanf(triprod1 / dotpr1));
	return area;
}

// Set values for projected pixel corners & update phi values in case of 2pi crossing.
inline DEVICE void retrievePixelCorners(const float2 *thphi, float *t, float *p, int &ind, const int M, bool &picheck, float offset) {
	t[0] = thphi[ind + M1].x;
	t[1] = thphi[ind].x;
	t[2] = thphi[ind + 1].x;
	t[3] = thphi[ind + M1 + 1].x;
	p[0] = thphi[ind + M1].y;
	p[1] = thphi[ind].y;
	p[2] = thphi[ind + 1].y;
	p[3] = thphi[ind + M1 + 1].y;

#pragma unroll
	for (int q = 0; q < 4; q++) {
		if (p[q] < 0) {
			ind = -1;
			return;
		}
		p[q] = fmodf(p[q]+offset, PI2);
	}
	// Check and correct for 2pi crossings.
	picheck = piCheck(p, .2f);
}

inline DEVICE void findLensingRedshift(const float *t, const float *p, const int M, const int ind, const float *camParam, 
										const float2 *viewthing, float &frac, float &redshft, float solidAngle) {
	if (solidAngle == 0.f) {
		float th1[3] = { t[0], t[1], t[2] };
		float ph1[3] = { p[0], p[1], p[2] };
		float th2[3] = { t[0], t[2], t[3] };
		float ph2[3] = { p[0], p[2], p[3] };
		solidAngle = calcAreax(th1, ph1) + calcAreax(th2, ph2);
	}

	float ver4[4] = { viewthing[ind].x, viewthing[ind + 1].x, viewthing[ind + M1 + 1].x, viewthing[ind + M1].x };
	float hor4[4] = { viewthing[ind].y, viewthing[ind + 1].y, viewthing[ind + M1 + 1].y, viewthing[ind + M1].y };
	float pixArea = calcArea(ver4, hor4);

	frac = pixArea / solidAngle;
	float thetaCam = (ver4[0] + ver4[1] + ver4[2] + ver4[3]) * .25f;
	float phiCam = (hor4[0] + hor4[1] + hor4[2] + hor4[3]) * .25f;
	redshft = redshift(thetaCam, phiCam, camParam);
}

inline DEVICE void wrapToPi(float &thetaW, float& phiW) {
	thetaW = fmodf(thetaW, PI2);
	while (thetaW < 0.f) thetaW += PI2;
	if (thetaW > PI) {
		thetaW -= 2.f * (thetaW - PI);
		phiW += PI;
	}
	while (phiW < 0.f) phiW += PI2;
	phiW = fmod(phiW, PI2);
}

inline DEVICE void findBlock(const float theta, const float phi, int g, const float2 *grid, const int GM, const int GN, int &i, int &j, int &gap, int level) {

	for (int s = 0; s < level + 1; s++) {
		int ngap = gap / 2;
		int k = i + ngap;
		int l = j + ngap;
		if (gap == 1 || grid[g*GM*GN + k*GM + l].x == -2) return;
		else {
			float thHalf = PI2*k / (1.f * GM);
			float phHalf = PI2*l / (1.f * GM);
			if (thHalf <= theta) i = k;
			if (phHalf <= phi) j = l;
			gap = ngap;
		}
	}
}

inline DEVICE float2 intersection(float ax, float ay, float bx, float by, float cx, float cy, float dx, float dy) {
	// Line AB represented as a1x + b1y = c1 
	double a1 = by - ay;
	double b1 = ax - bx;
	double c1 = a1*(ax)+b1*(ay);

	// Line CD represented as a2x + b2y = c2 
	double a2 = dy - cy;
	double b2 = cx - dx;
	double c2 = a2*(cx)+b2*(cy);

	double determinant = a1*b2 - a2*b1;
	if (determinant == 0) {
		return{ -1, -1 };
	}
	double x = (b2*c1 - b1*c2) / determinant;
	double y = (a1*c2 - a2*c1) / determinant;
	return{ x, y };
}

inline DEVICE float2 interpolateLinear(int i, int j, float percDown, float percRight, float2 *cornersCel) {
	float phi[4] = { cornersCel[0].y, cornersCel[1].y, cornersCel[2].y, cornersCel[3].y };
	float theta[4] = { cornersCel[0].x, cornersCel[1].x, cornersCel[2].x, cornersCel[3].x };

	piCheck(phi, 0.2f);
	float leftT = theta[0] + percDown * (theta[2] - theta[0]);
	float leftP = phi[0] + percDown * (phi[2] - phi[0]);
	float rightT = theta[1] + percDown * (theta[3] - theta[1]);
	float rightP = phi[1] + percDown * (phi[3] - phi[1]);
	float upT = theta[0] + percRight * (theta[1] - theta[0]);
	float upP = phi[0] + percRight * (phi[1] - phi[0]);
	float downT = theta[2] + percRight * (theta[3] - theta[2]);
	float downP = phi[2] + percRight * (phi[3] - phi[2]);
	return intersection(upT, upP, downT, downP, leftT, leftP, rightT, rightP);
}

inline DEVICE float2 hermite(float aValue, float2 const& aX0, float2 const& aX1, float2 const& aX2, float2 const& aX3,
	float aTension, float aBias) {
	/* Source:
	* http://paulbourke.net/miscellaneous/interpolation/
	*/

	float const v = aValue;
	float const v2 = v*v;
	float const v3 = v*v2;

	float const aa = (1.f + aBias)*(1.f - aTension) / 2.f;
	float const bb = (1.f - aBias)*(1.f - aTension) / 2.f;

	float const m0T = aa * (aX1.x - aX0.x) + bb * (aX2.x - aX1.x);
	float const m0P = aa * (aX1.y - aX0.y) + bb * (aX2.y - aX1.y);

	float const m1T = aa * (aX2.x - aX1.x) + bb * (aX3.x - aX2.x);
	float const m1P = aa * (aX2.y - aX1.y) + bb * (aX3.y - aX2.y);

	float const u0 = 2.f *v3 - 3.f*v2 + 1.f;
	float const u1 = v3 - 2.f*v2 + v;
	float const u2 = v3 - v2;
	float const u3 = -2.f*v3 + 3.f*v2;

	return{ u0*aX1.x + u1*m0T + u2*m1T + u3*aX2.x, u0*aX1.y + u1*m0P + u2*m1P + u3*aX2.y };
}

inline DEVICE float2 findPoint(int i, int j, int GM, int GN, int g, int offver, int offhor, int gap, const float2 *grid) {
	float2 gridpt = grid[GM*GN*g + i*GM + j];
	if (gridpt.x == -2 && gridpt.y == -2) {
		int j2 = (j + offhor*gap + GM) % GM;
		int i2 = i + offver*gap;
		float2 ij2 = grid[GM*GN*g + i2*GM + j2];
		if (ij2.x == -1 && ij2.y == -1) return{ -1, -1 };

		else if (ij2.x != -2 && ij2.y != -2) {
			int j0 = (j - offhor * gap + GM) % GM;
			int i0 = (i - offver * gap);

			float2 ij0 = grid[GM*GN*g + i0*GM + j0];
			if (ij0.x < 0) return{ -1, -1 };

			int jprev = (j - 3 * offhor * gap + GM) % GM;
			int jnext = (j + 3 * offhor * gap + GM) % GM;
			int iprev = i - offver * 3 * gap;
			int inext = i + offver * 3 * gap;
			if (offver != 0) {
				if (i2 == 0) {
					jnext = (j0 + GM / 2) % GM;
					inext = i0;
				}
				else if (i0 == GN - 1) {
					jprev = (j0 + GM / 2) % GM;
					iprev = i2;
				}
				else if (i2 == GN - 1) {
					inext = i0;
					jnext = (j0 + GM / 2) % GM;
				}
			}
			float2 ijprev = grid[GM*GN*g + iprev*GM + jprev];
			float2 ijnext = grid[GM*GN*g + inext*GM + jnext];

			if (ijprev.x > -2 && ijnext.x >-2) {
				float2 pt[4] = { ijprev, ij0, ij2, ijnext };
				if (pt[0].x != -1 && pt[3].x != -1) {
					piCheckTot(pt, 0.2f, 4);
					return hermite(0.5f, pt[0], pt[1], pt[2], pt[3], 0.f, 0.f);
				}
			}
			float2 pt[2] = { ij2, ij0 };
			piCheckTot(pt, 0.2f, 2);
			return{ .5f * (pt[0].x + pt[1].x), .5f * (pt[0].y + pt[1].y) };
		}
		else {
			int j0 = (j + gap) % GM;
			int j1 = (j - gap + GM) % GM;
			//if (i - gap < 0) return{ -1, -1 };

			float2 cornersCel2[12];

			cornersCel2[0] = grid[GM*GN*g + (i + gap)*GM + j0];
			cornersCel2[1] = grid[GM*GN*g + (i - gap)*GM + j0];
			cornersCel2[2] = grid[GM*GN*g + (i - gap)*GM + j1];
			cornersCel2[3] = grid[GM*GN*g + (i + gap)*GM + j1];

			for (int q = 0; q < 4; q++) {
				if (cornersCel2[q].x == -1 || cornersCel2[q].x == -2) return{ -1, -1 };
			}
			return interpolateHermite(i - gap, j1, 2 * gap, GM, GN, .5f, .5f, g, cornersCel2, grid);
		}
	}
	return gridpt;
}

inline DEVICE float2 interpolateHermite(const int i, const int j, int gap, const int GM, const int GN, const float percDown, const float percRight,
	const int g, float2 *cornersCel, const float2 *grid) {
	int k = i + gap;
	int l = (j + gap) % GM;
	int imin1 = i - gap;
	int kplus1 = k + gap;
	int jmin1 = (j - gap + GM) % GM;
	int lplus1 = (l + gap) % GM;
	int jx = j;
	int jy = j;
	int lx = l;
	int ly = l;

	if (i == 0) {
		jx = (j + GM / 2) % GM;
		lx = (jx + gap) % GM;
		imin1 = k;
	}
	else if (k == GN - 1) {
		jy = (j + GM / 2) % GM;
		ly = (jy + gap) % GM;
		kplus1 = i;
	}

	cornersCel[4] = findPoint(i, jmin1, GM, GN, g, 0, -1, gap, grid);		//4 upleft
	cornersCel[5] = findPoint(i, lplus1, GM, GN, g, 0, 1, gap, grid);		//5 upright
	cornersCel[6] = findPoint(k, jmin1, GM, GN, g, 0, -1, gap, grid);		//6 downleft
	cornersCel[7] = findPoint(k, lplus1, GM, GN, g, 0, 1, gap, grid);		//7 downright
	cornersCel[8] = findPoint(imin1, jx, GM, GN, g, -1, 0, gap, grid);		//8 lefthigh
	cornersCel[9] = findPoint(imin1, lx, GM, GN, g, -1, 0, gap, grid);		//9 righthigh
	cornersCel[10] = findPoint(kplus1, jy, GM, GN, g, 1, 0, gap, grid);		//10 leftdown
	cornersCel[11] = findPoint(kplus1, ly, GM, GN, g, 1, 0, gap, grid);		//11 rightdown

	for (int q = 4; q < 12; q++) {
		if (cornersCel[q].x == -1) return interpolateLinear(i, j, percDown, percRight, cornersCel);
	}
	piCheckTot(cornersCel, 0.2f, 12);

	float2 interpolateUp = hermite(percRight, cornersCel[4], cornersCel[0], cornersCel[1], cornersCel[5], 0.f, 0.f);
	float2 interpolateDown = hermite(percRight, cornersCel[6], cornersCel[2], cornersCel[3], cornersCel[7], 0.f, 0.f);
	float2 interpolateUpUp = { cornersCel[8].x + (cornersCel[9].x - cornersCel[8].x) * percRight,
		cornersCel[8].y + (cornersCel[9].y - cornersCel[8].y) * percRight };
	float2 interpolateDownDown = { cornersCel[10].x + (cornersCel[11].x - cornersCel[10].x) * percRight,
		cornersCel[10].y + (cornersCel[11].y - cornersCel[10].y) * percRight };
	//HERMITE FINITE
	return hermite(percDown, interpolateUpUp, interpolateUp, interpolateDown, interpolateDownDown, 0.f, 0.f);
}

inline DEVICE float2 interpolateSpline(int i, int j, int gap, int GM, int GN, const float thetaCam, const float phiCam, const int g,
	float2 *cornersCel, float *cornersCam, const float2 * grid) {

	for (int q = 0; q < 4; q++) {
		if (cornersCel[q].x == -1 && cornersCel[q].y == -1) return{ -1.f, -1.f };
	}

	float thetaUp = cornersCam[0];
	float thetaDown = cornersCam[2];
	float phiLeft = cornersCam[1];
	float phiRight = cornersCam[3];

	if (thetaUp == thetaCam) {
		if (phiLeft == phiCam) return cornersCel[0];
		if (phiRight == phiCam) return cornersCel[1];
		if (i == 0.f) return cornersCel[0];

	}
	else if (thetaDown == thetaCam) {
		if (phiLeft == phiCam) return cornersCel[2];
		if (phiRight == phiCam) return cornersCel[3];
	}

	float percDown = (thetaCam - thetaUp) / (thetaDown - thetaUp);
	float percRight = (phiCam - phiLeft) / (phiRight - phiLeft);
	return interpolateHermite(i, j, gap, GM, GN, percDown, percRight, g, cornersCel, grid);
}