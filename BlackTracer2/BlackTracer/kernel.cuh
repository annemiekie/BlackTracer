#pragma once

#define HOST __host__
#define DEVICE __device__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Const.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

extern void makeImage(const float2 *thphi, const int *pi, const float *stars, const int *starTree, 
	const int starSize, const float *camParam, const float *magnitude, const int treeLevel, 
	const int M, const int N, const int step, const cv::Mat csImage, const int G, const float gridStart,
	const float gridStep, 	const float *pixsize, const float2 *hit, const float2 *gradient, 
	const float2 *pixImgSize, const int GM, const int GN, const float2 *grid);

void cudaPrep(const float2 *thphi, const int *pi, const float *stars, const int *starTree, 
	const int starSize, const float *camParam, const float *magnitude, const int treeLevel, 
	const int M, const int N, const int step, const cv::Mat csImage, const int G, const float gridStart, 
	const float gridStep, const float *pixsize, const float2 *hit, const float2 *gradient,
	const float2 *pixImgSize, const int GM, const int GN, const float2 *grid);

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
//__device__ float redshift(float theta, float phi, const float *cam) {
//	float xCam = sin(theta)*cos(phi);
//	float zCam = cos(theta);
//	float yCam = sin(theta) * sin(phi);
//	float part = (1. - cam_speed*yCam);
//	float betaPart = sqrt(1 - cam_speed * cam_speed) / part;
//
//	float xFido = -sqrtf(1 - cam_speed*cam_speed) * xCam / part;
//	float zFido = -sqrtf(1 - cam_speed*cam_speed) * zCam / part;
//	float yFido = (-yCam + cam_speed) / part;
//	double k = sqrt(1 - cam_btheta*cam_btheta);
//	float phiFido = -xFido * cam_br / k + cam_bphi*yFido + cam_bphi*cam_btheta / k*zFido;
//
//	float eF = 1. / (cam_alpha + cam_w * cam_wbar * phiFido);
//	float b = eF * cam_wbar * phiFido;
//
//	return 1. / (betaPart * (1 - b*cam_w) / cam_alpha);
//}
inline DEVICE float redshift(float theta, float phi, const float *cam) {

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