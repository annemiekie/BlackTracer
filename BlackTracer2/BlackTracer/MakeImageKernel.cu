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
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
using namespace std;

/* ------------- DEFINITIONS & DECLARATIONS --------------*/
#pragma region define
#define TILE_W 4
#define TILE_H 4

#define ij (i*M+j)
#define N1Half (N/2+1)
#define N1 (N+1)
#define M1 (M+1)
#pragma endregion

__device__ void searchTree(const int *tree, const float *thphiPixMin, const float *thphiPixMax, const int treeLevel, int *searchNrs, int startNr, int &pos, int picheck) {
	float nodeStart[2] = { 0.f, 0.f + picheck*PI };
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

			float x_overlap = max(0.f, min(thphiPixMax[0], nodeStart[0] + nodeSize[0]) - max(thphiPixMin[0], nodeStart[0]));
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

// Set values for projected pixel corners & update phi values in case of 2pi crossing.
__device__ void retrievePixelCorners(const float2 *thphi, float *t, float *p, int &ind, const int M, bool &picheck, float offset) {
	t[0] = thphi[ind + M1].x;
	t[1] = thphi[ind].x;
	t[2] = thphi[ind + 1].x;
	t[3] = thphi[ind + M1 + 1].x;
	p[0] = thphi[ind + M1].y;
	p[1] = thphi[ind].y;
	p[2] = thphi[ind + 1].y;
	p[3] = thphi[ind + M1 + 1].y;

	if (p[0] < 0.f || p[1] < 0.f || p[2] < 0.f || p[3] < 0.f) {
		ind = -1;
		return;
	}

	#pragma unroll
	for (int q = 0; q < 4; q++) {
		if (p[q] < 0) return;
		p[q] += offset;
		if (p[q] > PI2) {
			p[q] = fmodf(p[q], PI2);
		}
	}
	// Check and correct for 2pi crossings.
	picheck = piCheck(p, .2f);
}

__device__ void findLensingRedshift(const float *t, const float *p, const int M, const int ind, const float *camParam, const float2 *viewthing, float &frac, float &redshft) {
	float th1[3] = { t[0], t[1], t[2] };
	float ph1[3] = { p[0], p[1], p[2] };
	float th2[3] = { t[0], t[2], t[3] };
	float ph2[3] = { p[0], p[2], p[3] };
	float solidAngle = calcAreax(th1, ph1) + calcAreax(th2, ph2);

	float ver4[4] = { viewthing[ind].x, viewthing[ind + 1].x, viewthing[ind + M1 + 1].x, viewthing[ind + M1].x };
	float hor4[4] = { viewthing[ind].y, viewthing[ind + 1].y, viewthing[ind + M1 + 1].y, viewthing[ind + M1].y };
	float pixArea = calcArea(ver4, hor4);

	frac = pixArea / solidAngle;
	float thetaCam = (ver4[0] + ver4[1] + ver4[2] + ver4[3]) * .25f;
	float phiCam = (hor4[0] + hor4[1] + hor4[2] + hor4[3]) * .25f;
	redshft = redshift(thetaCam, phiCam, camParam);
}

__device__ void wrapToPi(float &thetaW, float& phiW) {
	thetaW = fmod(thetaW, PI2);
	while (thetaW < 0) thetaW += PI2;
	if (thetaW > PI) {
		thetaW -= 2 * (thetaW - PI);
		phiW += PI;
	}
	while (phiW < 0) phiW += PI2;
	phiW = fmod(phiW, PI2);
}

__device__ void addTrails(const int starsToCheck, const int starSize, const int framenumber, int *stnums, float3 *trail,
						  const float2 *grad, int2 *stCache, const int q, const int i, const int j, const int M, const int trailnum,
						  const float part, const float frac, const float3 rgb) {
	if (starsToCheck < starSize / 100) {
		int cache = framenumber % 2;
		int loc = atomicAdd(&(stnums[q]), 1);
		if (loc < trailnum) stCache[trailnum*cache + 2 * (trailnum * q) + loc] = { i, j };

		float traildist = M;
		float angle = PI2;
		int2 prev;
		int num = -1;
		bool line = false;
		for (int w = 0; w <trailnum; w++) {
			int2 pr = stCache[trailnum*(1 - cache) + 2 * (trailnum * q) + w];
			if (pr.x < 0) break;
			int dx = (pr.x - i);
			int dy = (pr.y - j);
			int dxx = dx*dx;
			int dyy = dy*dy;
			if (dxx <= 1 && dyy <= 1) {
				line = false;
				break;
			}
			float2 gr;
			gr = grad[pr.x*M1 + pr.y];

			float dist = sqrtf(dxx + dyy);
			float div = (dist * sqrtf(gr.x*gr.x + gr.y*gr.y));
			float a1 = acosf((1.f*dx*gr.x + 1.f*dy*gr.y) / div);
			float a2 = acosf((1.f*dx*-gr.x + 1.f*dy*-gr.y) / div);
			float a = min(a1, a2);
			if (a > angle || a > PI*.25f || dist > 80) continue;
			else if (a < angle) {
				angle = a;
				traildist = dist;
				prev = pr;
				num = w;
				line = true;
			}
		}
		if (line) {
			int deltax = i - prev.x;
			int deltay = j - prev.y;
			int sgnDeltaX = deltax < 0 ? -1 : 1;
			int sgnDeltaY = deltay < 0 ? -1 : 1;
			float deltaerr = deltay == 0.f ? fabsf(deltax) : fabsf(deltax / (1.f*deltay));
			float error = 0.f;
			int y = prev.y;
			int x = prev.x;
			while (y != j || x != i) {
				if (error < 1.f) {
					y += sgnDeltaY;
					error += deltaerr;
				}
				if (error >= 0.5f) {
					x += sgnDeltaX;
					error -= 1.f;
				}
				float dist = distSq(x, i, y, j);
				float appMag = part - 2.5f * log10f(frac);
				float brightness = exp10f(-.4f * appMag);
				brightness *= ((traildist - sqrt(dist))*(traildist - sqrt(dist))) / (traildist*traildist);
				trail[x*M + y].x = brightness*rgb.x;
				trail[x*M + y].y = brightness*rgb.y;
				trail[x*M + y].z = brightness*rgb.z;
				if (dist <= 1.f) break;
			}
		}
	}
}

__global__ void distortStarMap(float3 *starLight, const float2 *thphi, const int *pi, const float *stars, const int *tree, 
							   const int starSize, const float *camParam, const float *magnitude, const int treeLevel,
							   const int M, const int N, const int step, float offset, int *search, int searchNr, int2 *stCache, 
							   int *stnums, float3 *trail, int trailnum, float2 *grad, int framenumber, float2 *viewthing) {

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	// Set starlight array to zero
	int filterW = step * 2 + 1;
	for (int u = 0; u <= 2 * step; u++) {
		for (int v = 0; v <= 2 * step; v++) {
			starLight[filterW*filterW * ij + filterW * u + v] = { 0.f, 0.f, 0.f };
		}
	}

	// Only compute if pixel is not black hole.
	if (pi[ij] >= 0) {

		// Set values for projected pixel corners & update phi values in case of 2pi crossing.
		float t[4], p[4];
		int ind = i * M1 + j;
		bool picheck = false;
		retrievePixelCorners(thphi, t, p, ind, M, picheck, offset);
		
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

		// Calculate redshift and lensing effect
		float redshft, frac;
		findLensingRedshift(t, p, M, ind, camParam, viewthing, frac, redshft);
		float red = 4.f * log10f(redshft);
		float maxDistSq = (step + .5f)*(step + .5f);

		// Calculate amount of stars to check
		int starsToCheck = 0;
		for (int s = 0; s < pos; s++) {
			int node = search[startnr + s];
			int startN = 0;
			if (node != 0 && ((node + 1) & node) != 0) {
				startN = tree[node - 1];
			}
			starsToCheck += (tree[node] - startN);
		}

		// Check stars in tree leaves
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
					interpolate(t[0], t[1], t[2], t[3], p[0], p[1], p[2], p[3], start, starp, sgn, i, j);

					float part = magnitude[2 * q] + red;
					float temp = 46.f / redshft * ((1.f / ((0.92f * magnitude[2*q+1]) + 1.7f)) + 
												   (1.f / ((0.92f * magnitude[2*q+1]) + 0.62f))) - 10.f;
					int index = max(0, min((int)floorf(temp), 1170));
					float3 rgb = { tempToRGB[3 * index] * tempToRGB[3 * index],
								   tempToRGB[3 * index + 1] * tempToRGB[3 * index + 1],
								   tempToRGB[3 * index + 2] * tempToRGB[3 * index + 2] };

					addTrails(starsToCheck, starSize, framenumber, stnums, trail, grad, stCache, q, i, j, M, trailnum, part, frac, rgb);

					for (int u = 0; u <= 2 * step; u++) {
						for (int v = 0; v <= 2 * step; v++) {
							float dist = distSq(-step + u + .5f, start, -step + v + .5f, starp);
							if (dist > maxDistSq) continue;
							else {
								float appMag = part - 2.5f * log10f(frac *gaussian(dist, step));
								float brightness = exp10f(-.4f * appMag);
								
								starLight[filterW*filterW * ij + filterW * u + v].x += brightness*rgb.x;
								starLight[filterW*filterW * ij + filterW * u + v].y += brightness*rgb.y;
								starLight[filterW*filterW * ij + filterW * u + v].z += brightness*rgb.z;
							}
						}
					}
				}
			}
		}
	}
}

__device__ void findBlock(float theta, float phi, int level, int g, float2 *grid, int GM, int GN, int &i, int &j, int &gap) {
	int ngap = gap / 2;
	int k = i + ngap;
	int l = j + ngap;
	if (grid[g*GM*GN + k*GM + l].x == -2 || ngap == 0) {
		return;
	}
	else {
		float thHalf = PI2*k / (1.f * GM);
		float phHalf = PI2*l / (1.f * GM);
		if (thHalf <= theta) i = k;
		if (phHalf <= phi) j = l;
		return findBlock(theta, phi, level + 1, g, grid, GM, GN, i, j, ngap);
	}
}

__device__ float2 intersection(float ax, float ay, float bx, float by, float cx, float cy, float dx, float dy) {
	// Line AB represented as a1x + b1y = c1 
	double a1 = by - ay;
	double b1 = ax - bx;
	double c1 = a1*(ax) + b1*(ay);

	// Line CD represented as a2x + b2y = c2 
	double a2 = dy - cy;
	double b2 = cx - dx;
	double c2 = a2*(dx) + b2*(cy);

	double determinant = a1*b2 - a2*b1;
	if (determinant == 0) {
		return{ -1, -1 };
	}
	double x = (b2*c1 - b1*c2) / determinant;
	double y = (a1*c2 - a2*c1) / determinant;
	return{ x, y };
}

__device__ float2 interpolateLinear(int i, int j, float percDown, float percRight, float2 *cornersCel) {
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

__device__ float2 interpolateSpline(int i, int j, float thetaCam, float phiCam, int g, float2 *cornersCel, float *cornersCam) {

	for (int q = 0; q < 4; q++) {
		if (cornersCel[q].x == -1) return{ -1, -1 };
	}

	double thetaUp = cornersCam[0];
	double thetaDown = cornersCam[2];
	double phiLeft = cornersCam[1];
	double phiRight = cornersCam[3];

	if (thetaUp == thetaCam) {
		if (phiLeft == phiCam) return cornersCel[0];
		if (phiRight == phiCam) return cornersCel[1];
		if (i == 0) return cornersCel[0];

	}
	else if (thetaDown == thetaCam) {
		if (phiLeft == phiCam) return cornersCel[2];
		if (phiRight == phiCam) return cornersCel[3];
	}

	double percDown = (thetaCam - thetaUp) / (thetaDown - thetaUp);
	double percRight = (phiCam - phiLeft) / (phiRight - phiLeft);
	//if (splines) return interpolateHermite(ij, percDown, percRight, gridNum, cornersCel);
	//else
	return interpolateLinear(i, j, percDown, percRight, cornersCel);
}

__global__ void pixInterpolation(float2 *viewthing, int *pix, int M, int N, int G, int g, float2 *thphi, float2 *grid, int GM, int GN) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	float theta = viewthing[i*M1 + j].x;
	float phi = viewthing[i*M1 + j].y;

	int lvlstart = 0;
	int half = (phi < PI)? 0 : 1;
	int a = 0;
	int b = half*GM / 2;
	int gap = GM / 4;
	//findBlock(theta, phi, lvlstart, g, grid, GM, GN, a, b, gap);

	//int k = a + gap;
	//int l = b + gap;
	//float factor = PI2 / (1.f*GM);
	//float cornersCam[4] = { factor*a, factor*b, factor*k, factor*l };
	//l = l % GM;
	//float2 cornersCel[4] = { grid[g*GN*GM + a*GM + b], grid[g*GN*GM + a*GM + l], grid[g*GN*GM + k*GM + b], grid[g*GN*GM + k*GM + l] };

	//float2 thphiInter = interpolateSpline(i, j, theta, phi, g, cornersCel, cornersCam);

	//if (thphiInter.x == -1 || thphiInter.y == -1 || theta == -1 || phi == -1) pix[i] = -1;
	//else wrapToPi(thphiInter.x, thphiInter.y);

	thphi[i*M1 + j] = { 0, 0 };// thphiInter;

}

__global__ void clearArrays(int* stnums, int2* stCache, const int frame, const int trailnum, const int starSize) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < starSize) {
		stnums[i] = 0;
		int c = frame % 2;
		for (int q = 0; q < trailnum; q++) {
			stCache[trailnum*(c)+2 * (trailnum * i) + q] = { -1, -1 };
			if (frame == 0) stCache[trailnum*(1-c)+2 * (trailnum * i) + q] = { -1, -1 };

		}
	}
}

__global__ void distortEnvironmentMap(const float2 *thphi, uchar4 *out, const int *pi, const int2 imsize,
									  const int M, const int N, float offset, float4* sumTable, const float *camParam,
									  int minmaxSize, int2 *minmaxPre, float *pixsize, float2 *viewthing) {

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	float4 color = { 0.f, 0.f, 0.f, 0.f };

	// Only compute if pixel is not black hole.
	if (pi[ij] >= 0) {

		float t[4], p[4];
		int ind = i * M1 + j;
		bool picheck = false;
		retrievePixelCorners(thphi, t, p, ind, M, picheck, offset);

		if (ind>0) {

			float pixSize = PI / float(imsize.x);
			float phMax = max(max(p[0], p[1]), max(p[2], p[3]));
			float phMin = min(min(p[0], p[1]), min(p[2], p[3]));
			int pixMax = int(phMax / pixSize);
			int pixMin = int(phMin / pixSize);
			int pixNum = pixMax - pixMin + 1;
			uint pixcount = 0;
			int2 *minmax = 0;
			int maxsize;

			if (imsize.y > 2000 || M > 2000) {
				minmaxSize = 0;
				maxsize = 1000;
			}
			else {
				minmax = &(minmaxPre[0]);
				maxsize = minmaxSize;
			}

			if (pixNum < maxsize) {
				if (imsize.y > 2000 || M > 2000) {
					int2 minmaxPost[1000];
					minmax = &(minmaxPost[0]);
				}
				int minmaxPos = ij*minmaxSize;
				for (int q = 0; q < pixNum; q++) {
					minmax[minmaxPos + q] = { imsize.x + 1, -100 };
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
					int min_ = minmax[minmaxPos + q].x;
					int max_ = minmax[minmaxPos + q].y;
					pixcount += (max_ - min_ + 1);
					int index = max_*imsize.y + (pixMin + q + imsize.y) % imsize.y;
					float4 maxColor = sumTable[index];
					index = max(0,min_ - 1)*imsize.y + (pixMin + q + imsize.y) % imsize.y;
					float4 minColor;
					if (index > 0) minColor = sumTable[index];
					else minColor = { 0.f, 0.f, 0.f, 0.f };
					color.x += maxColor.x - minColor.x;
					color.y += maxColor.y - minColor.y;
					color.z += maxColor.z - minColor.z;
					color.w += maxColor.w - minColor.w;

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
					color.w += maxColor.w - minColor.w;
				}
			}
			color.x = min(255.f, powf(color.x / color.w, 1.f / 2.2f));
			color.y = min(255.f, powf(color.y / color.w, 1.f / 2.2f));
			color.z = min(255.f, powf(color.z / color.w, 1.f / 2.2f));

			float redshft, frac;
			findLensingRedshift(t, p, M, ind, camParam, viewthing, frac, redshft);

			float H, S, P;
			RGBtoHSP(color.z / 255.f, color.y / 255.f, color.x / 255.f, H, S, P);
			P *= frac;
			P = redshft < 1.f ? P * 1.f / redshft : powf(P, redshft);
			HSPtoRGB(H, S, min(1.f, P), color.z, color.y, color.x);
		}
	}
	out[ij] = { min(255, int(color.z * 255)), min(255, int(color.y * 255)), min(255, int(color.x * 255)), 255 };
}

__global__ void sumStarLight(float3 *starLight, float3 *trail, float3 *out, int step, int M, int N, int filterW) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	float3 brightness = { 0.f, 0.f, 0.f };
	int start = max(0, step - i);
	int stop = min(2*step, step + N - i - 1);
	float factor = 100.f;
	for (int u = start; u <= stop; u++) {
		for (int v = 0; v <= 2 * step; v++) {
			brightness.x += factor*starLight[filterW*filterW*((i + u - step)*M + ((j + v - step + M) % M)) + filterW*filterW - (filterW * u + v + 1)].x;
			brightness.y += factor*starLight[filterW*filterW*((i + u - step)*M + ((j + v - step + M) % M)) + filterW*filterW - (filterW * u + v + 1)].y;
			brightness.z += factor*starLight[filterW*filterW*((i + u - step)*M + ((j + v - step + M) % M)) + filterW*filterW - (filterW * u + v + 1)].z;
		}
	}
	float factor2 = 25.f;
	brightness.x += factor2*trail[ij].x;
	brightness.y += factor2*trail[ij].y;
	brightness.z += factor2*trail[ij].z;
	trail[ij] = { 0.f, 0.f, 0.f };
	out[ij] = brightness;
}

__global__ void addDiffraction(float3 *starLight, const int M, const int N, const uchar3 *diffraction, const int filtersize) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	// check whether the pixel itself (halfway in the list) has a high value (higher than threshold) If yes, mark it.
	// In a second pass do the convolution with the pattern over all the pixels that got marked.
	float max = fmaxf(fmaxf(starLight[ij].x, starLight[ij].y), starLight[ij].z);
	if (max > 65025.f) {
		//if (trail[ij].x > 0.f) break;
		int filterhalf = filtersize / 2;
		int startx = 0;
		int endx = filtersize;
		if (i < filterhalf) startx = filterhalf - i;
		if (i >(N - filterhalf)) endx = N - i + filterhalf;
		float div = 5E6f;
		for (int q = startx; q < endx; q++) {
			for (int p = 0; p < filtersize; p++) {
				float3 diff = { starLight[ij].x / div *(float)(diffraction[q * filtersize + p].x* diffraction[q * filtersize + p].x),
					starLight[ij].y / div*(float)(diffraction[q * filtersize + p].y*diffraction[q * filtersize + p].y),
					starLight[ij].z / div*(float)(diffraction[q * filtersize + p].z*diffraction[q * filtersize + p].z) };
				atomicAdd(&(starLight[M * (i - filterhalf + q) + (j - filterhalf + p + M) % M].x), diff.x);
				atomicAdd(&(starLight[M * (i - filterhalf + q) + (j - filterhalf + p + M) % M].y), diff.y);
				atomicAdd(&(starLight[M * (i - filterhalf + q) + (j - filterhalf + p + M) % M].z), diff.z);

			}
		}
	}
}

__global__ void makePix(float3 *starLight, uchar4 *out, int M, int N, float2 *hit) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	//extra
	//float2 h = hit[ij];
	float disk_b = 0.f;
	float disk_g = 0.f;
	float disk_r = 0.f;
	//if (h.x > 0) {
	//	disk_b = 255.f*(h.x - 9.f) / (18.f - 9.f);
	//	disk_g = 255.f - disk_b;// h.y / PI2;
	//}// h.x > 0.f ? 255.f*(1.f - h.y / PI2) : 0.f;
	//if (h.y > PI) printf("%f, %f, %d, %d \n", h.x, h.y, i, j);
	//extra

	float3 sqrt_bright = { sqrtf(starLight[ij].x), sqrtf(starLight[ij].y), sqrtf(starLight[ij].z) };
	float max = fmaxf(fmaxf(sqrt_bright.x, sqrt_bright.y), sqrt_bright.z);

	if (max > 255.f) {
		sqrt_bright.y *= (255.f / max);
		sqrt_bright.z *= (255.f / max);
		sqrt_bright.x *= (255.f / max);
	}
	out[ij] = { min(255, (int)(sqrt_bright.z + disk_b)), min(255, (int)(sqrt_bright.y + disk_g)), min(255, (int)(sqrt_bright.x + disk_r)), 255 };
}

__global__ void addStarsAndBackground(uchar4 *stars, uchar4 *background, int M) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	float3 star = { (float)stars[ij].x * stars[ij].x, (float)stars[ij].y * stars[ij].y, (float)stars[ij].z * stars[ij].z };
	float3 bg = { (float)background[ij].x * background[ij].x, (float)background[ij].y * background[ij].y, (float)background[ij].z * background[ij].z };
	float p = 1.2;
	float3 out = { sqrtf(p*star.x + (2. - p)*bg.x), sqrtf(p*star.y + (2. - p)*bg.y), sqrtf(p*star.z + (2. - p)*bg.z)};
	//float max = fmaxf(fmaxf(out.x, out.y), out.z);
	//if (max > 255.f) {
	//	out.y *= (255.f / max);
	//	out.z *= (255.f / max);
	//	out.x *= (255.f / max);
	//}
	background[ij] = { min((int)out.z, 255), min( (int)out.y, 255), min((int)out.x, 255), 255 };

}

__global__ void stretchGrid(float2* thphi, float2* thphiOut, int M, int N, int start, bool hor, int2 *bbbh, int st_end) {

	int rc = (blockIdx.x * blockDim.x) + threadIdx.x;
	//if (rc == 0) printf("%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d\n", bbbh[0].x, bbbh[0].y, bbbh[1].x, bbbh[1].y, 
	//																		bbbh[2].x, bbbh[2].y, bbbh[3].x, bbbh[3].y, 
	//																		bbbh[4].x, bbbh[4].y, bbbh[5].x, bbbh[5].y);
	if ((hor && rc < N1) || (!hor && rc < M1)) {
		int lookupSize = hor ? bbbh[st_end * 2 + 2].y - bbbh[st_end * 2 + 2].x : bbbh[st_end * 2 + 3].y - bbbh[st_end * 2 + 3].x;
		int gridStart = hor ? bbbh[st_end * 2 + 2].x : bbbh[st_end * 2 + 3].x;
		int newSize = hor ? bbbh[0].y - bbbh[0].x : bbbh[1].y - bbbh[1].x;
		int newStart = hor ? bbbh[0].x : bbbh[1].x;
		int strt = max(0, newStart - int (newSize * 0.5f));
		int stop = newStart + int(newSize * 1.5f);
		int lookupStart = max(0, gridStart - int(lookupSize * 0.5f));
		int lookupStop = gridStart + int(lookupSize * 1.5f);
		float fraction = (lookupStop-lookupStart) / (1.f*(stop-strt));

		float count = 0.f;
		for (int cr = strt; cr < stop; cr++) {
			int lowBound = lookupStart + count;
			int pos = hor ? rc * M1 + lowBound : lowBound * M1 + rc;
			float2 val = thphi[start*M1*N1 + pos];
			hor ? pos++ : pos += M1;
			float2 nextVal = thphi[start*M1*N1 + pos];

			pos = hor ? rc * M1 + cr : cr * M1 + rc;
			if (val.x == -1 || nextVal.x == -1) {
				thphiOut[pos] = { -1, -1 };
			}
			else {
				if (nextVal.y < 0.2f*PI2 && val.y > 0.8f*PI2) 	nextVal.y += PI2;
				if (nextVal.y > 0.8f*PI2 && val.y < 0.2f*PI2) 	val.y += PI2;
				float perc = fmodf(count, 1.f);
				thphiOut[pos] = { (1.f - perc)*val.x + perc*nextVal.x, fmodf((1.f - perc)*val.y + perc*nextVal.y, PI2) };
			}
			count += fraction;
		}
		if (!hor) thphiOut[N*M1 + rc] = thphi[start*M1*N1+N*M1 + rc];
	}
}

__global__ void averageGrids(float2* thphi, float2 *startBlock, float2 *endBlock, float2 *thphiIn, int M, int N, int start, 
							 float perc, int2 *bbbh, float *camIn, float *camParam) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (i == 0 && j == 0) {
		for (int q = 0; q < 7; q++) camIn[q] = (1.f - perc)*camParam[7 * start + q] + perc*camParam[7 * (start + 1) + q];
	}
	if (i < N1 && j < M1) {
		float2 thphiS_block = startBlock[i*M1 + j];
		float2 thphiE_block = endBlock[i*M1 + j];
		if (thphiS_block.x == -1 || thphiE_block.x == -1) thphiIn[i*M1 + j] = { -1, -1 };
		else {
			bool picheck2 = false;
			if (fabsf(thphiS_block.y - thphiE_block.y) > PI) {
				if (thphiS_block.y < 0.2f*PI2 && thphiE_block.y > 0.8f*PI2) {
					picheck2 = true;
					thphiS_block.y += PI2;
				}
				else if (thphiE_block.y < 0.2f*PI2 && thphiS_block.y > 0.8f*PI2) {
					thphiE_block.y += PI2;
					picheck2 = true;
				}
			}
			float2 thphiS = thphi[start*M1*N1 + i*M1 + j];
			float2 thphiE = thphi[(start + 1)*M1*N1 + i*M1 + j];

			bool picheck = false;
			if (fabsf(thphiS.y - thphiE.y) > PI) {
				if (thphiS.y < 0.2f*PI2 && thphiE.y > 0.8f*PI2) {
					picheck = true;
					thphiS.y += PI2;
				}
				else if (thphiE.y < 0.2f*PI2 && thphiS.y > 0.8f*PI2) {
					thphiE.y += PI2;
					picheck = true;
				}
			}
			float a = 0.f;
			float radius_large = (.5f + 0.3f) *(bbbh[0].y - bbbh[0].x);
			float radius_small = (.5f + 0.2f) * (bbbh[0].y - bbbh[0].x);
			int2 center = { (bbbh[1].x + bbbh[1].y) / 2, (bbbh[0].x + bbbh[0].y) / 2 };
			float dist = sqrtf((i - center.x)*(i - center.x) + (j - center.y)*(j - center.y));
			if (dist - radius_large > 0.f) a = 1.f;
			else if (dist - radius_small > 0.f) a = (dist - radius_small) / (radius_large - radius_small);
			float2 average_block = { (1.f - perc)*thphiS_block.x + perc*thphiE_block.x, (1.f - perc)*thphiS_block.y + perc*thphiE_block.y };
			float2 average_rest = { (1.f - perc)*thphiS.x + perc*thphiE.x, (1.f - perc)*thphiS.y + perc*thphiE.y };
			if (picheck) average_rest.y = fmodf(average_rest.y, PI2);
			if (picheck2) average_block.y = fmodf(average_block.y, PI2);
			float2 average = { (1.f - a)*average_block.x + a*average_rest.x, (1.f - a)*average_block.y + a*average_rest.y };
			//float2 average = radius_large < dist ? average_rest : average_block;
			thphiIn[i*M1 + j] = average;
		}
	}

}

__global__ void averageShadow(const int* bh, int* bhIn, const int M, const int N, const int start, const float perc, const bool hor, int2 *bbbh) {
	// This kernel needs to give back the horizontal and vertical max size of the black holes
	// so 4 integer values. As well as the hor and vert size of the new black hole. 2 integers.
	// and starting points for all of these as well.

	int rc = (blockIdx.x * blockDim.x) + threadIdx.x;
	int bhS, bhE;
	int tot = M*N;
	int bhCheck = 0;
	int strt = -1;
	int bhSchange, bhEchange;
	int max = hor ? M : N;
	for (int cr = 0; cr < max; cr++) {
		int i = hor ? rc * M + cr : cr * M + rc;
		bhS = bh[start*tot + i];
		bhE = bh[(start+1)*tot + i];
		if (bhS < 0) {
			if (hor) { atomicMin(&(bbbh[2].x), cr); atomicMax(&(bbbh[2].y), cr);	} /// change to normal min max
			else { atomicMin(&(bbbh[3].x), cr); atomicMax(&(bbbh[3].y), cr); }
		} 
		if (bhE < 0) { 
			if (hor) { atomicMin(&(bbbh[4].x), cr);	atomicMax(&(bbbh[4].y), cr); }
			else { atomicMin(&(bbbh[5].x), cr); atomicMax(&(bbbh[5].y), cr); }
		}
		if (strt < 0) {
			if (bhS == 0 && bhE == 0) {
				bhIn[i] = 0;
			}
			else if (bhS < 0 && bhE < 0) {
				bhCheck = 1;
				bhIn[i] = -1;
			}
			else {
				strt = cr;
				bhSchange = bhS < 0 ? 0 : 1;
				bhEchange = bhE < 0 ? 0 : 1;
			}
		}
		else if ((bhS == 0 && bhE == 0 && bhCheck == 1) || (bhS < 0 && bhE < 0)) {
			int diff = cr - strt - bhCheck;
			float breakpoint = perc * diff;
			int mid = 0;
			if (bhCheck == bhSchange) mid = (int)(strt + breakpoint + 0.5f - bhCheck * 1);
			if (bhCheck == bhEchange) mid = (int)(cr - breakpoint + 0.5f - bhCheck * 1);
			for (int q = strt; q < mid; q++) {
				int pos = hor ? rc * M + q : q * M + rc;
				bhIn[pos] = -bhCheck;
			}
			int pos = hor ? rc * M + mid : mid * M + rc;
			if (hor) { atomicMin(&(bbbh[0].x), mid); atomicMax(&(bbbh[0].y), mid); }
			else { atomicMin(&(bbbh[1].x), mid); atomicMax(&(bbbh[1].y), mid); }

			bhIn[pos] = -1;
			for (int q = mid + 1; q <= cr; q++) {
				int pos = hor ? rc * M + q : q * M + rc;
				bhIn[pos] = bhCheck - 1;
			}
			strt = -1;
		}
	}
}

#pragma region device variables
// Device pointer variables
float2 *dev_thphi = 0;
float2 *dev_in = 0;
float *dev_st = 0;
int2 *dev_stCache = 0;
int *dev_stnums = 0;
float3 *dev_trail = 0;
float *dev_cam = 0;
float *dev_mag = 0;
uchar4 *dev_img = 0;
uchar4 *dev_img2 = 0;
int *dev_tree = 0;
int *dev_pi = 0;
int *dev_bhIn = 0;
float4 *dev_sumTable = 0;
float3 *dev_temp = 0;
int *dev_search = 0;
int2 *dev_minmax = 0;
int2 *dev_bbbh = 0;
float2 *dev_horStretch = 0;
float2 *dev_startBlock = 0;
float2 *dev_endBlock = 0;
float *dev_camIn = 0;
uchar3 *dev_diff = 0;
float3 *dev_starlight = 0;
float *dev_pixsize = 0;
float2 *dev_hit = 0;
float2 *dev_grad = 0;
float2 *dev_gradIn = 0;
float2 *dev_pixImgsize = 0;
float2 *dev_grid = 0;
int *dev_bh = 0;

// Other kernel variables
int dev_M, dev_N, dev_G, dev_minmaxnr;
int dev_diffSize;
float offset = 0.f;
float prec = 0.f;
int2 dev_imsize;
std::vector<int2> bbbh;
int dev_treelvl = 0;
int dev_trailnum = 0;
int dev_searchNr = 0;
int dev_step = 0;
int dev_filterW = 0;
int dev_starSize = 0;

#pragma endregion

cudaError_t cleanup() {
	cudaFree(dev_grid);
	cudaFree(dev_bh);
	cudaFree(dev_horStretch);
	cudaFree(dev_startBlock);
	cudaFree(dev_endBlock);
	cudaFree(dev_bbbh);
	cudaFree(dev_search);
	cudaFree(dev_minmax);
	cudaFree(dev_thphi);
	cudaFree(dev_in);
	cudaFree(dev_bhIn);
	cudaFree(dev_st);
	cudaFree(dev_stCache);
	cudaFree(dev_stnums);
	cudaFree(dev_tree);
	cudaFree(dev_pi);
	cudaFree(dev_cam);
	cudaFree(dev_camIn);
	cudaFree(dev_mag);
	cudaFree(dev_temp);
	cudaFree(dev_img);
	cudaFree(dev_img2);
	cudaFree(dev_pixsize);
	cudaFree(dev_hit);
	cudaFree(dev_trail);
	cudaFree(dev_diff);
	cudaFree(dev_starlight);
	cudaFree(dev_grad);
	cudaFree(dev_gradIn);
	cudaFree(dev_pixImgsize);
	cudaError_t cudaStatus = cudaDeviceReset();
	return cudaStatus;
}

void checkCudaStatus(cudaError_t cudaStatus, const char* message) {
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, message);
		printf("\n");
		cleanup();
	}
}

bool star = false;

#pragma region glutzooi
uchar *sprite = 0;
uchar4 *d_out = 0;
GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint vbo = 0;
GLuint vao = 0;
GLuint tex = 0;     // OpenGL texture object
GLuint diff = 0;
struct cudaGraphicsResource *cuda_pbo_resource;
int g_start_time;
int g_current_frame_number;
int gridInc = 0;
int gridRad = 0;
double radius = 5.f;
double inc = PI / 2.f;
bool moving = false;
bool setup = false;

double moveR = 0.2;
double moveT = PI / 32.;


void setupGrids() {

	if (!setup) {
	//	setup = true;
	//	BlackHole black = BlackHole(0.999);
	//	vector<Grid> grids(20);
	//	for (int q = 0; q < 5; q++) {
	//		double camIncPlus = inc + moveT*q;
//
	//		double camIncMin = inc - moveT*q;
	//		double camRadPlus = radius + moveR*q;
	//		double camRadMin = radius - moveR*q;
//
	//		double angle = (camIncPlus - inc)*PI / (2.f*moveT);
	//		if (angle > PI / 4.) angle = PI1_2 - angle;
	//		double btheta = sin(angle);
	//		double bphi = cos(angle);
	//		double br = btheta;
//
	//		Camera camRP = Camera(inc, 0., camRadPlus, -br, 0., bphi);
	//		grids[q] = Grid(10, 1, true, &camRP, &black);
//
	//		Camera camRM = Camera(inc, 0., camRadMin, br, 0., bphi);
	//		grids[5 + q] = Grid(10, 1, true, &camRM, &black);
//
	//		Camera camTP = Camera(camIncPlus, 0., radius, 0., btheta, bphi);
	//		grids[10 + q] = Grid(10, 1, true, &camTP, &black);
//
	//		Camera camTM = Camera(camIncMin, 0., radius, 0., -btheta, bphi);
	//		grids[15 + q] = Grid(10, 1, true, &camTM, &black);
	//	}
	//	//checkCudaStatus(cudaMemcpy(dev_thphi, thphi, G * (dev_M + 1)*(dev_N + 1) * sizeof(float2), cudaMemcpyHostToDevice), "cudaMemcpy failed! thphi ");
	}
}

void render() {
	float speed = 4.f;
	dim3 threadsPerBlock(TILE_H, TILE_W);
	dim3 numBlocks((dev_N - 1) / threadsPerBlock.x + 1, (dev_M - 1) / threadsPerBlock.y + 1);

	//if (setup && moving) {

	if (dev_G > 1) {
		int NN = dev_N;
		prec += 0.01f;
		prec = fmodf(prec, 5);
		int strt = (int)prec;
		float percentage = fmodf(prec, 1.f);
		int tpb = 32;
		int blocks = NN / tpb;
		averageShadow << < blocks, tpb >> >(dev_pi, dev_bhIn, dev_M, NN, strt, percentage, true, dev_bbbh);
		blocks = dev_M / tpb;
		averageShadow << < blocks, tpb >> >(dev_pi, dev_bhIn, dev_M, NN, strt, percentage, false, dev_bbbh);
		blocks = NN / tpb;
		averageShadow << < blocks, tpb >> >(dev_pi, dev_bhIn, dev_M, NN, strt, percentage, true, dev_bbbh);
		blocks = NN / tpb + 1;
		stretchGrid << <blocks, tpb >> >(dev_thphi, dev_horStretch, dev_M, NN, strt, true, dev_bbbh, 0);
		blocks = dev_M / tpb + 1;
		stretchGrid << <blocks, tpb >> >(dev_horStretch, dev_startBlock, dev_M, NN, 0, false, dev_bbbh, 0);
		blocks = NN / tpb + 1;
		stretchGrid << <blocks, tpb >> >(dev_thphi, dev_horStretch, dev_M, NN, strt + 1, true, dev_bbbh, 1);
		blocks = dev_M / tpb + 1;
		stretchGrid << <blocks, tpb >> >(dev_horStretch, dev_endBlock, dev_M, NN, 0, false, dev_bbbh, 1);
		dim3 numBlocks2(NN / threadsPerBlock.x + 1, dev_M / threadsPerBlock.y + 1);
		averageGrids << < numBlocks2, threadsPerBlock >> >(dev_thphi, dev_startBlock, dev_endBlock, dev_in, dev_M, NN, strt, percentage, dev_bbbh, dev_camIn, dev_cam);
		checkCudaStatus(cudaMemcpy(dev_bbbh, &bbbh[0], 6 * sizeof(int2), cudaMemcpyHostToDevice), "cudaMemcpy failed! bbbh");
		//checkCudaStatus(cudaMemcpy(dev_camIn, &camParam[0], 4 * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy failed! cam");

	}
	//speed = 1.f / camParam[0];

	offset += PI2 / (.25f*speed*dev_M);
	//offset += PI2 / (dev_M);
	offset = fmodf(offset, PI2);
	if (star) {

		int tpb = 32;
		int nb = dev_starSize / tpb + 1;
		clearArrays << < tpb, nb >> > (dev_stnums, dev_stCache, g_current_frame_number, dev_trailnum, dev_starSize);

		distortStarMap << <numBlocks, threadsPerBlock >> >(dev_temp, dev_in, dev_bhIn, dev_st, dev_tree, dev_starSize, 
														   dev_camIn, dev_mag, dev_treelvl, dev_M, dev_N, dev_step,
														   offset, dev_search, dev_searchNr, dev_stCache, dev_stnums, 
														   dev_trail, dev_trailnum, dev_grad, g_current_frame_number, dev_pixImgsize);

		sumStarLight << <numBlocks, threadsPerBlock >> >(dev_temp, dev_trail, dev_starlight, dev_step, dev_M, dev_N, dev_filterW);
		//addDiffraction << <numBlocks, threadsPerBlock >> >(dev_starlight, dev_M, dev_N, dev_diff, dev_diffSize);

		makePix << <numBlocks, threadsPerBlock >> >(dev_starlight, d_out, dev_M, dev_N, dev_hit);
	}
	else {
		distortEnvironmentMap << < numBlocks, threadsPerBlock >> >(dev_in, d_out, dev_bhIn, dev_imsize,
																   dev_M, dev_N, offset, dev_sumTable, dev_camIn,
																   dev_minmaxnr, dev_minmax, dev_pixsize, dev_pixImgsize);
	}
	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		glutExit();
	}
	double end_frame_time, end_rendering_time, waste_time;

	// wait until it is time to draw the current frame
	end_frame_time = g_start_time + (g_current_frame_number + 1) * 15.f;
	end_rendering_time = glutGet(GLUT_ELAPSED_TIME);
	waste_time = end_frame_time - end_rendering_time;
	if (waste_time > 0.0) Sleep(waste_time / 1000.);    // sleep parameter should be in seconds
	// update frame number
	g_current_frame_number++;
}

void processNormalKeys(unsigned char key, int x, int y) {
	if (key == 27)
		exit(0);
	else if (key == '=') {
		if (!moving) {
			gridRad++;
			moving = true;
			setupGrids();
		}
	}
	else if (key == '-') {
		if (!moving) {
			gridRad--;
			moving = true;
		}
	}
}

void processSpecialKeys(int key, int x, int y) {

	switch (key) {
	case GLUT_KEY_UP:
		if (!moving) {
			gridInc--;
			moving = true;
		}
		break;
	case GLUT_KEY_DOWN:
		if (!moving) {
			gridInc++;
			moving = true;
		}
		break;
	}
}

void drawTexture() {
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, dev_M, dev_N, 0, GL_RGBA,
		GL_UNSIGNED_BYTE, NULL);
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(0, 0);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(0, dev_N);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(dev_M, dev_N);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(dev_M, 0);
	glEnd();

	glBindVertexArray(vao);
	glBindTexture(GL_TEXTURE_2D, diff);
	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);

	glPointSize(100.0);
	glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
	glPointParameteri(GL_POINT_SPRITE_COORD_ORIGIN, GL_LOWER_LEFT);
	glEnable(GL_POINT_SPRITE);

	int diffstars = 1;
	glDrawArrays(GL_POINTS, 0, diffstars);

	glBindVertexArray(0);
	glDisable(GL_BLEND);
	glDisable(GL_POINT_SPRITE);
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
	g_start_time = glutGet(GLUT_ELAPSED_TIME);
	g_current_frame_number = 0;
	glewInit();
}

void initTexture() {
	glGenTextures(1, &diff);
	glBindTexture(GL_TEXTURE_2D, diff);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	int x,y,n;
	unsigned char *data = stbi_load("../pic/0.png", &x, &y, &n, STBI_rgb);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8, x, y, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
	glGenerateMipmap(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);

	float sprites[2] = { 700.f, 500.f };
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, 2*sizeof(GLfloat), sprites, GL_STREAM_DRAW);

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);
	glBindVertexArray(0);

	stbi_image_free(data);
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

void exitfunc() {
	if (pbo) {
		cudaGraphicsUnregisterResource(cuda_pbo_resource);
		glDeleteBuffers(1, &pbo);
		glDeleteTextures(1, &tex);
	}
}

#pragma endregion

void makeImage(const float2 *thphi, const int *pi, const float *stars, const int *starTree, 
			   const int starSize, const float *camParam, const float *mag, const int treeLevel,
			   const int M, const int N, const int step, const cv::Mat csImage, 
			   const int G, const float gridStart, const float gridStep, const float *pixsize, 
			   const float2 *hit, const float2 *gradient, const float2 *pixImgSize,
			   const int GM, const int GN, const float2 *grid) {

	cudaPrep(thphi, pi, stars, starTree, starSize, camParam, mag, treeLevel, M, N, step, csImage, 
		     G, gridStart, gridStep, pixsize, hit, gradient, pixImgSize, GM, GN, grid);

	//int foo = 1;
	//char * bar[1] = { " " };
	//initGLUT(&foo, bar);
	//gluOrtho2D(0, M, N, 0);
	//glutDisplayFunc(display);
	//// here are the new entries
	//glutKeyboardFunc(processNormalKeys);
	////glutSpecialFunc(processSpecialKeys);
	//initTexture();
	//initPixelBuffer();
	//glutMainLoop();
	//atexit(exitfunc);

	cudaError_t cudaStatus = cleanup();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}
}

void cudaPrep(const float2 *thphi, const int *pi, const float *stars, const int *tree, const int starSize, 
			  const float *camParam, const float *mag, const int treeLevel, const int M, const int N, 
			  const int step, cv::Mat celestImg, const int G, const float gridStart, const float gridStep, 
			  const float *pixsize, const float2 *hit, const float2 *gradient, const float2 *pixImgSize,
			  const int GM, const int GN, const float2 *grid) {
	#pragma region Set variables
	dev_M = M;
	dev_N = N;
	dev_G = G;
	dev_treelvl = treeLevel;
	dev_step = step;
	dev_starSize = starSize;
	dev_trailnum = 30;

	// Image and frame parameters
	int movielength = 4;
	vector<uchar4> image(N*M);
	vector<int> compressionParams;
	compressionParams.push_back(cv::IMWRITE_PNG_COMPRESSION);
	compressionParams.push_back(0);
	cv::Mat summedTable = cv::Mat::zeros(celestImg.size(), cv::DataType<cv::Vec4f>::type);

	// diffraction image
	int x, y, n;
	unsigned char *diffraction = stbi_load("../pic/0s.png", &x, &y, &n, STBI_rgb);
	dev_diffSize = x;

	// Choose which GPU to run on, change this on a multi-GPU system.
	checkCudaStatus(cudaSetDevice(0), "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");

	// Size parameters for malloc and memcopy
	dev_filterW = step * 2 + 1;
	int bhSize = M*N;
	int hitsize = M*N;
	int rastSize = M1*N1;
	int gridsize = GM*GN;
	int treeSize = (1 << (treeLevel + 1)) - 1;
	dev_searchNr = (int)powf(2, treeLevel / 3 * 2);
	int2 imsize = { celestImg.rows, celestImg.cols };
	dev_minmaxnr = (int) (imsize.y / 5.f);
	dev_imsize = imsize;

	// Everything for zooming & out of orbit
	vector<float2> in(rastSize);
	#pragma omp parallel for
	for (int q = 0; q < rastSize; q++) in[q] = thphi[q];
	vector<int> bhIn(bhSize);
	#pragma omp parallel for
	for (int q = 0; q < bhSize; q++) bhIn[q] = pi[q];
	vector<float2> gradIn(rastSize);

	#pragma omp parallel for
	for (int q = 0; q < rastSize; q++) gradIn[q] = gradient[q];
	vector<float> camIn(7);
	for (int q = 0; q < 7; q++) camIn[q] = camParam[q];

	bbbh = { { M, -1 }, { M, - 1 }, { M, -1 }, { M,- 1 }, { M, -1 }, { M, - 1 } };
	vector<float2> stretch(rastSize);

	// Summed image
	float pxsz = PI2/ (1.f*celestImg.cols);
	#pragma omp parallel for
	for (int q = 0; q < celestImg.cols; q++) {
		cv::Vec4f prev = { 0.f, 0.f, 0.f, 0.f };
		float phi[4] = { pxsz * q, pxsz*q, pxsz * (q+1), pxsz * (q+1) };

		for (int p = 0; p < celestImg.rows; p++) {
			float theta[4] = {pxsz * (p+1), pxsz*p, pxsz*p, pxsz*(p+1)};
			float area = calcArea(theta, phi);
			uchar4 pix = { celestImg.at<uchar3>(p, q).x, 
							celestImg.at<uchar3>(p, q).y, 
							celestImg.at<uchar3>(p, q).z, 255 };
			prev.val[0] += powf(pix.x, 2.2f)* area;
			prev.val[1] += powf(pix.y, 2.2f)* area;
			prev.val[2] += powf(pix.z, 2.2f)* area;
			prev.val[3] += area;
			summedTable.at<cv::Vec4f>(p, q) = prev;
		}
	}	
	float* sumTableData = (float*) summedTable.data;
	#pragma endregion

	#pragma region cudaMalloc
	checkCudaStatus(cudaMalloc((void**)&dev_sumTable, imsize.x*imsize.y * sizeof(float4)),		"cudaMalloc failed! sumtable");
	checkCudaStatus(cudaMalloc((void**)&dev_pi, G * bhSize * sizeof(int)),						"cudaMalloc failed! bhpi");
	checkCudaStatus(cudaMalloc((void**)&dev_bh, gridsize * sizeof(int)),						"cudaMalloc failed! bh");
	checkCudaStatus(cudaMalloc((void**)&dev_grid, G * gridsize * sizeof(float2)),					"cudaMalloc failed! grid");

	checkCudaStatus(cudaMalloc((void**)&dev_bhIn, bhSize * sizeof(int)),						"cudaMalloc failed! bhIn");
	checkCudaStatus(cudaMalloc((void**)&dev_pixsize, hitsize * sizeof(float)),					"cudaMalloc failed! pixsize");
	checkCudaStatus(cudaMalloc((void**)&dev_pixImgsize, hitsize * sizeof(float2)),				"cudaMalloc failed! view");
	checkCudaStatus(cudaMalloc((void**)&dev_tree, treeSize * sizeof(int)),						"cudaMalloc failed! tree");
	checkCudaStatus(cudaMalloc((void**)&dev_search, dev_searchNr * M * N * sizeof(int)),		"cudaMalloc failed! search");

	if (imsize.y <= 2000 && M < 2000)
		checkCudaStatus(cudaMalloc((void**)&dev_minmax, dev_minmaxnr * M * N * sizeof(int2)),	"cudaMalloc failed! minmax");
	checkCudaStatus(cudaMalloc((void**)&dev_img, N * M * sizeof(uchar4)),						"cudaMalloc failed! img");
	checkCudaStatus(cudaMalloc((void**)&dev_img2, N * M * sizeof(uchar4)),						"cudaMalloc failed! img2");
	checkCudaStatus(cudaMalloc((void**)&dev_temp, M * N * dev_filterW*dev_filterW * sizeof(float3)), "cudaMalloc failed! temp");
	checkCudaStatus(cudaMalloc((void**)&dev_starlight , M * N * sizeof(float3)),				"cudaMalloc failed! starlight");
	checkCudaStatus(cudaMalloc((void**)&dev_trail, M * N * sizeof(float3)),						"cudaMalloc failed! trail");
	checkCudaStatus(cudaMalloc((void**)&dev_thphi, G * rastSize * sizeof(float2)),				"cudaMalloc failed! thphi");
	checkCudaStatus(cudaMalloc((void**)&dev_grad, G * rastSize * sizeof(float2)),				"cudaMalloc failed! grad");
	checkCudaStatus(cudaMalloc((void**)&dev_hit, G * hitsize * sizeof(float2)),					"cudaMalloc failed! hit");
	checkCudaStatus(cudaMalloc((void**)&dev_horStretch, rastSize * sizeof(float2)),				"cudaMalloc failed! horstretch");
	checkCudaStatus(cudaMalloc((void**)&dev_startBlock, rastSize * sizeof(float2)),				"cudaMalloc failed! startBlock");
	checkCudaStatus(cudaMalloc((void**)&dev_endBlock, rastSize * sizeof(float2)),				"cudaMalloc failed! endBlock");
	checkCudaStatus(cudaMalloc((void**)&dev_in, rastSize * sizeof(float2)),						"cudaMalloc failed! in");
	checkCudaStatus(cudaMalloc((void**)&dev_gradIn, rastSize * sizeof(float2)),					"cudaMalloc failed! grad");
	checkCudaStatus(cudaMalloc((void**)&dev_st, starSize * 2 * sizeof(float)),					"cudaMalloc failed! stars");
	checkCudaStatus(cudaMalloc((void**)&dev_mag, starSize * 2 * sizeof(float)),					"cudaMalloc failed! mag");
	checkCudaStatus(cudaMalloc((void**)&dev_cam, 7 * G * sizeof(float)),						"cudaMalloc failed! cam");
	checkCudaStatus(cudaMalloc((void**)&dev_camIn, 7 * sizeof(float)),							"cudaMalloc failed! camIn");
	checkCudaStatus(cudaMalloc((void**)&dev_bbbh, 6 * sizeof(int2)),							"cudaMalloc failed! bbbh");
	checkCudaStatus(cudaMalloc((void**)&dev_stCache, 2 * starSize * dev_trailnum * sizeof(int2)),	"cudaMalloc failed! stCache");
	checkCudaStatus(cudaMalloc((void**)&dev_stnums, starSize * sizeof(int)),					"cudaMalloc failed! stnums");
	checkCudaStatus(cudaMalloc((void**)&dev_diff, dev_diffSize*dev_diffSize * sizeof(uchar3)), "cudaMalloc failed! diff");

	#pragma endregion


	#pragma region cudaMemcopy Host to Device
	checkCudaStatus(cudaMemcpy(dev_sumTable, (float4*)sumTableData, imsize.x*imsize.y * sizeof(float4), cudaMemcpyHostToDevice),	"cudaMemcpy failed! sumtable");
	checkCudaStatus(cudaMemcpy(dev_pi, pi, G * bhSize * sizeof(int), cudaMemcpyHostToDevice),										"cudaMemcpy failed! bhpi");
	checkCudaStatus(cudaMemcpy(dev_bhIn, &bhIn[0], bhSize * sizeof(int), cudaMemcpyHostToDevice),									"cudaMemcpy failed! bhpi");
	checkCudaStatus(cudaMemcpy(dev_pixsize, pixsize, hitsize * sizeof(float), cudaMemcpyHostToDevice),								"cudaMemcpy failed! pixsize");
	checkCudaStatus(cudaMemcpy(dev_bbbh, &bbbh[0], 6 * sizeof(int2), cudaMemcpyHostToDevice),										"cudaMemcpy failed! bbbh");
	checkCudaStatus(cudaMemcpy(dev_pixImgsize, pixImgSize, hitsize * sizeof(float2), cudaMemcpyHostToDevice),						"cudaMemcpy failed! view");
	checkCudaStatus(cudaMemcpy(dev_grid, grid, gridsize * G * sizeof(float2), cudaMemcpyHostToDevice),								"cudaMemcpy failed! grid");


	checkCudaStatus(cudaMemcpy(dev_tree, tree, treeSize * sizeof(int), cudaMemcpyHostToDevice),										"cudaMemcpy failed! tree");
	checkCudaStatus(cudaMemcpy(dev_thphi, thphi, G * rastSize * sizeof(float2), cudaMemcpyHostToDevice),							"cudaMemcpy failed! thphi ");
	checkCudaStatus(cudaMemcpy(dev_grad, gradient, G * rastSize * sizeof(float2), cudaMemcpyHostToDevice),							"cudaMemcpy failed! grad ");
	checkCudaStatus(cudaMemcpy(dev_hit, hit, G * hitsize* sizeof(float2), cudaMemcpyHostToDevice),									"cudaMemcpy failed! hit ");
	checkCudaStatus(cudaMemcpy(dev_in, &in[0], rastSize * sizeof(float2), cudaMemcpyHostToDevice),									"cudaMemcpy failed! in ");
	checkCudaStatus(cudaMemcpy(dev_gradIn, &gradIn[0], rastSize * sizeof(float2), cudaMemcpyHostToDevice),							"cudaMemcpy failed! in ");

	checkCudaStatus(cudaMemcpy(dev_horStretch, &stretch[0], rastSize * sizeof(float2), cudaMemcpyHostToDevice),						"cudaMemcpy failed! stretch ");
	checkCudaStatus(cudaMemcpy(dev_st, stars, starSize * 2 * sizeof(float), cudaMemcpyHostToDevice), 								"cudaMemcpy failed! stars ");
	checkCudaStatus(cudaMemcpy(dev_mag, mag, starSize * 2 * sizeof(float), cudaMemcpyHostToDevice), 								"cudaMemcpy failed! mag ");
	checkCudaStatus(cudaMemcpy(dev_cam, camParam, 7 * G * sizeof(float), cudaMemcpyHostToDevice), 									"cudaMemcpy failed! cam ");
	checkCudaStatus(cudaMemcpy(dev_camIn, &camIn[0], 7 * sizeof(float), cudaMemcpyHostToDevice),									"cudaMemcpy failed! camIn");
	checkCudaStatus(cudaMemcpy(dev_diff, (uchar3*)diffraction, dev_diffSize*dev_diffSize * sizeof(uchar3), cudaMemcpyHostToDevice), "cudaMemcpy failed! diffraction");

	#pragma endregion

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0.f;

	dim3 threadsPerBlock(TILE_H, TILE_W);
	dim3 numBlocks2(N / threadsPerBlock.x + 1, M / threadsPerBlock.y + 1);


	pixInterpolation << <numBlocks2, threadsPerBlock >> >(dev_pixImgsize, dev_bh, M, N, G, 0, dev_thphi, dev_grid, GM, GN);

	
	dim3 numBlocks((N - 1) / threadsPerBlock.x + 1, (M-1) / threadsPerBlock.y + 1);
	int fr = 0;

	for (int q = movielength*fr; q < movielength * (fr+1); q++) {
		float speed = 1.f/camParam[0];
		float offset = PI2*q / (.25f*speed*M);

		if (G > 1) {
			prec += 0.05f;
			prec = fmodf(prec, (float)dev_G - 1.f);
			int strt = (int)prec;

			//#pragma omp parallel for
			//for (int x = 0; x < rastSize; x++) gradIn[x] = gradient[x+rastSize*strt];
			//checkCudaStatus(cudaMemcpy(dev_gradIn, &gradIn[0], rastSize * sizeof(float2), cudaMemcpyHostToDevice), "cudaMemcpy failed! in ");
			// grad should be only the appropriate one for the grid, but right now only for non zoom working!!!

			float percentage = fmodf(prec, 1.f);
			int NN = N;
			int tpb = 32;
			int blocks = NN / tpb;
			averageShadow << < blocks, tpb >> >(dev_pi, dev_bhIn, M, NN, strt, percentage, true, dev_bbbh);
			blocks = M / tpb;
			averageShadow << < blocks, tpb >> >(dev_pi, dev_bhIn, M, NN, strt, percentage, false, dev_bbbh);
			blocks = NN / tpb;
			averageShadow << < blocks, tpb >> >(dev_pi, dev_bhIn, M, NN, strt, percentage, true, dev_bbbh);
			blocks = NN / tpb + 1;
			stretchGrid << <blocks, tpb >> >(dev_thphi, dev_horStretch, M, NN, strt, true, dev_bbbh, 0);
			blocks = M / tpb + 1;
			stretchGrid << <blocks, tpb >> >(dev_horStretch, dev_startBlock, M, NN, 0, false, dev_bbbh, 0);
			blocks = NN / tpb + 1;
			stretchGrid << <blocks, tpb >> >(dev_thphi, dev_horStretch, M, NN, strt + 1, true, dev_bbbh, 1);
			blocks = M / tpb + 1;
			stretchGrid << <blocks, tpb >> >(dev_horStretch, dev_endBlock, M, NN, 0, false, dev_bbbh, 1);
			dim3 numBlocks2(NN / threadsPerBlock.x + 1, M / threadsPerBlock.y + 1);
			averageGrids << < numBlocks2, threadsPerBlock >> >(dev_thphi, dev_startBlock, dev_endBlock, dev_in, M, NN, strt, percentage, dev_bbbh, dev_camIn, dev_cam);
			checkCudaStatus(cudaMemcpy(dev_bbbh, &bbbh[0], 6 * sizeof(int2), cudaMemcpyHostToDevice), "cudaMemcpy failed! bbbh");
			//checkCudaStatus(cudaMemcpy(dev_camIn, &camParam[0], 7 * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy failed! cam");
		}
		cudaEventRecord(start);
		if (star) {
			int tpb = 32;
			int nb = dev_starSize / tpb + 1;
			clearArrays << < nb, tpb >> > (dev_stnums, dev_stCache, q, dev_trailnum, dev_starSize);

			distortStarMap << <numBlocks, threadsPerBlock >> >(dev_temp, dev_in, dev_bhIn, dev_st, dev_tree, starSize, 
															   dev_camIn, dev_mag, treeLevel, M, N, step, offset, dev_search,
															   dev_searchNr, dev_stCache, dev_stnums, dev_trail, dev_trailnum, 
															   dev_grad, q, dev_pixImgsize);
			// grad should be only the appropriate one for the grid, but right now only for non zoom working!!!
			
			sumStarLight <<<numBlocks, threadsPerBlock>>>(dev_temp, dev_trail, dev_starlight, step, M, N, dev_filterW);
			addDiffraction << <numBlocks, threadsPerBlock >> >(dev_starlight, M, N, dev_diff, dev_diffSize);
			makePix << <numBlocks, threadsPerBlock >> >(dev_starlight, dev_img2, M, N, dev_hit);

		}
		else {
			distortEnvironmentMap << <numBlocks, threadsPerBlock >> >(dev_in, dev_img, dev_bhIn, imsize, M, N, 
																	  offset, dev_sumTable, dev_camIn, dev_minmaxnr, 
																	  dev_minmax, dev_pixsize, dev_pixImgsize);
		}
		addStarsAndBackground <<< numBlocks, threadsPerBlock >> > (dev_img2, dev_img, M);
	
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		cout << milliseconds << endl;

		#pragma region Check kernel errors
		// Check for any errors launching the kernel
		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			cleanup();
			return;
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching makeImageKernel!\n", cudaStatus);
			cleanup();
			return;
		}
		#pragma endregion
	
		// Copy output vector from GPU buffer to host memory.
		checkCudaStatus(cudaMemcpy(&image[0], dev_img, N * M *  sizeof(uchar4), cudaMemcpyDeviceToHost), "cudaMemcpy failed! Dev to Host");
		stringstream ss2;
		star ? ss2 << "movie/bh_" << N << "by" << M << "_" << starSize << "_stars_view_" << q << ".png" :
			ss2 << "bh_" << gridStart << "_" << N << "by" << M << "_" << celestImg.total() << "_" << q << ".png";

		string imgname = ss2.str();
		cv::Mat img = cv::Mat(N, M, CV_8UC4, (void*)&image[0]);
		cv::imwrite(imgname, img, compressionParams);
	}
}
