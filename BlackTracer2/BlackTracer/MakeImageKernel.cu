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
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_resize.h"
using namespace std;

/* ------------- DEFINITIONS & DECLARATIONS --------------*/
#pragma region define
#define TILE_W 4
#define TILE_H 4

#define ij (i*M+j)
#pragma endregion

__device__ void searchTree(const int *tree, const float *thphiPixMin, const float *thphiPixMax, const int treeLevel, 
						   int *searchNrs, int startNr, int &pos, int picheck) {
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
			if (a > angle || a > PI*.25f || dist > M/25) continue;
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

__global__ void distortStarMap(float3 *starLight, const float2 *thphi, const uchar *bh, const float *stars, const int *tree,
							   const int starSize, const float *camParam, const float *magnitude, const int treeLevel,
							   const int M, const int N, const int step, float offset, int *search, int searchNr, int2 *stCache, 
							   int *stnums, float3 *trail, int trailnum, float2 *grad, const int framenumber, const float2 *viewthing) {

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
	if (bh[ij] == 0) {

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
		findLensingRedshift(t, p, M, ind, camParam, viewthing, frac, redshft, 0.f);
		redshft = 1.f;
		float red = 0.f;// 4.f * log10f(redshft);
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

				//	addTrails(starsToCheck, starSize, framenumber, stnums, trail, grad, stCache, q, i, j, M, trailnum, part, frac, rgb);

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

__device__ float2 interpolatePix(const float theta, const float phi, const int M, const int N, const int g, const int gridlvl, 
								 const float2 *grid, const int GM, const int GN, int *gapsave, const int i, const int j) {
	int half = (phi < PI) ? 0 : 1;
	int a = 0;
	int b = half*GM / 2;
	int gap = GM / 2;

	findBlock(theta, phi, g, grid, GM, GN, a, b, gap, gridlvl);
	gapsave[i*M1 + j] = gap;

	int k = a + gap;
	int l = b + gap;

	float factor = PI2 / (1.f*GM);
	float cornersCam[4] = { factor*a, factor*b, factor*k, factor*l };
	l = l % GM;
	float2 nul = { -1, -1 };
	float2 cornersCel[12] = { grid[g*GN*GM + a*GM + b], grid[g*GN*GM + a*GM + l], grid[g*GN*GM + k*GM + b], grid[g*GN*GM + k*GM + l],
									nul, nul, nul, nul, nul, nul, nul, nul };
	float2 thphiInter = interpolateSpline(a, b, gap, GM, GN, theta, phi, g, cornersCel, cornersCam, grid);

	if (!thphiInter.x == -1 && thphiInter.y == -1) wrapToPi(thphiInter.x, thphiInter.y);

	return thphiInter;
}

__device__ __forceinline__ float atomicMinFloat(float * addr, float value) {
	float old;
	old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
		__uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));

	return old;
}

__device__ __forceinline__ float atomicMaxFloat(float * addr, float value) {
	float old;
	old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
		__uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

	return old;
}

__global__ void makeGrid(const int g, const int GM, const int GN, float2 *grid, const float2 *hashTable, const int2 *hashPosTag,
						  const int2 *offsetTable, const int2 *tableSize, const char count) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (i < GN && j < GM) {
		grid[count*GM*GN + i*GM + j] = hashLookup({ i, j }, hashTable, hashPosTag, offsetTable, tableSize, g);
	}
}

__global__ void camUpdate(const float alpha, const int g, const float *camParam, float *cam) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < 7) cam[i] = (1.f - alpha)*camParam[g * 7 + i] + alpha*camParam[(g + 1) * 7 + i];
}

__global__ void findBhCenter(const int GM, const int GN, const float2 *grid, float2 *bhBorder) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (i < GN && j < GM) {
		if (grid[i*GM + j].x == -1 || grid[GM*GN + i*GM + j].x == -1) {
			float gridsize = PI / (1.f*GN);
			atomicMinFloat(&(bhBorder[0].x), gridsize*(float)i);
			atomicMaxFloat(&(bhBorder[0].y), gridsize*(float)i);
			atomicMinFloat(&(bhBorder[1].x), gridsize*(float)j);
			atomicMaxFloat(&(bhBorder[1].y), gridsize*(float)j);
		}
	}
}

__global__ void findBhBorders(const int GM, const int GN, const float2 *grid, const int angleNum, float2 *bhBorder) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < angleNum * 2) {
		int ii = i / 2;
		float angle = PI2 / (1.f * angleNum) * 1.f* ii;
		float thetaChange = -sinf(angle);
		float phiChange = cosf(angle);
		float2 pt = { .5f*bhBorder[0].x + .5f*bhBorder[0].y, .5f*bhBorder[1].x + .5f*bhBorder[1].y };
		pt = { pt.x / PI2*GM, pt.y / PI2*GM };
		int2 gridpt = { int(pt.x), int(pt.y) };

		float2 gridB = { -2, -2 };
		float2 gridA = { -2, -2 };

		while (!(gridA.x > 0 && gridB.x == -1)) {
			gridB = gridA;
			pt.x += thetaChange;
			pt.y += phiChange;
			gridpt = { int(pt.x), int(pt.y) };
			gridA = grid[(i % 2)*GM*GN + gridpt.x *GM + gridpt.y];
		}

		bhBorder[2 + i] = { (pt.x - thetaChange)*PI2 / (1.f*GM), (pt.y - phiChange)*PI2 / (1.f*GM) };
	}
}

__global__ void displayborders(const int angleNum, float2 *bhBorder, uchar4 *out, const int M) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < angleNum * 2) {
		int x = int(bhBorder[i + 2].x / PI2 *1.f*M);
		int y = int(bhBorder[i + 2].y / PI2 *1.f*M);

		out[x*M + y] = {255* (i%2), 255*(1-i%2), 0, 255};
	}
}

__global__ void smoothBorder(const float2 *bhBorder, float2 *bhBorder2, const int angleNum) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < angleNum * 2) {
		if (i == 0) {
			bhBorder2[0] = bhBorder[0];
			bhBorder2[1] = bhBorder[1];
		}
		int prev = (i - 2 + 2 * angleNum) % (2 * angleNum);
		int next = (i + 2) % (2 * angleNum);
		bhBorder2[i + 2] = { 1.f / 3.f * (bhBorder[prev + 2].x + bhBorder[i + 2].x + bhBorder[next + 2].x),
							 1.f / 3.f * (bhBorder[prev + 2].y + bhBorder[i + 2].y + bhBorder[next + 2].y) };
	}
}

__global__ void pixInterpolation(const float2 *viewthing, const int M, const int N, const int Gr, float2 *thphi, const float2 *grid, 
								 const int GM, const int GN, const float hor, const float ver, int *gapsave, int gridlvl,
								 const float2 *bhBorder, const int angleNum, const float alpha) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (i < N1 && j < M1) {
		float theta = viewthing[i*M1 + j].x + ver;
		float phi = fmodf(viewthing[i*M1 + j].y + hor + PI2, PI2);
		if (Gr > 1) {
			float2 A, B;
			float2 center = { .5f*bhBorder[0].x + .5f*bhBorder[0].y, .5f*bhBorder[1].x + .5f*bhBorder[1].y };
			float stretchRad = max(bhBorder[0].y - bhBorder[0].x, bhBorder[1].x - bhBorder[1].y) * 0.75f;
			float centerdist = (theta - center.x)*(theta - center.x) + (phi - center.y)*(phi - center.y);
			if (centerdist < stretchRad*stretchRad) {
				float angle = atan2(center.x-theta, phi-center.y);
				angle = fmodf(angle + PI2, PI2);
				int angleSlot = angle / PI2 * angleNum;

				float2 bhBorderNew = { (1.f - alpha) * bhBorder[2*angleSlot+2].x + alpha * bhBorder[2*angleSlot +3].x,
									   (1.f - alpha) * bhBorder[2*angleSlot+2].y + alpha * bhBorder[2*angleSlot +3].y };

				if (centerdist <= (bhBorderNew.x - center.x)*(bhBorderNew.x - center.x) + (bhBorderNew.y - center.y)*(bhBorderNew.y - center.y)) {
					thphi[i*M1 + j] = { -1, -1 };
					return;
				}

				float tStoB = (center.x - stretchRad*sinf(angle) - bhBorderNew.x);
				float pStoB = (center.y + stretchRad*cosf(angle) - bhBorderNew.y);

				float thetaPerc = fabsf(tStoB) < 1E-5? 0 : 1.f - (theta - bhBorderNew.x) / tStoB;
				float phiPerc = fabsf(pStoB) < 1E-5 ? 0 : 1.f - (phi - bhBorderNew.y) / pStoB;
				float thetaA = theta - thetaPerc * (bhBorderNew.x - bhBorder[2 * angleSlot + 2].x);
				float phiA = phi - phiPerc * (bhBorderNew.y - bhBorder[2 * angleSlot + 2].y);
				float thetaB = theta - thetaPerc * (bhBorderNew.x - bhBorder[2 * angleSlot + 3].x);
				float phiB = phi - phiPerc * (bhBorderNew.y - bhBorder[2 * angleSlot + 3].y);

				A = interpolatePix(thetaA, phiA, M, N, 0, gridlvl, grid, GM, GN, gapsave, i, j);
				B = interpolatePix(thetaB, phiB, M, N, 1, gridlvl, grid, GM, GN, gapsave, i, j);
			}
			else {
				A = interpolatePix(theta, phi, M, N, 0, gridlvl, grid, GM, GN, gapsave, i, j);
				B = interpolatePix(theta, phi, M, N, 1, gridlvl, grid, GM, GN, gapsave, i, j);

			}
			if (A.x == -1 || B.x == -1) thphi[i*M1 + j] = { -1, -1 };
			else {

				if (A.y < .2f*PI2 && B.y > .8f*PI2) A.y += PI2;
				if (B.y < .2f*PI2 && A.y > .8f*PI2) B.y += PI2;

				if (isnan(B.x) || isnan(B.y)) printf("noooooooooB %d %d \n", i, j);
				if (isnan(A.x) || isnan(A.y)) printf("noooooooooA %d %d \n", i, j);

				thphi[i*M1 + j] = { (1.f - alpha)*A.x + alpha*B.x, fmodf((1.f - alpha)*A.y + alpha*B.y, PI2) };
			}
		}
		else {
			thphi[i*M1 + j] = interpolatePix(theta, phi, M, N, 0, gridlvl, grid, GM, GN, gapsave, i, j);
		}
	}
}

__global__ void makeGradField(const float2 *thphi, const int M, const int N, float2 *grad) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (i < N1 && j < M1) {
		float up = thphi[min(N, (i + 1))*M1 + j].x;
		float down = thphi[max(0, (i - 1))*M1 + j].x;
		float left = thphi[i*M1 + max(0, (j - 1))].x;
		float right = thphi[i*M1 + min(M1, j + 1)].x;
		float mid = thphi[i*M1 + j].x;
		if (mid > 0) {
			float xdir = 0;
			if (up > 0 && down > 0)
				xdir = .5f * (up - mid) - .5f*(down - mid);
			float ydir = 0;
			if (left>0 && right > 0)
				ydir = .5f*(right - mid) - .5f*(left - mid);
			float size = sqrtf(xdir*xdir + ydir*ydir);
			grad[i*M1 + j] = float2{ -ydir, xdir };
		}
	}
}

__global__ void findBlackPixels(const float2 *thphi, const int M, const int N, uchar *bh) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (i < N && j < M) {
		bool picheck = false;
		float t[4];
		float p[4];
		int ind = i*M1 + j;
		retrievePixelCorners(thphi, t, p, ind, M, picheck, 0.0f);
		if (ind == -1) bh[ij] = 1;
		else bh[ij] = 0;
	}
}

__global__ void findArea(const float2 *thphi, const int M, const int N, float *area) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (i < N && j < M) {
		bool picheck = false;
		float t[4];
		float p[4];
		int ind = i*M1 + j;
		retrievePixelCorners(thphi, t, p, ind, M, picheck, 0.0f);
		float th1[3] = { t[0], t[1], t[2] };
		float ph1[3] = { p[0], p[1], p[2] };
		float th2[3] = { t[0], t[2], t[3] };
		float ph2[3] = { p[0], p[2], p[3] };
		area[ij] = calcAreax(th1, ph1) + calcAreax(th2, ph2);

		//if (j > 273 && j < 279 && i>180 && j < 184) printf("%d %d %f %f %f %f \n", i, j, calcAreax(th1, ph1), calcAreax(th2, ph2));

	}
}

__global__ void smoothAreaH(float *areaSmooth, float *area, const uchar *bh, const int *gap, const int M, const int N) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (i < N && j < M) {
		if (bh[ij] == 1) return;

		int fs = max(max(gap[i*M1 + j], gap[i*M1 + M1 + j]), max(gap[i*M1 + j + 1], gap[i*M1 + M1 + j + 1]));
		fs = fs / 2;
		float sum = 0.f;
		int count = 0;

		for (int h = -fs; h <= fs; h++) {
			if (bh[i*M+(j + h + M) % M] == 0) {
				float ar = area[i*M + (j + h + M) % M];
				sum += ar;
				count++;
			}
		}

		areaSmooth[ij] = sum / (1.f*count);
	}
}

__global__ void smoothAreaV(float *areaSmooth, float *area, const uchar *bh, const int *gap, const int M, const int N) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (i < N && j < M) {
		if (bh[ij] == 1) return;

		int fs = max(max(gap[i*M1 + j], gap[i*M1 + M1 + j]), max(gap[i*M1 + j + 1], gap[i*M1 + M1 + j + 1]));
		fs = fs / 2;
		float sum = 0.f;
		int minv = max(0, i - fs);
		int maxv = min(N - 1, i + fs);
		int count = 0;

		for (int h = minv; h <= maxv; h++) {
			if (bh[h*M + j] == 0) {
				float ar = areaSmooth[h*M + j];
				sum += ar;
				count++;
			}
		}

		area[ij] = sum / (1.f*count);
	}
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

__global__ void distortEnvironmentMap(const float2 *thphi, uchar4 *out, const uchar *bh, const int2 imsize,
									  const int M, const int N, float offset, float4* sumTable, const float *camParam,
									  int minmaxSize, int2 *minmaxPre, float *solidangle, float2 *viewthing, bool lr) {

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	float4 color = { 0.f, 0.f, 0.f, 0.f };

	// Only compute if pixel is not black hole.
	if (bh[ij] == 0) {

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

					if (ApixP_local < 0) printf("i: %d j: %d q: %d pixmax: %d pixmin: %d pixNum: %d Apixti: %d ApixP_loc: %d ApixP: %f BpixP: %f ApixT: %f\n p: %f %f %f %f, t: %f %f %f %f \n", 
						i, j, q, pixMax, pixMin, pixNum, ApixTi, ApixP_local, ApixP, BpixP, ApixT, p[0], p[1], p[2], p[3], t[0], t[1], t[2], t[3]);

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
			findLensingRedshift(t, p, M, ind, camParam, viewthing, frac, redshft, solidangle[ij]);
			float H, S, P;
			RGBtoHSP(color.z / 255.f, color.y / 255.f, color.x / 255.f, H, S, P);
			if (lr) {
				P *= frac;
				P = redshft < 1.f ? P * 1.f / redshft : powf(P, redshft);
			}
			HSPtoRGB(H, S, min(1.f, P), color.z, color.y, color.x);
		}
	}
	//CHANGED
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

__global__ void addStarsAndBackground(uchar4 *stars, uchar4 *background, uchar4 *output, int M) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	float3 star = { (float)stars[ij].x * stars[ij].x, (float)stars[ij].y * stars[ij].y, (float)stars[ij].z * stars[ij].z };
	float3 bg = { (float)background[ij].x * background[ij].x, (float)background[ij].y * background[ij].y, (float)background[ij].z * background[ij].z };
	float p = 1.2f;
	float3 out = { sqrtf(p*star.x + (2.f - p)*bg.x), sqrtf(p*star.y + (2.f - p)*bg.y), sqrtf(p*star.z + (2.f - p)*bg.z)};
	//float max = fmaxf(fmaxf(out.x, out.y), out.z);
	//if (max > 255.f) {
	//	out.y *= (255.f / max);
	//	out.z *= (255.f / max);
	//	out.x *= (255.f / max);
	//}

	//  CHANGED
	output[ij] = { min((int)out.z, 255), min((int)out.y, 255), min((int)out.x, 255), 255 };
}

#pragma region device variables
// Device pointer variables
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
uchar *dev_bh = 0;
float4 *dev_sumTable = 0;
float3 *dev_temp = 0;
int *dev_search = 0;
int2 *dev_minmax = 0;
float *dev_camIn = 0;
uchar3 *dev_diff = 0;
float3 *dev_starlight = 0;
float *dev_area = 0;
float *dev_areaSmooth = 0;
float2 *dev_hit = 0;
float2 *dev_grad = 0;
float2 *dev_viewthing = 0;
float2 *dev_grid = 0;
int *dev_gap = 0;
float2 *dev_bhBorder = 0;
float2 *dev_bhBorder2 = 0;
float2 *dev_hashTable = 0;
int2 *dev_offsetTable = 0;
int2 *dev_hashPosTag = 0;
int2 *dev_tableSize = 0;

// Other kernel variables
int dev_M, dev_N, dev_G, dev_minmaxnr, dev_GM, dev_GN, dev_gridlvl, dev_angleNum;
float dev_viewAngle, dev_alpha;
int dev_diffSize;
float offset = 0.f;
float prec = 0.f;
int2 dev_imsize;
int dev_treelvl = 0;
int dev_trailnum = 0;
int dev_searchNr = 0;
int dev_step = 0;
int dev_filterW = 0;
int dev_starSize = 0;
int gnr = -1;

#pragma endregion

cudaError_t cleanup() {
	cudaFree(dev_grid);
	cudaFree(dev_bh);
	cudaFree(dev_search);
	cudaFree(dev_minmax);
	cudaFree(dev_in);
	cudaFree(dev_st);
	cudaFree(dev_stCache);
	cudaFree(dev_stnums);
	cudaFree(dev_tree);
	cudaFree(dev_cam);
	cudaFree(dev_camIn);
	cudaFree(dev_mag);
	cudaFree(dev_temp);
	cudaFree(dev_img);
	cudaFree(dev_img2);
	cudaFree(dev_area);
	cudaFree(dev_areaSmooth);
	cudaFree(dev_hit);
	cudaFree(dev_trail);
	cudaFree(dev_diff);
	cudaFree(dev_starlight);
	cudaFree(dev_grad);
	cudaFree(dev_viewthing);
	cudaFree(dev_gap);
	cudaFree(dev_bhBorder);
	cudaFree(dev_bhBorder2);
	cudaFree(dev_sumTable);
	cudaFree(dev_offsetTable);
	cudaFree(dev_hashTable);
	cudaFree(dev_tableSize);
	cudaFree(dev_hashPosTag);
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
bool dev_lr = true;
bool play = true;
float hor = 0.0f;
float ver = 0.0f;

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
bool moving = false;

void render() {
	float speed = 4.f;
	dim3 threadsPerBlock(TILE_H, TILE_W);
	dim3 numBlocks((dev_N - 1) / threadsPerBlock.x + 1, (dev_M - 1) / threadsPerBlock.y + 1);
	dim3 numBlocksM1(dev_N / threadsPerBlock.x + 1, dev_M / threadsPerBlock.y + 1);
	int tpb = 32;

	//if (dev_G > 1) {
	//	prec = fmodf(prec, (float)dev_G - 1.f);
	//	dev_alpha = fmodf(prec, 1.f);

	//	if (gnr != (int)prec) {
	//		gnr = (int)prec;
	//		dim3 numBlocks2((dev_GN - 1) / threadsPerBlock.x + 1, (dev_GM - 1) / threadsPerBlock.y + 1);
	//		findBhCenter <<<numBlocks2, threadsPerBlock>>>(gnr, dev_GM, dev_GN, dev_hashTable, dev_hashPosTag, dev_offsetTable, dev_tableSize, dev_bhBorder);
	//		findBhBorders << < dev_angleNum * 2 / tpb + 1, tpb >> >(gnr, dev_GM, dev_GN, dev_hashTable, dev_hashPosTag, dev_offsetTable, dev_tableSize, dev_angleNum, dev_bhBorder);
	//		smoothBorder << <dev_angleNum * 2 / tpb + 1, tpb >> >(dev_bhBorder, dev_bhBorder2, dev_angleNum);
	//		smoothBorder << <dev_angleNum * 2 / tpb + 1, tpb >> >(dev_bhBorder2, dev_bhBorder, dev_angleNum);
	//		smoothBorder << <dev_angleNum * 2 / tpb + 1, tpb >> >(dev_bhBorder, dev_bhBorder2, dev_angleNum);
	//		smoothBorder << <dev_angleNum * 2 / tpb + 1, tpb >> >(dev_bhBorder2, dev_bhBorder, dev_angleNum);
	//	}
	//	prec += 0.1f;
	//	cout << prec << endl;

	//}
	//displayborders << <dev_angleNum * 2 / tpb + 1, tpb >> >(dev_angleNum, dev_bhBorder, d_out, dev_M);
	//pixInterpolation << <numBlocksM1, threadsPerBlock >> >(dev_viewthing, dev_M, dev_N, dev_G, gnr,
	//													   dev_in, dev_grid, dev_GM, dev_GN, hor, ver, dev_gap, dev_gridlvl,
	//													   dev_bhBorder, dev_angleNum, dev_alpha);

	//findBlackPixels << <numBlocks, threadsPerBlock >> >(dev_in, dev_M, dev_N, dev_bh);
	//makeGradField << <numBlocksM1, threadsPerBlock >> > (dev_in, dev_M, dev_N, dev_grad);
	//findArea << <numBlocks, threadsPerBlock >> > (dev_in, dev_M, dev_N, dev_area);
	//smoothArea << <numBlocks, threadsPerBlock >> > (dev_areaSmooth, dev_area, dev_bh, dev_gap, dev_M, dev_N);

	//offset += PI2 / (.25f*speed*dev_M);
	//offset = fmodf(offset, PI2);
	//if (star) {
	//	int nb = dev_starSize / tpb + 1;
	//	clearArrays << < tpb, nb >> > (dev_stnums, dev_stCache, g_current_frame_number, dev_trailnum, dev_starSize);

	//	distortStarMap << <numBlocks, threadsPerBlock >> >(dev_temp, dev_in, dev_bh, dev_st, dev_tree, dev_starSize, 
	//													   dev_camIn, dev_mag, dev_treelvl, dev_M, dev_N, dev_step,
	//													   offset, dev_search, dev_searchNr, dev_stCache, dev_stnums, 
	//													   dev_trail, dev_trailnum, dev_grad, g_current_frame_number, dev_viewthing);

	//	sumStarLight << <numBlocks, threadsPerBlock >> >(dev_temp, dev_trail, dev_starlight, dev_step, dev_M, dev_N, dev_filterW);
	//	addDiffraction << <numBlocks, threadsPerBlock >> >(dev_starlight, dev_M, dev_N, dev_diff, dev_diffSize);
	//	makePix << <numBlocks, threadsPerBlock >> >(dev_starlight, dev_img2, dev_M, dev_N, dev_hit);


	//	distortEnvironmentMap << < numBlocks, threadsPerBlock >> >(dev_in, dev_img, dev_bh, dev_imsize,
	//															   dev_M, dev_N, offset, dev_sumTable, dev_camIn,
	//															   dev_minmaxnr, dev_minmax, dev_areaSmooth, dev_viewthing, dev_lr);
	//	addStarsAndBackground << < numBlocks, threadsPerBlock >> > (dev_img2, dev_img, d_out, dev_M);
	//}
	//else {
	//	distortEnvironmentMap << < numBlocks, threadsPerBlock >> >(dev_in, d_out, dev_bh, dev_imsize,
	//		dev_M, dev_N, offset, dev_sumTable, dev_camIn,
	//		dev_minmaxnr, dev_minmax, dev_areaSmooth, dev_viewthing, dev_lr);
	//}

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
			//gridRad++;
			moving = true;
			//setupGrids();
		}
	}
	else if (key == 'l') {
		dev_lr = !dev_lr;
	}
}

void processSpecialKeys(int key, int x, int y) {
	float maxangle = (PI - dev_viewAngle / (1.f*dev_M) * dev_N) / 2.f;

	switch (key) {
	case GLUT_KEY_UP:
		if (ver + 0.01f <= maxangle) {
			ver += 0.01f;
		}
		break;
	case GLUT_KEY_DOWN:
		if (fabs(ver - 0.01f) <= maxangle) {
			ver -= 0.01f;
		}
		break;
	case GLUT_KEY_LEFT:
		hor += 0.01f;
		break;
	case GLUT_KEY_RIGHT:
		hor -= 0.01f;
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

	//glBindVertexArray(vao);
	//glBindTexture(GL_TEXTURE_2D, diff);
	//glEnable(GL_BLEND);
	//glBlendFunc(GL_ONE, GL_ONE);

	//glPointSize(100.0);
	//glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
	//glPointParameteri(GL_POINT_SPRITE_COORD_ORIGIN, GL_LOWER_LEFT);
	//glEnable(GL_POINT_SPRITE);

	//int diffstars = 1;
	//glDrawArrays(GL_POINTS, 0, diffstars);

	//glBindVertexArray(0);
	//glDisable(GL_BLEND);
	//glDisable(GL_POINT_SPRITE);
	//glDisable(GL_TEXTURE_2D);

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

	//int x,y,n;
	//unsigned char *data = stbi_load("../pic/0.png", &x, &y, &n, STBI_rgb);

	//glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8, x, y, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
	//glGenerateMipmap(GL_TEXTURE_2D);
	//glBindTexture(GL_TEXTURE_2D, 0);

	//float sprites[2] = { 700.f, 500.f };
	//glGenBuffers(1, &vbo);
	//glBindBuffer(GL_ARRAY_BUFFER, vbo);
	//glBufferData(GL_ARRAY_BUFFER, 2*sizeof(GLfloat), sprites, GL_STREAM_DRAW);

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);
	glBindVertexArray(0);

	//stbi_image_free(data);
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

void makeImage(const float *stars, const int *starTree, 
			   const int starSize, const float *camParam, const float *mag, const int treeLevel,
			   const int M, const int N, const int step, const cv::Mat csImage, 
			   const int G, const float gridStart, const float gridStep, 
			   const float2 *hit, const float2 *viewthing, const float viewAngle,
			   const int GM, const int GN, const int gridlvl,
			   const int2 *offsetTables, const float2 *hashTables, const int2 *hashPosTag, const int2 *tableSizes, const int otsize, const int htsize) {

	cudaPrep(stars, starTree, starSize, camParam, mag, treeLevel, M, N, step, csImage, 
		     G, gridStart, gridStep, hit, viewthing, viewAngle, GM, GN, gridlvl,
			 offsetTables, hashTables, hashPosTag, tableSizes, otsize, htsize);

	if (play) {
		int foo = 1;
		char * bar[1] = { " " };
		initGLUT(&foo, bar);
		gluOrtho2D(0, M, N, 0);
		glutDisplayFunc(display);
		// here are the new entries
		glutKeyboardFunc(processNormalKeys);
		glutSpecialFunc(processSpecialKeys);
		initTexture();
		initPixelBuffer();
		glutMainLoop();
		atexit(exitfunc);
	}

	cudaError_t cudaStatus = cleanup();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		cleanup();
		//if (abort) exit(code);
	}
}

void cudaPrep(const float *stars, const int *tree, const int starSize, 
			  const float *camParam, const float *mag, const int treeLevel, const int M, const int N, 
			  const int step, cv::Mat celestImg, const int G, const float gridStart, const float gridStep, 
			  const float2 *hit, const float2 *viewthing, const float viewAngle,
			  const int GM, const int GN, const int gridlvl,
			  const int2 *offsetTables, const float2 *hashTables, const int2 *hashPosTag, const int2 *tableSizes, const int otsize, const int htsize) {
	#pragma region Set variables
	printf("Setting cuda variables...\n");

	dev_M = M;
	dev_N = N;
	dev_viewAngle = viewAngle;
	dev_G = G;
	dev_GM = GM;
	dev_GN = GN;
	dev_treelvl = treeLevel;
	dev_step = step;
	dev_starSize = starSize;
	dev_trailnum = 30;
	dev_gridlvl = gridlvl;
	dev_angleNum = 1000;
	dev_alpha = 0.f;

	// Image and frame parameters
	int movielength = 30;
	vector<uchar4> image(N*M);
	vector<int> compressionParams;
	compressionParams.push_back(cv::IMWRITE_PNG_COMPRESSION);
	compressionParams.push_back(0);
	cv::Mat summedTable = cv::Mat::zeros(celestImg.size(), cv::DataType<cv::Vec4f>::type);

	// diffraction image
	int x, y, n;
	unsigned char *diffraction = stbi_load("../pic/0s.png", &x, &y, &n, STBI_rgb);
	dev_diffSize = x;// M / 16;
	//unsigned char *diffraction = (unsigned char*)malloc(x*y*n*sizeof(unsigned char));
	//stbir_resize_uint8(starimg, x, y, 0, diffraction, dev_diffSize, dev_diffSize, 0, n);

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
	vector<float> camIn(7);
	for (int q = 0; q < 7; q++) camIn[q] = camParam[q];

	vector<float2> bhBorder((dev_angleNum + 1) * 2);
	bhBorder[0] = { 100, 0 };
	bhBorder[1] = { 100, 0 };

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

	printf("Allocating cuda memory...\n");

	#pragma region cudaMalloc		
	checkCudaStatus(cudaMalloc((void**)&dev_viewthing, rastSize * sizeof(float2)),				"cudaMalloc failed! view");
	checkCudaStatus(cudaMalloc((void**)&dev_grid, 2 * gridsize * sizeof(float2)),				"cudaMalloc failed! grid");
	checkCudaStatus(cudaMalloc((void**)&dev_hashTable, htsize * sizeof(float2)),				"cudaMalloc failed! ht");
	checkCudaStatus(cudaMalloc((void**)&dev_offsetTable, otsize * sizeof(int2)),				"cudaMalloc failed! ot");
	checkCudaStatus(cudaMalloc((void**)&dev_tableSize, G * sizeof(int2)),						"cudaMalloc failed! ts");
	checkCudaStatus(cudaMalloc((void**)&dev_hashPosTag, htsize * sizeof(int2)),					"cudaMalloc failed! hp");

	checkCudaStatus(cudaMalloc((void**)&dev_bh, bhSize * sizeof(uchar)),							"cudaMalloc failed! bh");
	checkCudaStatus(cudaMalloc((void**)&dev_in, rastSize * sizeof(float2)),						"cudaMalloc failed! in");
	checkCudaStatus(cudaMalloc((void**)&dev_bhBorder, (dev_angleNum+1) * 2 * sizeof(float2)),	"cudaMalloc failed! bhBorder");
	checkCudaStatus(cudaMalloc((void**)&dev_bhBorder2, (dev_angleNum + 1) * 2 * sizeof(float2)), "cudaMalloc failed! bhBorder");

	checkCudaStatus(cudaMalloc((void**)&dev_cam, 7 * G * sizeof(float)),						"cudaMalloc failed! cam");
	checkCudaStatus(cudaMalloc((void**)&dev_camIn, 7 * sizeof(float)),							"cudaMalloc failed! camIn");

	checkCudaStatus(cudaMalloc((void**)&dev_gap, rastSize * sizeof(int)),						"cudaMalloc failed! grid");
	checkCudaStatus(cudaMalloc((void**)&dev_grad, rastSize * sizeof(float2)),					"cudaMalloc failed! grad");
	checkCudaStatus(cudaMalloc((void**)&dev_hit, G * hitsize * sizeof(float2)),					"cudaMalloc failed! hit");
	checkCudaStatus(cudaMalloc((void**)&dev_areaSmooth, bhSize * sizeof(float)),				"cudaMalloc failed! areasm");
	checkCudaStatus(cudaMalloc((void**)&dev_area, bhSize * sizeof(float)),						"cudaMalloc failed! area");	

	checkCudaStatus(cudaMalloc((void**)&dev_sumTable, imsize.x*imsize.y * sizeof(float4)),		"cudaMalloc failed! sumtable");
	if (imsize.y <= 2000 && M < 2000 && N < 1000)
		checkCudaStatus(cudaMalloc((void**)&dev_minmax, dev_minmaxnr * M * N * sizeof(int2)),	"cudaMalloc failed! minmax");
	checkCudaStatus(cudaMalloc((void**)&dev_img, N * M * sizeof(uchar4)),						"cudaMalloc failed! img");

	checkCudaStatus(cudaMalloc((void**)&dev_img2, N * M * sizeof(uchar4)),						"cudaMalloc failed! img2");
	checkCudaStatus(cudaMalloc((void**)&dev_temp, M * N * dev_filterW*dev_filterW * sizeof(float3)), "cudaMalloc failed! temp");
	checkCudaStatus(cudaMalloc((void**)&dev_starlight , M * N * sizeof(float3)),				"cudaMalloc failed! starlight");
	checkCudaStatus(cudaMalloc((void**)&dev_trail, M * N * sizeof(float3)),						"cudaMalloc failed! trail");
	checkCudaStatus(cudaMalloc((void**)&dev_st, starSize * 2 * sizeof(float)),					"cudaMalloc failed! stars");
	checkCudaStatus(cudaMalloc((void**)&dev_mag, starSize * 2 * sizeof(float)),					"cudaMalloc failed! mag");
	checkCudaStatus(cudaMalloc((void**)&dev_stCache, 2 * starSize * dev_trailnum * sizeof(int2)), "cudaMalloc failed! stCache");
	checkCudaStatus(cudaMalloc((void**)&dev_stnums, starSize * sizeof(int)),					"cudaMalloc failed! stnums");
	checkCudaStatus(cudaMalloc((void**)&dev_diff, dev_diffSize*dev_diffSize * sizeof(uchar3)), "cudaMalloc failed! diff");
	checkCudaStatus(cudaMalloc((void**)&dev_tree, treeSize * sizeof(int)),						"cudaMalloc failed! tree");
	checkCudaStatus(cudaMalloc((void**)&dev_search, dev_searchNr * M * N * sizeof(int)),		"cudaMalloc failed! search");
	#pragma endregion

	printf("Copying into cuda memory...\n");

	#pragma region cudaMemcopy Host to Device
	checkCudaStatus(cudaMemcpy(dev_sumTable, (float4*)sumTableData, imsize.x*imsize.y * sizeof(float4), cudaMemcpyHostToDevice),	"cudaMemcpy failed! sumtable");
	checkCudaStatus(cudaMemcpy(dev_viewthing, viewthing, rastSize * sizeof(float2), cudaMemcpyHostToDevice),						"cudaMemcpy failed! view");
	checkCudaStatus(cudaMemcpy(dev_hashTable, hashTables, htsize * sizeof(float2), cudaMemcpyHostToDevice),							"cudaMemcpy failed! ht");
	checkCudaStatus(cudaMemcpy(dev_offsetTable, offsetTables, otsize * sizeof(int2), cudaMemcpyHostToDevice),						"cudaMemcpy failed! ot");
	checkCudaStatus(cudaMemcpy(dev_tableSize, tableSizes, G * sizeof(int2), cudaMemcpyHostToDevice),								"cudaMemcpy failed! ts");
	checkCudaStatus(cudaMemcpy(dev_hashPosTag, hashPosTag, htsize * sizeof(int2), cudaMemcpyHostToDevice),							"cudaMemcpy failed! hp");

	checkCudaStatus(cudaMemcpy(dev_bhBorder, &bhBorder[0], (dev_angleNum + 1) * 2 * sizeof(float2), cudaMemcpyHostToDevice),		"cudaMemcpy failed! bhBorder");

	checkCudaStatus(cudaMemcpy(dev_tree, tree, treeSize * sizeof(int), cudaMemcpyHostToDevice),										"cudaMemcpy failed! tree");
	checkCudaStatus(cudaMemcpy(dev_hit, hit, G * hitsize* sizeof(float2), cudaMemcpyHostToDevice),									"cudaMemcpy failed! hit ");

	checkCudaStatus(cudaMemcpy(dev_st, stars, starSize * 2 * sizeof(float), cudaMemcpyHostToDevice), 								"cudaMemcpy failed! stars ");
	checkCudaStatus(cudaMemcpy(dev_mag, mag, starSize * 2 * sizeof(float), cudaMemcpyHostToDevice), 								"cudaMemcpy failed! mag ");
	checkCudaStatus(cudaMemcpy(dev_cam, camParam, 7 * G * sizeof(float), cudaMemcpyHostToDevice), 									"cudaMemcpy failed! cam ");
	checkCudaStatus(cudaMemcpy(dev_diff, (uchar3*)diffraction, dev_diffSize*dev_diffSize * sizeof(uchar3), cudaMemcpyHostToDevice), "cudaMemcpy failed! diffraction");
	#pragma endregion

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0.f;

	dim3 threadsPerBlock(TILE_H, TILE_W);
	dim3 numBlocks((N - 1) / threadsPerBlock.x + 1, (M - 1) / threadsPerBlock.y + 1);
	dim3 numBlocksM1(N / threadsPerBlock.x + 1, M / threadsPerBlock.y + 1);

	int fr = 0;
	int tpb = 32;
	printf("Completed cuda preparation.\n");
	dim3 numBlocks2((GN - 1) / threadsPerBlock.x + 1, (GM - 1) / threadsPerBlock.y + 1);
	if (G == 1) {
		makeGrid << < numBlocks2, threadsPerBlock >> >(0, GM, GN, dev_grid, dev_hashTable, dev_hashPosTag, dev_offsetTable, dev_tableSize, 0);
		dev_camIn = dev_cam;
	}

	for (int q = movielength*fr; q < movielength* (fr + 1); q++) {
		float speed = 1.f/camParam[0];
		float offset = PI2*q / (.25f*speed*M);

		cudaEventRecord(start);
		if (G > 1) {
			prec = fmodf(prec, (float)dev_G - 1.f);
			dev_alpha = fmodf(prec, 1.f);
			//cout << q << " " << 0.2f*prec+5.0f << " " << dev_alpha << endl;
			if (gnr != (int)prec) {
				gnr = (int)prec;

				makeGrid << < numBlocks2, threadsPerBlock >> >(gnr, GM, GN, dev_grid, dev_hashTable, dev_hashPosTag, dev_offsetTable, dev_tableSize, 0);
				makeGrid << < numBlocks2, threadsPerBlock >> >(gnr + 1, GM, GN, dev_grid, dev_hashTable, dev_hashPosTag, dev_offsetTable, dev_tableSize, 1);

				findBhCenter << < numBlocks2, threadsPerBlock >> >(GM, GN, dev_grid, dev_bhBorder);
				findBhBorders << < dev_angleNum * 2 / tpb + 1, tpb >> >(GM, GN, dev_grid, dev_angleNum, dev_bhBorder);
				smoothBorder << <dev_angleNum * 2 / tpb + 1, tpb >> >(dev_bhBorder, dev_bhBorder2, dev_angleNum);
				smoothBorder << <dev_angleNum * 2 / tpb + 1, tpb >> >(dev_bhBorder2, dev_bhBorder, dev_angleNum);
				smoothBorder << <dev_angleNum * 2 / tpb + 1, tpb >> >(dev_bhBorder, dev_bhBorder2, dev_angleNum);
				smoothBorder << <dev_angleNum * 2 / tpb + 1, tpb >> >(dev_bhBorder2, dev_bhBorder, dev_angleNum);
			}
			camUpdate << <1, 8>> >(dev_alpha, gnr, dev_cam, dev_camIn);
			prec += .2f;
		}
		else gnr = 0;

		//displayborders << <dev_angleNum * 2 / tpb + 1, tpb >> >(dev_angleNum, dev_bhBorder, dev_img, M);
		pixInterpolation << <numBlocksM1, threadsPerBlock >> >(dev_viewthing, M, N, dev_G, 
																dev_in, dev_grid, GM, GN, hor, ver, dev_gap, gridlvl,
																dev_bhBorder, dev_angleNum, dev_alpha);

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		cout << "Image construction time in ms: " << milliseconds << endl;


		findBlackPixels << <numBlocks, threadsPerBlock >> >(dev_in, M, N, dev_bh);
		findArea << <numBlocks, threadsPerBlock >> > (dev_in, M, N, dev_area);
		smoothAreaH << <numBlocks, threadsPerBlock >> > (dev_areaSmooth, dev_area, dev_bh, dev_gap, M, N);
		smoothAreaV << <numBlocks, threadsPerBlock >> > (dev_areaSmooth, dev_area, dev_bh, dev_gap, M, N);

		if (star) {
			int nb = dev_starSize / tpb + 1;
			clearArrays << < nb, tpb >> > (dev_stnums, dev_stCache, q, dev_trailnum, dev_starSize);
			makeGradField << <numBlocksM1, threadsPerBlock >> > (dev_in, M, N, dev_grad);
			distortStarMap << <numBlocks, threadsPerBlock >> >(dev_temp, dev_in, dev_bh, dev_st, dev_tree, starSize,
															   dev_camIn, dev_mag, treeLevel, M, N, step, offset, dev_search,
															   dev_searchNr, dev_stCache, dev_stnums, dev_trail, dev_trailnum,
														       dev_grad, q, dev_viewthing);
			sumStarLight << <numBlocks, threadsPerBlock >> >(dev_temp, dev_trail, dev_starlight, step, M, N, dev_filterW);
			addDiffraction << <numBlocks, threadsPerBlock >> >(dev_starlight, M, N, dev_diff, dev_diffSize);
			makePix << <numBlocks, threadsPerBlock >> >(dev_starlight, dev_img2, M, N, dev_hit);
		}
		distortEnvironmentMap << <numBlocks, threadsPerBlock >> >(dev_in, dev_img, dev_bh, imsize, M, N,
																	offset, dev_sumTable, dev_camIn, dev_minmaxnr,
																	dev_minmax, dev_area, dev_viewthing, dev_lr);

		if (star) addStarsAndBackground <<< numBlocks, threadsPerBlock >> > (dev_img2, dev_img, dev_img, M);


		#pragma region Check kernel errors
		//gpuErrchk(cudaPeekAtLastError());

		// Check for any errors launching the kernel
		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			cleanup();
			return;
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Device synchronize failed: %s\n", cudaGetErrorString(cudaStatus));
			cleanup();
			return;
		}
		#pragma endregion
	
		// Copy output vector from GPU buffer to host memory.
		checkCudaStatus(cudaMemcpy(&image[0], dev_img, N * M *  sizeof(uchar4), cudaMemcpyDeviceToHost), "cudaMemcpy failed! Dev to Host");
		stringstream ss2;
		ss2 << "bh_" << gridStart << "_" << N << "by" << M << "_" << celestImg.total() << "_" << q << ".png";

		string imgname = ss2.str();
		cv::Mat img = cv::Mat(N, M, CV_8UC4, (void*)&image[0]);
		cv::imwrite(imgname, img, compressionParams);
	}
	prec = 0.f;
	gnr = -1;
}
