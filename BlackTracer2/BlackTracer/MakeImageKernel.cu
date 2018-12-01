#include "kernel.cuh"
#include <iostream>
#include <stdint.h>
#include <iomanip>
#include <algorithm>
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <chrono>

//#include <GL/glew.h>
//#include <GL/freeglut.h>
//#include <cuda_gl_interop.h>
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
#define ij (i*M+j)
#define N1Half (N/2+1)
#define N1 (N+1)
#define M1 (M+1)
#define cam_speed cam[0]
#define cam_alpha cam[1]
#define cam_w cam[2]
#define cam_wbar cam[3]

__constant__ const float pixSep[3] = {-.5f, .5f, 1.5f};

extern int makeImage(float *out, const float2 *thphi, const int *pi, const float *ver, const float *hor,
	const float *stars, const int *starTree, const int starSize, const float *camParam, const float *magnitude, const int treeLevel,
	const bool symmetry, const int M, const int N, const int step);

cudaError_t cudaPrep(float *out, const float2 *thphi, const int *pi, const float *ver, const float *hor,
	const float *stars, const int *starTree, const int starSize, const float *camParam, const float *magnitude, const int treeLevel,
	const bool symmetry, const int M, const int N, const int step);

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
	while (((fabs(start - midT) > error) || (fabs(starp - midP) > error)) && count < 10) {
		count++;

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
__device__ bool starInPolygon(float t[4], float p[4], float start, float starp, int sgn) {
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
__device__ bool piCheck(float(&p)[4], float factor) {
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
/// Corrects phi values for 2-pi crossings.
/// </summary>
/// <param name="p">The phi values to check.</param>
/// <param name="factor">The factor to check if a point is close to the border.</param>
/// <returns></returns>
__device__ bool piCorrect(float(&p)[4], float factor) {
	float factor2 = PI2 * factor;
	#pragma unroll
	for (int q = 0; q < 4; q++) {
		if (p[q] < factor2) {
			p[q] += PI2;
		}
	}
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
	return exp(-1.*dist);//0.8*exp(-2*dist);
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

__global__ void makeImageKernel(float *out, const float2 *thphi, const int *pi, const float *hor, const float *ver,
	const float *stars, const int *tree, const int starSize, const float *camParam, const float *magnitude, const int treeLevel,
							const bool symmetry, const int M, const int N, const int step, int t) {

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
	float sum = 0.f;

	// Only compute if pixel is not black hole.
	int pibh = (symmetry && i >= N / 2) ? pi[(N - 1 - i)*M + j] : pi[ij];
	if (pibh >= 0) {

		// Set values for projected pixel corners & update phi values in case of 2pi crossing.
		#pragma region Retrieve pixel corners

		float t[4];
		float p[4];
		// Check and correct for 2pi crossings.
		bool picheck = false;

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
		if (pibh > 0) picheck = piCheck(p, .2f);

		//int idx2 = 2 * (threadIdx.x * RASTER_W + threadIdx.y);
		//int R2 = RASTER_W * 2;
		//float t[4] = { s_tp[idx2 + R2],		s_tp[idx2],		s_tp[idx2 + 2],	s_tp[idx2 + R2 + 2] };
		//float p[4] = { s_tp[idx2 + R2 + 1], s_tp[idx2 + 1], s_tp[idx2 + 3], s_tp[idx2 + R2 + 3] };

		#pragma endregion

		// Calculate orientation and size of projected polygon (positive -> CW, negative -> CCW)
		float orient = (t[1] - t[0]) * (p[1] + p[0]) + (t[2] - t[1]) * (p[2] + p[1]) +
			(t[3] - t[2]) * (p[3] + p[2]) + (t[0] - t[3]) * (p[0] + p[3]);
		int sgn = orient < 0 ? -1 : 1;

		// Search where in the star tree the bounding box of the projected pixel falls.
		const float thphiPixMax[2] = { max(max(t[0], t[1]), max(t[2], t[3])),
			max(max(p[0], p[1]), max(p[2], p[3])) };
		const float thphiPixMin[2] = { min(min(t[0], t[1]), min(t[2], t[3])),
			min(min(p[0], p[1]), min(p[2], p[3])) };

		float pixVertSize = ver[i + 1] - ver[i];
		float pixHorSize = hor[i + 1] - hor[i];
		float pixSize = pixVertSize * pixHorSize;
		float pixposT = i + .5f;
		float pixposP = j + .5f;
		float frac = pixSize / (fabs(orient) * .5f);
		float maxDistSq = (step + .5f)*(step + .5f);
		if (picheck) {
			for (int q = 0; q < starSize; q++) {
				float start = stars[2 * q];
				float starp = stars[2 * q + 1];
				bool starInPoly = starInPolygon(t, p, start, starp, sgn);
				if (!starInPoly && starp < PI2 * .2f) {
					starp += PI2;
					starInPoly = starInPolygon(t, p, start, starp, sgn);
				}
				if (starInPoly) {
					interpolate(t[0], t[1], t[2], t[3], p[0], p[1], p[2], p[3], start, starp, sgn);
					sum += 1.f;
					// put starlight in own pixel
					float thetaCam = ver[i] + pixVertSize * start;
					float phiCam = hor[j] + pixHorSize * starp;
					float temp = magnitude[q] + 10.f * log10(redshift(thetaCam, phiCam, camParam));
					// m2 = m1 - 2.5(log(frac)) where frac = brightnessfraction of b2/b1;
					for (int u = -step; u <= step; u++) {
						for (int v = -step; v <= step; v++) {
							float dist = distSq(pixSep[step + u], start, pixSep[step + v], starp);
							if (dist > maxDistSq) continue;
							else {
								float appMag = temp - 2.5f * log10(frac*gaussian(dist));
								// i+1 for extra row top (and bottom), +u and +v for filter location
								// (j+v+M)&(M-1) to wrap in horizontal direction
								atomicAdd(&out[(i + 1 + u)*M + ((j + v + M)&(M - 1))], exp10f(-appMag*0.4f));
							}
						}
					}
				}
			}
		}
		else {
			int node = searchTree(tree, thphiPixMin, thphiPixMax, treeLevel);
			// Check set of stars in tree
			int startN = 0;
			if (node != 0 && ((node + 1) & node) != 0) {
				startN = tree[node - 1];
			}
			for (int q = startN; q < tree[node]; q++) {
				float start = stars[2 * q];
				float starp = stars[2 * q + 1];

				if (starInPolygon(t, p, start, starp, sgn)) {
					interpolate(t[0], t[1], t[2], t[3], p[0], p[1], p[2], p[3], start, starp, sgn);
					sum += 1.f;
					// put starlight in own pixel
					float thetaCam = ver[i] + pixVertSize * start;
					float phiCam = hor[j] + pixHorSize * starp;
					float temp = magnitude[q] + 10.f * log10(redshift(thetaCam, phiCam, camParam));
					// m2 = m1 - 2.5(log(frac)) where frac = brightnessfraction of b2/b1;
					for (int u = -step; u <= step; u++) {
						for (int v = -step; v <= step; v++) {
							float dist = distSq(pixSep[step+u], start, pixSep[step+v], starp);
							if (dist > maxDistSq) continue;
							else {
								float appMag = temp	- 2.5f * log10(frac*gaussian(dist));
								// i+1 for extra row top (and bottom), +u and +v for filter location
								// (j+v+M)&(M-1) to wrap in horizontal direction
								atomicAdd(&out[(i + 1 + u)*M + ((j + v + M)&(M - 1))], exp10f(-appMag*0.4f));
								//if (i > 498) printf("%f \t %f \t %d \t %d \t %f \t %f \t %f \t %f \t %f \t %f \t %f \n", appMag, dist, u, v, pixSep[step+u], pixSep[step+v], orient, frac, pixHorSize, pixVertSize, temp);
								//out[ij] += 1.f;

							}
						}
					}
				}
			}
		}
	}
	//out[ij].x = sum;
	//out[ij].y = sum;
	//out[ij].z = sum;
	//out[ij].w = sum;
}

int makeImage(float *out, const float2 *thphi, const int *pi, const float *ver, const float *hor,
	const float *stars, const int *starTree, const int starSize, const float *camParam, const float *mag, const int treeLevel,
			const bool symmetry, const int M, const int N, const int step) {
	cudaError_t cudaStatus = cudaPrep(out, thphi, pi, ver, hor, stars, starTree, starSize, camParam, mag, treeLevel, symmetry, M, N, step);
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
					const bool symmetry, const int M, const int N, const int step) {

	vector<uchar4> image((N + step * 2)*M);
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
	cv::Mat img = cv::Mat((N + 2 * step), M, CV_8UC4, (void*)&image[0]);
	cv::namedWindow("bh", cv::WINDOW_AUTOSIZE);
	
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	int bhpiSize = symmetry ? M*N / 2 : M*N;
	int rastSize = symmetry ? M1*N1Half : M1*N1;
	int treeSize = (1 << (treeLevel + 1)) - 1;

	#pragma region malloc
	cudaStatus = cudaMalloc((void**)&dev_pi, bhpiSize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! tb");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_tree, treeSize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! tc");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_out, M * (N + 2 * step) * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! td");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_thphi, rastSize * sizeof(float2));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! te");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_st, starSize * 2 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! stf");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_mag, starSize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! stf");
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
		fprintf(stderr, "cudaMalloc failed! ver");
		goto Error;
	}
	#pragma endregion

	// Copy input vectors from host memory to GPU buffers.
	#pragma region memcpyHtD

	cudaStatus = cudaMemcpy(dev_pi, pi, bhpiSize * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_tree, tree, treeSize * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_thphi, thphi, rastSize * sizeof(float2), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_st, stars, starSize * 2 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_mag, mag, starSize * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_cam, camParam, 4 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_ver, ver, N1 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_hor, hor, M1 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	#pragma endregion

	// Launch a kernel on the GPU with one thread for each element.
	dim3 threadsPerBlock(TILE_H, TILE_W);
	// Only works if img width and height is dividable by 16
	dim3 numBlocks(N / threadsPerBlock.x, M / threadsPerBlock.y);

	//for (int q = 0; q < 100; q++) {
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);

		makeImageKernel << <numBlocks, threadsPerBlock >> >(dev_out, dev_thphi, dev_pi, dev_hor, dev_ver,
			dev_st, dev_tree, starSize, dev_cam, dev_mag, treeLevel,
			symmetry, M, N, step, 0);
		cudaEventRecord(stop);

		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		cout << "time to launch kernel " << milliseconds << endl;

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
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}

		#pragma region memcpyDtH
		auto start_time = std::chrono::high_resolution_clock::now();

		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(out, dev_out, M * (N + 2 * step) *  sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
		auto end_time = std::chrono::high_resolution_clock::now();
		cout << " time memcpy in microseconds: " << std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() << endl;

		#pragma endregion

		cv::imshow("bh", img);
		cv::waitKey(0);
	//}



	Error:
		cudaFree(dev_thphi);
		cudaFree(dev_st);
		cudaFree(dev_tree);
		cudaFree(dev_pi);
		cudaFree(dev_out);
		cudaFree(dev_cam);
		cudaFree(dev_mag);
		cudaDeviceReset();
		return cudaStatus;
}

//// texture and pixel objects
//GLuint pbo = 0;     // OpenGL pixel buffer object
//GLuint tex = 0;     // OpenGL texture object
//struct cudaGraphicsResource *cuda_pbo_resource;
//const unsigned int W = 2048;
//const unsigned int H = 1024;
//uchar4 *d_out = 0;
//
//void render() {
//
//   // Launch a kernel on the GPU with one thread for each element.
//   dim3 threadsPerBlock(TILE_H, TILE_W);
//   // Only works if img width and height is dividable by 16
//   dim3 numBlocks(H / threadsPerBlock.x, W / threadsPerBlock.y);
//   int time = glutGet(GLUT_ELAPSED_TIME);
//   makeImageKernel << <numBlocks, threadsPerBlock >> >(d_out, dev_out, dev_thphi, dev_pi, dev_hor, dev_ver,
//	   dev_st, dev_tree, dev_starSize, dev_cam, dev_mag, 9,
//	   true, W, H, 1, time);   
//}
//
// void drawTexture() {
//   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA,
//                GL_UNSIGNED_BYTE, NULL);
//   glEnable(GL_TEXTURE_2D);
//   glBegin(GL_QUADS);
//   glTexCoord2f(0.0f, 0.0f); glVertex2f(0, 0);
//   glTexCoord2f(0.0f, 1.0f); glVertex2f(0, H);
//   glTexCoord2f(1.0f, 1.0f); glVertex2f(W, H);
//   glTexCoord2f(1.0f, 0.0f); glVertex2f(W, 0);
//   glEnd();
//   glDisable(GL_TEXTURE_2D);
//
//}
//
//void display() {
//   render();
//   drawTexture();
//   glutSwapBuffers();
//}
//
//void initGLUT(int *argc, char **argv) {
//   glutInit(argc, argv);
//   glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
//   glutInitWindowSize(W, H);
//   glutCreateWindow("whooptiedoe");
// //#ifndef __APPLE__
//   glewInit();
// //#endif
//
//}
//
//void initPixelBuffer() {
//   glGenBuffers(1, &pbo);
//   glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
//   glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * W*H*sizeof(GLubyte), 0,
//                GL_STREAM_DRAW);
//   glGenTextures(1, &tex);
//   glBindTexture(GL_TEXTURE_2D, tex);
//   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
//   cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo,
//                                cudaGraphicsMapFlagsWriteDiscard);
//
//   cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
//   cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL,
//	   cuda_pbo_resource);
//   cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
//
//}
//void exitfunc() {
//	if (pbo) {
//		cudaGraphicsUnregisterResource(cuda_pbo_resource);
//		glDeleteBuffers(1, &pbo);
//		glDeleteTextures(1, &tex);
//	}
//}
//
//int makeImage(float *out, const float *thphi, const int *pi, const float *ver, const float *hor,
//	const float *stars, const int *starTree, const int starSize, const float *camParam, const float *mag, const int treeLevel,
//	const bool symmetry, const int M, const int N, const int step) {
//	cudaError_t cudaStatus = cudaPrep(out, thphi, pi, ver, hor, stars, starTree, starSize, camParam, mag, treeLevel, symmetry, M, N, step);
//	int foo = 1;
//	char * bar[1] = { " " };
//	initGLUT(&foo, bar);
//	gluOrtho2D(0, 2048, 1024, 0);
//	glutDisplayFunc(display);
//	initPixelBuffer();
//	glutMainLoop();
//	atexit(exitfunc);
//	return 0;
//}