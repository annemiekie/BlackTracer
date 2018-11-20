#include "kernel.cuh"
#include <iostream>
#include <stdint.h>
#include <iomanip>
#include <algorithm>
#include <thrust/device_vector.h>
using namespace std;

#define L(x,y) __launch_bounds__(x,y)

#define M 1024
#define N 512

#define TILE_W 8
#define TILE_H 8
#define RASTER_W (TILE_W + 1)
#define SHARE (TILE_W*TILE_H+TILE_W+TILE_H+1)*2
//#define R 1
//#define D (R*2+1)
//#define S (D*D)
//#define BLOCK_W (TILE_W+(2*R))
//#define BLOCK_H (TILE_W+(2*R))

#define ij i*M+j

#define tp(x,y) tp[2*x+y] 

#define PI2 6.283185307179586476
#define PI 3.141592653589793238

extern int integration_wrapper(float *out, float *thphi, const int arrSize, const int *bh, const int *pi,
							   const int size, const float *stars, const int *starTree, const int starSize);

cudaError_t cudaPrep(float *out, float *thphi, const int arrSize, const int *bh, const int *pi,
					 const int size, const float *stars, const int *starTree, const int starsize);

__device__ bool checkCrossProduct(float t_a, float t_b, float p_a, float p_b, 
								  float starTheta, float starPhi, int sgn) {
	float c1t = (float)sgn * (t_a - t_b);
	float c1p = (float)sgn * (p_a - p_b);
	float c2t = sgn ? starTheta - t_b : starTheta - t_a;
	float c2p = sgn ? starPhi - p_b : starPhi - p_a;
	return (c1t * c2p - c2t * c1p > 0);
}

__device__ void interpolate(float t0, float t1, float t2, float t3, 
							float p0, float p1, float p2, float p3, 
							float start, float starp, int sgn) {

	float error = 0.00001;

	float midT = (t0 + t1 + t2 + t3) * .25f;
	float midP = (p0 + p1 + p2 + p3) * .25f;

	float starInPixY = 0.5f;
	float starInPixX = 0.5f;

	float perc = 0.5;
	int count = 0;
	while ((fabs(start - midT) > error) || (fabs(starp - midP) > error) && count < 100) {
		count++;
		float half01T = (t0 + t1) * .5;
		float half23T = (t2 + t3) * .5;
		float half12T = (t2 + t1) * .5;
		float half03T = (t0 + t3) * .5;
		float half01P = (p0 + p1) * .5;
		float half23P = (p2 + p3) * .5;
		float half12P = (p2 + p1) * .5;
		float half03P = (p0 + p3) * .5;

		float line01to23T = half23T - half01T;
		float line03to12T = half12T - half03T;
		float line01to23P = half23P - half01P;
		float line03to12P = half12P - half03P;

		float line01toStarT = start - half01T;
		float line03toStarT = start - half03T;
		float line01toStarP = starp - half01P;
		float line03toStarP = starp - half03P;

		int a = (line03to12T * line03toStarP - line03toStarT * line03to12P) > 0 ? 1 : -1;
		int b = (line01to23T * line01toStarP - line01toStarT * line01to23P) > 0 ? 1 : -1;

		perc *= 0.5;

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
		float midT = (t0 + t1 + t2 + t3) * .25f;
		float midP = (p0 + p1 + p2 + p3) * .25f;
	}
	start = starInPixX;
	starp = starInPixY;
}

__device__ bool starInPixel(float t[4], float p[4], float start, float starp, int sgn) {
	#pragma unroll
	for (int q = 0; q < 4; q++) {
		if (!checkCrossProduct(t[q], t[(q + 1) % 4], p[q], p[(q + 1) % 4], start, starp, sgn)) return false;
	}
	return true;
}

__device__ bool piCheck(float (&p)[4], float factor) {
	float factor1 = PI2*(1.f-factor);
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

__global__ void interpolateKernel(float *out, float *thphi, const int *bh, const int *pi,  
								  float *stars, const int *tree, const int starSize) {

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	//__shared__ int tree[TREESIZE];
	//int lidx = threadIdx.x *blockDim.y + threadIdx.y;
	//while (lidx < TREESIZE) {
	//	tree[lidx] = _tree[lidx];
	//	lidx += blockDim.y * blockDim.x;
	//}

	__shared__ float s_tp[SHARE];
	int idx = threadIdx.x * blockDim.y + threadIdx.y;
	int blockStart = (blockIdx.x * blockDim.x) * (M + 1) + (blockIdx.y * blockDim.y);
	float raster = (float)RASTER_W;
	while (idx < SHARE / 2) {
		float idxf = idx + 0.000001;
		int mod = (int) fmod(idxf, raster);
		int ind = 2 * (blockStart + mod + (M + 1) * (int)(idxf / raster));
		s_tp[2 * idx] = thphi[ind];
		s_tp[2 * idx + 1] = thphi[ind + 1];
		idx += blockDim.y * blockDim.x;
	}

	__syncthreads();

	int sum = 0;
	if (bh[ij] == 0) {

		//#pragma region indices
		//int ind = 2 * i*(M + 1) + 2 * j;
		//int M2 = (M + 1) * 2;
		//float t[4] = { thphi[ind + M2], thphi[ind], thphi[ind + 2], thphi[ind + M2 + 2] };
		//float p[4] = { thphi[ind + M2 + 1], thphi[ind + 1], thphi[ind + 3], thphi[ind + M2 + 3] };

		int idx2 = 2 * (threadIdx.x * RASTER_W + threadIdx.y);
		int R2 = RASTER_W * 2;
		float t[4] = { s_tp[idx2 + R2],		s_tp[idx2],		s_tp[idx2 + 2],	s_tp[idx2 + R2 + 2] };
		float p[4] = { s_tp[idx2 + R2 + 1], s_tp[idx2 + 1], s_tp[idx2 + 3], s_tp[idx2 + R2 + 3] };

		#pragma endregion

		bool picheck = false;
		if (pi[ij] == 1) {
			picheck = piCheck(p, .2f);
		}

		// Orientation is positive if CW, negative if CCW
		float orient = (t[1] - t[0]) * (p[1] + p[0]) + (t[2] - t[1]) * (p[2] + p[1]) +
					   (t[3] - t[2]) * (p[3] + p[2]) + (t[0] - t[3]) * (p[0] + p[3]);
		int sgn = orient < 0 ? -1 : 1;

		const float thphiPixMax[2] = { max(max(t[0], t[1]), max(t[2], t[3])), 
									   max(max(p[0], p[1]), max(p[2], p[3])) };
		const float thphiPixMin[2] = { min(min(t[0], t[1]), min(t[2], t[3])),
									   min(min(p[0], p[1]), min(p[2], p[3])) };

		float nodeStart[2] = { 0., 0. };
		float nodeSize[2] = { PI, PI2 };
		int level = 0;
		int node = 0;

		while (level < TREELEVEL) {
			int star_n = tree[node];
			if (node != 0 && ((node + 1) & node) != 0) {
				star_n -= tree[node - 1];
			}
			if (star_n == 0) break;
			level++;
			int tp = level % 2;
			nodeSize[tp] = nodeSize[tp] * .5;

			float check = nodeStart[tp] + nodeSize[tp];
			bool lu = thphiPixMin[tp] < check;
			bool rd = thphiPixMax[tp] > check;
			if (lu && rd) {
				break;
			}
			else if (lu) node = node * 2 + 1;
			else if (rd) {
				node = node * 2 + 2;
				nodeStart[tp] = nodeStart[tp] + nodeSize[tp];
			}
		}

		int start = 0;
		if (node != 0 && ((node + 1) & node) != 0) {
			start = tree[node - 1];
		}
		for (int q = start; q < tree[node]; q++) {
			float start = stars[2 * q];
			float starp = stars[2 * q + 1];
			if (starInPixel(t, p, start, starp, sgn)) {
				sum++;
				interpolate(t[0], t[1], t[2], t[3], p[0], p[1], p[2], p[3], start, starp, sgn);
			}
			else if (picheck && starp < PI2 * .2f) {
				starp += PI2;
				if (starInPixel(t, p, start, starp, sgn)) {
					sum++;
					interpolate(t[0], t[1], t[2], t[3], p[0], p[1], p[2], p[3], start, starp, sgn);
				}
			}
		}
	}
	out[ij] = sum;
}

int integration_wrapper(float *out, float *thphi, const int arrSize, const int *bh, const int *pi, const int size, 
						const float *stars, const int *starTree, const int starSize) {
	cudaError_t cudaStatus = cudaPrep(out, thphi, arrSize, bh, pi, size, stars, starTree, starSize);
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

cudaError_t cudaPrep(float *out, float *thphi, const int arrSize, const int *bh, const int *pi, const int pixSize, 
					 const float *st, const int *tree, const int starSize) {
	float *dev_thphi = 0;
	float *dev_st = 0;
	float *dev_out = 0;
	int *dev_tree = 0;
	int *dev_bh = 0;
	int *dev_pi = 0;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	
	#pragma region malloc
	cudaStatus = cudaMalloc((void**)&dev_bh, pixSize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! t");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_pi, pixSize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! t");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_tree, TREESIZE * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! t");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_out, pixSize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! t");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_thphi, arrSize * 2 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! t");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_st, starSize * 2 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! st");
		goto Error;
	}

	#pragma endregion

	// Copy input vectors from host memory to GPU buffers.
	#pragma region memcpyHtD
	cudaStatus = cudaMemcpy(dev_bh, bh, pixSize * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_pi, pi, pixSize * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_tree, tree, TREESIZE * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_thphi, thphi, arrSize * 2 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_st, st, starSize * 2 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	#pragma endregion

	// Launch a kernel on the GPU with one thread for each element.
	//int threadsPerBlock = 256;
	//int numBlocks = ((int)pixSize + threadsPerBlock - 1) / threadsPerBlock;
	//int sharedMemSize = 2 * sizeof(float);
	dim3 threadsPerBlock(TILE_H, TILE_W);
	// Only works if img width and height is dividable by 16
	dim3 numBlocks(N / threadsPerBlock.x, M / threadsPerBlock.y);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	interpolateKernel << <numBlocks, threadsPerBlock >> >(dev_out, dev_thphi, dev_bh, dev_pi, dev_st, dev_tree, starSize);
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

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(out, dev_out, pixSize * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	#pragma endregion

Error:
	cudaFree(dev_thphi);
	cudaFree(dev_st);
	cudaFree(dev_tree);
	cudaFree(dev_bh);
	cudaFree(dev_pi);
	cudaFree(dev_out);
	cudaDeviceReset();
	return cudaStatus;
}
