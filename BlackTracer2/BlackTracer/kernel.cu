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
//#define R 1
//#define D (R*2+1)
//#define S (D*D)
//#define BLOCK_W (TILE_W+(2*R))
//#define BLOCK_H (TILE_W+(2*R))

#define ij i*M+j

#define tp(x,y) tp[2*x+y] 

#define PI2 6.283185307179586476
#define PI 3.141592653589793238

extern int integration_wrapper(double *out, double *thphi, const int arrSize, const int *bh, const int *pi,
							   const int size, const double *stars, const int *starTree, const int starSize);

cudaError_t cudaPrep(double *out, double *thphi, const int arrSize, const int *bh, const int *pi,
					 const int size, const double *stars, const int *starTree, const int starsize);

__device__ bool checkCrossProduct(double t_a, double t_b, double p_a, double p_b, 
								  double starTheta, double starPhi, int sgn) {
	double c1t = (double)sgn * (t_a - t_b);
	double c1p = (double)sgn * (p_a - p_b);
	double c2t = sgn ? starTheta - t_b : starTheta - t_a;
	double c2p = sgn ? starPhi - p_b : starPhi - p_a;
	return (c1t * c2p - c2t * c1p > 0);
}

__device__ void interpolate(double t0, double t1, double t2, double t3, 
							double p0, double p1, double p2, double p3, 
							double start, double starp, int sgn) {

	double error = 0.00001;

	double midT = (t0 + t1 + t2 + t3) / 4.f;
	double midP = (p0 + p1 + p2 + p3) / 4.f;

	double starInPixY = 0.5f;
	double starInPixX = 0.5f;

	double perc = 0.5;
	int count = 0;
	while ((fabs(start - midT) > error) || (fabs(starp - midP) > error) && count < 100) {
		count++;
		double half01T = (t0 + t1) * .5;
		double half23T = (t2 + t3) * .5;
		double half12T = (t2 + t1) * .5;
		double half03T = (t0 + t3) * .5;
		double half01P = (p0 + p1) * .5;
		double half23P = (p2 + p3) * .5;
		double half12P = (p2 + p1) * .5;
		double half03P = (p0 + p3) * .5;

		double line01to23T = half23T - half01T;
		double line03to12T = half12T - half03T;
		double line01to23P = half23P - half01P;
		double line03to12P = half12P - half03P;

		double line01toStarT = start - half01T;
		double line03toStarT = start - half03T;
		double line01toStarP = starp - half01P;
		double line03toStarP = starp - half03P;

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
		double midT = (t0 + t1 + t2 + t3) * .25;
		double midP = (p0 + p1 + p2 + p3) * .25;
	}
	start = starInPixX;
	starp = starInPixY;
}

__device__ bool starInPixel(double t0, double t1, double t2, double t3,
							double p0, double p1, double p2, double p3,
							double start, double starp, int sgn) {
	return checkCrossProduct(t0, t1, p0, p1, start, starp, sgn) &&
		   checkCrossProduct(t1, t2, p1, p2, start, starp, sgn) &&
		   checkCrossProduct(t2, t3, p2, p3, start, starp, sgn) &&
		   checkCrossProduct(t3, t0, p3, p0, start, starp, sgn);
}

__device__ bool piCheck(double& p0, double& p1, double& p2, double& p3, double factor) {
	double factor1 = PI2*(1.-1./factor);
	bool picheck = false;
	if (p0 > factor1 || p1 > factor1 || p2 > factor1 || p3 > factor1) {
		double factor2 = PI2*1./factor;
		if (p0 < factor2) {
			p0 += PI2;
			picheck = true;
		}
		if (p1 < factor2) {
			p1 += PI2;
			picheck = true;
		}
		if (p2 < factor2) {
			p2 += PI2;
			picheck = true;
		}
		if (p3 < factor2) {
			p3 += PI2;
			picheck = true;
		}
		return picheck;
	}
}

__global__ void interpolateKernel(double *out, double *thphi, const int *bh, const int *pi,  
								  double *stars, const int *_tree, const int starSize) {

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	__shared__ int tree[TREESIZE];
	int lidx = threadIdx.x *blockDim.y + threadIdx.y;
	while (lidx < TREESIZE) {
		tree[lidx] = _tree[lidx];
		lidx += blockDim.y;
	}
	__syncthreads();

	int sum = 0;
	if (bh[ij] == 0) {

		#pragma region indices
		int ind = 2*i*(M + 1) + 2*j;
		int M2 = M * 2;
		double t0 = thphi[ind + M2 + 2];
		double t1 = thphi[ind];
		double t2 = thphi[ind + 2];
		double t3 = thphi[ind + M2 + 4];
		double p0 = thphi[ind + M2 + 3];
		double p1 = thphi[ind + 1];
		double p2 = thphi[ind + 3];
		double p3 = thphi[ind + M2 + 5];
		#pragma endregion

		bool picheck = false;
		if (pi[ij] == 1) {
			picheck = piCheck(p0, p1, p2, p3, 5.);
		}

		// Orientation is positive if CW, negative if CCW
		double orient = (t1 - t0) * (p1 + p0) + (t2 - t1) * (p2 + p1) +
				 (t3 - t2) * (p3 + p2) + (t0 - t3) * (p0 + p3);
		int sgn = orient < 0 ? -1 : 1;

		const double thphiPixMax[2] = { max(max(t0, t1), max(t2, t3)), 
										max(max(p0, p1), max(p2, p3)) };
		const double thphiPixMin[2] = { min(min(t0, t1), min(t2, t3)), 
										min(min(p0, p1), min(p2, p3)) };

		double nodeStart[2] = { 0., 0. };
		double nodeSize[2] = { PI, PI2 };
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
			nodeSize[tp] = nodeSize[tp] / 2.;

			double check = nodeStart[tp] + nodeSize[tp];
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
			double start = stars[2 * q];
			double starp = stars[2 * q + 1];
			if (starInPixel(t0, t1, t2, t3, p0, p1, p2, p3, start, starp, sgn)) {
				sum++;
				interpolate(t0, t1, t2, t3, p0, p1, p2, p3, start, starp, sgn);
			}
			else if (picheck && starp < PI2 / 5.) {
				starp += PI2;
				if (starInPixel(t0, t1, t2, t3, p0, p1, p2, p3, start, starp, sgn)) {
					sum++;
					interpolate(t0, t1, t2, t3, p0, p1, p2, p3, start, starp, sgn);
				}
			}
		}
	}
	out[ij] = sum;
}

int integration_wrapper(double *out, double *thphi, const int arrSize, const int *bh, const int *pi, const int size, 
						const double *stars, const int *starTree, const int starSize) {
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

cudaError_t cudaPrep(double *out, double *thphi, const int arrSize, const int *bh, const int *pi, const int pixSize, 
					 const double *st, const int *tree, const int starSize) {
	double *dev_thphi = 0;
	double *dev_st = 0;
	double *dev_out = 0;
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

	cudaStatus = cudaMalloc((void**)&dev_out, pixSize * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! t");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_thphi, arrSize * 2 * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! t");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_st, starSize * 2 * sizeof(double));
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

	cudaStatus = cudaMemcpy(dev_thphi, thphi, arrSize * 2 * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_st, st, starSize * 2 * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	#pragma endregion

	// Launch a kernel on the GPU with one thread for each element.
	//int threadsPerBlock = 256;
	//int numBlocks = ((int)pixSize + threadsPerBlock - 1) / threadsPerBlock;
	//int sharedMemSize = 2 * sizeof(double);
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
	cudaStatus = cudaMemcpy(out, dev_out, pixSize * sizeof(double), cudaMemcpyDeviceToHost);
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
