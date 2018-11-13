#include "kernel.cuh"
#include <iostream>
#include <stdint.h>
#include <iomanip>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
using namespace std;

#define L(x,y) __launch_bounds__(x,y)

extern int integration_wrapper(double *t0, double *p0, double *t1, double *p1, double *t2, double *p2, double *t3, double *p3, const int *bh, const int *pi,
								const int size, const double *starTheta, const double *starPhi, const int starSize);
cudaError_t cudaPrep(double *t0, double *p0, double *t1, double *p1, double *t2, double *p2, double *t3, double *p3, const int *bh, const int *pi,
					const int size, const double *st, const double *sp, const int starsize);


DEVICE bool checkCrossProduct(double t0, double t1, double p0, double p1, double starTheta, double starPhi, int sgn) {
	double c1t = (double)sgn * (t0 - t1);
	double c1p = (double)sgn * (p0 - p1);
	double c2t = sgn ? starTheta - t1 : starTheta - t0;
	double c2p = sgn ? starPhi - p1 : starPhi - p0;
	return (c1t * c2p - c2t * c1p > 0);
}

__device__ void interpolate(double t0, double t1, double t2, double t3, double p0, double p1, double p2, double p3, double start, double starp, int sgn) {

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

__global__ void interpolateKernel(double *t0, double *p0, double *t1, double *p1, double *t2, double *p2, double *t3, double *p3, const int *bh, const int *pi,
									const int n, const double *starTheta, const double *starPhi, const int starSize) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	//__shared__ Grid grid;
	//grid = *_grid;

	__shared__ double PI;
	PI = 3.141592653589793238;
	__shared__ double PI2;
	PI2 = PI*2.;
	
	for (int i = index; i < n; i += stride) {
		double orient = 0.;
		int sum = 0;
		bool picheck = false;
		if (bh[i] == 0) {
			if (pi[i] == 1) {
				double factor1 = PI2*0.8;
				if (p0[i] > factor1 || p1[i] > factor1 || p2[i] > factor1 || p3[i] > factor1) {
					double factor2 = PI2*0.2;
					if (p0[i] < factor2) {
						p0[i] += PI2;
						picheck = true;
					}
					if (p1[i] < factor2) {
						p1[i] += PI2;
						picheck = true;
					}
					if (p2[i] < factor2) {
						p2[i] += PI2;
						picheck = true;
					}
					if (p3[i] < factor2) {
						p3[i] += PI2;
						picheck = true;
					}
				}
			}

			// Orientation is positive if CW, negative if CCW
			orient = (t1[i] - t0[i]) *(p1[i] + p0[i]) +
				(t2[i] - t1[i]) *(p2[i] + p1[i]) +
				(t3[i] - t2[i]) *(p3[i] + p2[i]) +
				(t0[i] - t3[i]) *(p0[i] + p3[i]);
			int sgn = 1;
			if (orient < 0) {
				sgn = -1;
			}
			for (int q = 0; q < starSize; q++) {
				double start = starTheta[q];
				double starp = starPhi[q];
				if (checkCrossProduct(t0[i], t1[i], p0[i], p1[i], start, starp, sgn) &&
					checkCrossProduct(t1[i], t2[i], p1[i], p2[i], start, starp, sgn) &&
					checkCrossProduct(t2[i], t3[i], p2[i], p3[i], start, starp, sgn) &&
					checkCrossProduct(t3[i], t0[i], p3[i], p0[i], start, starp, sgn)) {
					sum++;
					interpolate(t0[i], t1[i], t2[i], t3[i], p0[i], p1[i], p2[i], p3[i], start, starp, sgn);
				}
				else if (picheck && starp < PI2 / 5.) {
					starp += PI2;
					if (checkCrossProduct(t0[i], t1[i], p0[i], p1[i], start, starp, sgn) &&
						checkCrossProduct(t1[i], t2[i], p1[i], p2[i], start, starp, sgn) &&
						checkCrossProduct(t2[i], t3[i], p2[i], p3[i], start, starp, sgn) &&
						checkCrossProduct(t3[i], t0[i], p3[i], p0[i], start, starp, sgn)) {
						sum++;
						interpolate(t0[i], t1[i], t2[i], t3[i], p0[i], p1[i], p2[i], p3[i], start, starp, sgn);
					}
				}
			}

		}
		t0[i] = orient;
		p0[i] = sum;
	}
}

int integration_wrapper(double *t0, double *p0, double *t1, double *p1, double *t2, double *p2, double *t3, double *p3, 
						const int *bh, const int *pi, const int size, const double *starTheta, const double *starPhi, const int starSize) {
	cudaError_t cudaStatus = cudaPrep(t0, p0, t1, p1, t2, p2, t3, p3, bh, pi, size, starTheta, starPhi, starSize);
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

void checkGpuMem() {
	float free_m, total_m, used_m;
	size_t free_t, total_t;
	cudaMemGetInfo(&free_t, &total_t);
	free_m = (uint32_t)free_t / 1048576.0f;
	total_m = (uint32_t)total_t / 1048576.0f;

	used_m = total_m - free_m;
	printf("  mem free %d .... %f MB mem total %d....%f MB mem used %f MB\n", free_t, free_m, total_t, total_m, used_m);

}

cudaError_t cudaPrep(double *t0, double *p0, double *t1, double *p1, double *t2, double *p2, double *t3, double *p3, 
					const int *bh, const int *pi, const int size, const double *st, const double *sp, const int starsize) {
	double *dev_t1 = 0;
	double *dev_p1 = 0;
	double *dev_t2 = 0;
	double *dev_p2 = 0;
	double *dev_t3 = 0;
	double *dev_p3 = 0;
	double *dev_t0 = 0;
	double *dev_p0 = 0;
	double *dev_st = 0;
	double *dev_sp = 0;
	int *dev_bh = 0;
	int *dev_pi = 0;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_bh, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! t");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_pi, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! t");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_t0, size * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! t");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_p0, size * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! p");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_t1, size * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! t");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_p1, size * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! p");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_t2, size * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! t");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_p2, size * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! p");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_t3, size * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! t");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_p3, size * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! p");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_st, starsize * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! st");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_sp, starsize * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! sp");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_bh, bh, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_pi, pi, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_t0, t0, size * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_p0, p0, size * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_t1, t1, size * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_p1, p1, size * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_t2, t2, size * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_p2, p2, size * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_t3, t3, size * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_p3, p3, size * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_st, st, starsize * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_sp, sp, starsize * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	int blockSize = 256;
	int numBlocks = ((int)size + blockSize - 1) / blockSize;
	interpolateKernel << <numBlocks, blockSize >> >(dev_t0, dev_p0, dev_t1, dev_p1, dev_t2, dev_p2, dev_t3, dev_p3, dev_bh, dev_pi, size, dev_st, dev_sp, starsize);
	//rayIntegrateKernel<<<1,1>>>(size, dev_t, dev_p, dev_cam);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(t0, dev_t0, size * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(p0, dev_p0, size * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_t0);
	cudaFree(dev_p0);
	cudaFree(dev_t1);
	cudaFree(dev_p1);	
	cudaFree(dev_t2);
	cudaFree(dev_p2);	
	cudaFree(dev_t3);
	cudaFree(dev_p3);
	cudaFree(dev_sp);
	cudaFree(dev_st);
	cudaFree(dev_bh);
	cudaFree(dev_pi);
	cudaDeviceReset();
	return cudaStatus;
}