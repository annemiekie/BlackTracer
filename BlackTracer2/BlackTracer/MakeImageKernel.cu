#include "kernel.cuh"
#include <iostream>
#include <stdint.h>
#include <iomanip>
#include <algorithm>
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

#define DIST 1.5f
#define DISTsq (DIST*DIST)

extern int makeImage(int *out, const float *thphi, const int *bh, const int *pi,
	const int size, const float *stars, const int *starTree, const int starSize,
	const float *camParam, const float *magnitude, const bool symmetry, const int M, const int N);

cudaError_t cudaPrep(int *out, const float *thphi, const int *bh, const int *pi,
	const float *stars, const int *starTree, const int starSize,
	const float *camParam, const float *magnitude, const bool symmetry, const int M, const int N);

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
__device__ float euclideanDistSq(float t_a, float t_b, float p_a, float p_b) {
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
__device__ int searchTree(const int *tree, const float *thphiPixMin, const float *thphiPixMax) {
	float nodeStart[2] = { 0.f, 0.f };
	float nodeSize[2] = { PI, PI2 };
	int node = 0;

	for (int level = 1; level <= TREELEVEL; level++) {
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

__global__ void makeImageKernel(int *out, const float *thphi, const int *bh, const int *pi, 
							const float *stars, const int *tree, const int starSize, const float *camParam, const float *magnitude, 
							const bool symmetry, const int M, const int N) {

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
	int sum = 0;

	// Only compute if pixel is not black hole.
	int pibh = (symmetry && i >= N / 2) ? pi[(N - 1 - i)*M + j] : pi[ij];
	if (pibh >= 0) {

		// Set values for projected pixel corners & update phi values in case of 2pi crossing.
		#pragma region Retrieve pixel corners

		float t[4];
		float p[4];
		// Check and correct for 2pi crossings.
		bool picheck = false;
		int M2 = (M + 1) * 2;

		if (symmetry && i >= N / 2) {
			int ix = N - 1 - i;
			int ind = ix * M2 + 2 * j;
			p[0] = thphi[ind + 1];
			p[1] = thphi[ind + M2 + 1];
			p[2] = thphi[ind + M2 + 3];
			p[3] = thphi[ind + 3];
			t[0] = PI - thphi[ind];
			t[1] = PI - thphi[ind + M2];
			t[2] = PI - thphi[ind + M2 + 2];
			t[3] = PI - thphi[ind + 2];

			//if (pibh == 1) picheck = piCheck(p, .2f);
		}
		else {
			int ind = i * M2 + 2 * j;
			t[0] = thphi[ind + M2];
			t[1] = thphi[ind];
			t[2] = thphi[ind + 2];
			t[3] = thphi[ind + M2 + 2];
			p[0] = thphi[ind + M2 + 1];
			p[1] = thphi[ind + 1];
			p[2] = thphi[ind + 3];
			p[3] = thphi[ind + M2 + 3];

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

		int node = searchTree(tree, thphiPixMin, thphiPixMax);

		// Check set of stars in tree
		int startN = 0;
		if (node != 0 && ((node + 1) & node) != 0) {
			startN = tree[node - 1];
		}

		if (tree[node] - startN > 0) {
			float pixVertSize = 1.f;
			float pixHorSize = 1.f;
			float pixSize = pixVertSize * pixHorSize;
			float pixposT = i + .5f;
			float pixposP = j + .5f;
			float frac = pixSize / (fabs(orient) * .5f);
			if (!picheck) {
				for (int q = startN; q < tree[node]; q++) {
					float start = stars[2 * q];
					float starp = stars[2 * q + 1];
					if (starInPolygon(t, p, start, starp, sgn)) {
						interpolate(t[0], t[1], t[2], t[3], p[0], p[1], p[2], p[3], start, starp, sgn);
						// put starlight in own pixel
						float thetaCam = t[1] + pixVertSize * start;
						float phiCam = p[1] + pixHorSize * starp;
						// m2 = m1 - 2.5(log(frac)) where frac = brightnessfraction of b2/b1;
						float appMag = magnitude[q] + 10 * log10(redshift(thetaCam, phiCam, camParam)) 
										- 2.5 * log10(frac*gaussian(starp*starp + start*start));
						//atomicAdd(&out[ij], pow(10., -appMag*0.4));

						// put starlight in surrounding pixels

						//float distsq = euclideanDistSq(pixposT + start, , pixposP + starp, );
						//if (distsq > DISTsq) continue;

						sum++;
					}
				}
			}
			else {
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
						sum++;
					}
				}
			}
		}
	}
	out[ij] = sum;
}

int makeImage(int *out, const float *thphi, const int *bh, const int *pi, const float *stars, 
			const int *starTree, const int starSize, const float *camParam, const float *mag, 
			const bool symmetry, const int M, const int N) {
	cudaError_t cudaStatus = cudaPrep(out, thphi, bh, pi, stars, starTree, starSize, camParam, mag, symmetry, M, N);
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

cudaError_t cudaPrep(int *out, const float *thphi, const int *bh, const int *pi, const float *stars, 
					const int *tree, const int starSize, const float *camParam, const float *mag, 
					const bool symmetry, const int M, const int N) {

	float *dev_thphi = 0;
	float *dev_st = 0;
	float *dev_cam = 0;
	float *dev_mag = 0;
	int *dev_out = 0;
	int *dev_tree = 0;
	int *dev_bh = 0;
	int *dev_pi = 0;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	int bhpiSize = symmetry ? M*N / 2 : M*N;
	int rastSize = symmetry ? M1*N1Half : M1*N1;

	#pragma region malloc
	cudaStatus = cudaMalloc((void**)&dev_bh, bhpiSize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! ta");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_pi, bhpiSize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! tb");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_tree, TREESIZE * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! tc");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_out, M * N * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! td");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_thphi, rastSize * 2 * sizeof(float));
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
		fprintf(stderr, "cudaMalloc failed! ste");
		goto Error;
	}

	#pragma endregion

	// Copy input vectors from host memory to GPU buffers.
	#pragma region memcpyHtD
	cudaStatus = cudaMemcpy(dev_bh, bh, bhpiSize * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_pi, pi, bhpiSize * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_tree, tree, TREESIZE * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_thphi, thphi, rastSize * 2 * sizeof(float), cudaMemcpyHostToDevice);
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
	#pragma endregion

	// Launch a kernel on the GPU with one thread for each element.
	dim3 threadsPerBlock(TILE_H, TILE_W);
	// Only works if img width and height is dividable by 16
	dim3 numBlocks(N / threadsPerBlock.x, M / threadsPerBlock.y);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	makeImageKernel << <numBlocks, threadsPerBlock >> >(dev_out, dev_thphi, dev_bh, dev_pi, 
														dev_st, dev_tree, starSize, dev_cam, dev_mag, 
														symmetry, M, N);
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
	cudaStatus = cudaMemcpy(out, dev_out, M * N * sizeof(int), cudaMemcpyDeviceToHost);
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
	cudaFree(dev_cam);
	cudaFree(dev_mag);
	cudaDeviceReset();
	return cudaStatus;
}
