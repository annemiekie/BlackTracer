#pragma once
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdint.h> 
#include "Grid.h"
#include "Viewer.h"
#include "Metric.h"
#include "Const.h"
#include "StarProcessor.h"
#include "Camera.h"
#include "kernel.cuh"
#include <chrono>

using namespace cv;
using namespace std;

extern int makeImage(float *out, const float2 *thphi, const int *pi, const float *ver, const float *hor,
					const float *stars, const int *starTree, const int starSize, const float *camParam, const float *magnitude, const int treeLevel,
					const bool symmetry, const int M, const int N, const int step, const Mat csImage);

/// <summary>
/// Class which handles all the computations that have to do with the distortion
/// of the input image and / or list of stars, given a computed grid and user input.
/// Can be used to produce a distorted output image.
/// </summary>
class Distorter
{
private:

	/** ------------------------------ VARIABLES ------------------------------ **/
	#pragma region variables
	/// <summary>
	/// Grid with computed rays from camera sky to celestial sky.
	/// </summary>
	Grid* grid;

	/// <summary>
	/// View window defined by the user (default: spherical panorama).
	/// </summary>
	Viewer* view;

	/// <summary>
	/// Camera defined by the user.
	/// </summary>
	Camera* cam;

	/// <summary>
	/// Star info including binary tree over
	/// list with stars (point light sources) present in the celestial sky.
	/// </summary>
	StarProcessor* starTree;

	bool symmetry = false;

	#pragma endregion

	/** ---------------------------- INTERPOLATING ---------------------------- **/
	#pragma region interpolating

	/// <summary>
	/// Recursively finds the block the specified theta,phi on the camera sky fall into.
	/// </summary>
	/// <param name="ij">The key of the current block the search is in.</param>
	/// <param name="theta">The theta on the camera sky.</param>
	/// <param name="phi">The phi on the camera sky.</param>
	/// <param name="level">The current level to search.</param>
	/// <returns>The key of the block.</returns>
	uint64_t findBlock(uint64_t ij, double theta, double phi, int level) {

		if (grid->blockLevels[ij] <= level) {
			return ij;
		}
		else {
			int gap = (int)pow(2, grid->MAXLEVEL - level);
			uint32_t i = i_32;
			uint32_t j = j_32;
			uint32_t k = i + gap / 2;
			uint32_t l = j + gap / 2;
			double thHalf = PI2*k / grid->M;
			double phHalf = PI2*l / grid->M;
			if (thHalf < theta) i = k;
			if (phHalf < phi) j = l;
			return findBlock(i_j, theta, phi, level + 1);
		}
	};

	/// <summary>
	/// Find the starting block of the grid and the starting level of the block.
	/// The starting block number depends on the symmetry and the value of phi.
	/// </summary>
	/// <param name="lvlstart">The lvlstart.</param>
	/// <param name="phi">The phi.</param>
	/// <returns></returns>
	int findStartingBlock(int& lvlstart, double phi) {
		if (!grid->equafactor) {
			lvlstart = 1;
			if (phi > PI) {
				return (phi > 3.f * PI1_2) ? 3 : 2;
			}
			else if (phi > PI1_2) return 1;
		}
		else {
			if (phi > PI) return 1;
		}
		return 0;
	}

	/// <summary>
	/// Interpolates the celestial sky values for the corners of the block with key ij
	/// to find the celestial sky value for the provided theta-phi point that lies within
	/// this block on the camera sky.
	/// </summary>
	/// <param name="ij">The key to the block the pixelcorner is located in.</param>
	/// <param name="thetaCam">Theta value of the pixel corner (on the camera sky).</param>
	/// <param name="phiCam">Phi value of the pixel corner (on the camera sky).</param>
	/// <returns>The Celestial sky position the provided theta-phi value will map to.</returns>
	Point2d interpolate(uint64_t ij, double thetaCam, double phiCam) {
		vector<double> cornersCam(4);
		vector<Point2d> cornersCel(4);
		calcCornerVals(ij, cornersCel, cornersCam);

		double error = 0.00001;
		double thetaUp = cornersCam[0];
		double thetaDown = cornersCam[2];
		double phiLeft = cornersCam[1];
		double phiRight = cornersCam[3];

		double thetaMid = 1000;
		double phiMid = 1000;
		if (thetaUp == thetaCam) {
			if (phiLeft == phiCam) return cornersCel[0];
			if (phiRight == phiCam) return cornersCel[1];
		}
		else if (thetaDown == thetaCam) {
			if (phiLeft == phiCam) return cornersCel[2];
			if (phiRight == phiCam) return cornersCel[3];
		}

		if (grid->crossings2pi[ij] || metric::check2PIcross(cornersCel, 5.)) metric::correct2PIcross(cornersCel, 5.);
		Point2d lu = cornersCel[0];
		Point2d ru = cornersCel[1];
		Point2d ld = cornersCel[2];
		Point2d rd = cornersCel[3];
		if (lu == Point2d(-1, -1) || ru == Point2d(-1, -1) || ld == Point2d(-1, -1) || rd == Point2d(-1, -1)) return Point2d(-1, -1);

		while ((fabs(thetaCam - thetaMid) > error) || (fabs(phiCam - phiMid) > error)) {
			thetaMid = (thetaUp + thetaDown) * HALF;
			phiMid = (phiRight + phiLeft) * HALF;
			Point2d mall = (lu + ld + rd + ru)*(1.f / 4.f);
			if (thetaCam > thetaMid) {
				Point2d md = (rd + ld) * HALF;
				thetaUp = thetaMid;
				if (phiCam > phiMid) {
					phiLeft = phiMid;
					lu = mall;
					ru = (ru + rd)*HALF;
					ld = md;
				}
				else {
					phiRight = phiMid;
					ru = mall;
					rd = md;
					lu = (lu + ld)*HALF;
				}
			}
			else {
				thetaDown = thetaMid;
				Point2d mu = (ru + lu) * HALF;
				if (phiCam > phiMid) {
					phiLeft = phiMid;
					ld = mall;
					lu = mu;
					rd = (ru + rd)*HALF;
				}
				else {
					phiRight = phiMid;
					rd = mall;
					ru = mu;
					ld = (lu + ld)*HALF;
				}
			}
		}
		Point2d meanVal = (ru + lu + rd + ld)*(1. / 4.);
		metric::wrapToPi(meanVal.x, meanVal.y);
		return meanVal;
	}

	/// <summary>
	/// Calculates the camera sky and celestial sky corner values for a given block.
	/// </summary>
	/// <param name="ij">The key to the block.</param>
	/// <param name="cornersCel">Vector to store the celestial sky positions.</param>
	/// <param name="cornersCam">Vector to store the camera sky positions.</param>
	void calcCornerVals(uint64_t ij, vector<Point2d>& cornersCel, vector<double>& cornersCam) {
		int level = grid->blockLevels[ij];
		int gap = (int)pow(2, grid->MAXLEVEL - level);
		uint32_t i = i_32;
		uint32_t j = j_32;
		uint32_t k = i + gap;
		// l should convert to 2PI at end of grid for correct interpolation
		uint32_t l = j + gap;
		double factor = PI2 / grid->M;
		cornersCam = { factor*i, factor*j, factor*k, factor*l };
		// but for lookup l should be the 0-point
		l = l % grid->M;
		cornersCel = { grid->CamToCel[ij], grid->CamToCel[i_l], grid->CamToCel[k_j], grid->CamToCel[k_l] };
	}
	#pragma endregion

	/** -------------------------------- STARS -------------------------------- **/
	#pragma region stars

	void cudaTest() {
		// fill arrays with data
		int N = view->pixelheight;
		int M = view->pixelwidth;
		int H = symmetry ? N/2 : N;
		int W = M;
		int H1 = H + 1;
		int W1 = W + 1;
		vector<float2> thphi(H1*W1);
		vector<int> pix(H1*W1);
		vector<int> pi(H*W);
		vector<float> ver(N + 1);
		vector<float> hor(W1);

		//#pragma omp parallel for
		for (int t = 0; t < H1; t++) {
			double theta = view->ver[t];
			for (int p = 0; p < W1; p++) {
				double phi = view->hor[p];
				int lvlstart = 0;
				int quarter = findStartingBlock(lvlstart, phi);
				uint64_t ij = findBlock(grid->startblocks[quarter], theta, phi, lvlstart);
				Point2d thphiInter = interpolate(ij, theta, phi);
				//uint32_t k = (uint32_t) t;
				//uint32_t j = (uint32_t) p;
				//uint64_t ij = (uint64_t)k << 32 | j;
				//Point2d thphiInter = grid->CamToCel[ij];
				int i = t*W1 + p;
				if (thphiInter != Point2d(-1, -1)) {
					pix[i] = 0;// grid->crossings2pi[i_j] ? 1 : 0;
				}
				else {
					pix[i] = -4;
				}
				thphi[i].x = thphiInter.x;
				thphi[i].y = thphiInter.y;
			}
		}

		// Fill pi with black hole and picrossing info
		#pragma omp parallel for
		for (int t = 0; t < H; t++) {
			int pi1 = pix[t*W1] + pix[(t + 1)*W1];
			for (int p = 0; p < W; p++) {
				int i = t*W + p;
				int pi2 = pix[t*W1 + p + 1] + pix[(t + 1)*W1 + p + 1];
				pi[i] = pi1 + pi2;
				pi1 = pi2;
			}
		}

		for (int t = 0; t < N+1; t++) {
			ver[t] = view->ver[t];
		}
		for (int t = 0; t < W1; t++) {
			hor[t] = view->hor[t];
		}

		int step = 1;
		vector<float> out(N*M);
		makeImage(&out[0], &thphi[0], &pi[0], &ver[0], &hor[0],
			&(starTree->starPos[0]), &(starTree->binaryStarTree[0]), starTree->starSize, cam->getParamArray(), 
			&(starTree->starMag[0]), starTree->treeLevel, symmetry, M, N, step, starTree->imgWithStars);
	}
	#pragma endregion

public:

	/// <summary>
	/// Initializes a new empty instance of the <see cref="Distorter"/> class.
	/// </summary>
	Distorter() {};

	/// <summary>
	/// Initializes a new instance of the <see cref="Distorter"/> class.
	/// </summary>
	/// <param name="deformgrid">The grid with mappings from camera sky to celestial sky.</param>
	/// <param name="viewer">The viewer that produces the output image.</param>
	/// <param name="_stars">The star info including binary tree with stars to display.</param>
	/// <param name="camera">The camera the output image is viewed from.</param>
	Distorter(Grid* deformgrid, Viewer* viewer, StarProcessor* _stars, Camera* camera) {
		// Load variables
		grid = deformgrid;
		view = viewer;
		starTree = _stars;
		cam = camera;
		if (grid->equafactor == 0) symmetry = true;

		// Precomputations on the celestial sky image.
		rayInterpolater();
	};

	/// <summary>
	/// Computes everything the distorter is supposed to compute.
	/// </summary>
	void rayInterpolater() {
		auto start_time = std::chrono::high_resolution_clock::now();
		cout << "Interpolating..." << endl;
		cudaTest();
		auto end_time = std::chrono::high_resolution_clock::now();
		cout << " time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << endl;
	};


	/**	EXTRA METHODS - NOT NECESSARY **/
	#pragma region unused

	void drawBlocks(string filename) {
		Mat gridimg(2 * view->pixelheight + 2, 2 * view->pixelwidth + 2, CV_8UC1, Scalar(255));
		double sj = 1.*gridimg.cols / grid->M;
		double si = 1.*gridimg.rows / ((grid->N - 1)*(2 - grid->equafactor) + 1);
		for (auto block : grid->blockLevels) {
			uint64_t ij = block.first;
			int level = block.second;
			int gap = pow(2, grid->MAXLEVEL - level);
			uint32_t i = i_32;
			uint32_t j = j_32;
			uint32_t k = i_32 + gap;
			uint32_t l = j_32 + gap;

			line(gridimg, Point2d(j*sj, i*si), Point2d(j*sj, k*si), Scalar(0), 1, CV_AA);
			line(gridimg, Point2d(l*sj, i*si), Point2d(l*sj, k*si), Scalar(0), 1, CV_AA);
			line(gridimg, Point2d(j*sj, i*si), Point2d(l*sj, i*si), Scalar(0), 1, CV_AA);
			line(gridimg, Point2d(j*sj, k*si), Point2d(l*sj, k*si), Scalar(0), 1, CV_AA);
		}
		namedWindow("blocks", WINDOW_AUTOSIZE);
		imshow("blocks", gridimg);
		imwrite(filename, gridimg);
		waitKey(0);
	}
	#pragma endregion

	/// <summary>
	/// Finalizes an instance of the <see cref="Distorter"/> class.
	/// </summary>
	~Distorter() {	};
	
};