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

extern void makeImage(const float2 *thphi, const int *pi, const float *ver, const float *hor,
					const float *stars, const int *starTree, const int starSize, const float *camParam, 
					const float *magnitude, const int treeLevel, const bool symmetry, 
					const int M, const int N, const int step, const Mat csImage, const int G, 
					const float gridStart, const float gridStep, const float *pixsize, const float2 *hits,
					const float2 *gradient);

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
	vector<Grid>* grids;

	/// <summary>
	/// View window defined by the user (default: spherical panorama).
	/// </summary>
	Viewer* view;

	/// <summary>
	/// Camera defined by the user.
	/// </summary>
	vector<Camera>* cams;

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
	uint64_t findBlock(uint64_t ij, double theta, double phi, int level, int gridNum) {

		if ((*grids)[gridNum].blockLevels[ij] <= level) {
			return ij;
		}
		else {
			int gap = (int)pow(2, (*grids)[gridNum].MAXLEVEL - level);
			uint32_t i = i_32;
			uint32_t j = j_32;
			uint32_t k = i + gap / 2;
			uint32_t l = j + gap / 2;
			double thHalf = PI2*k / (1. * (*grids)[gridNum].M);
			double phHalf = PI2*l / (1. * (*grids)[gridNum].M);
			if (thHalf <= theta) i = k;
			if (phHalf <= phi) j = l;
			return findBlock(i_j, theta, phi, level + 1, gridNum);
		}
	};

	/// <summary>
	/// Find the starting block of the grid and the starting level of the block.
	/// The starting block number depends on the symmetry and the value of phi.
	/// </summary>
	/// <param name="lvlstart">The lvlstart.</param>
	/// <param name="phi">The phi.</param>
	/// <returns></returns>
	int findStartingBlock(int& lvlstart, double phi, int gridNum) {
		if (!(*grids)[gridNum].equafactor) {
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

	Point2d const hermite(double aValue, Point2d const& aX0, Point2d const& aX1, Point2d const& aX2, Point2d const& aX3, double aTension, double aBias) {
		/* Source:
		* http://paulbourke.net/miscellaneous/interpolation/
		*/

		double const v = aValue;
		double const v2 = v*v;
		double const v3 = v*v2;

		double const aa = (1. + aBias)*(1. - aTension) / 2.;
		double const bb = (1. - aBias)*(1. - aTension) / 2.;

		Point2d const m0 = aa * (aX1 - aX0) + bb * (aX2 - aX1);
		Point2d const m1 = aa * (aX2 - aX1) + bb * (aX3 - aX2);

		double const u0 = 2. *v3 - 3.*v2 + 1.;
		double const u1 = v3 - 2.*v2 + v;
		double const u2 = v3 - v2;
		double const u3 = -2.*v3 + 3.*v2;

		return u0*aX1 + u1*m0 + u2*m1 + u3*aX2;
	}

	Point2d intersection(Point2d Aa, Point2d B, Point2d C, Point2d D) {
		// Line AB represented as a1x + b1y = c1 
		double a1 = B.y - Aa.y;
		double b1 = Aa.x - B.x;
		double c1 = a1*(Aa.x) + b1*(Aa.y);

		// Line CD represented as a2x + b2y = c2 
		double a2 = D.y - C.y;
		double b2 = C.x - D.x;
		double c2 = a2*(C.x) + b2*(C.y);

		double determinant = a1*b2 - a2*b1;
		if (determinant == 0) {
			return Point2d(-1, -1);;
		}
		double x = (b2*c1 - b1*c2) / determinant;
		double y = (a1*c2 - a2*c1) / determinant;
		return Point2d(x, y);
	}

	bool find(uint64_t ij, int gridNum) {
		return (*grids)[gridNum].CamToCel.find(ij) != (*grids)[gridNum].CamToCel.end();
	}

	Point2d findPoint(uint32_t i, uint32_t j, int offver, int offhor, int gridNum, int gap) {
		uint64_t ij = (uint64_t)i << 32 | j;
		if (!find(ij,gridNum)) {
			uint32_t j2 = (j + offhor*gap + (*grids)[gridNum].M) % (*grids)[gridNum].M;
			uint32_t i2 = i + offver*gap;
			ij = (uint64_t)i2 << 32 | j2;

			if (find(ij, gridNum)) {
				if ((*grids)[gridNum].CamToCel[ij] == Point2d(-1, -1)) return Point2d(-1, -1);

				uint32_t j0 = (j - offhor * gap + (*grids)[gridNum].M) % (*grids)[gridNum].M;
				uint32_t i0 = (i - offver * gap);
				uint64_t ij0 = (uint64_t)i0 << 32 | j0;

				if (!find(ij0, gridNum)) return Point2d(-1, -1);
				if ((*grids)[gridNum].CamToCel[ij0] == Point2d(-1, -1)) return Point2d(-1, -1);


				uint32_t jprev = (j - 3 * offhor * gap + (*grids)[gridNum].M) % (*grids)[gridNum].M;
				uint32_t jnext = (j + 3 * offhor * gap + (*grids)[gridNum].M) % (*grids)[gridNum].M;
				uint32_t iprev = i - offver * 3 * gap;
				uint32_t inext = i + offver * 3 * gap;
				uint64_t ijprev = (uint64_t)iprev << 32 | jprev;
				uint64_t ijnext = (uint64_t)inext << 32 | jnext;
				bool a = false;
				bool b = false;
				if (offver!=0) {
					if (i2 == 0) {
						jnext = (j0 + (*grids)[gridNum].M / 2) % (*grids)[gridNum].M;
						inext = i0;
					}
					else if (i0 == (*grids)[gridNum].N - 1) {
						if ((*grids)[gridNum].equafactor==1) jprev = (j0 + (*grids)[gridNum].M / 2) % (*grids)[gridNum].M;
						else a = true;
						iprev = i2;
					} 
					else if (i2 == (*grids)[gridNum].N - 1) {
						inext = i0;
						if ((*grids)[gridNum].equafactor ==1) jnext = (j0 + (*grids)[gridNum].M / 2) % (*grids)[gridNum].M;
						else b = true;
					}
				}

				if (find(ijprev, gridNum) && find(ijnext, gridNum)) {
					vector<Point2d> check = { (*grids)[gridNum].CamToCel[ijprev], (*grids)[gridNum].CamToCel[ij0],
											(*grids)[gridNum].CamToCel[ij], (*grids)[gridNum].CamToCel[ijnext] };
					if (check[0] != Point2d(-1, -1) || check[3] != Point2d(-1, -1)) {
						if (metric::check2PIcross(check, 5.)) metric::correct2PIcross(check, 5.);
						if (a) check[3].x = PI - check[3].x;
						if (b) check[0].x = PI - check[0].x;
						return  hermite(0.5, check[0], check[1], check[2], check[3], 0., 0.);
					}
				}
				vector<Point2d> check = { (*grids)[gridNum].CamToCel[ij], (*grids)[gridNum].CamToCel[ij0] };
				if (metric::check2PIcross(check, 5.)) metric::correct2PIcross(check, 5.);
				return .5 * (check[0] + check[1]);
			}
			else {
				uint32_t j0 = (j + gap) % (*grids)[gridNum].M;
				uint32_t j1 = (j - gap + (*grids)[gridNum].M) % (*grids)[gridNum].M;
				uint64_t ij0 = (uint64_t)(i + gap) << 32 | j0;
				uint64_t ij1 = (uint64_t)(i - gap) << 32 | j0;		
				uint64_t ij2 = (uint64_t)(i - gap) << 32 | j1;
				uint64_t ij3 = (uint64_t)(i + gap) << 32 | j1;
				if (!find(ij0, gridNum) || !find(ij1, gridNum) || !find(ij2, gridNum) || !find(ij3, gridNum)) 
					return Point2d(-1, -1);
				Point2d answer  = interpolate4(ij2, .5, .5, gridNum);
				if (answer == Point2d(-1, -1)) {
					vector<Point2d> check = { (*grids)[gridNum].CamToCel[ij0], (*grids)[gridNum].CamToCel[ij1], 
											  (*grids)[gridNum].CamToCel[ij2], (*grids)[gridNum].CamToCel[ij3] };
					for (int q = 0; q < check.size(); q++) {
						if (check[q] == Point2d(-1, -1)) return Point2d(-1, -1);
					}
					if (metric::check2PIcross(check, 5.)) metric::correct2PIcross(check, 5.);
					return .25 * (check[0] + check[1] + check[2] + check[3]);
				}
				else return answer;
			}
		}
		else {
			return (*grids)[gridNum].CamToCel[ij];
		}
	}

	Point2d interpolate4(const uint64_t ij, const double percDown, const double percRight, const int gridNum) {
		vector<double> cornersCam(4);
		vector<Point2d> cornersCel(4);
		calcCornerVals(ij, cornersCel, cornersCam, gridNum);
		for (int q = 0; q < 4; q++) {
			if (cornersCel[q] == Point2d(-1, -1)) return Point2d(-1, -1);
		}
		return interpolateHermite(ij, percDown, percRight, gridNum, cornersCel);
	}

	Point2d interpolate(const uint64_t ij, const double thetaCam, const double phiCam, const int gridNum, bool splines) {
		vector<double> cornersCam(4);
		vector<Point2d> cornersCel(4);
		calcCornerVals(ij, cornersCel, cornersCam, gridNum);
		for (int q = 0; q < 4; q++) {
			if (cornersCel[q] == Point2d(-1, -1)) return Point2d(-1, -1);
		}

		double thetaUp = cornersCam[0];
		double thetaDown = cornersCam[2];
		double phiLeft = cornersCam[1];
		double phiRight = cornersCam[3];

		if (thetaUp == thetaCam) {
			if (phiLeft == phiCam) return cornersCel[0];
			if (phiRight == phiCam) return cornersCel[1];
			if (i_32 == 0) return cornersCel[0];

		}
		else if (thetaDown == thetaCam) {
			if (phiLeft == phiCam) return cornersCel[2];
			if (phiRight == phiCam) return cornersCel[3];
		}

		double percDown = (thetaCam - thetaUp) / (thetaDown - thetaUp);
		double percRight = (phiCam - phiLeft) / (phiRight - phiLeft);
		if (splines) return interpolateHermite(ij, percDown, percRight, gridNum, cornersCel);
		else return interpolateLinear(ij, percDown, percRight, cornersCel);
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
	Point2d interpolateLinear(uint64_t ij, double percDown, double percRight, vector<Point2d> &cornersCel) {
		if (metric::check2PIcross(cornersCel, 5.)) metric::correct2PIcross(cornersCel, 5.);
		Point2d left = cornersCel[0] + percDown * (cornersCel[2] - cornersCel[0]);
		Point2d right = cornersCel[1] + percDown * (cornersCel[3] - cornersCel[1]);
		Point2d up = cornersCel[0] + percRight * (cornersCel[1] - cornersCel[0]);
		Point2d down = cornersCel[2] + percRight * (cornersCel[3] - cornersCel[2]);
		return intersection(up, down, left, right);
	}

	Point2d interpolateHermite(const uint64_t ij, const double percDown, const double percRight, 
							   const int gridNum, vector<Point2d> &cornersCel) {
		int gap = (int)pow(2, (*grids)[gridNum].MAXLEVEL - (*grids)[gridNum].blockLevels[ij]);

		uint32_t i = i_32;
		uint32_t j = j_32;
		uint32_t k = i + gap;
		uint32_t l = (j + gap) % (*grids)[gridNum].M;
		uint32_t imin1 = i - gap;
		uint32_t kplus1 = k + gap;
		uint32_t jmin1 = (j - gap + (*grids)[gridNum].M) % (*grids)[gridNum].M;
		uint32_t lplus1 = (l + gap) % (*grids)[gridNum].M;;
		uint32_t jx = j;
		uint32_t jy = j;
		uint32_t lx = l;
		uint32_t ly = l;
		bool half = false;

		if (i == 0) {
			jx = (j + (*grids)[gridNum].M / 2) % (*grids)[gridNum].M;
			lx = (jx + gap) % (*grids)[gridNum].M;
			imin1 = k;
		}
		else if (k == (*grids)[gridNum].N - 1) {
			if ((*grids)[gridNum].equafactor) {
				jy = (j + (*grids)[gridNum].M / 2) % (*grids)[gridNum].M;
				ly = (jy + gap) % (*grids)[gridNum].M;
			}
			else half = true;
			kplus1 = i;
		}

		cornersCel.push_back(findPoint(i, jmin1, 0, -1, gridNum, gap));		//4 upleft
		cornersCel.push_back(findPoint(i, lplus1, 0, 1, gridNum, gap));		//5 upright
		cornersCel.push_back(findPoint(k, jmin1, 0, -1, gridNum, gap));		//6 downleft
		cornersCel.push_back(findPoint(k, lplus1, 0, 1, gridNum, gap));		//7 downright
		cornersCel.push_back(findPoint(imin1, jx, -1, 0, gridNum, gap));	//8 lefthigh
		cornersCel.push_back(findPoint(imin1, lx, -1, 0, gridNum, gap));	//9 righthigh
		cornersCel.push_back(findPoint(kplus1, jy, 1, 0, gridNum, gap));	//10 leftdown
		cornersCel.push_back(findPoint(kplus1, ly, 1, 0, gridNum, gap));	//11 rightdown
		if (half) {
			cornersCel[10].x = PI - cornersCel[10].x;
			cornersCel[11].x = PI - cornersCel[11].x;
		}

		for (int q = 4; q < cornersCel.size(); q++) {
			if (cornersCel[q] == Point2d(-1, -1)) return interpolateLinear(ij, percDown, percRight, cornersCel);
		}
		if (metric::check2PIcross(cornersCel, 5.)) metric::correct2PIcross(cornersCel, 5.);

		Point2d interpolateUp = hermite(percRight, cornersCel[4], cornersCel[0], cornersCel[1], cornersCel[5], 0.f, 0.f);
		Point2d interpolateDown = hermite(percRight, cornersCel[6], cornersCel[2], cornersCel[3], cornersCel[7], 0.f, 0.f);
		Point2d interpolateUpUp = cornersCel[8] + (cornersCel[9] - cornersCel[8]) * percRight;
		Point2d interpolateDownDown = cornersCel[10] + (cornersCel[11] - cornersCel[10]) * percRight;
		//HERMITE FINITE
		return hermite(percDown, interpolateUpUp, interpolateUp, interpolateDown, interpolateDownDown, 0.f, 0.f);
	}

	float calcArea(float t[4], float p[4]) {
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

	float calcAreax(float t[3], float p[3]) {
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
		float y[3] = { yi[0],yi[1], yi[2] };
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

	Point2d getHits(uint64_t ij, int gridNum) {
		int level = (*grids)[gridNum].blockLevels[ij];
		int gap = (int)pow(2, (*grids)[gridNum].MAXLEVEL - level);
		uint32_t i = i_32;
		uint32_t j = j_32;
		uint32_t k = i + gap;
		// l should convert to 2PI at end of grid for correct interpolation
		uint32_t l = (j + gap)% (*grids)[gridNum].M;
		Point2d a = (*grids)[gridNum].CamToAD[ij];
		Point2d b = (*grids)[gridNum].CamToAD[i_l];
		Point2d c = (*grids)[gridNum].CamToAD[k_j];
		Point2d d = (*grids)[gridNum].CamToAD[k_l];
		Point2d mean = (a + b + c + d)/4.;
		return mean;
	}

	/// <summary>
	/// Calculates the camera sky and celestial sky corner values for a given block.
	/// </summary>
	/// <param name="ij">The key to the block.</param>
	/// <param name="cornersCel">Vector to store the celestial sky positions.</param>
	/// <param name="cornersCam">Vector to store the camera sky positions.</param>
	void calcCornerVals(uint64_t ij, vector<Point2d>& cornersCel, vector<double>& cornersCam, int gridNum) {
		int level = (*grids)[gridNum].blockLevels[ij];
		int gap = (int)pow(2, (*grids)[gridNum].MAXLEVEL - level);
		uint32_t i = i_32;
		uint32_t j = j_32;
		uint32_t k = i + gap;
		// l should convert to 2PI at end of grid for correct interpolation
		uint32_t l = j + gap;
		double factor = PI2 / (1.0*(*grids)[gridNum].M);
		cornersCam = { factor*i, factor*j, factor*k, factor*l };
		// but for lookup l should be the 0-point
		l = l % (*grids)[gridNum].M;
		cornersCel = { (*grids)[gridNum].CamToCel[ij], (*grids)[gridNum].CamToCel[i_l], (*grids)[gridNum].CamToCel[k_j], (*grids)[gridNum].CamToCel[k_l] };
	}
	#pragma endregion

	/** -------------------------------- STARS -------------------------------- **/
	#pragma region stars

	void cudaTest() {
		// fill arrays with data

		auto start_time = std::chrono::high_resolution_clock::now();

		int Gr = grids->size();
		int N = view->pixelheight;
		int M = view->pixelwidth;
		int H = symmetry ? N/2 : N;
		int W = M;
		int H1 = H + 1;
		int W1 = W + 1;
		vector<float2> thphi(Gr*H1*W1);
		vector<int> pi(Gr*H*W);
		vector<float> ver(N + 1);
		vector<float> hor(W1);
		vector<float> camParams;
		vector<float2> hit(Gr*H*W);
		cout.precision(17);
		vector<int> gap(H1*W1);
		vector<float> pixsizeIn(2*H*W);
		vector<float> pixsizeOut(Gr*H*W);
		vector<float2> gradient_field(Gr*H1*W1);
		Mat mat_gradient_field = Mat::zeros(H1, W1, CV_32FC2);

		for (int g = 0; g < Gr; g++) {
			vector<int> pix(H1*W1);
			vector<Point2d> hx(H1*W1);
			//for (int aaah = 0; aaah < 100; aaah++) {
			auto start_time2 = std::chrono::high_resolution_clock::now();
			#pragma omp parallel for 
			for (int t = 0; t < H1; t++) {
				double theta = view->ver[t];
				for (int p = 0; p < W1; p++) {
					double phi = view->hor[p];
					int lvlstart = 0;
					int quarter = findStartingBlock(lvlstart, phi, g);

					uint64_t ij = findBlock((*grids)[g].startblocks[quarter], theta, phi, lvlstart, g);
					Point2d thphiInter = interpolate(ij, theta, phi, g, true);

					int i = t*W1 + p;
					// for filter
					int level = (*grids)[g].blockLevels[ij];
					gap[i] = (int)pow(2, (*grids)[g].MAXLEVEL - level);

					//if (t < 16 && (p>968 && p<990)) {
					//	cout << t << " " << p << " " << thphiInter.x - diff.x << " " << thphiInter.y - diff.y << endl;// " " << (*grids)[g].CamToCel[ij].x << " " << (*grids)[g].CamToCel[ij].y << endl;
					//	diff = thphiInter;
					//}
					//uint32_t k = (uint32_t)t;
					//uint32_t j = (uint32_t)p % W;
					//uint64_t ij = (uint64_t)k << 32 | j;
					//Point2d thphiInter = (*grids)[g].CamToCel[ij];
					if (thphiInter == Point2d(-1, -1)) {
						pix[i] = -1;
					}
					else {
						metric::wrapToPi(thphiInter.x, thphiInter.y);
					}
					thphi[H1*W1*g + i].x = thphiInter.x;
					thphi[H1*W1*g + i].y = thphiInter.y;
					//hx[i] = getHits(ij, g);
				}
			}
			auto end_time2 = std::chrono::high_resolution_clock::now();
			cout << std::chrono::duration_cast<std::chrono::milliseconds>(end_time2 - start_time2).count() << endl;
			//}
			// Fill pi with black hole and picrossing info
			#pragma omp parallel for
			for (int t = 0; t < H; t++) {
				int pi1 = pix[t*W1] + pix[(t + 1)*W1];
				Point2d h1 = hx[t*W1] + hx[(t + 1)*W1];
				for (int p = 0; p < W; p++) {
					int i = t*W + p;
					int pi2 = pix[t*W1 + p + 1] + pix[(t + 1)*W1 + p + 1];
					Point2d h2 = hx[t*W1 + p + 1] + hx[(t + 1)*W1 + p + 1];
					pi[H*W*g + i] = pi1 + pi2;
					Point2d hmean = (h1 + h2) / 4.f;
					hit[H*W*g + i].x = hmean.x;
					hit[H*W*g + i].y = hmean.y;
					pi1 = pi2;
					h1 = h2;
				}
			}

			int smoothnumber = 0;// 2;
			
			// calc area array
			#pragma omp parallel for
			for (int t = 0; t < H; t++) {
				for (int p = 0; p < W; p++) {
					int i = t*W1 + p;
					float theta[4] = { thphi[i].x, thphi[i + 1].x, thphi[i + W1 + 1].x, thphi[i + W1].x };
					float phi[4] = { thphi[i].y, thphi[i + 1].y, thphi[i + W1 + 1].y, thphi[i + W1].y };
					if (pi[H*W*g + t*W + p] >= 0) {
						for (int q = 0; q < 4; q++) {
							if (phi[q] > .8*PI2) {
								for (int r = 0; r < 4; r++) {
									if (phi[r] < .2*PI2) phi[r] += PI2;
								}
								break;
							}
						}
						float th1[3] = { theta[0], theta[1], theta[2] };
						float ph1[3] = { phi[0], phi[1], phi[2] };
						float th2[3] = { theta[0], theta[2], theta[3] };
						float ph2[3] = { phi[0], phi[2], phi[3] };
						float area1 = calcAreax(th1, ph1);
						float area2 = calcAreax(th2, ph2);
						float area = area1 + area2;
			
						if (smoothnumber>0) pixsizeIn[t*W + p] = area;
						else pixsizeOut[g*H*W + t*W + p] = area;
					}
				}
			}
			
			// smoothing filter for area
			for (int st = 0; st < smoothnumber; st++) {
				#pragma omp parallel for
				for (int t = 0; t < H; t++) {
					for (int p = 0; p < W; p++) {
						if (pi[H*W*g + t*W + p] < 0) continue;
						int fs = max(max(gap[t*W1 + p], gap[t*W1 + W1 + p]), max(gap[t*W1 + p + 1], gap[t*W1 + W1 + p + 1]));
						fs = fs / 2;
						float sum = 0.;
						int minv = max(0, t - fs);
						int maxv = min(H - 1, t + fs);
						int count = 0;
			
						for (int v = minv; v <= maxv; v++) {
							for (int h = -fs; h <= fs; h++) {
								if (pi[H*W*g + v*W + (p + h + W) % W] >= 0) {
									sum += pixsizeIn[(st % 2)*H*W + v*W + (p + h + W) % W];
									count++;
								}
							}
						}
						if (st < smoothnumber - 1) {
							pixsizeIn[(1 - (st % 2))*H*W + t*W + p] = sum / (1.*count);
						}
						else {
							pixsizeOut[g*H*W + t*W + p] = sum / (1.*count);
							//if (t == 50) {
								//cout << p << " " << pixsizeOut[g*H*W + t*W + p] << endl;
							//}
						}
					}
				}
			}

			camParams.push_back((*cams)[g].speed);
			camParams.push_back((*cams)[g].alpha);
			camParams.push_back((*cams)[g].w);
			camParams.push_back((*cams)[g].wbar);


			#pragma omp parallel for
			for (int t = 0; t < H1;t++) {
				for (int p = 0; p < W1; p++) {
					float up = thphi[min(H1 - 1, (t + 1))*W1 + p].x;
					float down = thphi[max(0, (t - 1))*W1 + p].x;
					float left = thphi[t*W1 + max(0, (p - 1))].x;
					float right = thphi[t*W1 + min(W1, p + 1)].x;
					float mid = thphi[t*W1 + p].x;
					if (mid > 0) {
						float xdir = 0;
						//if (mid < .2f*PI2) {
						//	if (up > .8f*PI2 || down > .8f*PI2 || left > .8f*PI2 || right > .8f*PI2) {
						//		mid += PI2;
						//		if (up < .2f *PI2) up += PI2;
						//		if (down < .2f *PI2) down += PI2;
						//		if (left < .2f *PI2) left += PI2;
						//		if (right < .2f *PI2) right += PI2;
						//	}
						//}
						//if (mid > .8f*PI2) {
						//	if (up < .2f *PI2) up += PI2;
						//	if (down < .2f *PI2) down += PI2;
						//	if (left < .2f *PI2) left += PI2;
						//	if (right < .2f *PI2) right += PI2;
						//}
						if (up > 0 && down > 0)
							xdir = .5f * (up - mid) - .5f*(down - mid);
						float ydir = 0;
						if (left>0 && right > 0)
							ydir = .5f*(right - mid) - .5f*(left - mid);
						float size = sqrt(xdir*xdir + ydir*ydir);
						xdir /= size;
						ydir /= size;

						gradient_field[H1*W1*g +t*W1 +p] = float2{ -ydir, xdir };
					}
				}
			}


		}

		for (int t = 0; t < N+1; t++) {
			ver[t] = view->ver[t];
		}
		for (int t = 0; t < W1; t++) {
			hor[t] = view->hor[t];
		}

		int step = 1;
		float gridStart = (*cams)[0].r;
		float gridStep = ((*cams)[Gr-1].r - gridStart) / (1.f*Gr);

		auto end_time = std::chrono::high_resolution_clock::now();
		cout << "Interpolated grid in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms!" << endl << endl;



		makeImage(&thphi[0], &pi[0], &ver[0], &hor[0],
			&(starTree->starPos[0]), &(starTree->binaryStarTree[0]), starTree->starSize, &camParams[0], 
			&(starTree->starMag[0]), starTree->treeLevel, symmetry, M, N, step, starTree->imgWithStars, 
			Gr, gridStart, gridStep, &pixsizeOut[0], &hit[0], &gradient_field[0]);

		Mat g_img = Mat::zeros(H, W1, DataType<Vec3b>::type);
		for (int i = 0; i < H; i += 16) {
			for (int j = 0; j < W1; j += 16) {
				Point2f p(j, i);
				Point2f grad = mat_gradient_field.at<Point2f>(p) * 10.f;
				if (fabs(grad.x) > fabs(grad.y) && fabs(grad.x) > 20.f) {
					grad.y *= fabs(20.f / grad.x);
					grad.x = metric::sgn(grad.x)*20.f;
				}
				else if (fabs(grad.y) > fabs(grad.x) && fabs(grad.y) > 20.f) {
					grad.x *= fabs(20.f / grad.y);
					grad.y = metric::sgn(grad.y)*20.f;
				}
				Point2f p2(grad+p);
				arrowedLine(g_img, p, p2, Scalar(255, 0, 0), 1.5, 8, 0, 0.1);
			}
		}
		namedWindow("vector", WINDOW_AUTOSIZE);
		imshow("vector", g_img);
		waitKey(0);
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
	Distorter(vector<Grid>* deformgrid, Viewer* viewer, StarProcessor* _stars, vector<Camera>* camera) {
		// Load variables
		grids = deformgrid;
		view = viewer;
		starTree = _stars;
		cams = camera;

		if ((*grids)[0].equafactor == 0) symmetry = true;

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

	void drawBlocks(string filename, int g) {
		//Mat gridimg(2 * view->pixelheight + 2, 2 * view->pixelwidth + 2, CV_8UC1, Scalar(255));
		Mat gridimg((*grids)[g].N, (*grids)[g].M, CV_8UC1, Scalar(255));
		double sj = 1.;// *gridimg.cols / (*grids)[g].M;
		double si = 1.;// *gridimg.rows / ((*grids)[g].N - 1);/// *(2 - (*grids)[g].equafactor) + 1);
		for (auto block : (*grids)[g].blockLevels) {
			uint64_t ij = block.first;
			int level = block.second;
			int gap = pow(2, (*grids)[g].MAXLEVEL - level);
			uint32_t i = i_32;
			uint32_t j = j_32;
			uint32_t k = i_32 + gap;
			uint32_t l = j_32 + gap;

			//if (gap < 128) {
				//cout << gap << endl;
				//cout << i << endl;
				line(gridimg, Point2d(j*sj, i*si), Point2d(j*sj, k*si), Scalar(0), 1);// , CV_AA);
				line(gridimg, Point2d(l*sj, i*si), Point2d(l*sj, k*si), Scalar(0), 1);// , CV_AA);
				line(gridimg, Point2d(j*sj, i*si), Point2d(l*sj, i*si), Scalar(0), 1);// , CV_AA);
				line(gridimg, Point2d(j*sj, k*si), Point2d(l*sj, k*si), Scalar(0), 1);// , CV_AA);
			//}
		}
		//namedWindow("blocks", WINDOW_AUTOSIZE);
		//imshow("blocks", gridimg);
		imwrite(filename, gridimg);
		//waitKey(0);
	}
	#pragma endregion

	/// <summary>
	/// Finalizes an instance of the <see cref="Distorter"/> class.
	/// </summary>
	~Distorter() {	};
	
};