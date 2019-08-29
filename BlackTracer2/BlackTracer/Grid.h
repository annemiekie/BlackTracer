#pragma once

#include "opencv2/imgcodecs/imgcodecs.hpp"
#include <cmath>
#include <vector>
#include "Metric.h"
#include "Camera.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <unordered_map>
#include <iterator>
#include <stdint.h> 
#include "Const.h"
#include "Code.h"

using namespace cv;
#include "kernel.cuh"
using namespace std;

#define PRECCELEST 0.01
#define ERROR 0.001//1e-6

class Grid
{
private:
	#pragma region private
	/** ------------------------------ VARIABLES ------------------------------ **/

	// Cereal settings for serialization
	friend class cereal::access;
	template < class Archive >
	void serialize(Archive & ar)
	{
		ar(MAXLEVEL, N, M, CamToCel, CamToAD, blockLevels, startblocks, equafactor);
	}

	// Camera & Blackhole
	const Camera* cam;
	const BlackHole* black;

	// Hashing functions (2 options)
	struct hashing_func {
		uint64_t operator()(const uint64_t& key) const {
			uint64_t v = key * 3935559000370003845 + 2691343689449507681;

			v ^= v >> 21;
			v ^= v << 37;
			v ^= v >> 4;

			v *= 4768777513237032717;

			v ^= v << 20;
			v ^= v >> 41;
			v ^= v << 5;

			return v;
		}
	};
	struct hashing_func2 {
		uint64_t  operator()(const uint64_t& key) const{
			uint64_t x = key;
			x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
			x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
			x = x ^ (x >> 31);
			return x;
		}
	};

	// Set of blocks to be checked for division
	unordered_set<uint64_t, hashing_func2> checkblocks;

	bool disk = false;

	/** ------------------------------ POST PROCESSING ------------------------------ **/

	#pragma region post processing

	/// <summary>
	/// Returns if a location lies within the boundaries of the provided polygon.
	/// </summary>
	/// <param name="point">The point (theta, phi) to evaluate.</param>
	/// <param name="thphivals">The theta-phi coordinates of the polygon corners.</param>
	/// <param name="sgn">The winding order of the polygon (+ for CW, - for CCW).</param>
	/// <returns></returns>
	bool pointInPolygon(Point2d& point, vector<Point2d>& thphivals, int sgn) {
		for (int q = 0; q < 4; q++) {
			Point2d vecLine = sgn * (thphivals[q] - thphivals[(q + 1) % 4]);
			Point2d vecPoint = sgn ? (point - thphivals[(q + 1) % 4]) : (point - thphivals[q]);
			if (vecLine.cross(vecPoint) < 0) {
				return false;
			}
		}
		return true;
	}

	/// <summary>
	/// Fixes the t-vertices in the grid.
	/// </summary>
	/// <param name="block">The block to check and fix.</param>
	void fixTvertices(pair<uint64_t, int> block) {
		int level = block.second;
		if (level == MAXLEVEL) return;
		uint64_t ij = block.first;
		if (CamToCel[ij]_phi < 0) return;

		int gap = pow(2, MAXLEVEL - level);
		uint32_t i = i_32;
		uint32_t j = j_32;
		uint32_t k = i + gap;
		uint32_t l = (j + gap) % M;

		checkAdjacentBlock(ij, k_j, level, 1, gap);
		checkAdjacentBlock(ij, i_l, level, 0,gap);
		checkAdjacentBlock(i_l, k_l, level, 1, gap);
		checkAdjacentBlock(k_j, k_l, level, 0, gap);
	}

	/// <summary>
	/// Recursively checks the edge of a block for adjacent smaller blocks causing t-vertices.
	/// Adjusts the value of smaller block vertices positioned on the edge to be halfway
	/// inbetween the values at the edges of the larger block.
	/// </summary>
	/// <param name="ij">The key for one of the corners of the block edge.</param>
	/// <param name="ij2">The key for the other corner of the block edge.</param>
	/// <param name="level">The level of the block.</param>
	/// <param name="udlr">1=down, 0=right</param>
	/// <param name="lr">1=right</param>
	/// <param name="gap">The gap at the current level.</param>
	void checkAdjacentBlock(uint64_t ij, uint64_t ij2, int level, int udlr, int gap) {
		uint32_t i = i_32 + udlr * gap / 2;
		uint32_t j = j_32 + (1-udlr) * gap / 2;
		auto it = CamToCel.find(i_j);
		if (it == CamToCel.end())
			return;
		else {
			uint32_t jprev = (j_32 - (1-udlr) * gap + M) % M;
			uint32_t jnext = (j_32 + (1 - udlr) * 2 * gap) % M;
			uint32_t iprev = i_32 - udlr * gap;
			uint32_t inext = i_32 + 2 * udlr * gap;
			
			bool half = false;

			if (i_32 == 0) {
				jprev = (j_32 + M / 2) % M;
				iprev = gap;
			}
			else if (inext > N - 1) {
				inext = i_32;
				if (equafactor) jnext = (j_32 + M / 2) % M;
				else half = true;
			}
			uint64_t ijprev = (uint64_t)iprev << 32 | jprev;
			uint64_t ijnext = (uint64_t)inext << 32 | jnext;

			bool succes = false;
			if (find(ijprev) && find(ijnext)) {
				vector<Point2d> check = { CamToCel[ijprev], CamToCel[ij], CamToCel[ij2], CamToCel[ijnext] };
				if (CamToCel[ijprev] != Point2d(-1, -1) && CamToCel[ijnext] != Point2d(-1, -1)) {
					succes = true;
					if (half) check[3].x = PI - check[3].x;
					if (metric::check2PIcross(check, 5.)) metric::correct2PIcross(check, 5.);
					CamToCel[i_j] = hermite(0.5, check[0], check[1], check[2], check[3], 0., 0.);
				}
			}
			if (!succes) {
				vector<Point2d> check = { CamToCel[ij], CamToCel[ij2] };
				if (metric::check2PIcross(check, 5.)) metric::correct2PIcross(check, 5.);
				CamToCel[i_j] = 1. / 2.*(check[1] + check[0]);
			}
			if (level + 1 == MAXLEVEL) return;
			checkAdjacentBlock(ij, i_j, level + 1, udlr, gap / 2);
			checkAdjacentBlock(i_j, ij2, level + 1, udlr, gap / 2);
		}
	}

	bool find(uint64_t ij) {
		return CamToCel.find(ij) != CamToCel.end();
	}

	Point2d const hermite(double aValue, Point2d const& aX0, Point2d const& aX1, Point2d const& aX2, Point2d const& aX3, double aTension, double aBias) {
		/* Source:
		* http://paulbourke.net/miscellaneous/interpolation/
		*/

		double const v = aValue;
		double const v2 = v*v;
		double const v3 = v*v2;

		double const aa = (double(1) + aBias)*(double(1) - aTension) / double(2);
		double const bb = (double(1) - aBias)*(double(1) - aTension) / double(2);

		Point2d const m0 = aa * (aX1 - aX0) + bb * (aX2 - aX1);
		Point2d const m1 = aa * (aX2 - aX1) + bb * (aX3 - aX2);

		double const u0 = double(2) *v3 - double(3)*v2 + double(1);
		double const u1 = v3 - double(2)*v2 + v;
		double const u2 = v3 - v2;
		double const u3 = double(-2)*v3 + double(3)*v2;

		return u0*aX1 + u1*m0 + u2*m1 + u3*aX2;
	}

	#pragma endregion

	/** -------------------------------- RAY TRACING -------------------------------- **/

	/// <summary>
	/// Prints the grid cam.
	/// </summary>
	/// <param name="level">The level.</param>
	void printGridCam(int level) {
		cout.precision(2);
		cout << endl;

		int gap = (int)pow(2, MAXLEVEL - level);
		for (uint32_t i = 0; i < N; i += gap) {
			for (uint32_t j = 0; j < M; j += gap) {
				double val = CamToCel[i_j]_theta;
				if (val>1e-10)
					cout << setw(4) << val / PI;
				else
					cout << setw(4) << 0.0;
			}
			cout << endl;
		}

		cout << endl;
		for (uint32_t i = 0; i < N; i += gap) {
			for (uint32_t j = 0; j < M; j += gap) {
				double val = CamToCel[i_j]_phi;
				if (val>1e-10)
					cout << setw(4) << val / PI;
				else
					cout << setw(4) << 0.0;
			}
			cout << endl;
		}
		cout << endl;

		//for (uint32_t i = 0; i < N; i += gap) {
		//	for (uint32_t j = 0; j < M; j += gap) {
		//		int val = CamToAD[i_j];
		//		cout << setw(4) << val;
		//	}
		//	cout << endl;
		//}
		//cout << endl;

		cout.precision(10);
	}

	/// <summary>
	/// Makes the start blocks.
	/// </summary>
	void makeStartBlocks() {
		int gap = (int)pow(2, MAXLEVEL - 1 + equafactor);
		for (uint32_t i = 0; i < N - 1; i += gap) {
			for (uint32_t j = 0; j < M; j += gap) {
				startblocks.push_back(i_j);
			}
		}
	}

	/// <summary>
	/// Raytraces this instance.
	/// </summary>
	void raytrace() {
		int gap = (int)pow(2, MAXLEVEL - STARTLVL);
		int s = (1 + equafactor);

		vector<uint64_t> ijstart(s);

		ijstart[0] = 0;
		if (equafactor) ijstart[1] = (uint64_t)(N - 1) << 32;

		cout << "Computing Level " << STARTLVL << "..." << endl;
		callKernel(ijstart);

		for (uint32_t j = 0; j < M; j += gap) {
			uint32_t i, l, k;
			i = l = k = 0;
			CamToCel[i_j] = CamToCel[k_l];
			if (disk) CamToAD[i_j] = CamToAD[k_l];
			checkblocks.insert(i_j);
			if (equafactor) {
				i = k = N - 1;
				CamToCel[i_j] = CamToCel[k_l];
				if (disk) CamToAD[i_j] = CamToAD[k_l];
			}
		}

		integrateFirst(gap);
		adaptiveBlockIntegration(STARTLVL);
	}

	/// <summary>
	/// Integrates the first blocks.
	/// </summary>
	/// <param name="gap">The gap at the current trace level.</param>
	void integrateFirst(const int gap) {
		vector<uint64_t> toIntIJ;

		for (uint32_t i = gap; i < N - equafactor; i += gap) {
			for (uint32_t j = 0; j < M; j += gap) {
				toIntIJ.push_back(i_j);
				if (i == N - 1);// && !equafactor);
				else if (MAXLEVEL == STARTLVL) blockLevels[i_j] = STARTLVL;
				else checkblocks.insert(i_j);
			}
		}
		callKernel(toIntIJ);

	}

	/// <summary>
	/// Fills the grid map with the just computed raytraced values.
	/// </summary>
	/// <param name="ijvals">The original keys for which rays where traced.</param>
	/// <param name="s">The size of the vectors.</param>
	/// <param name="thetavals">The computed theta values (celestial sky).</param>
	/// <param name="phivals">The computed phi values (celestial sky).</param>
	void fillGridCam(const vector<uint64_t>& ijvals, const size_t s, vector<double>& thetavals, 
		vector<double>& phivals, vector<double>& hitr, vector<double>& hitphi) {
		for (int k = 0; k < s; k++) {
			CamToCel[ijvals[k]] = Point2d(thetavals[k], phivals[k]);
			if (disk) CamToAD[ijvals[k]] = Point2d(hitr[k], hitphi[k]);
		}
	}

	/// <summary>
	/// Calls the kernel.
	/// </summary>
	/// <param name="ijvec">The ijvec.</param>
	void callKernel(const vector<uint64_t>& ijvec) {
		size_t s = ijvec.size();
		vector<double> theta(s), phi(s);

		for (int q = 0; q < s; q++) {
			uint64_t ij = ijvec[q];
			theta[q] = (double)i_32 / (N - 1) * PI / (2 - equafactor);
			phi[q] = (double)j_32 / M * PI2;
		}
		if (disk) {
			vector<double> hitr(s), hitphi(s);
			integration_wrapper(theta, phi, hitr, hitphi, s);
		}
		else {
			integration_wrapper(theta, phi, s);
			vector<double> e1, e2;
			fillGridCam(ijvec, s, theta, phi, e1, e2);
		}
	}

	/// <summary>
	/// Returns if a block needs to be refined.
	/// </summary>
	/// <param name="i">The i position of the block.</param>
	/// <param name="j">The j position of the block.</param>
	/// <param name="gap">The current block gap.</param>
	/// <param name="level">The current block level.</param>
	bool refineCheck(const uint32_t i, const uint32_t j, const int gap, const int level) {
		uint32_t k = i + gap;
		uint32_t l = (j + gap) % M;
		if (disk) {
			double a = CamToAD[i_j].x;
			double b = CamToAD[k_j].x;
			double c = CamToAD[i_l].x;
			double d = CamToAD[k_l].x;
			if (a > 0 || b > 0 || c > 0 || d > 0) {
				return true;
			}
		}

		double th1 = CamToCel[i_j]_theta;
		double th2 = CamToCel[k_j]_theta;
		double th3 = CamToCel[i_l]_theta;
		double th4 = CamToCel[k_l]_theta;

		double ph1 = CamToCel[i_j]_phi;
		double ph2 = CamToCel[k_j]_phi;
		double ph3 = CamToCel[i_l]_phi;
		double ph4 = CamToCel[k_l]_phi;

		double diag = (th1 - th4)*(th1 - th4) + (ph1 - ph4)*(ph1 - ph4);
		double diag2 = (th2 - th2)*(th2 - th3) + (ph2 - ph3)*(ph2 - ph3);

		double max = std::max(diag, diag2);

		if (level < 6 && max>1E-10) return true;
		if (max > PRECCELEST) return true;

		// If no refinement necessary, save level at position.
		blockLevels[i_j] = level;
		return false;

	};

	/// <summary>
	/// Fills the toIntIJ vector with unique instances of theta-phi combinations.
	/// </summary>
	/// <param name="toIntIJ">The vector to store the positions in.</param>
	/// <param name="i">The i key - to be translated to theta.</param>
	/// <param name="j">The j key - to be translated to phi.</param>
	void fillVector(vector<uint64_t>& toIntIJ, uint32_t i, uint32_t j) {
		auto iter = CamToCel.find(i_j);
		if (iter == CamToCel.end()) {
			toIntIJ.push_back(i_j);
			CamToCel[i_j] = Point2d(-10, -10);
		}
	}

	/// <summary>
	/// Adaptively raytraces the grid.
	/// </summary>
	/// <param name="level">The current level.</param>
	void adaptiveBlockIntegration(int level) {

		//size_t checksize = checkblocks.size();
		while (level < MAXLEVEL) {
			if (level<5) printGridCam(level);
			cout << "Computing level " << level + 1 << "..." << endl;

			if (checkblocks.size() == 0) return;

			unordered_set<uint64_t, hashing_func2> todo;
			vector<uint64_t> toIntIJ;

			for (auto ij : checkblocks) {

				uint32_t gap = (uint32_t)pow(2, MAXLEVEL - level);
				uint32_t i = i_32;
				uint32_t j = j_32;
				uint32_t k = i + gap / 2;
				uint32_t l = j + gap / 2;
				j = j % M;

				if (refineCheck(i, j, gap, level)) {
					fillVector(toIntIJ, k, j);
					fillVector(toIntIJ, k, l);
					fillVector(toIntIJ, i, l);
					fillVector(toIntIJ, i + gap, l);
					fillVector(toIntIJ, k, (j + gap) % M);
					todo.insert(i_j);
					todo.insert(k_j);
					todo.insert(k_l);
					todo.insert(i_l);
				}

			}
			callKernel(toIntIJ);
			level++;
			checkblocks = todo;
		}

		for (auto ij : checkblocks)
			blockLevels[ij] = level;
	}

	/// <summary>
	/// Raytraces the rays starting in camera sky from the theta, phi positions defined
	/// in the provided vectors.
	/// </summary>
	/// <param name="theta">The theta positions.</param>
	/// <param name="phi">The phi positions.</param>
	/// <param name="n">The size of the vectors.</param>
	void integration_wrapper(vector<double>& theta, vector<double>& phi, int n) {
		double thetaS = cam->theta;
		double phiS = cam->phi;
		double rS = cam->r;
		double sp = cam->speed;

		#pragma loop(hint_parallel(8))
		#pragma loop(ivdep)
		for (int i = 0; i<n; i++) {

			double xCam = sin(theta[i])*cos(phi[i]);
			double yCam = sin(theta[i])*sin(phi[i]);
			double zCam = cos(theta[i]);

			double yFido = (-yCam + sp) / (1 - sp*yCam);
			double xFido = -sqrtf(1 - sp*sp) * xCam / (1 - sp*yCam);
			double zFido = -sqrtf(1 - sp*sp) * zCam / (1 - sp*yCam);

			double k = sqrt(1 - cam->btheta*cam->btheta);
			double rFido = xFido * cam->bphi/k + cam->br*yFido + cam->br*cam->btheta/k*zFido;
			double thetaFido = cam->btheta*yFido-k*zFido;
			double phiFido = -xFido * cam->br / k + cam->bphi*yFido + cam->bphi*cam->btheta / k*zFido;
			//double rFido = xFido;
			//double thetaFido = -zFido;
			//double phiFido = yFido;

			double eF = 1. / (cam->alpha + cam->w * cam->wbar * phiFido);

			double pR = eF * cam->ro * rFido / sqrtf(cam->Delta);
			double pTheta = eF * cam->ro * thetaFido;
			double pPhi = eF * cam->wbar * phiFido;

			double b = pPhi;
			double q = pTheta*pTheta + cos(thetaS)*cos(thetaS)*(b*b / (sin(thetaS)*sin(thetaS)) - *metric::asq);

			theta[i] = -1;
			phi[i] = -1;
			if (metric::checkCelest(pR, rS, thetaS, b, q))
				metric::rkckIntegrate1(rS, thetaS, phiS, pR, b, q, pTheta, theta[i], phi[i]);
		}
	}

	void integration_wrapper(vector<double>& theta, vector<double>& phi, vector<double>& hitr, vector<double>& hitphi, int n) {
		double thetaS = cam->theta;
		double phiS = cam->phi;
		double rS = cam->r;
		double sp = cam->speed;

		#pragma loop(hint_parallel(8))
		#pragma loop(ivdep)
		//#pragma omp parallel for
		for (int i = 0; i<n; i++) {

			double xCam = sin(theta[i])*cos(phi[i]);
			double yCam = sin(theta[i])*sin(phi[i]);
			double zCam = cos(theta[i]);

			double yFido = (-yCam + sp) / (1 - sp*yCam);
			double xFido = -sqrtf(1 - sp*sp) * xCam / (1 - sp*yCam);
			double zFido = -sqrtf(1 - sp*sp) * zCam / (1 - sp*yCam);

			double rFido = xFido;
			double thetaFido = -zFido;
			double phiFido = yFido;

			double eF = 1. / (cam->alpha + cam->w * cam->wbar * phiFido);

			double pR = eF * cam->ro * rFido / sqrtf(cam->Delta);
			double pTheta = eF * cam->ro * thetaFido;
			double pPhi = eF * cam->wbar * phiFido;

			double b = pPhi;
			double q = pTheta*pTheta + cos(thetaS)*cos(thetaS)*(b*b / (sin(thetaS)*sin(thetaS)) - *metric::asq);

			bool bh = false;
			hitr[i] = 0;
			hitphi[i] = 0;
			if (!metric::checkCelest(pR, rS, thetaS, b, q)) bh = true;
			metric::rkckIntegrate2(rS, thetaS, phiS, pR, b, q, pTheta, theta[i], phi[i], hitr[i], hitphi[i], bh);
			if (bh) {
				theta[i] = -1;
				phi[i] = -1;
			}
		}
	}

	#pragma endregion private

public:

	/// <summary>
	/// 1 if rotation axis != camera axis, 0 otherwise
	/// </summary>
	int equafactor;

	/// <summary>
	/// N = max vertical rays, M = max horizontal rays.
	/// </summary>
	int MAXLEVEL, N, M, STARTN, STARTM, STARTLVL;

	/// <summary>
	/// Mapping from camera sky position to celestial angle.
	/// </summary>
	unordered_map <uint64_t, Point2d, hashing_func2> CamToCel;

	unordered_map <uint64_t, Point2d, hashing_func2> CamToAD;


	/// <summary>
	/// Mapping from block position to level at that point.
	/// </summary>
	unordered_map <uint64_t, int, hashing_func2> blockLevels;

	/// <summary>
	/// Largest blocks making up the first level.
	/// </summary>
	vector<uint64_t> startblocks;

	/// <summary>
	/// Initializes an empty new instance of the <see cref="Grid"/> class.
	/// </summary>
	Grid() {};

	/// <summary>
	/// Initializes a new instance of the <see cref="Grid"/> class.
	/// </summary>
	/// <param name="maxLevelPrec">The maximum level for the grid.</param>
	/// <param name="startLevel">The start level for the grid.</param>
	/// <param name="angle">If the camera is not on the symmetry axis.</param>
	/// <param name="camera">The camera.</param>
	/// <param name="bh">The black hole.</param>
	Grid(const int maxLevelPrec, const int startLevel, const bool angle, const Camera* camera, const BlackHole* bh) {
		MAXLEVEL = maxLevelPrec;
		STARTLVL = startLevel;
		cam = camera;
		black = bh;
		equafactor = angle ? 1. : 0.;

		N = (uint32_t)round(pow(2, MAXLEVEL) / (2 - equafactor) + 1);
		STARTN = (uint32_t)round(pow(2, startLevel) / (2 - equafactor) + 1);
		M = (2 - equafactor) * 2 * (N - 1);
		STARTM = (2 - equafactor) * 2 * (STARTN - 1);

		makeStartBlocks();
		raytrace();
		for (auto block : blockLevels) {
			fixTvertices(block);
		}
	};

	/** UNUSED */
	#pragma region unused

	/// <summary>
	/// Checks for block if it has a high chance of crossing the 2pi border
	/// and calculates the orientation of the block.
	/// </summary>
	/// <param name="block">The block.</param>
	//void orientation2piChecks(pair<uint64_t, int> block) {

	//	int level = block.second;
	//	uint64_t ij = block.first;
	//	if (CamToCel[ij]_phi < 0) return;

	//	uint32_t i = i_32;
	//	uint32_t j = j_32;

	//	uint32_t gap = (uint32_t)pow(2, MAXLEVEL - level);

	//	uint32_t k = i + gap;
	//	uint32_t l = (j + gap) % M;

	//	vector<Point2d> thphivals = { CamToCel[i_j], CamToCel[k_j], CamToCel[k_l], CamToCel[i_l] };

	//	// Check per block if it turns clockwise or counterclockwise --> store per block
	//	// Orientation is positive if CW, negative if CCW
	//	double orientation = 0;
	//	for (int q = 0; q < 4; q++) {
	//		orientation += (thphivals[(q + 1) % 4]_theta - thphivals[q]_theta)
	//			*(thphivals[(q + 1) % 4]_phi + thphivals[q]_phi);
	//	}

	//	int sgn = metric::sgn(orientation);

	//	// Check for blocks that have corners in the range close to 2pi and close to 0
	//	bool suspect = metric::check2PIcross(thphivals, 5.);
	//	crossings2pi[ij] = false;

	//	// Check if a ray down the middle will also fall in the middle of the projection
	//	if (suspect) {
	//		float i_new = (float)i + gap / 2.;
	//		float j_new = (float)j + gap / 2.;
	//		vector<double> theta = { (double)i_new / (N - 1) * PI / (2 - equafactor) };
	//		vector<double> phi = { (double)j_new / M * PI2 };
	//		integration_wrapper(theta, phi, 1);
	//		Point2d thphi = Point2d(theta[0], phi[0]);

	//		// If the ray does not fall into the projection there is a very high chance of
	//		// a 2pi crossing, the orientation is also inverse in this case.
	//		if (!pointInPolygon(thphi, thphivals, sgn)) {
	//			sgn = -sgn;
	//			crossings2pi[ij] = true;
	//		}
	//	}
	//	blockOrientation[ij] = sgn;
	//}

	double maxvec(vector<double>& val) {
		double max = -20;
		for (int i = 0; i < val.size(); ++i) {
			if (val[i] > max) max = val[i];
		}
	}

	void writeToFile(string filename) {
		ofstream raytracedata;
		raytracedata.open(filename);
		int nstarters = startblocks.size();
		raytracedata << CamToCel.size() << " " << N << " " << M << " " << MAXLEVEL << " " << nstarters << '\n';
		uint64_t ij;
		for (int q = 0; q < nstarters; q++) {
			ij = startblocks[q];
			raytracedata << i_32 << " " << j_32 << endl;
		}
		for (auto entry : CamToCel) {
			ij = entry.first;
			Point2d thphi = entry.second;
			uint32_t i = i_32;
			uint32_t j = j_32;
			int level = 0;
			auto it = blockLevels.find(ij);
			if (it != blockLevels.end())
				level = blockLevels[ij];
			double theta = thphi_theta;
			double phi = thphi_phi;
			raytracedata << i << " " << j << " " << theta << " " << phi << " " << level << '\n';
		}
		raytracedata.close();
	}

	bool fileError = 0;
	Grid(string filename) {
		ifstream file;
		file.open(filename);
		uint32_t i, j;
		double theta, phi;
		int numEntries, level, numStart;
		if (file.is_open()) {
			cout << "Reading grid from file..." << endl;
			file >> numEntries;
			file >> N;
			file >> M;
			file >> MAXLEVEL;
			file >> numStart;
			for (int q = 0; q < numStart; q++) {
				file >> i;
				file >> j;
				startblocks.push_back(i_j);
			}
			equafactor = 1;
			if (numStart > 2) equafactor = 0;
			int maxdots = 64;
			for (int q = 0; q < maxdots; q++) {
				cout << "-";
			} cout << "|" << endl;
			int dotspot = numEntries / maxdots;
			for (int q = 0; q < numEntries; q++) {
				if (q%dotspot == 0) cout << ".";
				file >> i;
				file >> j;
				file >> theta;
				file >> phi;
				file >> level;
				CamToCel[i_j] = Point2d(theta, phi);
				if (level != 0) blockLevels[i_j] = level;
			}
			cout << endl;
		}
		else {
			cout << "No such file exists!" << endl;
			fileError = 1;
		}
		file.close();
	};
	#pragma endregion

	/// <summary>
	/// Finalizes an instance of the <see cref="Grid"/> class.
	/// </summary>
	~Grid() {};
};


