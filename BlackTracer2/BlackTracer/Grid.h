#pragma once

#include "opencv2/imgcodecs/imgcodecs.hpp"
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cmath>
#include <vector>
#include "Metric.h"
#include "Camera.h"
#include "BlackHole.h"
#include <iomanip>
#include <unordered_map>
#include <unordered_set>
#include <iterator>
#include <stdint.h> 
#include "Const.h"
#include "Code.h"
#include "PSHOffsetTable.h"
#include <chrono>
#include <numeric>
#include <fstream>
#define PRECCELEST 0.015
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
		ar(MAXLEVEL, N, M, hasher);
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
	std::unordered_set<uint64_t, hashing_func2> checkblocks;

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
	bool pointInPolygon(cv::Point2d& point, vector<cv::Point2d>& thphivals, int sgn) {
		for (int q = 0; q < 4; q++) {
			cv::Point2d vecLine = sgn * (thphivals[q] - thphivals[(q + 1) % 4]);
			cv::Point2d vecPoint = sgn ? (point - thphivals[(q + 1) % 4]) : (point - thphivals[q]);
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
				vector<cv::Point2d> check = { CamToCel[ijprev], CamToCel[ij], CamToCel[ij2], CamToCel[ijnext] };
				if (CamToCel[ijprev] != cv::Point2d(-1, -1) && CamToCel[ijnext] != cv::Point2d(-1, -1)) {
					succes = true;
					if (half) check[3].x = PI - check[3].x;
					if (metric::check2PIcross(check, 5.)) metric::correct2PIcross(check, 5.);
					CamToCel[i_j] = hermite(0.5, check[0], check[1], check[2], check[3], 0., 0.);
				}
			}
			if (!succes) {
				vector<cv::Point2d> check = { CamToCel[ij], CamToCel[ij2] };
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

	cv::Point2d const hermite(double aValue, cv::Point2d const& aX0, cv::Point2d const& aX1, cv::Point2d const& aX2, cv::Point2d const& aX3, double aTension, double aBias) {
		/* Source:
		* http://paulbourke.net/miscellaneous/interpolation/
		*/

		double const v = aValue;
		double const v2 = v*v;
		double const v3 = v*v2;

		double const aa = (double(1) + aBias)*(double(1) - aTension) / double(2);
		double const bb = (double(1) - aBias)*(double(1) - aTension) / double(2);

		cv::Point2d const m0 = aa * (aX1 - aX0) + bb * (aX2 - aX1);
		cv::Point2d const m1 = aa * (aX2 - aX1) + bb * (aX3 - aX2);

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
		std::cout.precision(2);
		std::cout << endl;

		int gap = (int)pow(2, MAXLEVEL - level);
		for (uint32_t i = 0; i < N; i += gap) {
			for (uint32_t j = 0; j < M; j += gap) {
				double val = CamToCel[i_j]_theta;
				if (val>1e-10)
					std::cout << setw(4) << val / PI;
				else
					std::cout << setw(4) << 0.0;
			}
			std::cout << endl;
		}

		std::cout << endl;
		for (uint32_t i = 0; i < N; i += gap) {
			for (uint32_t j = 0; j < M; j += gap) {
				double val = CamToCel[i_j]_phi;
				if (val>1e-10)
					std::cout << setw(4) << val / PI;
				else
					std::cout << setw(4) << 0.0;
			}
			std::cout << endl;
		}
		std::cout << endl;

		std::cout << endl;
		for (uint32_t i = 0; i < N; i += gap) {
			for (uint32_t j = 0; j < M; j += gap) {
				double val = steps[i*M+j];
				std::cout << setw(4) << val;
			}
			std::cout << endl;
		}
		std::cout << endl;

		int sum = 0;
		int sumnotnull = 0;
		int countnotnull = 0;
		ofstream myfile;
		myfile.open("steps.txt");
		for (int i = 0; i < N*M; i++) {
			sum += steps[i];
			if (steps[i] > 0) {
				sumnotnull += steps[i];
				countnotnull++;
			}
			myfile << steps[i] << "\n";
		}
		myfile.close();
		cout << "steeeeps" << sum << endl;
		cout << "steeeepsnotnull" << sumnotnull << endl;

		cout << "ave" << (float)sum / (float)(M*(N + 1)) << endl;
		cout << "avenotnull" << (float)sumnotnull / (float)(countnotnull) << endl;

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
	/// Raytraces this instance.
	/// </summary>
	void raytrace() {
		int gap = (int)pow(2, MAXLEVEL - STARTLVL);
		int s = (1 + equafactor);

		vector<uint64_t> ijstart(s);

		ijstart[0] = 0;
		if (equafactor) ijstart[1] = (uint64_t)(N - 1) << 32;

		if (print) std::cout << "Computing Level " << STARTLVL << "..." << endl;
		callKernel(ijstart);

		for (uint32_t j = 0; j < M; j += gap) {
			uint32_t i, l, k;
			i = l = k = 0;
			CamToCel[i_j] = CamToCel[k_l];
			steps[i*M + j] = steps[0];
			checkblocks.insert(i_j);
			if (equafactor) {
				i = k = N - 1;
				CamToCel[i_j] = CamToCel[k_l];
				steps[i*M + j] = steps[0];

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
		vector<double>& phivals, vector<double>& hitr, vector<double>& hitphi, vector<int>& step) {
		for (int k = 0; k < s; k++) {
			CamToCel[ijvals[k]] = cv::Point2d(thetavals[k], phivals[k]);
			uint64_t ij = ijvals[k];
			steps[i_32*M+j_32] = step[k];
			//if (disk) CamToAD[ijvals[k]] = cv::Point2d(hitr[k], hitphi[k]);
		}
	}

	template <typename T, typename Compare>
	std::vector<std::size_t> sort_permutation(
		const std::vector<T>& vec, const std::vector<T>&vec1,
		Compare& compare) {
		std::vector<std::size_t> p(vec.size());
		std::iota(p.begin(), p.end(), 0);
		std::sort(p.begin(), p.end(),
			[&](std::size_t i, std::size_t j) { return compare(vec[i], vec[j], vec1[i], vec1[j]); });
		return p;
	}
	template <typename T>
	std::vector<T> apply_permutation(
		const std::vector<T>& vec,
		const std::vector<std::size_t>& p) {
		std::vector<T> sorted_vec(vec.size());
		std::transform(p.begin(), p.end(), sorted_vec.begin(),
			[&](std::size_t i) { return vec[i]; });
		return sorted_vec;
	}


	/// <summary>
	/// Calls the kernel.
	/// </summary>
	/// <param name="ijvec">The ijvec.</param>
	void callKernel(vector<uint64_t>& ijvec) {
		size_t s = ijvec.size();
		vector<double> theta(s), phi(s);
		vector<int> step(s);
		for (int q = 0; q < s; q++) {
			uint64_t ij = ijvec[q];
			theta[q] = (double)i_32 / (N - 1) * PI / (2 - equafactor);
			phi[q] = (double)j_32 / M * PI2;
		}

		auto start_time = std::chrono::high_resolution_clock::now();
		integration_wrapper(theta, phi, s, step);
		vector<double> e1, e2;
		fillGridCam(ijvec, s, theta, phi, e1, e2, step);
		auto end_time = std::chrono::high_resolution_clock::now();
		int count = 0;
		for (int q = 0; q < s; q++) if (step[q] != 0) count++;
		cout << "CPU: " << count << "rays in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms!" << endl << endl;
		//}
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
		double diag2 = (th2 - th3)*(th2 - th3) + (ph2 - ph3)*(ph2 - ph3);

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
			CamToCel[i_j] = cv::Point2d(-10, -10);
		}
	}

	/// <summary>
	/// Adaptively raytraces the grid.
	/// </summary>
	/// <param name="level">The current level.</param>
	void adaptiveBlockIntegration(int level) {

		while (level < MAXLEVEL) {
			if (level<5 && print) printGridCam(level);
			if (print) std::cout << "Computing level " << level + 1 << "..." << endl;

			if (checkblocks.size() == 0) return;

			std::unordered_set<uint64_t, hashing_func2> todo;
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
	void integration_wrapper(vector<double>& theta, vector<double>& phi, const int n, vector<int>& step) {
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
			step[i] = 0;
			if (metric::checkCelest(pR, rS, thetaS, b, q)) {
				metric::rkckIntegrate1(rS, thetaS, phiS, pR, b, q, pTheta, theta[i], phi[i], step[i]);
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
	std::unordered_map <uint64_t, cv::Point2d, hashing_func2> CamToCel;

	std::vector<int> steps;

	std::unordered_map <uint64_t, cv::Point2d, hashing_func2> CamToAD;

	PSHOffsetTable hasher;

	//PSHOffsetTable hasher;

	/// <summary>
	/// Mapping from block position to level at that point.
	/// </summary>
	std::unordered_map <uint64_t, int, hashing_func2> blockLevels;

	bool print = false;

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
		equafactor = angle ? 1 : 0;

		N = (uint32_t)round(pow(2, MAXLEVEL) / (2 - equafactor) + 1);
		STARTN = (uint32_t)round(pow(2, startLevel) / (2 - equafactor) + 1);
		M = (2 - equafactor) * 2 * (N - 1);
		STARTM = (2 - equafactor) * 2 * (STARTN - 1);
		steps = vector<int>(M*N);
		raytrace();
		//printGridCam(5);

		for (auto block : blockLevels) {
			fixTvertices(block);
		}
		if (startLevel != maxLevelPrec) saveAsGpuHash();
	};


	//void callKernelTEST(const Camera* camera, const BlackHole* bh, size_t s) {
	//	cam = camera;
	//	black = bh;
	//	M = 2048;
	//	N = 512;
	//	equafactor = 0;
	//	vector<double> theta(s), phi(s);
	//	int num = 10;
	//	for (int q = 0; q < s; q++) {
	//		theta[q] = (double)(num) / (N - 1) * PI / (2 - equafactor);
	//		phi[q] = (double)(num) / M * PI2;
	//	}
	//	integration_wrapper(theta, phi, s);
	//}

	void saveAsGpuHash() {
		if (print) cout << "Computing Perfect Hash.." << endl;

		vector<int2> elements;
		vector<float2> data;
		for (auto entry : CamToCel) {
			elements.push_back({ (uint32_t)(entry.first >> 32), uint32_t(entry.first) });
			data.push_back({ entry.second.x, entry.second.y });
		}
		hasher = PSHOffsetTable(elements, data);

		if (print) cout << "Completed Perfect Hash" << endl;
	}

	/// <summary>
	/// Finalizes an instance of the <see cref="Grid"/> class.
	/// </summary>
	~Grid() {};
};


