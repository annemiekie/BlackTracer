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
using namespace std;

#define PRECCELEST 0.01
#define ERROR 0.001//1e-6

class Grid
{
private:
	// Cereal settings for serialization
	friend class cereal::access;
	template < class Archive >
	void serialize(Archive & ar)
	{
		ar(MAXLEVEL, N, M, CamToCel, crossings2pi, blockOrientation, blockLevels, startblocks, equafactor);
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

public:
	// 1 if rotation axis != camera axis, 0 otherwise
	int equafactor;

	// N = max vertical blocks, M = max horizontal blocks.
	int MAXLEVEL, N, M, STARTN, STARTM, STARTLVL;

	// Mapping from camera sky position to celestial angle.
	unordered_map <uint64_t, Point2d, hashing_func2> CamToCel;

	// Mapping from block position to level at that point.
	unordered_map <uint64_t, int, hashing_func2> blockLevels;

	// Mapping from block position to block orientation.
	unordered_map <uint64_t, int, hashing_func2> blockOrientation;

	// Set of from camera sky positions with 2pi problem.
	unordered_map <uint64_t, bool, hashing_func2> crossings2pi;

	// Largest blocks making up the first level.
	vector<uint64_t> startblocks;

	Grid(){};

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
			orientation2piChecks(block);
		}
	};

	void orientation2piChecks(pair<uint64_t, int> block) {
		// in distorter store per pixelcorner if in cw or ccw block
		// for every pixel check if all in cw or ccw block, act accordingly
		// if different blocks

		// might even be possible to find the locations of the inversions with this info
		// by finding connected regions of cw and ccw blocks

		int level = block.second;
		uint64_t ij = block.first;
		if (CamToCel[ij]_phi < 0) return;

		uint32_t i = i_32;
		uint32_t j = j_32;

		uint32_t gap = (uint32_t)pow(2, MAXLEVEL - level);

		uint32_t k = i + gap;
		uint32_t l = (j + gap) % M;

		vector<Point2d> thphivals = { CamToCel[i_j], CamToCel[k_j], CamToCel[k_l], CamToCel[i_l] };

		// Check per block if it turns clockwise or counterclockwise --> store per block
		// Orientation is positive if CW, negative if CCW
		double orientation = 0;
		for (int q = 0; q < 4; q++) {
			orientation += (thphivals[(q + 1) % 4]_theta - thphivals[q]_theta)
				*(thphivals[(q + 1) % 4]_phi + thphivals[q]_phi);
		}

		int sgn = metric::sgn(orientation);

		// Check for blocks that have corners in the range close to 2pi and close to 0
		bool suspect = check2PIcross(thphivals, 5.);
		crossings2pi[ij] = false;

		// Check if a ray down the middle will also fall in the middle of the projection
		if (suspect) {
			float i_new = (float)i + gap / 2.;
			float j_new = (float)j + gap / 2.;
			vector<double> theta = { (double)i_new / (N - 1) * PI / (2 - equafactor) };
			vector<double> phi = { (double)j_new / M * PI2 };
			integration_wrapper(theta, phi, 1);
			Point2d thphi = Point2d(theta[0], phi[0]);

			// If the ray does not fall into the projection there is a very high chance of
			// a 2pi crossing, the orientation is also inverse in this case.
			if (!pointInPolygon(thphi, thphivals, sgn)) {
				sgn = -sgn;
				crossings2pi[ij] = true;
			}
		}
		blockOrientation[ij] = sgn;
	}

	/**
	 * Returns if a location lies within the boundaries of the provided polygon.
	 */
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
	
	bool check2PIcross(vector<Point2d>& poss, float factor) {
		for (int i = 0; i < poss.size(); i++) {
			if (poss[i]_phi > PI2*(1. - 1. / factor))
				return true;
		}
		return false;
	};

	void correct2PIcross(Point2d& poss, float factor) {
		if (poss.y < PI2*(1. / factor) && poss.y >= 0)
			poss.y += PI2;
	};
	
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

		checkAdjacentBlock(ij, k_j, level, 1, 0, gap);
		checkAdjacentBlock(ij, i_l, level, 0, 1, gap);
		checkAdjacentBlock(i_l, k_l, level, 1, 0, gap);
		checkAdjacentBlock(k_j, k_l, level, 0, 1, gap);
	}

	void checkAdjacentBlock(uint64_t ij, uint64_t ij2, int level, int ud, int lr, int gap) {
		uint32_t i = i_32 + ud * gap / 2;
		uint32_t j = j_32 + lr * gap / 2;
		auto it = CamToCel.find(i_j);
		if (it == CamToCel.end())
			return;
		else {
			CamToCel[i_j] = 1. / 2.*(CamToCel[ij] + CamToCel[ij2]);
			if (level + 1 == MAXLEVEL) return;
			checkAdjacentBlock(ij, i_j, level + 1, ud, lr, gap / 2);
			checkAdjacentBlock(i_j, ij2, level + 1, ud, lr, gap / 2);
		}
	}

	void printGridCam(int level) {
		cout.precision(2);
		cout << endl;

		int gap = (int)pow(2, MAXLEVEL - level);
		for (uint32_t i = 0; i < N; i += gap) {
			for (uint32_t j = 0; j < M; j += gap) {
				double val = CamToCel[i_j]_theta;
				if (val>1e-10)
					cout << setw(4) << val/PI;
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
					cout << setw(4) << val/PI;
				else
					cout << setw(4) << 0.0;
			}
			cout << endl;
		}
		cout << endl;

		cout.precision(10);
	}

	void makeStartBlocks() {
		int gap = (int)pow(2, MAXLEVEL - 1 + equafactor);
		for (uint32_t i = 0; i < N - 1; i += gap) {
			for (uint32_t j = 0; j < M; j += gap) {
				startblocks.push_back(i_j);
			}
		}
	}
	
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
			i=l=k= 0;
			CamToCel[i_j] = CamToCel[k_l];
			checkblocks.insert(i_j);
			if (equafactor) {
				i = k = N - 1;
				CamToCel[i_j] = CamToCel[k_l];
			}
		}

		integrateFirst(gap);
		adaptiveBlockIntegration(STARTLVL);
	}

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

	void fillGridCam(const vector<uint64_t>& ijvals, const size_t s, vector<double>& thetavals, vector<double>& phivals) {
		for (int k = 0; k < s; k++)
			CamToCel[ijvals[k]] = Point2d(thetavals[k], phivals[k]);
	}

	void callKernel(const vector<uint64_t>& ijvec) {
		size_t s = ijvec.size();
		vector<double> theta(s), phi(s);
		
		for (int q= 0; q < s; q++) {
			uint64_t ij = ijvec[q];
			theta[q] = (double)i_32 / (N - 1) * PI / (2-equafactor);
			phi[q] = (double)j_32 / M * PI2;
		}
		integration_wrapper(theta, phi, s);
		
		fillGridCam(ijvec, s, theta, phi);
	}

	/**
	 * Check if a block needs to be refined.
	 * 
	 */
	bool refineCheck(const uint32_t i, const uint32_t j, const int gap, const int level) {
		uint32_t k = i + gap;
		uint32_t l = (j + gap) % M;

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

	void fillVector(vector<uint64_t>& toIntIJ, uint32_t i, uint32_t j) {
		auto iter = CamToCel.find(i_j);
		if (iter == CamToCel.end()) {
			toIntIJ.push_back(i_j);
			CamToCel[i_j] = Point2d(-10, -10);
		}
	}

	void adaptiveBlockIntegration(int level) {

		//size_t checksize = checkblocks.size();
		while(level < MAXLEVEL) {
			if (level<5) printGridCam(level);
			cout << "Computing level " << level + 1 << "..." << endl;

			if (checkblocks.size() == 0) return;

			unordered_set<uint64_t, hashing_func2> todo;
			vector<uint64_t> toIntIJ;

			for (auto ij: checkblocks) {

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

	void integration_wrapper(vector<double>& theta, vector<double>& phi, int n)
	{
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

			double rFido = xFido;
			double thetaFido = -zFido;
			double phiFido = yFido;

			double eF = 1. / (cam->alpha + cam->w * cam->wbar * phiFido);

			double pR = eF * cam->ro * rFido / sqrtf(cam->Delta);
			double pTheta = eF * cam->ro * thetaFido;
			double pPhi = eF * cam->wbar * phiFido;

			double b = pPhi;
			double q = pTheta*pTheta + cos(thetaS)*cos(thetaS)*(b*b / (sin(thetaS)*sin(thetaS)) - *metric::asq);

			theta[i] = -1;
			phi[i] = -1;

			if (metric::checkCelest(pR, rS, thetaS, b, q)) {
				metric::rkckIntegrate(rS, thetaS, phiS, pR, b, q, pTheta, theta[i], phi[i]);
			}
		}
	}

	/** UNUSED */
	#pragma region unused

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

	~Grid() {};
};


