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
//#include "kernel.h"

using namespace cv;
using namespace std;

#define PRECCELEST 0.01
#define ERROR 0.001//1e-6

//extern void integration_wrapper(double *theta, double *phi, int size, const Camera* cam, const double afactor);

class Grid
{
private:
	friend class cereal::access;
	template < class Archive >
	void serialize(Archive & ar)
	{
		ar(MAXLEVEL, N, M, CamToCel, blockLevels, startblocks, equafactor);
	}
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
	const Camera* cam;
	const BlackHole* black;
	int raycount;
	int counter = 0;
	//vector<uint64_t> checkblocks;
	unordered_set<uint64_t, hashing_func2> checkblocks;

public:
	int equafactor;
	int MAXLEVEL, N, M, STARTN, STARTM, STARTLVL;
	unordered_map <uint64_t, Point2d, hashing_func2> CamToCel;
	unordered_map <uint64_t, int, hashing_func2> blockLevels;
	vector<uint64_t> startblocks;
	bool fileError = 0;

	Grid(){};

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

	Grid(const int maxLevelPrec, const int startLevel, const bool angle, const Camera* camera, const BlackHole* bh) {
		raycount = 0;
		MAXLEVEL = maxLevelPrec;
		STARTLVL = startLevel;
		equafactor = 0.;
		if (angle) equafactor = 1.;
		N = (uint32_t)round(pow(2, MAXLEVEL) / (2-equafactor) + 1);
		STARTN = (uint32_t)round( pow(2, startLevel) / (2 - equafactor) + 1);
		M = (2 - equafactor) * 2 * (N - 1);
		STARTM = (2 - equafactor) * 2 * (STARTN - 1);
		//CamToCel.reserve(N*M*2);
		//blockLevels.reserve(N*M * 2);
		cam = camera;
		black = bh;
		makeStartBlocks();
		//time_t start = time(NULL);
		raytrace();
		//cout << "TEST " << time(NULL) - start << endl;
	};

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

	void fillGridCam(const vector<uint64_t>& ijvals, const size_t s, vector<double>& thetavals, vector<double>& phivals) {//double* thetavals, double* phivals) {
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

	double maxvec(vector<double>& val) {
		double max=-20;
		for (int i = 0; i < val.size(); ++i) {
			if (val[i] > max) max = val[i];
		}
	}

	void maxmin(double &max, double &min) {
		double temp;
		if (max < min) {
			temp = max;
			max = min;
			min = temp;
		}
	}

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

		//double side1 = metric::sq(th1 - th3) + metric::sq(ph1 - ph3);
		//if (side1 > PRECCELEST) return true;

		double diag = (th1 - th4)*(th1 - th4) + (ph1 - ph4)*(ph1 - ph4);
		double diag2 = (th2 - th2)*(th2 - th3) + (ph2 - ph3)*(ph2 - ph3);
		maxmin(diag, diag2);
		if (level < 6 && diag>1E-10) return true;
		if (diag > PRECCELEST) return true;
		//double side2 = metric::sq(th1 - th2) + metric::sq(ph1 - ph2);
		//double side3 = metric::sq(th4 - th3) + metric::sq(ph4 - ph3);
		//double side4 = metric::sq(th4 - th2) + metric::sq(ph4 - ph2);
		//double cosa1 = (side1 + diag - side4) / (2 * sqrt(side1*diag));
		//double cosa2 = (side2 + diag - side3) / (2 * sqrt(side2*diag));

		//maxmin(side1, side2);
		//if (side1 > PRECCELEST) return true;
		//if (side1 > 4.*side2 || diag > 6.*side2 || 2.*diag < side1) return true;
		//maxmin(cosa1, cosa2);
		//if (cosa1 > 1.0 || cosa2 < 0.4) return true;

		blockLevels[i_j] = level;
		return false;

	};

	void fillVector(vector<uint64_t>& toIntIJ, uint32_t i, uint32_t j) {
		auto iter = CamToCel.find(i_j);
		if (iter == CamToCel.end()) {
			toIntIJ.push_back(i_j);
			CamToCel[i_j] = Point2d(-10, -10);
			//raycount++;
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

	void integration_wrapper(vector<double>& theta, vector<double>& phi, int n)//double *theta, double *phi, int n)
	{
		double thetaS = cam->theta;
		double phiS = cam->phi;
		double rS = cam->r;
		double sp = cam->speed;

		//printf("metric a {%f} \n", *metric::a);
		//printf("metric asq {%f} \n", *metric::asq);

#pragma loop(hint_parallel(8))
#pragma loop(ivdep)
		for (int i = 0; i<n; i++) {

			//	printf("computing {%i} \n", i);

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
			//	printf("{%i,%lf,%lf,%lf,%lf} \n", i, xCam, xFido, b, q);
			theta[i] = -1;
			phi[i] = -1;

			if (metric::checkCelest(pR, rS, thetaS, b, q)) {
				//printf("%i passed check \n",i);
				metric::rkckIntegrate(rS, thetaS, phiS, pR, b, q, pTheta, theta[i], phi[i]);
			}
		}
	}
/*	void adaptiveBlockIntegration(int level) {
		int h = 0;
		int lvl = level;
		size_t checksize = checkblocks.size();
		for (;;) {
			if (adaptiveBlockIntegration(lvl, h, checksize) == 0) break;
		}
	}
	*/
	/*int adaptiveBlockIntegration(int& level, int& h, size_t& checksize) {
		if (checksize == 0) return 0;

		if (level == (MAXLEVEL)) {
			for (auto ij : checkblocks) {
				blockLevels[ij] = level;
			}
			//delete checkblocks?
			return 0;
		}

		if (h == 0) {
			if (level<5) printGridCam(level);
			else cout << endl;
			int newlevelrays = 3 * (int)checksize;
			cout << "checksize " << checksize << endl;
			cout << "Computing Level " << level + 1 << endl;
			for (int q = 0; q < ceil(1.f*newlevelrays / (MAXINTEGRATE + 5)); q++){
				cout << "-";
			}
			cout << "|" << endl;
		}

		cout << ".";

		vector<uint32_t> toIntI;
		vector<uint32_t> toIntJ;

		uint32_t gap = (uint32_t)pow(2, MAXLEVEL - level);

		while (raycount <= MAXINTEGRATE) {
			if (h == checksize) {
				level++;
				checksize = checkblocks.size();
				h = 0;
				break;
			}

			uint64_t ij = checkblocks[h];
			uint32_t i = i_32;
			uint32_t j = j_32;
			j = j % M;

			if (refineCheck(i, j, gap, level)) {
				fillVectors(toIntI, toIntJ, i + gap / 2, j, true);
				fillVectors(toIntI, toIntJ, i + gap / 2, j + gap / 2, true);
				fillVectors(toIntI, toIntJ, i, j + gap / 2, true);
				fillVectors(toIntI, toIntJ, i + gap, j + gap / 2, false);
				fillVectors(toIntI, toIntJ, i + gap / 2, (j + gap) % M, false);
				h++;
			}
			else {
				checkblocks.erase(checkblocks.begin() + h);
				checksize--;
			}
		}
		//cout << level << " callkernel start "  << time(NULL) << endl;
		callKernel(toIntI, toIntJ);
		//cout << "callkernel end " << time(NULL) << endl;
		raycount = 0;
		return 1;

	};*/

	~Grid() {};
};


