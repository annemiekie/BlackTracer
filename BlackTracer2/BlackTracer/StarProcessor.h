#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "Metric.h"
#include "Const.h"
#include "Star.h"
#include <vector>

using namespace cv;
using namespace std;

class StarProcessor {
public:
	int *binaryStarTree;
	double **starPos;
	vector<Star> stars;
	int starSize;
	StarProcessor(string filename) {
		stars = readStars(filename);
		starSize = stars.size();
		vector< vector<double> > starThphi;
		for (int i = 0; i < stars.size(); i++) {
			starThphi.push_back({ stars[i].theta, stars[i].phi });
		}

		starPos = new double*[stars.size()];
		for (int i = 0; i < stars.size(); i++) {
			starPos[i] = new double[2];
		}

		binaryStarTree = new int[TREESIZE];
		double thphi[2] = { 0, 0 };
		double size[2] = { PI, PI2 };
		makeTree(starThphi, 0, thphi, size, 0);

		//bool check = false;
		//int found = 0;
		//for (int i = 0; i < starSize; i++) {
		//	double theta = stars[i].theta;
		//	double phi = stars[i].phi;
		//	check = false;
		//	for (int i = 0; i < starSize; i++) {
		//		if (theta == starPos[i][0] && phi == starPos[i][1]) {
		//			check = true;
		//			found++;
		//			break;
		//		}
		//	}
		//	if (check = false) break;
		//}
		
	}

	~StarProcessor() {};

private:
	/* Read stars from file into vector. */
	vector<Star> readStars(string filename) {
		vector<Star> stars;
		ifstream file;
		file.open(filename);
		double ra, dec, x, mag, theta, phi;
		int numEntries, level, numStart;
		vector<Vec3b> colors = { Vec3b(255, 255, 0), Vec3b(255, 0, 0), Vec3b(0, 0, 255) };
		if (file.is_open()) {
			cout << "Reading stars from file..." << endl;
			while (!file.eof()) {
				file >> ra;
				file >> dec;
				file >> mag;
				file >> x;
				file >> x;
				file >> x;
				phi = dec / PI;
				theta = ra / PI;
				metric::wrapToPi(theta, phi);
				double x = rand() / static_cast<double>(RAND_MAX + 1);
				int rand = static_cast<int>(x * 2);
				Star star = Star(phi, theta, mag, colors[rand]);
				stars.push_back(star);
			}
		}
		else {
			cout << "No such file exists!" << endl;
		}
		file.close();
		return stars;
	};

	void makeTree(vector< vector<double> > stars, int level, double thphi[], double size[], int writePos) {
		int stsize = stars.size();
		int searchPos = 0;
		if (writePos != 0 && ((writePos + 1) & writePos) != 0) {
			searchPos = binaryStarTree[writePos - 1];
		}
		binaryStarTree[writePos] = searchPos + stsize;
		if (level == TREELEVEL) {
			for (int i = 0; i < stsize; i++) {
				starPos[i + searchPos][0] = stars[i][0];
				starPos[i + searchPos][1] = stars[i][1];
			}
			return;
		}
		if (stsize == 0) {
			level++;
			makeTree(stars, level, thphi, size, writePos * 2 + 1);
			makeTree(stars, level, thphi, size, writePos * 2 + 2);
			return;
		}

		level++;
		int n = level % 2;
		size[n] = size[n] / 2.;
		vector< vector<double> > starsLU;
		vector< vector<double> > starsRD;

		double check = thphi[n]+size[n];
		for (int i = 0; i < stsize; i++) {
			if ( stars[i][n] < check) {		
				starsLU.push_back(stars[i]);
			}
			else {
				starsRD.push_back(stars[i]);
			}
		}
		makeTree(starsLU, level, thphi, size, writePos * 2 + 1);
		thphi[n] += size[n];
		makeTree(starsRD, level, thphi, size, writePos * 2 + 2);
	}
};