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
	float *starPos;
	float *starMag;
	vector<Star> stars;
	int starSize;
	StarProcessor(string filename) {
		stars = readStars(filename);
		starSize = stars.size();
		vector< vector<float> > starThphi;
		for (int i = 0; i < stars.size(); i++) {
			starThphi.push_back({ stars[i].theta, stars[i].phi, stars[i].magnitude });
		}
		starMag = new float[stars.size()];
		starPos = new float[stars.size() * 2];
		binaryStarTree = new int[TREESIZE];
		float thphi[2] = { 0, 0 };
		float size[2] = { PI, PI2 };
		makeTree(starThphi, 0, thphi, size, 0);
	}

	~StarProcessor() {

	};

private:
	/* Read stars from file into vector. */
	vector<Star> readStars(string filename) {
		vector<Star> stars;
		ifstream file;
		file.open(filename);
		float ra, dec, x, mag, theta, phi;
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
				phi = dec * PI / 180.;
				theta = ra * PI / 180;
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

	void makeTree(vector< vector<float> > stars, int level, float thphi[2], float size[2], int writePos) {
		int stsize = stars.size();
		int searchPos = 0;
		if (writePos != 0 && ((writePos + 1) & writePos) != 0) {
			searchPos = binaryStarTree[writePos - 1];
		}
		binaryStarTree[writePos] = searchPos + stsize;
		if (level == TREELEVEL) {
			for (int i = 0; i < stsize; i++) {
				starPos[(i + searchPos) * 2] = stars[i][0];
				starPos[(i + searchPos) * 2 + 1] = stars[i][1];
				starMag[i + searchPos] = stars[i][2];
			}
			return;
		}
		level++;
		int n = level % 2;

		size[n] = size[n] * .5;
		vector< vector<float> > starsLU;
		vector< vector<float> > starsRD;

		float check = thphi[n] + size[n];
		for (int i = 0; i < stsize; i++) {
			if ( stars[i][n] < check) {		
				starsLU.push_back(stars[i]);
			}
			else {
				starsRD.push_back(stars[i]);
			}
		}
		float sizeNew[2] = { size[0], size[1] };
		float thphiNew[2] = { thphi[0], thphi[1] };
		makeTree(starsLU, level, thphiNew, sizeNew, writePos * 2 + 1);
		float thphiNew2[2] = { thphi[0], thphi[1] };
		float sizeNew2[2] = { size[0], size[1] };

		thphiNew2[n] += size[n];
		makeTree(starsRD, level, thphiNew2, sizeNew2, writePos * 2 + 2);
	}
};