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
	// Cereal settings for serialization
	friend class cereal::access;
	template < class Archive >
	void serialize(Archive & ar) {
		ar(binaryStarTree, starPos, starMag, starSize, treeSize, treeLevel);
	}

	struct _star {
		float theta;
		float phi;
		float magnitude;
	};

	vector<int> binaryStarTree;
	vector<float> starPos;
	vector<float> starMag;
	vector<Star> starVec;
	vector<_star> starStruct;
	int starSize;
	int treeSize = 0;
	int treeLevel = 0;

	StarProcessor() {}

	StarProcessor(string filename, int _treeLevel) {
		readStars2(filename);
		starSize = starVec.size();
		vector< vector<float> > starThphi(starSize, vector<float>(4));
		for (int i = 0; i < starVec.size(); i++) {
			starThphi[i] = { starVec[i].theta, starVec[i].phi, starVec[i].magnitude, starVec[i].color};
		}
		//starStruct.resize(starVec.size());
		starMag.resize(starSize*2);
		starPos.resize(starSize*2);
		treeLevel = _treeLevel;
		treeSize = (1 << (treeLevel + 1))-1;
		binaryStarTree.resize(treeSize);
		float thphi[2] = { 0, 0 };
		float size[2] = { PI, PI2 };
		makeTree(starThphi, 0, thphi, size, 0);
	}

	~StarProcessor() {

	};

private:

	/* Read stars from file into vector. */
	void readStars2(string filename) {
		ifstream file;
		file.open(filename);
		float ra, dec, x, mag, col;
		if (file.is_open()) {
			cout << "Reading stars from file..." << endl;
			while (!file.eof()) {
				file >> mag;
				file >> col;
				file >> ra;
				file >> dec;
				file >> x;

				dec = PI1_2 - dec;
				//metric::wrapToPi(dec, ra);
				Star star = Star(ra, dec, mag, col);
				starVec.push_back(star);
			}
		}
		else {
			cout << "No such file exists!" << endl;
		}
		file.close();
	};
	/* Read stars from file into vector. */
	void readStars(string filename) {
		ifstream file;
		file.open(filename);
		float ra, dec, x, mag, theta, phi;
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
				Star star = Star(phi, theta, mag, x);
				starVec.push_back(star);
			}
		}
		else {
			cout << "No such file exists!" << endl;
		}
		file.close();
	};

	void makeTree(vector< vector<float> > stars, int level, float thphi[2], float size[2], int writePos) {
		int stsize = stars.size();
		int searchPos = 0;
		if (writePos != 0 && ((writePos + 1) & writePos) != 0) {
			searchPos = binaryStarTree[writePos - 1];
		}
		binaryStarTree[writePos] = searchPos + stsize;
		if (level == treeLevel) {
			for (int i = 0; i < stsize; i++) {
				starPos[(i + searchPos) * 2] = stars[i][0];
				starPos[(i + searchPos) * 2 + 1] = stars[i][1];
				starMag[(i + searchPos) * 2] = stars[i][2];
				starMag[(i + searchPos) * 2 + 1] = stars[i][3];
				//starStruct[i + searchPos] = { stars[i][0], stars[i][1], stars[i][2] };
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