#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "Metric.h"
#include "Const.h"
#include <vector>
#include <chrono>
#include "cuda_runtime.h"

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

	vector<int> binaryStarTree;
	vector<float> starPos;
	vector<float> starMag;
	int starSize;
	int treeSize = 0;
	int treeLevel = 0;
	Mat imgWithStars;
	StarProcessor() {}

	StarProcessor(string filename, int _treeLevel, string imgfile, string starImg, int magnitudeCut) {
		vector< vector<float> > starVec;
		readStars2(filename, starVec);
		vector< vector<float> > starThphi;
		vector< vector<float> > lowStars;
		for (int i = 0; i < starVec.size(); i++) {
			if (starVec[i][2] > magnitudeCut) {
				lowStars.push_back(starVec[i]);
			}
			else {
				starThphi.push_back(starVec[i]);
			}
		}

		//for (int i = 0; i < starVec.size(); i+=2) starThphi.push_back(starVec[i]);

		imgWithStars = imread(imgfile);
		//addLowLightStarsToImage(lowStars, imgWithStars);
		imwrite(starImg, imgWithStars);

		starSize = starThphi.size();
		starMag.resize(starSize * 2);
		starPos.resize(starSize * 2);
		treeLevel = _treeLevel;
		treeSize = (1 << (treeLevel + 1)) - 1;
		binaryStarTree.resize(treeSize);
		float thphi[2] = { 0, 0 };
		float size[2] = { PI, PI2 };
		makeTree(starThphi, 0, thphi, size, 0);


	}

	~StarProcessor() {};

private:

	/* Read stars from file into vector. */
	void readStars2(string filename, vector< vector<float> > &starVec) {
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
				metric::wrapToPi(dec, ra);
				starVec.push_back({dec, ra, mag, col});
			}
		}
		else {
			cout << "No such file exists!" << endl;
		}
		file.close();
	};

	float gaussian(float dist) {
		return expf(-2.f*dist) * 0.318310f;
	}
	float distSq(float t_a, float t_b, float p_a, float p_b) {
		return (t_a - t_b)*(t_a - t_b) + (p_a - p_b)*(p_a - p_b);
	}
	
	/// <summary>
	/// Adds the low light stars to image.
	/// </summary>
	/// <param name="lowstars">The lowstars.</param>
	/// <param name="imgWithStars">The img with stars.</param>
	void addLowLightStarsToImage(vector< vector<float> >& lowstars, Mat &imgWithStars) {
		int width = imgWithStars.cols;
		int height = imgWithStars.rows;
		Mat stars = Mat(height, width, DataType<Vec3i>::type);
		for (int i = 0; i < lowstars.size(); i++) {
			float magnitude = lowstars[i][2];
			float bvcolor = lowstars[i][3];
			float thetastar = lowstars[i][0];
			float phistar = lowstars[i][1];
			float pixPosTheta = (thetastar / PI* 1.f*height);
			float pixPosPhi = (phistar / PI2* 1.f*width);
			int pixHor = (int)pixPosPhi;
			int pixVert = (int)pixPosTheta;

			float temp = 46.f * ((1.f / ((0.92f * bvcolor) + 1.7f)) + (1.f / ((0.92f * bvcolor) + 0.62f))) - 10.f;
			int index = max(0, min((int)floorf(temp), 1172));
			float3 rgb = { tempToRGBx[3 * index] * tempToRGBx[3 * index], 
						   tempToRGBx[3 * index + 1] * tempToRGBx[3 * index + 1], 
						   tempToRGBx[3 * index + 2] * tempToRGBx[3 * index + 2] };
			int step = 2;
			float maxDistSq = (step + .5f)*(step + .5f);

			int start = max(0, -(pixVert - step ));
			int diff = pixVert + step - height + 1;
			int stop = (pixVert + step >= height) ? 2*step-diff : 2*step;
			for (int u = start; u <= stop; u++) {
				for (int v = 0; v <= 2 * step; v++) {
					float dist = distSq(-step + u + .5f, pixPosTheta-pixVert, -step + v + .5f, pixPosPhi-pixHor);
					if (dist > maxDistSq) continue;
					else {
						float appMag = -.4f * (magnitude - 2.5f * log10f(gaussian(dist)));
						float brightness = 400.f*pow(10.f,appMag);
						stars.at<Vec3i>(-step + u + pixVert, (-step + v + pixHor + width) % width)[0] = brightness*rgb.z;
						stars.at<Vec3i>(-step + u + pixVert, (-step + v + pixHor + width) % width)[1] = brightness*rgb.y;
						stars.at<Vec3i>(-step + u + pixVert, (-step + v + pixHor + width) % width)[2] = brightness*rgb.x;
					}
				}
			}
		}

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int orig_b = imgWithStars.at<Vec3b>(i, j)[0];
				int orig_g = imgWithStars.at<Vec3b>(i, j)[1];
				int orig_r = imgWithStars.at<Vec3b>(i, j)[2];

				int sum_b = min(255, (int)sqrt(stars.at<Vec3i>(i, j)[0] + orig_b*orig_b));
				int sum_g = min(255, (int)sqrt(stars.at<Vec3i>(i, j)[1] + orig_g*orig_g));
				int sum_r = min(255, (int)sqrt(stars.at<Vec3i>(i, j)[2] + orig_r*orig_r));
				imgWithStars.at<Vec3b>(i, j)[0] = sum_b;
				imgWithStars.at<Vec3b>(i, j)[1] = sum_g;
				imgWithStars.at<Vec3b>(i, j)[2] = sum_r;
			}
		}
	}

	/// <summary>
	/// Makes the tree.
	/// </summary>
	/// <param name="stars">The stars.</param>
	/// <param name="level">The level.</param>
	/// <param name="thphi">The thphi.</param>
	/// <param name="size">The size.</param>
	/// <param name="writePos">The write position.</param>
	void makeTree(vector< vector<float> >& stars, int level, float thphi[2], float size[2], int writePos) {
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