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

	vector<int> binaryStarTree;
	vector<float> starPos;
	vector<float> starMag;
	vector<Star> starVec;
	int starSize;
	int treeSize = 0;
	int treeLevel = 0;
	Mat imgWithStars;
	StarProcessor() {}

	StarProcessor(string filename, int _treeLevel, string imgfile, string starImg, int magnitudeCut) {
		readStars2(filename);
		vector< vector<float> > starThphi;
		vector< vector<float> > lowStars;
		for (int i = 0; i < starVec.size(); i++) {
			if (starVec[i].magnitude > magnitudeCut) {
				lowStars.push_back({ starVec[i].theta, starVec[i].phi, starVec[i].magnitude, starVec[i].color });
			}
			else {
				starThphi.push_back({ starVec[i].theta, starVec[i].phi, starVec[i].magnitude, starVec[i].color });
			}
		}
		imgWithStars = imread(imgfile);
		addLowLightStarsToImage(lowStars, imgWithStars);
		imwrite(starImg, imgWithStars);

		starSize = starThphi.size();
		starMag.resize(starSize * 2);
		starPos.resize(starSize * 2);
		treeLevel = _treeLevel;
		treeSize = (1 << (treeLevel + 1))-1;
		binaryStarTree.resize(treeSize);
		float thphi[2] = { 0, 0 };
		float size[2] = { PI, PI2 };
		makeTree(starThphi, 0, thphi, size, 0);

	}

	~StarProcessor() {

	};

	inline
		int count_trailing_zeros(uint32_t aX) {
#	if defined(__GNUC__) // GCC
		return aX ? __builtin_ctz(aX) : 32;

#	elif defined(_MSC_VER) // Visual Studio
		/* Untested, too lazy to boot into Windows. MSVC has two possible options
		* here:
		*
		*   - __lzcnt() (maps to LZCNT from BMI1 (Haswell++)
		*   - _BitScaneReverse() (maps to BSR from ancient x86)
		*
		* __lzcnt() works - been using it elsewhere, but requires a Haswell or
		* newer CPU that supports BMI1 instructions. _BitScanReverse() is kinda
		* a guess from the docs, so verify that it works as I think it should.
		*/
#	if 0
		return __tzcnt(aX);  // Haswell and newer (BMI1) instruction
#	else
		unsigned long ret;
		_BitScanForward(&ret, aX);
		return aX ? ret : 32;
#	endif

#	else // Unknown compiler
#	error "Got no CLZ intrinsic for your compiler"
		/* Either find the intrinsic for your compiler, or go to Bit Twiddeling
		* Hacks and grab the "fallback" implementation.
		*/
#endif // ~ compiler
	}


	void searchTree() {
		float thphiPixMin[2] = { 0.0, 6.2 };
		float thphiPixMax[2] = { 0.1, 6.3 };
		vector<int> searchNrs(20);
		bool picheck = true;
		float nodeStart[2] = { 0.f, PI };
		float nodeSize[2] = { PI, PI2 };
		float bbsize = (thphiPixMax[0] - thphiPixMin[0]) * (thphiPixMax[1] - thphiPixMin[1]);
		int node = 0;
		uint bitMask = powf(2, treeLevel);
		int level = 0;
		int lvl = 0;
		int pos = 0;
		while (bitMask != 0) {
			bitMask &= ~(1UL << (treeLevel - level));

			for (lvl = level+1; lvl <= treeLevel; lvl++) {
				int star_n = binaryStarTree[node];
				if (node != 0 && ((node + 1) & node) != 0) {
					star_n -= binaryStarTree[node - 1];
				}
				int tp = lvl & 1;

				float x_overlap = std::max(0.f, std::min(thphiPixMax[0], nodeStart[0] + nodeSize[0]) - std::max(thphiPixMin[0], nodeStart[0]));
				float y_overlap = std::max(0.f, std::min(thphiPixMax[1], nodeStart[1] + nodeSize[1]) - std::max(thphiPixMin[1], nodeStart[1]));
				float overlapArea = x_overlap * y_overlap;
				bool size = overlapArea / (nodeSize[0] * nodeSize[1]) > 0.8f;
				nodeSize[tp] = nodeSize[tp] * .5f;

				float check = nodeStart[tp] + nodeSize[tp];
				bool lu = thphiPixMin[tp] < check;
				bool rd = thphiPixMax[tp] >= check;
				if (lvl == 1 && picheck) {
					bool tmp = lu;
					lu = rd;
					rd = tmp;
				}

				if (star_n == 0) {
					node = node * 2 + 1; break;
				}

				if (lvl == treeLevel || (rd && lu && size)) {
					if (rd) {
						searchNrs[pos] = node * 2 + 2;
						pos++;
					}
					if (lu) {
						searchNrs[pos] = node * 2 + 1;
						pos++;
					}
					node = node * 2 + 1;
					break;
				}
				else {
					node = node * 2 + 1;
					if (rd) bitMask |= 1UL << (treeLevel - lvl);
					if (lu && lvl == 1 && picheck) nodeStart[1] += nodeSize[1];
					if (!lu) break;
				}
			}
			level = treeLevel - count_trailing_zeros(bitMask);
			if (level >= 0) {
				int diff = lvl - level;
				for (int i = 0; i < diff; i++) {
					int tp = (lvl - i) & 1;
					if (!(node & 1)) nodeStart[tp] -= nodeSize[tp];
					nodeSize[tp] = nodeSize[tp] * 2.f;
					node = (node - 1) / 2;
				}
				node++;
				int tp = level & 1;
				if (picheck && level == 1) nodeStart[tp] -= nodeSize[tp];
				else nodeStart[tp] += nodeSize[tp];

			}
		}
	}

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
				metric::wrapToPi(dec, ra);
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

	float gaussian(float dist) {
		return expf(-2.f*dist) * 0.318310f;
	}
	float distSq(float t_a, float t_b, float p_a, float p_b) {
		return (t_a - t_b)*(t_a - t_b) + (p_a - p_b)*(p_a - p_b);
	}

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
		//vector<int> compressionParams;
		//compressionParams.push_back(cv::IMWRITE_PNG_COMPRESSION);
		//compressionParams.push_back(0);

		//imwrite("merp.png", imgWithStars);
	}

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