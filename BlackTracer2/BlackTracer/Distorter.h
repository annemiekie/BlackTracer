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
#include <chrono>
#include <iostream>
#include <fstream>
#include "cuda_runtime.h"
using namespace cv;
using namespace std;

extern void makeImage(const float *stars, const int *starTree, 
					  const int starSize, const float *camParam, const float *magnitude, const int treeLevel, 
					  const int M, const int N, const int step, const Mat csImage, const int G, 
					  const float gridStart, const float gridStep, const float2 *hits,
					  const float2 *viewthing, const float viewAngle, const int GM, const int GN, const float2 *grid, const int gridlvl,
					  const int2 *offsetTables, const float2 *hashTables, const int2 *hashPosTag, const int2 *tableSizes, const int otsize, const int htsize);

extern void callback();

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

	/** -------------------------------- STARS -------------------------------- **/
	#pragma region stars


	void cudaTest() {
		// fill arrays with data

		auto start_time = std::chrono::high_resolution_clock::now();

		int Gr = grids->size();
		int N = view->pixelheight;
		int M = view->pixelwidth;
		vector<float> camParams(7*Gr);
		vector<float2> hit(Gr*M*N);
		int GM = (*grids)[0].M;
		int GN = (*grids)[0].N;
		vector<float2> grid(GM*GN*Gr);

		// How to find the blackhole border in this manner (?)
		// maybe a really quick descend to determine where the black hole is for both grids -- store as uchar1
		// go down to blocksize == 1 and then select the blocks that have bh & !bh components
		// make a list of all of them (allocate memory in advance)
		// new kernel: find min and max
		// new kernel: find for each one on the edge which angle slot it belongs to -- not possible!!
		//				because the angle slots do not match the number and exact spread of borderblocks
		// maybe divide&conquer to find for every angle the right block and therefore theta/phi values?

		vector<float> hashTable;// = (*grids)[0].hasher.hashTable;
		vector<int> offsetTable;// = (*grids)[0].hasher.offsetTable;
		vector<int> hashPosTag;
		vector<int2> tableSizes(Gr);
		
		for (int g = 0; g < Gr; g++) {
			//(*grids)[g].saveAsGpuHash();
			hashTable.insert(hashTable.end(), (*grids)[g].hasher.hashTable1.begin(), (*grids)[g].hasher.hashTable1.end() );
			offsetTable.insert(offsetTable.end(), (*grids)[g].hasher.offsetTable1.begin(), (*grids)[g].hasher.offsetTable1.end() );
			hashPosTag.insert(hashPosTag.end(), (*grids)[g].hasher.hashPosTag1.begin(), (*grids)[g].hasher.hashPosTag1.end());
			tableSizes[g] = { (*grids)[g].hasher.hashTableWidth, (*grids)[g].hasher.offsetTableWidth };

			cout << tableSizes[g].x << " " << tableSizes[g].y << endl;
		}

		for (int g = 0; g < Gr; g++) {
			vector<float> camParamsG = (*cams)[g].getParamArray();
			for (int cp = 0; cp < 7; cp++) {
				camParams[g * 7 + cp] = camParamsG[cp];
			}
			for (uint32_t i = 0; i < GN; i++) {
				for (uint32_t j = 0; j < GM; j++) {

					//int2 index = (*grids)[g].hasher.hashFunc({ i, j });
					//int2 compareIndex = (*grids)[g].hasher.hashPosTag[index.x*(*grids)[g].hasher.hashTableWidth + index.y];
					//float2 data =		(*grids)[g].hasher.hashTable[index.x*(*grids)[g].hasher.hashTableWidth + index.y];
					//bool compare = false;
					//if (i == compareIndex.x && j == compareIndex.y) compare = true;
					

					if ((*grids)[g].CamToCel.find(i_j) == (*grids)[g].CamToCel.end()) {
						grid[g*GM*GN + i*GM + j] = { -2, -2 };
						//if (compare) cout << "Oh no 1" << endl;
					}
					else {
						grid[g*GM*GN + i*GM + j] = { (*grids)[g].CamToCel[i_j].x, (*grids)[g].CamToCel[i_j].y };
						//if (!compare) cout << "Oh no 2" << endl;
						//else if (data.x != grid[g*GM*GN + i*GM + j].x || data.y != grid[g*GM*GN + i*GM + j].y)
						//	cout << "Oh no 3 " << data.x << " " << grid[g*GM*GN + i*GM + j].x << " " << data.y << " " << grid[g*GM*GN + i*GM + j].y;
					}
				}
			}
		}

		int step = 1;
		float gridStart = (*cams)[0].r;
		float gridStep = ((*cams)[Gr-1].r - gridStart) / (1.f*Gr-1.f);

		cout << "Copied grid..." << endl;

		//float* x = &hashTable[0];
		//float2* y = (float2*)x;
		//for (int i = 0; i< hashTable.size())

		makeImage(&(starTree->starPos[0]), &(starTree->binaryStarTree[0]), 
				  starTree->starSize, &camParams[0], &(starTree->starMag[0]), starTree->treeLevel, 
				  M, N, step, starTree->imgWithStars, Gr, gridStart, gridStep, &hit[0], 
				  &(view->viewMatrix[0]), view->viewAngleWide, GM, GN, &grid[0], (*grids)[0].MAXLEVEL,
				  (int2*)&offsetTable[0], (float2*)(&hashTable[0]), (int2*)&hashPosTag[0], &tableSizes[0], offsetTable.size()/2, hashTable.size()/2);
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

		cudaTest();
	};

	void drawBlocks(string filename, int g) {
		Mat gridimg((*grids)[g].N, (*grids)[g].M, CV_8UC1, Scalar(255));
		for (auto block : (*grids)[g].blockLevels) {
			uint64_t ij = block.first;
			int level = block.second;
			int gap = pow(2, (*grids)[g].MAXLEVEL - level);
			uint32_t i = i_32;
			uint32_t j = j_32;
			uint32_t k = i_32 + gap;
			uint32_t l = j_32 + gap;
			line(gridimg, Point2d(j, i), Point2d(j, k), Scalar(0), 1);
			line(gridimg, Point2d(l, i), Point2d(l, k), Scalar(0), 1);
			line(gridimg, Point2d(j, i), Point2d(l, i), Scalar(0), 1);
			line(gridimg, Point2d(j, k), Point2d(l, k), Scalar(0), 1);
		}
		imwrite(filename, gridimg);
	}

	/// <summary>
	/// Finalizes an instance of the <see cref="Distorter"/> class.
	/// </summary>
	~Distorter() {	};
	
};