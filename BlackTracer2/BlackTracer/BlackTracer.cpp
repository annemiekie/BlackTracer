#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdio.h>
#include <stdint.h> 
#include <sstream>
#include <string>
#include <stdlib.h>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/unordered_map.hpp>

#include "BlackHole.h"
#include "Camera.h"
#include "Grid.h"
#include "Distorter.h"
#include "StarProcessor.h"
#include <vector>

using namespace std;

/* Serialize grid with given filename. */
void write(string filename, Grid grid) {
	{
		ofstream ofs(filename, ios::out | ios::binary);
		cereal::BinaryOutputArchive oarch(ofs);
		oarch(grid);
	}

};

void writeStars(string filename, StarProcessor starProcessor) {
	{
		ofstream ofs(filename, ios::out | ios::binary);
		cereal::BinaryOutputArchive oarch(ofs);
		oarch(starProcessor);
	}
}

/* Count and display number of blocks per level in provided grid. */
void gridLevelCount(Grid& grid, int maxlevel) {
	vector<int> check(maxlevel + 1);
	for (int p = 1; p < maxlevel + 1; p++)
		check[p] = 0;
	for (auto block : grid.blockLevels)
		check[block.second]++;
	for (int p = 1; p < maxlevel + 1; p++)
		cout << "lvl " << p << " blocks: " << check[p] << endl;
	cout << endl << "Total rays: " << grid.CamToCel.size() << endl << endl;
}

int main()
{
	/* ---------------------- VARIABLE SETTINGS ----------------------- */
	#pragma region setting of all variables
	// Output precision
	//std::cout.precision(5);


	//vector<int> compressionParams;
	//compressionParams.push_back(cv::IMWRITE_PNG_COMPRESSION);
	//compressionParams.push_back(0);
	//cv::Mat compare = cv::imread("../pic/compare.png");
	//cv::Mat compare2 = cv::imread("../pic/compare2.png");
	//cv::Mat imgMINUS = compare - compare2;
	//imgMINUS = imgMINUS * 4;
	//cv::imwrite("comparison.png", imgMINUS, compressionParams);

	// If a spherical panorama output is used.
	bool sphereView = false;
	// If the camera axis is tilted wrt the rotation axis.
	bool angleview = false;
	// If a custom user speed is used.
	bool userSpeed = false;

	// Output window size in pixels.
	int windowWidth = 1000;
	int windowHeight = 1000;
	if (sphereView) windowHeight = (int)floor(windowWidth / 2);

	// Viewer settings.
	double viewAngle = PI/2.;
	double offset[2] = { 0., .25*PI1_4};

	// Image location.
	string image = "../pic/rainbow.png";
	string gridimage = "../pic/disk.png";
	
	// Star file location.
	string starLoc = "stars/sterren.txt";
	// Star binary tree depth.
	int treeLevel = 8;
	int magnitudeCut = 1000;

	double br = 0.;
	double bphi = 1.;
	double btheta = 0.;

	// Rotation speed.
	double afactor = 0.999;
	// Optional camera speed.
	double camSpeed = 0.00001;
	// Camera distance from black hole.
	//double camRadius = 5.0;
	double gridDist = 0.2;
	double2 camRadiusExt = { 2.6, 2.6 };
	double gridIncDist = PI / 32.;
	double2 camIncExt = { PI/2., PI/2. };// PI / 32.};
	//int gridNum = abs(camRadiusExt.y - camRadiusExt.x) / gridDist + 1;
	int gridNum = abs(camIncExt.y - camIncExt.x) / gridIncDist + 1;

	// Amount of tilt of camera axis wrt rotation axis.
	double camTheta = PI1_2;// -PI / 64.;
	if (!angleview) camTheta = PI1_2;
	//if (camTheta != PI1_2) angleview = true;
	// Amount of rotation around the axis.
	double camPhi = 0.;

	// Level settings for the grid.
	int startlevel = 1;
	int maxlevel = 10;
	#pragma endregion

	/* -------------------- INITIALIZATION CLASSES -------------------- */
	//#pragma region initializing black hole and camera
	
	BlackHole black = BlackHole(afactor);
	cout << "Initiated Black Hole " << endl;
	vector<Camera> cams;
	vector<Grid> grids(gridNum);
	for (int q = 0; q < gridNum; q++) {
		Camera cam;
		double camRad = camRadiusExt.x;
		//if (gridNum >1) camRad += 1.0*q*(camRadiusExt.y - camRadiusExt.x) / (gridNum - 1.0);
		double camInc = camIncExt.x;
		if (gridNum >1) camInc += 1.0*q*(camIncExt.y - camIncExt.x) / (gridNum - 1.0);

		double l = abs(camIncExt.y - camIncExt.x);
		double half = (camIncExt.y + camIncExt.x) / 2.;
		double angle = (camInc - min(camIncExt.y, camIncExt.x))*PI / (2.*l);
		if (angle > PI / 4.) angle = PI1_2 - angle;
		//btheta = sin(angle);
		//bphi = cos(angle);
		//if (camIncExt.y > camIncExt.x) btheta = -btheta;

		cout << btheta << " " << bphi << endl;

		if (userSpeed) cam = Camera(camTheta, camPhi, camRad, camSpeed);
		else cam = Camera(camInc, camPhi, camRad, br, btheta, bphi);
		cams.push_back(cam);
		cout << "Initiated Camera at Radius " << camRad << endl;
		cout << "Initiated Camera at Inclination " << camInc/PI << "pi" << endl;

		/* ------------------ GRID LOADING / COMPUTATION ------------------ */
		#pragma region loading grid from file or computing new grid
		// Filename for grid.
		// Create an output string stream
		std::ostringstream streamObj3;
		// Set Fixed -Point Notation
		streamObj3 << std::fixed;
		// Set precision to 2 digits
		streamObj3 << std::setprecision(1);
		//Add double to stream
		streamObj3 << camRad;
		// Get string from output string stream
		std::string strObj3 = streamObj3.str();

		stringstream ss;
		ss << "grids/" << "rayTraceLvl" << startlevel << "to" << maxlevel << "Pos" << strObj3 << "_" << camInc / PI << "_"
			<< camPhi / PI << "Speed" << afactor << "_sp_.grid";
		string filename = ss.str();

		// Try loading existing grid file, if fail compute new grid.
		ifstream ifs(filename, ios::in | ios::binary);

		time_t tstart = time(NULL);
		auto start_time = std::chrono::high_resolution_clock::now();
		if (ifs.good()) {
			cout << "Scanning gridfile..." << endl;
			{
				// Create an input archive
				cereal::BinaryInputArchive iarch(ifs);
				iarch(grids[q]);
			}
			auto end_time = std::chrono::high_resolution_clock::now();
			time_t tend = time(NULL);
			cout << "Scanned grid in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms!" << endl << endl;
		}
		else {
			cout << "Computing new grid file..." << endl << endl;

			tstart = time(NULL);
			cout << "Start = " << tstart << endl << endl;

			cout << "Raytracing grid..." << endl;
			grids[q] = Grid(maxlevel, startlevel, angleview, &cam, &black);
			cout << endl << "Computed grid!" << endl << endl;

			time_t tend = time(NULL);
			auto end_time = std::chrono::high_resolution_clock::now();
			cout << "End = " << tend << endl;
			cout << "Computed grid in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms!" << endl << endl;

			cout << "Writing to file..." << endl << endl;
			write(filename, grids[q]);
		}

		gridLevelCount(grids[q], maxlevel);
	}
	#pragma endregion

	/* --------------------- INITIALIZATION VIEW ---------------------- */
	#pragma region initializing view

	Viewer view = Viewer(viewAngle, offset[0], offset[1], windowWidth, windowHeight, sphereView);
	cout << "Initiated Viewer " << endl;

	#pragma endregion

	/* -------------------- INITIALIZATION STARS ---------------------- */
	#pragma region initializing stars

	// Reading stars into vector
	StarProcessor starProcessor;

	// Filename for grid.
	stringstream ss1;
	ss1 << "stars/" << "starProcessor_l" << treeLevel << "_m" << magnitudeCut << ".star";
	string filename = ss1.str();
	stringstream ss2;
	ss2 << "stars/starProcessor_starImg.png";

	// Try loading existing grid file, if fail compute new grid.
	ifstream ifs1(filename, ios::in | ios::binary);

	string starImgName = ss2.str();
	ifstream ifs2(starImgName);

	time_t tstart = time(NULL);
	if(!ifs1.good() || !ifs2.good()) {
		cout << "Computing new star file..." << endl;

		auto start_time = std::chrono::high_resolution_clock::now();
		starProcessor = StarProcessor(starLoc, treeLevel, image, starImgName, magnitudeCut);

		auto end_time = std::chrono::high_resolution_clock::now();
		cout << "Calculated star file in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms!" << endl << endl;

		cout << "Writing to file..." << endl << endl;
		writeStars(filename, starProcessor);
	}
	else {
		cout << "Scanning starfile..." << endl;
		{
			// Create an input archive
			cereal::BinaryInputArchive iarch(ifs1);
			iarch(starProcessor);
		}
		time_t tend = time(NULL);
		cout << "Scanned stars in " << tend - tstart << " s!" << endl << endl;

		starProcessor.imgWithStars = imread(starImgName);
	}
	#pragma endregion

	/* ----------------------- DISTORTING IMAGE ----------------------- */
	#pragma region distortion
	tstart = time(NULL);

	// Optional computation of splines from grid.
	cout << "Initiated Distorter " << endl;	
	Distorter spacetime = Distorter(&grids, &view, &starProcessor, &cams);
	spacetime.drawBlocks("blocks.png", 0);

	cout << "Computed distorted image!" << endl << endl;
	time_t tend = time(NULL);
	cout << "Visualising time: " << tend - tstart << endl;
	#pragma endregion


	/* ------------------------- SAVING IMAGE ------------------------- */
	#pragma region saving image
	//stringstream ss2;
	//ss2 << "rayTraceLvl" << startlevel << "to" << maxlevel << "Pos" << camRadius 
	//	<< "_" << camTheta / PI << "_" << camPhi / PI << "Speed" << afactor << "stars.png";
	//string imgname = ss2.str();

	//spacetime.saveImg(imgname);
	#pragma endregion

	return 0;
}