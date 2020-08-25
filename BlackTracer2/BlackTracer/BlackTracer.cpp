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
//using namespace std::;

/// <summary>
/// Serialize grid with given filename.
/// </summary>
/// <param name="filename">The filename.</param>
/// <param name="grid">The grid.</param>
void write(string filename, Grid grid) {
	{
		ofstream ofs(filename, ios::out | ios::binary);
		cereal::BinaryOutputArchive oarch(ofs);
		oarch(grid);
	}

};

/// <summary>
/// Serialize the starprocessor instance to a file.
/// </summary>
/// <param name="filename">The filename.</param>
/// <param name="starProcessor">The star processor.</param>
void writeStars(string filename, StarProcessor starProcessor) {
	{
		ofstream ofs(filename, ios::out | ios::binary);
		cereal::BinaryOutputArchive oarch(ofs);
		oarch(starProcessor);
	}
}

/// <summary>
/// Prints the number of blocks for each level, and total rays of a grid.
/// </summary>
/// <param name="grid">The grid.</param>
/// <param name="maxlevel">The maxlevel.</param>
void gridLevelCount(Grid& grid, int maxlevel) {
	vector<int> check(maxlevel + 1);
	for (int p = 1; p < maxlevel + 1; p++)
		check[p] = 0;
	for (auto block : grid.blockLevels)
		check[block.second]++;
	for (int p = 1; p < maxlevel + 1; p++)
		std::cout << "lvl " << p << " blocks: " << check[p] << std::endl;
	std::cout << endl << "Total rays: " << grid.CamToCel.size() << std::endl << std::endl;
}

/// <summary>
/// Compares two images and gives the difference error.
/// Prints error info and writes difference image.
/// </summary>
/// <param name="filename1">First image.</param>
/// <param name="filename2">Second image.</param>
void compare(string filename1, string filename2) {
	vector<int> compressionParams;
	compressionParams.push_back(cv::IMWRITE_PNG_COMPRESSION);
	compressionParams.push_back(0);
	cv::Mat compare = cv::imread(filename1);
	compare.convertTo(compare, CV_32F);
	cv::Mat compare2 = cv::imread(filename2);
	compare2.convertTo(compare2, CV_32F);
	cv::Mat imgMINUS = (compare - compare2);
	cv::Mat imgabs = cv::abs(imgMINUS);
	cv::Scalar sum = cv::sum(imgabs);

	double minVal;
	double maxVal;
	Point minLoc;
	Point maxLoc;
	cv::Mat m_out;
	cv::transform(imgabs, m_out, cv::Matx13f(1, 1, 1));
	minMaxLoc(m_out, &minVal, &maxVal, &minLoc, &maxLoc);

	std::cout << 1.f*(sum[0] + sum[1] + sum[2]) / (255.f * 1920 * 960 * 3) << std::endl;
	std::cout << minVal << " " << maxVal / (255.f*3.f) << std::endl;

	cv::Mat m_test;
	cv::transform(compare, m_test, cv::Matx13f(1, 1, 1));
	minMaxLoc(m_test, &minVal, &maxVal, &minLoc, &maxLoc);
	std::cout << minVal << " " << maxVal / (255.f*3.f) << std::endl;
	imgMINUS = 4 * imgMINUS;
	imgMINUS = cv::Scalar::all(255) - imgMINUS;
	cv::imwrite("comparisonINV.png", imgMINUS, compressionParams);
}

int main()
{
	/* ---------------------- VARIABLE SETTINGS ----------------------- */
	#pragma region setting of all variables
	// If a spherical panorama output is used.
	bool sphereView = false;
	// If the camera axis is tilted wrt the rotation axis.
	bool angleview = true;
	// If a custom user speed is used.
	bool userSpeed = false;

	// Output window size in pixels.
	int windowWidth = 1920;
	int windowHeight = 1080;
	if (sphereView) windowHeight = (int)floor(windowWidth / 2);

	// Viewer settings.
	double viewAngle = PI / 2.;
	double offset[2] = { 0., .25*PI1_4 };

	// Image location.
	string image = "../pic/adobe2.jpeg";
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
	double2 camSpeedExt = { 0.32001, 0.00001 };
	double gridSpDist = 0.02;

	// Camera distance from black hole.
	double camRadius = 5.;
	double gridDist = 0.2;
	double2 camRadiusExt = { camRadius, 1.5 };
	double gridIncDist = PI / 32.;
	double2 camIncExt = { PI / 2., PI - 1E-6 };

	int gridNum = 1.;
	int changeType = -1; // 0 radius, 1 inclination, 2 speed
	if (changeType == 0) gridNum = 1. + round(abs(camRadiusExt.y - camRadiusExt.x) / gridDist);
	else if (changeType == 1) gridNum = 1. + round(abs(camIncExt.y - camIncExt.x) / gridIncDist);
	else if (changeType == 2) gridNum = 1. + round(abs(camSpeedExt.y - camSpeedExt.x) / gridSpDist);

	// Amount of tilt of camera axis wrt rotation axis.
	double camTheta = PI1_2;
	if (camTheta != PI1_2) angleview = true;
	// Amount of rotation around the axis.
	double camPhi = 0;

	// Level settings for the grid.
	int startlevel = 1;
	int maxlevel = 10;


	/* ---------------------- USER SETTINGS ----------------------- */
	char yn;
	double var;

	std::cout << "<---------------------------------------->" << std::endl;
	std::cout << "Welcome to BlackTracer, a Black Hole simulator!" << std::endl;
	std::cout << "The different parameters in the simulation can be set\n by hand of you can choose the standard version." << std::endl << std::endl;
	std::cout << "Do you want to choose your own parameters? (y/n)" << std::endl;
	std::cin >> yn; std::cout << std::endl;
	if (yn == 'y') {
		std::cout << "Black Hole Parameters:" << std::endl;
		std::cout << "Spin between 0.001-0.999: ";
		std::cin >> var; std::cout << std::endl;
	}

	std::cout << "<---------------------------------------->" << std::endl;



	std::cout << "" << std::endl;



	#pragma endregion

	/* -------------------- INITIALIZATION CLASSES -------------------- */
	//#pragma region initializing black hole and camera
	
	BlackHole black = BlackHole(afactor);
	std::cout << "Initiated Black Hole " << std::endl;
	vector<Camera> cams;
	vector<Grid> grids(gridNum);
	for (int q = 0; q < gridNum; q++) {
		Camera cam;
		double camRad = camRadiusExt.x;
		double camInc = camIncExt.x;
		double camSpeed = camSpeedExt.x;
		if (gridNum > 1) {
			if (changeType == 0) camRad += 1.0*q*(camRadiusExt.y - camRadiusExt.x) / (gridNum - 1.0);
			if (changeType == 1) camInc += 1.0*q*(camIncExt.y - camIncExt.x) / (gridNum - 1.0);
			if (changeType == 2) camSpeed += 1.0*q*(camSpeedExt.y - camSpeedExt.x) / (gridNum - 1.0);
		}
		if (userSpeed) cam = Camera(camTheta, camPhi, camRad, camSpeed);
		else cam = Camera(camInc, camPhi, camRad, br, btheta, bphi);
		cams.push_back(cam);
		std::cout << "Initiated Camera at Radius " << camRad << std::endl;
		std::cout << "Initiated Camera at Inclination " << camInc / PI << "pi" << std::endl;
		
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
		if (!userSpeed) ss << "grids/" << "rayTraceLvl" << startlevel << "to" << maxlevel << "Pos" << strObj3 << "_" << camInc / PI << "_"
			<< camPhi / PI << "Spin" << afactor << "_sp_.grid";
		else ss << "grids/" << "rayTraceLvl" << startlevel << "to" << maxlevel << "Pos" << strObj3 << "_" << camInc / PI << "_"
			<< camPhi / PI << "Spin" << afactor << "Speed" << camSpeed <<".grid";
		string filename = ss.str();

		// Try loading existing grid file, if fail compute new grid.
		ifstream ifs(filename, ios::in | ios::binary);


		auto start_time = std::chrono::high_resolution_clock::now();
		if (ifs.good()) {
			std::cout << "Scanning gridfile..." << std::endl;
			{
				// Create an input archive
				cereal::BinaryInputArchive iarch(ifs);
				iarch(grids[q]);
			}
			auto end_time = std::chrono::high_resolution_clock::now();
			std::cout << "Scanned grid in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms!" << std::endl << std::endl;
		}
		else {
			std::cout << "Computing new grid file..." << std::endl << std::endl;

			std::cout << "Raytracing grid..." << std::endl;
			grids[q] = Grid(maxlevel, startlevel, angleview, &cam, &black);
			std::cout << endl << "Computed grid!" << std::endl << std::endl;

			auto end_time = std::chrono::high_resolution_clock::now();
			std::cout << "Computed grid in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms!" << std::endl << std::endl;
			std::cout << "Writing to file..." << std::endl << std::endl;
			write(filename, grids[q]);

			gridLevelCount(grids[q], maxlevel);
		}

	}
	#pragma endregion
	//vector<uchar3> imagemat = vector<uchar3>(2048 * 1025);
	//for (int q = 0; q < 2048 * 1025; q++) {
	//	int steps = grids[0].steps[q];
	//	if (steps == 0) imagemat[q] = { 255, 255, 255 };
	//	else if (steps <= 20) imagemat[q] = {84, 1, 68};
	//	else if (steps <= 30) imagemat[q] = { 136, 69, 64 };
	//	else if (steps <= 40) imagemat[q] = { 141, 96, 52 };
	//	else if (steps <= 50) imagemat[q] = { 142, 121, 41 };
	//	else if (steps <= 60) imagemat[q] = { 140, 148, 32 };
	//	else if (steps <= 70) imagemat[q] = { 132, 168, 34 };
	//	else if (steps <= 80) imagemat[q] = { 111, 192, 70 };
	//	else if (steps <= 90) imagemat[q] = { 84, 208, 117 };
	//	else if (steps <= 100) imagemat[q] = { 38, 223, 189 };
	//	else imagemat[q] = { 33, 231, 249 };
	//}

	//string imgname = "heatmap.jpg";
	//cv::Mat img = cv::Mat(1025, 2048, CV_8UC3, (void*)&imagemat[0]);
	//cv::imwrite(imgname, img);

	/* --------------------- INITIALIZATION VIEW ---------------------- */
	#pragma region initializing view

	Viewer view = Viewer(viewAngle, offset[0], offset[1], windowWidth, windowHeight, sphereView);
	std::cout << "Initiated Viewer " << std::endl;

	#pragma endregion

	/* -------------------- INITIALIZATION STARS ---------------------- */
	#pragma region initializing stars

	// Reading stars into vector
	StarProcessor starProcessor;

	// Filename for stars and image.
	stringstream ss1;
	ss1 << "stars/" << "starProcessor_l" << treeLevel << "_m" << magnitudeCut << ".star";
	string filename = ss1.str();
	stringstream ss2;
	ss2 << "stars/starProcessor_starImg.png";

	// Try loading existing grid file, if fail compute new grid.
	ifstream ifs1(filename, ios::in | ios::binary);

	//string starImgName = ss2.str();
	ifstream ifs2(image);

	time_t tstart = time(NULL);
	if(!ifs1.good() || !ifs2.good()) {
		std::cout << "Computing new star file..." << std::endl;

		auto start_time = std::chrono::high_resolution_clock::now();
		starProcessor = StarProcessor(starLoc, treeLevel, image, image, magnitudeCut);

		auto end_time = std::chrono::high_resolution_clock::now();
		std::cout << "Calculated star file in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms!" << std::endl << std::endl;

		std::cout << "Writing to file..." << std::endl << std::endl;
		writeStars(filename, starProcessor);
	}
	else {
		std::cout << "Scanning starfile..." << std::endl;
		{
			// Create an input archive
			cereal::BinaryInputArchive iarch(ifs1);
			iarch(starProcessor);
		}
		time_t tend = time(NULL);
		std::cout << "Scanned stars in " << tend - tstart << " s!" << std::endl << std::endl;

		starProcessor.imgWithStars = imread(image);
	}
	#pragma endregion

	/* ----------------------- DISTORTING IMAGE ----------------------- */
	#pragma region distortion
	// Optional computation of splines from grid.
	std::cout << "Initiated Distorter " << std::endl;
	Distorter spacetime = Distorter(&grids, &view, &starProcessor, &cams);\
	spacetime.drawBlocks("blocks7.png", 0);

	#pragma endregion

	return 0;
}