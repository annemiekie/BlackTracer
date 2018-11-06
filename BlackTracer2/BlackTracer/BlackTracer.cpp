#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdio.h>
#include <stdint.h> 
#include <sstream>
#include <string>

#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/unordered_map.hpp>

#include "BlackHole.h"
#include "Camera.h"
#include "Grid.h"
#include "Splines.h"
#include "Distorter.h"
#include "Star.h"
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

/* Count and display number of blocks per level in provided grid. */
void gridLevelCount(Grid& grid, int maxlevel) {
	vector<int> check(maxlevel + 1);
	for (int p = 1; p < maxlevel + 1; p++)
		check[p] = 0;
	for (auto block : grid.blockLevels)
		check[block.second]++;
	for (int p = 1; p < maxlevel + 1; p++)
		cout << "lvl " << p << " blocks: " << check[p] << endl;
}

int main()
{
	/* ---------------------- VARIABLE SETTINGS ----------------------- */
	#pragma region setting of all variables
	// Output precision
	std::cout.precision(5);

	// If spline Interpolation needs to be performed.
	bool splineInter = false;
	// If a spherical panorama output is used.
	bool sphereView = true;
	// If the camera axis is tilted wrt the rotation axis.
	bool angleview = false;
	// If a custom user speed is used.
	bool userSpeed = false;

	// Output window size in pixels.
	int windowWidth = 1000;
	int windowHeight = 700;
	if (sphereView) windowHeight = (int)floor(windowWidth / 2);

	// Viewer settings.
	double viewAngle = PI/2.;
	double offset[2] = { 0, .5*PI1_4};

	// Image location.
	string image = "../pic/artstarsb2.png";
	string starLoc = "catalog_text";

	// Rotation speed.
	double afactor = 0.999;
	// Optional camera speed.
	double camSpeed = 0.;
	// Camera distance from black hole.
	double camRadius = 4.;
	// Amount of tilt of camera axis wrt rotation axis.
	double camTheta = PI1_2;
	if (camTheta != PI1_2) angleview = true;
	// Amount of rotation around the axis.
	double camPhi = 0.;

	// Level settings for the grid.
	int maxlevel = 11;
	int startlevel = 1;
	#pragma endregion

	/* -------------------- INITIALIZATION CLASSES -------------------- */
	#pragma region initializing black hole and camera
	
	BlackHole black = BlackHole(afactor);
	cout << "Initiated Black Hole " << endl;

	Camera cam;
	if (userSpeed) cam = Camera(camTheta, camPhi, camRadius, camSpeed);
	else cam = Camera(camTheta, camPhi, camRadius);
	cout << "Initiated Camera " << endl;
	#pragma endregion

	/* ------------------ GRID LOADING / COMPUTATION ------------------ */
	#pragma region loading grid from file or computing new grid
	Grid grid;

	// Filename for grid.
	stringstream ss;
	ss << "rayTraceLvl" << startlevel << "to" << maxlevel << "Pos" << camRadius << "_" << camTheta / PI << "_" 
		<< camPhi / PI << "Speed" << afactor;
	string filename = ss.str();

	// Try loading existing grid file, if fail compute new grid.
	ifstream ifs(filename + ".grid", ios::in | ios::binary);

	time_t tstart = time(NULL);
	if (ifs.good()) {
		cout << "Scanning gridfile..." << endl;
		{
			// Create an input archive
			cereal::BinaryInputArchive iarch(ifs);
			iarch(grid);
		}
		time_t tend = time(NULL);
		cout << "Scanned grid in " << tend - tstart << " s!" << endl << endl;
	}
	else {
		cout << "Computing new grid file..." << endl << endl;

		time_t tstart = time(NULL);
		cout << "Start = " << tstart << endl << endl;

		cout << "Raytracing grid..." << endl;
		grid = Grid(maxlevel, startlevel, angleview, &cam, &black);
		cout << endl << "Computed grid!" << endl << endl;

		time_t tend = time(NULL);
		cout << "End = " << tend << endl;
		cout << "Time = " << tend - tstart << endl << endl;

		cout << "Writing to file..." << endl << endl;
		write(filename + ".grid", grid);
	}

	gridLevelCount(grid, maxlevel);
	#pragma endregion

	/* --------------------- INITIALIZATION VIEW ---------------------- */
	#pragma region initializing view

	Viewer view = Viewer(viewAngle, offset[0], offset[1], windowWidth, windowHeight, sphereView);
	cout << "Initiated Viewer " << endl;

	// Reading stars into vector
	vector<Star> stars = readStars(starLoc);
	#pragma endregion

	/* ----------------------- DISTORTING IMAGE ----------------------- */
	#pragma region distortion
	tstart = time(NULL);

	// Optional computation of splines from grid.
	Splines splines;
	if (splineInter) splines = Splines(&grid, &view);
	
	Distorter spacetime = Distorter(image, &grid, &view, &splines, &stars, splineInter, &cam);
	cout << "Initiated Distorter " << endl;
	cout << "Distorting image..." << endl << endl;
	spacetime.rayInterpolater();
	cout << "Computed distorted image!" << endl << endl;
	time_t tend = time(NULL);
	cout << "Visualising time: " << tend - tstart << endl;
	#pragma endregion

	/* ------------------------- SAVING IMAGE ------------------------- */
	#pragma region saving image
	stringstream ss2;
	ss2 << "rayTraceLvl" << startlevel << "to" << maxlevel << "Pos" << camRadius 
		<< "_" << camTheta / PI << "_" << camPhi / PI << "Speed" << afactor << "stars.png";
	string imgname = ss2.str();

	spacetime.saveImg(imgname);
	#pragma endregion

	return 0;
}

