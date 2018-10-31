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

void write(string filename, Grid grid) {

	{
		ofstream ofs(filename, ios::out | ios::binary);
		cereal::BinaryOutputArchive oarch(ofs);
		oarch(grid);
	}

};

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
	std::cout.precision(5);

	bool splineInter = false;
	bool sphereView = true;
	bool angleview = false;
	bool userSpeed = false;
	int windowWidth = 1000;
	int windowHeight = 700;
	if (sphereView) windowHeight = (int)floor(windowWidth / 2);
	double viewAngle = PI/2.;
	double offset[2] = { 0, .5*PI1_4};
	string image = "../pic/artstarsb2.png";

	double afactor = 0.999;
	double camSpeed = 0.;
	double camRadius = 4.;
	double camTheta = PI1_2;
	if (camTheta != PI1_2) angleview = true;
	double camPhi = 0.;

	int maxlevel = 6;
	int startlevel = 1;
	stringstream ss;
	ss << "rayTraceLvl" << startlevel << "to" << maxlevel << "Pos" << camRadius << "_" << camTheta / PI << "_" 
		<< camPhi / PI << "Speed" << afactor;
	string filename = ss.str();

	/* Initiating Grid, Black Hole and Camera */
	Grid grid;

	BlackHole black = BlackHole(afactor);
	cout << "Initiated Black Hole " << endl;

	Camera cam;
	if (userSpeed) cam = Camera(camTheta, camPhi, camRadius, camSpeed);
	else cam = Camera(camTheta, camPhi, camRadius);
	cout << "Initiated Camera " << endl;

	ifstream ifs(filename + ".grid", ios::in | ios::binary);
	ifstream ifs2(filename + ".txt");

	time_t tstart = time(NULL);
	if (ifs.good()) {
		cout << "Scanning gridfile..." << endl;
		{
				cereal::BinaryInputArchive iarch(ifs); // Create an input archive
				iarch(grid);

		}
		time_t tend = time(NULL);
		cout << "Scanned grid in " << tend - tstart << " s!" << endl << endl;
	}
	else if (ifs2.good()) {
		cout << "Scanning txtfile..." << endl;

		ifs2.close();
		grid = Grid(filename + ".txt");
		time_t tend = time(NULL);
		cout << "Scanned grid in " << tend - tstart << " s!" << endl << endl;
		write(filename + ".grid", grid);
	}
	else {
		cout << "Computing new file..." << endl << endl;

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
		return 0;
	}

	gridLevelCount(grid, maxlevel);

	tstart = time(NULL);

	Viewer view = Viewer(viewAngle, offset[0], offset[1], windowWidth, windowHeight, sphereView);
	cout << "Initiated Viewer " << endl;

	Splines splines;
	if (splineInter) splines = Splines(&grid, &view);

	vector<Star> stars = readStars("catalog_text");

	Distorter spacetime = Distorter(image, &grid, &view, &splines, &stars, splineInter, &cam);
	cout << "Initiated Distorter " << endl;

	cout << "Distorting image..." << endl << endl;
	spacetime.rayInterpolater();
	cout << "Computed distorted image!" << endl << endl;
	time_t tend = time(NULL);
	cout << "Visualising time: " << tend - tstart << endl;

	stringstream ss2;
	ss2 << "rayTraceLvl" << startlevel << "to" << maxlevel << "Pos" << camRadius 
		<< "_" << camTheta / PI << "_" << camPhi / PI << "Speed" << afactor << "stars.png";
	string imgname = ss2.str();


	spacetime.saveImg(imgname);

	return 0;
}

