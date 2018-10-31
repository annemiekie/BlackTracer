#pragma once

#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdint.h> 
#include <vector>
#include "Const.h"
#include "SplineBlock.h"
#include "Bezier.h"
#include "Grid.h"
#include "Viewer.h"
#include "Code.h"
#include <chrono>
#include <thread>

using namespace std;
using namespace cv;

class Splines
{
private:
	bool debug = false;
	Grid* grid;
	Viewer* view;
public:
	unordered_map <uint64_t, SplineBlock > CamToSpline;
	Splines(){};

	Splines(Grid* _grid, Viewer* _view) {
		grid = _grid;
		view = _view;
		makeSplineBlocks();
		//viewBlocks();
	}
	
	void print(vector<Point2d> vec) {
		for (int i = 0; i < vec.size(); i++) {
			cout << vec[i] << endl;
		}
	}

	void viewBlocks() {
		int pixh = view->pixelheight * 2;
		int pixw = view->pixelwidth * 2;
		Mat img(pixh, pixw, CV_8UC3, Vec3b(255,255,255));

		//namedWindow("SplineBlock", WINDOW_NORMAL);
		//imshow("SplineBlock", img);
		//waitKey(0);

		for (int j = 0; j < grid->M; j++) {
			for (int i = 0; i < grid->N; i++) {
				auto it = CamToSpline.find(i_j);
				if (it != CamToSpline.end()) {
					SplineBlock splblk = CamToSpline[i_j];
					//img += splblk.draw(img.rows, img.cols);
					Vec3b color = Vec3b(255, 0, 0);
					if (1.0*j / grid->M < 0.187 || 1.0*j / grid->M > 0.65)//color = Vec3b(255, 255, 0);
					visualiseSplineBlock(splblk, img,Vec3b(0,0,0));
				}
			}
		}


		namedWindow("SplineBlock", WINDOW_NORMAL);
		imshow("SplineBlock", img);
		waitKey(0);
	}

	void visualiseSplineBlock(SplineBlock splblk, Mat img, Vec3b color) {

		splblk.draw(color, img);

		Bezier upDown = splblk.interpolate(0.2, splblk.up, splblk.down, splblk.left, splblk.right);
		Bezier leftRight = splblk.interpolate(0.5, splblk.left, splblk.right, splblk.up, splblk.down);

		upDown.img(img, Vec3b(0, 0, 255));
		leftRight.img(img, Vec3b(0, 0, 255));
		
		/*double phmin, phmax, thmin, thmax;
		upDown.makeBoundBox(phmax, phmin, thmax, thmin);
		upDown.drawBoundBox(img,phmax, phmin, thmax, thmin);
		leftRight.makeBoundBox(phmax, phmin, thmax, thmin);
		leftRight.drawBoundBox(img, phmax, phmin, thmax, thmin);*/

		Point2d cross;
		splblk.interpolate(0.5, 0.5, cross);
		//cout << cross << endl;
		
		namedWindow("SplineBlock", WINDOW_NORMAL);
		imshow("SplineBlock", img);
		waitKey(0);
	}

	bool fillsplUD(vector<Point2d>& spline, vector<int>& splits, int i, int i32, int j32, int fac) {
		for (int j = j32 - fac; j <= j32 + fac + 1; j++) {
			uint32_t l;
			if (j < 0) l = j + grid->M;
			else l = j%grid->M;
			//if (debug) cout << l << endl;

			Point2d thphi = grid->CamToCel[i_l];
			if (thphi_theta > 0 && thphi_phi > 0)
				spline.push_back(thphi);
			else if (j < j32)
				splits[i - i32] += 1;
			else if (j <= j32 + 1)
				return 0;
		}
		return 1;
	}

	bool fillsplLR(vector<Point2d>& spline, vector<int>& splits, int j, int i32, int j32, int fac) {
		for (int i = i32 - fac; i <= i32 + fac + 1; i++) {
			uint32_t l = j;
			if (i < 0)
				splits[j - j32 + 2] += 1;
			else if (i <= grid->N) {
				l = l%grid->M;
				Point2d thphi = grid->CamToCel[i_l];
				if (thphi_theta > 0 && thphi_phi > 0)
					spline.push_back(thphi);
				else if (i < i32)
					splits[j - j32 + 2] += 1;
				else if (i <= i32 + 1)
					return 0;
			}
		}
		return 1;
	}

	void makeSplineBlocks() {
		for (auto block : grid->blockLevels) {
			makeSplineBlock(block.first);
		}
	}

	bool checkLargeVal(vector<Point2d>& vec, float factor) {
		for (int i = 0; i < vec.size(); i++) {
			if (vec[i]_phi > PI2*(1. - 1. / factor))
				return true;
		}
		return false;
	}

	bool checkSmallAndfix(vector<Point2d>& vec, float factor) {
		bool check = false;
		for (int i = 0; i < vec.size(); i++) {
			if (vec[i]_phi < PI2*(1. / factor)) {
				vec[i]_phi += PI2;
				check = true;
			}
		}
		return check;
	}

	bool checkMaxdiff(vector<Point2d>& vec, float factor) {
		float max = 0.f;
		for (int i = 0; i < vec.size()-1; i++) {
			float val = (vec[i]_phi - vec[i + 1]_phi)*(vec[i]_phi - vec[i + 1]_phi) + 
				(vec[i]_theta - vec[i + 1]_theta)*(vec[i]_theta - vec[i + 1]_theta);
			if (val>max) max = val;
		}
		//cout << max << endl;
		return (max > PI2 / factor);
	}


	bool check2PIcross(vector<Point2d>& vec1, vector<Point2d>& vec2, vector<Point2d>& vec3, vector<Point2d>& vec4, float factor) {
		bool check1, check2, check3, check4;
		check1 = check2 = check3 = check4 = true;
		if (checkLargeVal(vec1, factor) || checkLargeVal(vec2, factor) || checkLargeVal(vec3, factor) || checkLargeVal(vec4, factor)) {
			//factor /= 2.0; 
			check1 = checkSmallAndfix(vec1, factor);
			check2 = checkSmallAndfix(vec2, factor);
			check3 = checkSmallAndfix(vec3, factor);
			check4 = checkSmallAndfix(vec4, factor);
		}
		return (check1 || check2 || check3 || check4);
	};
	
	void makeSplineBlock(uint64_t ij) {

		int fac = 1;
		vector<Point2d> splThetaUp, splPhiLeft, splThetaDown, splPhiRight;
		vector<int> splits = { 0, 0, 0, 0 };
		int i32 = int(i_32);
		int j32 = int(j_32);

		if (debug) cout << i32 << " " << j32 << endl;

		if (!fillsplUD(splThetaUp, splits, i32, i32, j32, fac) ||
			!fillsplUD(splThetaDown, splits, i32+1, i32, j32, fac) ||
			!fillsplLR(splPhiLeft, splits, j32, i32, j32, fac) ||
			!fillsplLR(splPhiRight, splits, j32 + 1, i32, j32, fac)) return;

		if (debug) {
			cout << "filled vectors" << endl;
			printspline(splThetaUp);
			printspline(splThetaDown);
			printspline(splPhiLeft);
			printspline(splPhiRight);
		}

		bool check = check2PIcross(splThetaUp, splPhiLeft, splThetaDown, splPhiRight, 5);
		float maxdiff = 3.;
		if (checkMaxdiff(splThetaUp, maxdiff) || checkMaxdiff(splPhiLeft, maxdiff) ||
			checkMaxdiff(splThetaDown, maxdiff) || checkMaxdiff(splPhiRight, maxdiff)) return;
		
		if (debug) {
			cout << "checked 2PI" << endl;
			if (check) {
				printspline(splThetaUp);
				printspline(splThetaDown);
				printspline(splPhiLeft);
				printspline(splPhiRight);
			}
		}

		vector<Point2d> p1, p2;

		controlPoints(splThetaUp, p1, p2);
		if (!checkspline(splThetaUp, p1, p2)) return;
		int pos = fac - splits[0];
		SplineBlock splblock = SplineBlock(Bezier(splThetaUp[pos], splThetaUp[pos + 1], p1[pos], p2[pos]));

		controlPoints(splThetaDown, p1, p2);
		if (!checkspline(splThetaDown, p1, p2)) return;
		pos = fac - splits[1];
		splblock.setDown(Bezier(splThetaDown[pos], splThetaDown[pos + 1], p1[pos], p2[pos]));

		controlPoints(splPhiLeft, p1, p2);
		if (!checkspline(splPhiLeft, p1, p2)) return;
		pos = fac - splits[2];
		splblock.setLeft(Bezier(splPhiLeft[pos], splPhiLeft[pos + 1], p1[pos], p2[pos]));

		controlPoints(splPhiRight, p1, p2);
		if (!checkspline(splPhiRight, p1, p2)) return;
		pos = fac - splits[3];
		splblock.setRight(Bezier(splPhiRight[pos], splPhiRight[pos + 1], p1[pos], p2[pos]));

		if (debug) cout << "made splines" << endl;
		
		CamToSpline[ij] = splblock;

		if (debug) {
			int pixh = view->pixelheight * 2;
			int pixw = view->pixelwidth * 2;
			Mat img(pixh, pixw, CV_8UC1, Scalar(0));
			visualiseSplineBlock(splblock, img, Vec3b(0,0,0));
		}
	}


	double pointsDist(Point2d p1, Point2d p2) {
		return (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y);
	}

	bool checkspline(vector<Point2d>& spline, vector<Point2d>& p1, vector<Point2d>& p2) {
		for (int i = 0; i < spline.size() - 1; i++) {
			double dist = pointsDist(spline[i], spline[i + 1]);
			if (dist < 1E-6) return true;
			else if (pointsDist(spline[i], p1[i]) > 2.*dist) return false;
			else if (pointsDist(spline[i + 1], p2[i]) > 2.*dist) return false;
		}
		return true;
	}




	void checkTvertices() {

		for (auto splblk : CamToSpline) {
			uint64_t ij = splblk.first;
			int level = grid->blockLevels(ij);
			if (level == grid->MAXLEVEL) return;

			int gap = pow(2, grid->MAXLEVEL - level);
			uint32_t i = i_32;
			uint32_t j = j_32;
			uint32_t k = i + gap;
			uint32_t l = (j + gap) % grid->M;

			checkAdjacentBlock(ij, ij, i_l, level, 0, 1, gap, 1);
			checkAdjacentBlock(ij, k_j, k_l, level, 0, 1, gap, 2);
			checkAdjacentBlock(ij, ij, k_j, level, 1, 0, gap, 3);
			checkAdjacentBlock(ij, i_l, k_l, level, 1, 0, gap, 4);

		}
	}

	void checkAdjacentBlock(uint64_t ind, uint64_t ij, uint64_t ij2, int level, int ud, int lr, int gap, int pos) {
		uint32_t i = i_32 + ud * gap / 2;
		uint32_t j = j_32 + lr * gap / 2;
		auto it = grid->CamToCel.find(i_j);
		if (it == grid->CamToCel.end())
			return;
		else {
			//grid->CamToCel[i_j] = 1. / 2.*(grid->CamToCel[ij] + grid->CamToCel[ij2]);
			// FIND MIDPOINT:
			SplineBlock splblk = CamToSpline[ind];
		//	Point2d midpoint = splblk.interpolate(0.5, 0.5);
			Bezier spl1, spl2;
			switch (pos) {
			case 1: splblk.up.split(0.5, spl1, spl2); 

				break;
			case 2: splblk.down.split(0.5, spl1, spl2); break;
			case 3: splblk.left.split(0.5, spl1, spl2); break;
			case 4: splblk.right.split(0.5, spl1, spl2); break;
			}
			// MAKE NEW BLOCKS WITH HALF SPLINES
			if (level + 1 == grid->MAXLEVEL) return;
			//checkAdjacentBlock(ij, i_j, level + 1, ud, lr, gap / 2);
			//checkAdjacentBlock(i_j, ij2, level + 1, ud, lr, gap / 2);
		}
	}







	void printspline(vector<Point2d> spline) {
		cout << "spliine " << endl;
		for (int i = 0; i < spline.size(); i++) {
			cout << spline[i] << endl;
		}
	}



	void makeBlackSpline() {
		unordered_map<uint64_t, vector<Point2d>> boundBlocks;
		vector<uint64_t> blockindices;

		for (auto block : grid->blockLevels) {
			checkBlock(block.first, boundBlocks, blockindices);
		}
		//cout << boundBlocks.size() << endl;
		vector<Point2d> blackpoints;
		uint64_t ij = blockindices[0];
		blockindices.erase(blockindices.begin());
		blackpoints.push_back(boundBlocks[ij][0]);
		blackpoints.push_back(boundBlocks[ij][1]);
		while (blockindices.size() > 0) {
			Point2d checkpoint = blackpoints[blackpoints.size() - 1];
			//int i = i_32;
			//int j = j_32;
			for (int q = 0; q < blockindices.size(); q++) {
				ij = blockindices[q];
				if (boundBlocks[ij][0] == checkpoint) {
					blackpoints.push_back(boundBlocks[ij][1]);
					blockindices.erase(blockindices.begin() + q);
					break;
				}
				else if (boundBlocks[ij][1] == checkpoint) {
					blackpoints.push_back(boundBlocks[ij][0]);
					blockindices.erase(blockindices.begin() + q);
					break;
				}
				if (q == blockindices.size() - 1) {
					cout << "error in computing blackspline" << endl;
					return;
				}
			}
		}
		int smoothness = floor(1. / 10. * grid->N);
		cout << smoothness << endl;
		for (int q = 0; q < 3 * smoothness; q++) {
			blackpoints.push_back(blackpoints[q]);
		}

		vector<Point2d> p1, p2;
		vector<Bezier> blackspline;
		vector<Point2d> blackpoints2;
		for (int q = 1; q < blackpoints.size(); q += smoothness) {
			blackpoints2.push_back(blackpoints[q]);
		}

		controlPoints(blackpoints2, p1, p2);
		for (int q = 1; q < blackpoints2.size() - 2; q++) {
			blackspline.push_back(Bezier(blackpoints2[q], blackpoints2[q + 1], p1[q], p2[q]));
		}
		visualiseSplines(blackspline);
	}

	void visualiseSplines(vector<Bezier> splinevector) {
		Mat img(view->pixelheight, view->pixelwidth, CV_8UC1, Scalar(0));
		Mat img2 = img.clone();

#pragma omp parallel for
		for (int p = 0; p < splinevector.size(); p++) {
			vector<Point2d> curvePoints = splinevector[p].draw(img.cols, img.rows, 64);
			for (int q = 0; q < curvePoints.size() - 1; q++){
				double phi1 = curvePoints[q].x;
				double phi2 = curvePoints[q + 1].x;
				line(img, curvePoints[q], curvePoints[q + 1], Scalar(255), 1, CV_AA);
			}
			img += img2;
		}
		imshow("Spline", img);
		waitKey(0);
	}

	void checkBlock(uint64_t ij, unordered_map<uint64_t, vector<Point2d>>& boundBlocks, vector<uint64_t>& blockindices) {
		vector<double> cornersCam;
		vector<Point2d> cornersCel;
		calcCornerVals(ij, cornersCel, cornersCam);
		vector<int> negvals;

		for (int i = 0; i < 4; i++) {
			if (cornersCel[i]_theta < 0) {
				negvals.push_back(i);
			}
		}
		if (negvals.size() == 0 || negvals.size() == 4) return;
		else {
			double thetaUp = cornersCam[0];
			double thetaDown = cornersCam[2];
			double phiLeft = cornersCam[1];
			double phiRight = cornersCam[3];
			double thetaMid = (thetaUp + thetaDown) * HALF;
			double phiMid = (phiRight + phiLeft) * HALF;
			blockindices.push_back(ij);
			switch (negvals.size()) {
			case 2:
				switch (abs(negvals[0] - negvals[1])) {
				case 1: boundBlocks[ij] = vector < Point2d > {Point2d(thetaMid, phiLeft), Point2d(thetaMid, phiRight)};
						return;
				case 2: boundBlocks[ij] = vector < Point2d > {Point2d(thetaUp, phiMid), Point2d(thetaDown, phiMid)};
						return;
				}
			case 3:
				switch (negvals[0] + negvals[1] + negvals[2]) {
				case 3: boundBlocks[ij] = vector < Point2d > {Point2d(thetaMid, phiRight), Point2d(thetaDown, phiMid)};
						return;
				case 4: boundBlocks[ij] = vector < Point2d > {Point2d(thetaMid, phiLeft), Point2d(thetaDown, phiMid)};
						return;
				case 5: boundBlocks[ij] = vector < Point2d > {Point2d(thetaMid, phiRight), Point2d(thetaUp, phiMid)};
						return;
				case 6: boundBlocks[ij] = vector < Point2d > {Point2d(thetaMid, phiLeft), Point2d(thetaUp, phiMid)};
						return;
				}
			case 1:
				switch (negvals[0]) {
				case 3: boundBlocks[ij] = vector < Point2d > {Point2d(thetaMid, phiRight), Point2d(thetaDown, phiMid)};
						return;
				case 2: boundBlocks[ij] = vector < Point2d > {Point2d(thetaMid, phiLeft), Point2d(thetaDown, phiMid)};
						return;
				case 1: boundBlocks[ij] = vector < Point2d > {Point2d(thetaMid, phiRight), Point2d(thetaUp, phiMid)};
						return;
				case 0: boundBlocks[ij] = vector < Point2d > {Point2d(thetaMid, phiLeft), Point2d(thetaUp, phiMid)};
						return;
				}
			}
		}
	}

	void calcCornerVals(uint64_t ij, vector<Point2d>& cornersCel, vector<double>& cornersCam) {
		int level = grid->blockLevels[ij];
		int gap = (int)pow(2, grid->MAXLEVEL - level);
		uint32_t i = i_32;
		uint32_t j = j_32;
		uint32_t k = i + gap;
		// l should convert to 2PI at end of grid for correct interpolation
		uint32_t l = j + gap;
		double factor = PI2 / grid->M;
		cornersCam = { factor*i, factor*j, factor*k, factor*l };
		// but for lookup l should be the 0-point
		l = l % grid->M;
		cornersCel = { grid->CamToCel[ij], grid->CamToCel[i_l], grid->CamToCel[k_j], grid->CamToCel[k_l] };
	}


	//https://www.codeproject.com/Articles/31859/Draw-a-Smooth-Curve-through-a-Set-of-D-Points-wit 
	//Oleg V.Polikarpotchkin, Peter Lee, 24 Mar 2009
	void controlPoints(vector<Point2d>& knots, vector<Point2d> &p1, vector<Point2d> &p2) {
		size_t n = knots.size() - 1;
		p1.resize(n);
		p2.resize(n);
		if (n == 1) {
			// 3P1 = 2P0 + P3
			p1[0] = (2. * knots[0] + knots[1]) / 3.;
			// P2 = 2P1 – P0
			p2[0] = 2 * p1[0] - knots[0];
			return;
		}
		vector<Point2d> r(n);
		for (int i = 1; i < n - 1; i++)
			r[i] = 4. * knots[i] + 2. * knots[i + 1];
		r[0] = knots[0] + 2. * knots[1];
		r[n - 1] = (8. * knots[n - 1] + knots[n]) / 2.;

		// First control point
		solvesystem(r, p1);

		for (int i = 0; i < n - 1; ++i)
			p2[i] = 2. * knots[i + 1] - p1[i + 1];

		p2[n - 1] = (knots[n] + p1[n - 1]) / 2.;
	};

	void solvesystem(vector<Point2d> r, vector<Point2d>& x) {
		size_t n = r.size();
		vector<double> tmp(n);
		double b = 2.;
		x[0] = r[0] / b;
		for (int i = 1; i < n; i++) {// Decomposition and forward substitution.
			tmp[i] = 1. / b;
			b = (i < n - 1 ? 4.0 : 3.5) - tmp[i];
			x[i] = (r[i] - x[i - 1]) / b;
		}

		for (int i = 1; i < n; i++) {// Decomposition and forward substitution.
			x[n - i - 1] -= tmp[n - i] * x[n - i];
		}

	};

	~Splines(){};
};