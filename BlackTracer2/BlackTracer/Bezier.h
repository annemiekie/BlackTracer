#pragma once

#include "opencv2/imgcodecs/imgcodecs.hpp"
#include <cmath>
#include <iostream>
#include <stdint.h> 
#include <vector>
#include "Const.h"
#include "Code.h"

using namespace std;
using namespace cv;

class Bezier
{
private:


public:
	vector<Point2d> P;
	Bezier(){};

	Bezier(Point2d start, Point2d end, Point2d contr1, Point2d contr2) {
		P.resize(4);
		P[0] = start;
		P[1] = contr1;
		P[2] = contr2;
		P[3] = end;
	};

	Point2d start(){
		return P[0];
	}

	Point2d end(){
		return P[3];
	}

	Point2d pointOnLine(double t) {
		return this->start() + t*(this->end() - this->start());
	}

	void getDeriv(Point2d &startDir, Point2d &endDir) {
		startDir = spherCoorAt(0.01) - spherCoorAt(0.);
		endDir = spherCoorAt(0.99) - spherCoorAt(1.);
	}

	double maxminGoldSec(double s, double t, int tp, bool max) {
		double gr = (sqrt(5.0) + 1.) / 2.;

		double ax = s;
		double b = t;
		double c = b - (b - ax) / gr;
		double d = ax + (b - ax) / gr;

		while (fabs(c - d) > 0.0001) {
			if ((max) ? Vec2d(spherCoorAt(c))[tp] > Vec2d(spherCoorAt(d))[tp] :
						Vec2d(spherCoorAt(c))[tp] < Vec2d(spherCoorAt(d))[tp]) {
				b = d;
			}
			else {
				ax = c;
			}
			c = b - (b - ax) / gr;
			d = ax + (b - ax) / gr;
		}
		return (ax + b) / 2.;
	}

	void max(double s, double t, double& maxT, double& maxP, double& minT, double& minP) {
		maxT = maxminGoldSec(s, t, 0, 1);
		maxP = maxminGoldSec(s, t, 1, 1);
		minT = maxminGoldSec(s, t, 0, 0);
		maxP = maxminGoldSec(s, t, 1, 0);
	}

	Point2d getDerivAtPoint(double t) {
		return spherCoorAt(t) - spherCoorAt(t+0.01);
	}

	double distStartEnd() {
		Point2d diff = this->end() - this->start();
		return sqrt(diff.x*diff.x + diff.y*diff.y);
	}

	vector<Point2d> getControl() {
		vector<Point2d> cont = { P[1], P[2] };
		return cont;
	}

	Point2d spherCoorAt(double t) {
		double s = 1 - t;
		return s*s*s*P[0] + 3 * s*s*t*P[1] + 3 * s*t*t*P[2] + t*t*t*P[3];
	}

	double xCoorAt(double t) {
		double s = 1 - t;
		return s*s*s*P[0].x + 3 * s*s*t*P[1].x + 3 * s*t*t*P[2].x + t*t*t*P[3].x;
	}

	double yCoorAt(double t) {
		double s = 1 - t;
		return s*s*s*P[0].y + 3 * s*s*t*P[1].y + 3 * s*t*t*P[2].y + t*t*t*P[3].y;
	}

	Point2d cartCoorAt(int xdim, int ydim, double t) {
		Point2d thphi = spherCoorAt(t);
		double posx = 1. * xdim * thphi_phi / (PI2);
		double posy = 1. * ydim * thphi_theta / PI;
		return Point2d(posx, posy);
	}

	vector<Point2d> drawCart(int detail){
		vector<Point2d> curvePoints;
		for (int i = 0; i < detail + 1; i++) {
			curvePoints.push_back(spherCoorAt(1.*i / detail));
		}
		return curvePoints;
	}

	vector<Point2d> draw(int xdim, int ydim, int detail){
		vector<Point2d> curvePoints;
		for (int i = 0; i <= detail; i++) {
			curvePoints.push_back(cartCoorAt(xdim, ydim, 1.*i/detail));
		}
		return curvePoints;
	}

	void img(Mat& img, Vec3b color) {
		vector<Point2d> curvePoints = draw(img.cols, img.rows, 32);
		for (int q = 0; q < curvePoints.size() - 1; q++){
			curvePoints[q].x = fmod(curvePoints[q].x, 1.*img.cols);
			curvePoints[q + 1].x = fmod(curvePoints[q + 1].x, 1.*img.cols);
			if (fabs(curvePoints[q].x - curvePoints[q + 1].x) < (img.cols / 8))
				line(img, curvePoints[q], curvePoints[q + 1], color, 1, CV_AA);
		}
	};

	void split(double t, Bezier& spl1, Bezier& spl2) {
		//spl1.P.resize(4);
		//spl2.P.resize(4);
		vector<Point2d> newP;
		vector<Point2d> oldP = P;
		for (int i = 0; i < 3; i++) {
			newP.resize(3 - i);
			for (int j = 0; j < 3 - i; j++) {
				if (j == 0) spl1.P.push_back(oldP[j]);
				if (j == 2 - i) spl2.P.push_back(oldP[j+1]);
				newP[j] = (1. - t) * oldP[j] + t * oldP[j+1];
			}
			oldP = newP;
		}

		spl1.P.push_back(oldP[0]);
		spl2.P.push_back(oldP[0]);
	}

	vector<double> extreme(vector<double>& Pxy) {
		vector<double> ext;

		double a = 3.*(-Pxy[0] + 3.*Pxy[1] - 3.*Pxy[2] + Pxy[3]);
		double b = 6.*(Pxy[0] - 2.*Pxy[1] + Pxy[2]);
		double c = 3.*(Pxy[1] - Pxy[0]);
		double aa = 2.*(b - a);
		double bb = 2.*(c - b);
		ext.push_back(-bb / aa);

		double squa = b*b - 4.*a*c;
		if (squa < 0) return ext;
		else {
			ext.push_back((-b + sqrt(squa)) / (2.*a));
			ext.push_back((-b - sqrt(squa)) / (2.*a));
		}
		return ext;
	}

	void drawBoundBox(Mat& img, double phmax, double phmin, double thmax, double thmin) {
		double sx = 1.*img.cols / PI2;
		double sy = 1.*img.rows / PI;
		line(img, Point2d(phmin*sx, thmin*sy), Point2d(phmax*sx, thmin*sy), Scalar(0), 1, CV_AA);
		line(img, Point2d(phmin*sx, thmin*sy), Point2d(phmin*sx, thmax*sy), Scalar(0), 1, CV_AA);
		line(img, Point2d(phmax*sx, thmin*sy), Point2d(phmax*sx, thmax*sy), Scalar(0), 1, CV_AA);
		line(img, Point2d(phmin*sx, thmax*sy), Point2d(phmax*sx, thmax*sy), Scalar(0), 1, CV_AA);
	}

	void makeBoundBox(double& phmax, double& phmin, double& thmax, double& thmin) {
		vector<double> extremeph = extreme(vector < double > {P[0]_phi, P[1]_phi, P[2]_phi, P[3]_phi});
		vector<double> extremeth = extreme(vector < double > {P[0]_theta, P[1]_theta, P[2]_theta, P[3]_theta});
		
		extremeph.push_back(0.);
		extremeph.push_back(1.);
		extremeth.push_back(0.);
		extremeth.push_back(1.);
		phmin = thmin = 100.;
		phmax = thmax = -1.;
		for (int i = 0; i < extremeph.size(); i++) {
			double t = extremeph[i];
			if (t < 0. || t>1.) continue;
			double phatt = yCoorAt(t);
			if (phatt < phmin) phmin = phatt;
			if (phatt > phmax) phmax = phatt;
		}
		for (int i = 0; i < extremeth.size(); i++) {
			double t = extremeth[i];
			if (t < 0. || t>1.) continue;
			double thatt = xCoorAt(t);
			if (thatt < thmin) thmin = thatt;
			if (thatt > thmax) thmax = thatt;
		}
	}

	~Bezier(){};
};