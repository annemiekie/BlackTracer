#pragma once

#include "opencv2/imgcodecs/imgcodecs.hpp"
#include <iostream>
#include <stdint.h> 
#include <vector>
#include "Const.h"
#include "Bezier.h"
#include "Code.h"

using namespace std;
using namespace cv;

class SplineBlock
{
private:
public:
	Bezier up;
	Bezier down;
	Bezier left;
	Bezier right;
	int complete = 0;

	SplineBlock(){};

	SplineBlock(Bezier _up) {
		up = _up;
		complete++;
	}


	void setUp(Bezier _up) {
		up = _up;
		complete++;
	}


	void setDown(Bezier _down) {
		down = _down;
		complete++;
	}

	void setLeft(Bezier _left) {
		left = _left;
		complete++;
	}

	void setRight(Bezier _right) {
		right = _right;
		complete++;
	}

	SplineBlock(Bezier _up, Bezier _down, Bezier _left, Bezier _right) {
		up = _up;
		down = _down;
		left = _left;
		right = _right;
		complete = 4;
	};

	bool interpolate(double thPerc, double phPerc, Point2d& cross) {
		Bezier upDown = interpolate(thPerc, up, down, left, right);
		Bezier leftRight = interpolate(phPerc, left, right, up, down);
		return intersect(upDown, leftRight, cross,0);
	};

	bool intersect(Bezier& spl1, Bezier& spl2, Point2d& cross, int counter) {
		if (counter == 200) return false;
		double phmax1, phmin1, phmax2, phmin2, thmax1, thmax2, thmin2, thmin1;
		spl1.makeBoundBox(phmax1, phmin1, thmax1, thmin1);
		spl2.makeBoundBox(phmax2, phmin2, thmax2, thmin2);

		//cout << phmax1 << " " << phmin1 << " " << phmax2 << " " << phmin2 << " " <<
		//	thmax1 << " " << thmax2 << " " << thmin2 << " " << thmin1 << endl;
		//Mat img(1080, 1920, CV_8UC3, Vec3b(255,255,255));
		//spl1.drawBoundBox(img, phmax1, phmin1, thmax1, thmin1);
		//spl2.drawBoundBox(img, phmax2, phmin2, thmax2, thmin2);
		//spl1.img(img, Vec3b(255, 0, 0));
		//spl2.img(img, Vec3b(0, 255, 0));

		//namedWindow("boundbox", WINDOW_NORMAL);
		//imshow("boundbox", img);
		//waitKey(0);

		if (fabs(phmin1 - phmin2) < 0.00001 && fabs(thmin1 - thmin2) < 0.00001) {
			cross = Point2d(thmin1, phmin1);
			return true;
		}
		else if (phmax1 < phmin2) return false;
		else if (phmin1 > phmax2) return false; // a is right of b
		else if (thmax1 < thmin2) return false; // a is above b
		else if (thmin1 > thmax2) return false; // a is below b
		else {
			Bezier spl11, spl12, spl22, spl21;
			spl1.split(0.5, spl11, spl12);
			spl2.split(0.5, spl21, spl22);
			return (intersect(spl11, spl21, cross, counter + 1) || intersect(spl11, spl22, cross, counter + 1) ||
				intersect(spl12, spl21, cross, counter + 1) || intersect(spl12, spl22, cross, counter + 1));
		}
	};

	void meanDeriv(Bezier line1, Bezier line2, Point2d &startDir, Point2d &endDir, double perc) {
		Point2d s1, s2, e1, e2;
		line1.getDeriv(s1, e1);
		line2.getDeriv(s2, e2);
		startDir = perc*s2 + (1.-perc)*s1;
		endDir = perc*e2 + (1. - perc)*e1;
	};

	Bezier interpolate(double perc, Bezier b1, Bezier b2, Bezier l1, Bezier l2) {
		Point2d startDir, endDir;
		double scale = 25.;
		meanDeriv(b1, b2, startDir, endDir, perc);
		Point2d start = l1.spherCoorAt(perc);
		Point2d end = l2.spherCoorAt(perc);

		Point2d contr1 = start + scale*startDir;
		Point2d contr2 = end + scale*endDir;

		return Bezier(start, end, contr1, contr2);
	};



	void draw(Vec3b color, Mat& img) {
		up.img(img, color);
		down.img(img, color);
		left.img(img, color);
		right.img(img, color);
	};

	~SplineBlock(){};
};