#pragma once

#include <stdint.h> 
#include <vector>
#include "Const.h"

using namespace std;


class Viewer
{
private:
public:
	vector<double> ver, hor;
	double viewAngle;
	int pixelwidth, pixelheight;

	Viewer(){};

	Viewer(double viewangle, int pixw, int pixh, bool sphere) {
		viewAngle = viewangle;
		pixelwidth = pixw;
		pixelheight = pixh;

		ver = vector<double>(pixelheight + 1);
		hor = vector<double>(pixelwidth + 1);

		if (sphere) makeSphereView();
		else makeCameraView(0,0);
	};

	Viewer(double viewangle, double offUp, double offRight, int pixw, int pixh, bool sphere) {
		viewAngle = viewangle;
		pixelwidth = pixw;
		pixelheight = pixh;

		//+1 to complete the last pixel, because every pixel has 4 corners
		ver = vector<double>(pixelheight + 1);
		hor = vector<double>(pixelwidth + 1);

		if (sphere) makeSphereView();
		else makeCameraView(offUp, offRight);
	};

	void makeSphereView() {
		for (int i = 0; i < ver.size(); i++) {
			ver[i] = 1.*i* PI / pixelheight;
		}
		for (int j = 0; j < hor.size(); j++) {
			hor[j] = 1.*j * PI2 / pixelwidth;
		}
	}

	void turnVer(float offsetVer) {
		if (fabs(offsetVer) >(PI - viewAngle)*HALF) {
			cout << "Error, offsetUp too big" << endl;
			return;
		}
		for (int i = 0; i<ver.size(); i++) {
			ver[i] -= offsetVer;
		}
	}

	void turnHor(float offsetHor) {
		for (int i = 0; i<hor.size(); i++) {
			double phi = hor[i];
			phi = fmod(phi - offsetHor, PI2);
			while (phi < 0) phi += PI2;
			hor[i] = phi;
		}
	}

	void makeCameraView(double offsetVer, double offsetHor) {
		if (fabs(offsetVer) >(PI - viewAngle)/2.) {
			cout << "Error, offsetUp too big" << endl;
			return;
		}

		double xdist = 1.f;
		double yleft = tan(viewAngle / 2.);
		double step = (2. * yleft) / pixelwidth;
		double zup = yleft*pixelheight / pixelwidth;

		for (int i = 0; i < pixelheight / 2; i++) {
			double zval = zup - i*step;
			double theta = atan(1. / zval) - offsetVer;

			ver[i] = theta;
		}
		ver[pixelheight / 2] = PI1_2 - offsetVer;
		for (int i = pixelheight/2+1; i < ver.size(); i++) {
			double zval = zup - i*step;
			double theta = atan(1. / zval) + PI - offsetVer;
			ver[i] = theta;
		}

		for (int j = 0; j < pixelwidth / 2; j++) {
			double yval = yleft - j*step;
			double phi = atan(1. / yval) + PI1_2 - offsetHor;
			hor[hor.size() - j - 1] = phi;
		}
		hor[pixelwidth / 2] = PI - offsetHor;
		for (int j = pixelwidth / 2 + 1; j < hor.size(); j++) {
			double yval = yleft - j*step;
			double phi = atan(1. / yval) + PI + PI1_2 - offsetHor;
			hor[hor.size() - j - 1] = phi;
		}
	};
	
	~Viewer(){};
};