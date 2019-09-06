#pragma once

#include <stdint.h> 
#include <vector>
#include "Const.h"
#include "cuda_runtime.h"
using namespace std;


class Viewer
{
private:
public:
	vector<float2> viewMatrix;
	double viewAngleUp, viewAngleWide;
	int pixelwidth, pixelheight;

	Viewer(){};

	Viewer(double viewangle, double offUp, double offRight, int pixw, int pixh, bool sphere) {
		pixelwidth = pixw;
		pixelheight = pixh;

		//+1 to complete the last pixel, because every pixel has 4 corners
		viewMatrix = vector<float2>((pixelheight + 1) * (pixelwidth + 1));

		if (sphere) makeSphereView();
		else {
			viewAngleWide = viewangle;
			viewAngleUp = 1.*pixh * viewAngleWide / (1.*pixw);
			makeCameraView(offUp, offRight);
		}
	};

	//vector<float2> makeEquaView() {
	//	int H = pixelheight;
	//	int H1 = H + 1;
	//	vector<float2> viewequa(H1*H1);
	//	for (int i = 0; i < H1; i++) {
	//		for (int j = 0; j < H1; j++) {
	//			float xval = 1.f*i / (1.f*H)*PI - PI1_2;
	//			float yval = 1.f*j / (1.f*H)*PI - PI1_2;
	//			float ro = sqrtf(xval*xval + yval*yval);
	//			float2 answer;
	//			if (ro > PI1_2) answer = { -1, -1 };
	//			else {
	//				float plus = j > H / 2 ? 0 : PI;
	//				ro = sqrtf(xval*xval + yval*yval);
	//				answer = { ro, atanf(-xval / yval) + plus };
	//				metric::wrapToPi(answer.x, answer.y);
	//			}
	//			viewMatrix[i*H1 + j] = answer;
	//		}
	//	}
	//}

	//void makeHalfEquaView() {
	//	int H = pixelheight;
	//	int H1 = H + 1;
	//	vector<float2> viewequa(H1*H1);
	//	for (int i = 0; i < H1; i++) {
	//		for (int j = 0; j < H1; j++) {
	//			float xval = -2.f*i / (1.f*H) + 1.f;
	//			float yval = -2.f*j / (1.f*H) + 1.f;
	//			float ro = sqrtf(xval*xval + yval*yval);
	//			float2 answer;
	//			if (ro > 1) answer = { -1, -1 };
	//			else {
	//				float plus = j > H / 2 ? 0 : PI;
	//				ro = sqrtf(xval*xval + yval*yval);
	//				float cosc = cosf(asinf(ro));
	//				answer = { fabs(asinf(xval) - PI1_2), atanf(yval / cosc) + PI };
	//				metric::wrapToPi(answer.x, answer.y);
	//			}
	//			viewMatrix[i*H1 + j] = answer;
	//		}
	//	}

	//	//return viewequa;
	//}

	void makeSphereView() {
		for (int i = 0; i < pixelheight + 1; i++) {
			for (int j = 0; j < pixelwidth + 1; j++) {
				viewMatrix[i*(pixelwidth + 1) + j] = { 1.f*i* PI / pixelheight, 1.f*j * PI2 / pixelwidth };
			}
		}
	}

	//void turnVer(double offsetVer) {
	//	if (fabs(offsetVer) >(PI - viewAngle)*HALF) {
	//		cout << "Error, offsetUp too big" << endl;
	//		return;
	//	}
	//	for (int i = 0; i<ver.size(); i++) {
	//		ver[i] -= offsetVer;
	//	}
	//}

	//void turnHor(double offsetHor) {
	//	for (int i = 0; i<hor.size(); i++) {
	//		double phi = hor[i];
	//		phi = fmod(phi - offsetHor, PI2);
	//		while (phi < 0) phi += PI2;
	//		hor[i] = phi;
	//	}
	//}

	void makeCameraView(double offsetVer, double offsetHor) {
		if (fabs(offsetVer) > (PI-viewAngleUp)/2.) {
			cout << "Error, offsetUp too big" << endl;
			return;
		}

		float xdist = 1.f;
		float yleft = tan(viewAngleWide / 2.f);
		float step = (2.f * yleft) / pixelwidth;
		float zup = yleft*pixelheight / pixelwidth;

		for (int i = 0; i < pixelheight+1; i++) {
			float zval = zup - i*step;
			float halfH = i*1.f / (pixelheight / 2.f);
			float theta = halfH == 1.f ? PI1_2 - offsetVer : atan(1. / zval) + (halfH>1.f)*PI - offsetVer;
			for (int j = 0; j < pixelwidth + 1; j++) {

				float yval = yleft - j*step;
				float halfV = j*1.f / (pixelwidth / 2.f);
				float phi = halfV == 1.f ? PI - offsetHor : atan(1. / yval) + (halfV>1.f)*PI + PI1_2 - offsetHor;

				viewMatrix[(pixelwidth + 1) * i + pixelwidth - j] = { theta, phi };
			}
		}
	};
	
	~Viewer(){};
};