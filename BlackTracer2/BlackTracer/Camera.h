#pragma once
#include "Metric.h"


class Camera
{
public:
	double theta, phi, r; 
	double speed, br, btheta, bphi;

	double alpha, w, wbar, Delta, ro;

	Camera(){};

	//Camera(Camera &c){
	//	theta = c.theta;
	//	phi = c.phi;
	//	r = c.r;
	//	speed = c.speed;
	//	alpha = c.alpha;
	//	w = c.w;
	//	wbar = c.wbar;
	//	Delta = c.Delta;
	//	ro = c.ro;
	//};

	Camera(double theCam, double phiCam, double radfactor, double speedCam)
	{
		theta = theCam;
		phi = phiCam;
		r = radfactor;
		speed = speedCam;

		bphi = 1;
		btheta = 0;
		br = 0;
		initforms();
	};

	Camera(double theCam, double phiCam, double radfactor, double _br, double _btheta, double _bphi)
	{
		theta = theCam;
		phi = phiCam;
		r = radfactor;

		bphi = _bphi;
		btheta = _btheta;
		br = _br;

		speed = metric::calcSpeed(r, theta);
		initforms();
	};

	void initforms() {
		alpha = metric::_alpha(this->r, this->theta);
		w = metric::_w(this->r, this->theta);
		wbar = metric::_wbar(this->r, this->theta);
		Delta = metric::_Delta(this->r);
		ro = metric::_ro(this->r, this->theta);

	};

	double getDistFromBH(float mass) {
		return mass*r;
	};

	vector<float> getParamArray() {
		vector<float> camera(7);
		camera[0] = speed;
		camera[1] = alpha;
		camera[2] = w;
		camera[3] = wbar;
		camera[4] = br;
		camera[5] = btheta;
		camera[6] = bphi;

		return camera;
	};

	~Camera(){};
};


