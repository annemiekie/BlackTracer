#pragma once
#include "Metric.h"


class Camera
{
public:
	double theta, phi, r; 
	double speed;

	double alpha, w, wbar, Delta, ro;

	Camera(){};

	Camera(Camera &c){
		theta = c.theta;
		phi = c.phi;
		r = c.r;
		speed = c.speed;
		alpha = c.alpha;
		w = c.w;
		wbar = c.wbar;
		Delta = c.Delta;
		ro = c.ro;
	};

	Camera(double theCam, double phiCam, double radfactor, double speedCam)
	{
		theta = theCam;
		phi = phiCam;
		r = radfactor;
		speed = speedCam;
		initforms();
	};

	Camera(double theCam, double phiCam, double radfactor)
	{
		theta = theCam;
		phi = phiCam;
		r = radfactor;

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



	~Camera(){};
};


