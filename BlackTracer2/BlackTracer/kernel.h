#include "Camera.h"
#include "Metric.h"
#include <iostream>
#include <stdint.h>
#include <iomanip>

void integration_wrapper(double *theta, double *phi, int n, const Camera* cam, const double afactor)
{
	double thetaS = cam->theta;
	double phiS = cam->phi;
	double rS = cam->r;
	double sp = cam->speed;

	//printf("metric a {%f} \n", *metric::a);
	//printf("metric asq {%f} \n", *metric::asq);

#pragma loop(hint_parallel(8))
#pragma loop(ivdep)
	for (int i = 0; i<n; i ++) {

		//	printf("computing {%i} \n", i);

		double xCam = sin(theta[i])*cos(phi[i]);
		double yCam = sin(theta[i])*sin(phi[i]);
		double zCam = cos(theta[i]);

		double yFido = (-yCam + sp) / (1 - sp*yCam);
		double xFido = -sqrtf(1 - sp*sp) * xCam / (1 - sp*yCam);
		double zFido = -sqrtf(1 - sp*sp) * zCam / (1 - sp*yCam);

		double rFido = xFido;
		double thetaFido = -zFido;
		double phiFido = yFido;

		double eF = 1. / (cam->alpha + cam->w * cam->wbar * phiFido);

		double pR = eF * cam->ro * rFido / sqrtf(cam->Delta);
		double pTheta = eF * cam->ro * thetaFido;
		double pPhi = eF * cam->wbar * phiFido;

		double b = pPhi;
		double q = pTheta*pTheta + cos(thetaS)*cos(thetaS)*(b*b / (sin(thetaS)*sin(thetaS)) - *metric::asq);
		//	printf("{%i,%lf,%lf,%lf,%lf} \n", i, xCam, xFido, b, q);
		theta[i] = -1;
		phi[i] = -1;

		if (metric::checkCelest(pR, rS, thetaS, b, q)) {
			//printf("%i passed check \n",i);
			metric::rkckIntegrate(rS, thetaS, phiS, pR, b, q, pTheta, theta[i], phi[i]);
		}
	}
}