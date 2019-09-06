#pragma once

#include <iostream>
#include <string>
#include "Const.h"
#include "Code.h"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace std;

#define rVar var[0]
#define thetaVar var[1]
#define phiVar var[2]
#define pRVar var[3]
#define pThetaVar var[4]

#define drdz varOut[0]
#define dtdz varOut[1]
#define dpdz varOut[2]
#define dprdz varOut[3]
#define dptdz varOut[4]

#define A *a
#define Asq *asq

#define SAFETY 0.9
#define PGROW -0.2
#define PSHRNK -0.25
#define ERRCON 1.89e-4
#define MAXSTP 1000
#define TINY 1.0e-30
#define ADAPTIVE 5.0

namespace metric {
	double starta = 0.0;
	double startasq = 0.0;
	double* a = &starta;
	double* asq = &startasq;

	const double b21 = 0.2,
		b31 = 3.0 / 40.0, b32 = 9.0 / 40.0, b41 = 0.3, b42 = -0.9, b43 = 1.2,
		b51 = -11.0 / 54.0, b52 = 2.5, b53 = -70.0 / 27.0, b54 = 35.0 / 27.0,
		b61 = 1631.0 / 55296.0, b62 = 175.0 / 512.0, b63 = 575.0 / 13824.0,
		b64 = 44275.0 / 110592.0, b65 = 253.0 / 4096.0, c1 = 37.0 / 378.0,
		c3 = 250.0 / 621.0, c4 = 125.0 / 594.0, c6 = 512.0 / 1771.0,
		dc5 = -277.00 / 14336.0;
	const double dc1 = 37.0 / 378.0 - 2825.0 / 27648.0, dc3 = 250.0 / 621.0 - 18575.0 / 48384.0,
		dc4 = 125.0 / 594.0 - 13525.0 / 55296.0, dc6 = 512.0 / 1771.0 - 0.25;

	static void setAngVel(double afactor) {
		A = afactor;
		Asq = afactor*afactor;
	}

	static double sq(double x) {
		return x*x;
	}

	static double sq3(double x) {
		return x*x*x;
	}

	inline double _Delta(double r) {
		return sq(r) - 2. * r + Asq;
	};

	inline double _Sigma(double r, double theta) {
		return sqrt(sq(sq(r) + Asq) - Asq*_Delta(r) * sq(sin(theta)));
	};

	inline double _w(double r, double theta) {
		return 2. * A * r / sq(_Sigma(r, theta));
	};

	inline double _ro(double r, double theta) {
		return sqrt(sq(r) + Asq*sq(cos(theta)));
	};

	static double _rosq(double r, double theta) {
		return sq(r) + Asq*sq(cos(theta));
	};

	inline double _wbar(double r, double theta) {
		return _Sigma(r, theta)*sin(theta) / _ro(r, theta);
	};

	inline double _alpha(double r, double theta) {
		return _ro(r, theta) * sqrt(_Delta(r)) / _Sigma(r, theta);
	};

	static double _P(double r, double b) {
		return sq(r) + Asq - A*b;
	}

	static double _R(double r, double theta, double b, double q) {
		return sq(_P(r, b)) - _Delta(r)*(sq((b - A)) + q);
	};

	static double _BigTheta(double r, double theta, double b, double q) {
		return q - sq(cos(theta))*(sq(b) / sq(sin(theta)) - Asq);
	};

	inline double calcSpeed(double r, double theta) {
		double omega = 1. / (A + pow(r, 1.5));
		double sp= _wbar(r, theta) / _alpha(r, theta) * (omega - _w(r, theta));
		return sp;
	}

	static double findMinGoldSec(double theta, double bval, double qval, double ax, double b, double tol) {
		double gr = (sqrt(5.0) + 1.) / 2.;
		double c = b - (b - ax) / gr;
		double d = ax + (b - ax) / gr;
		while (fabs(c - d) > tol) {
			if (_R(c, theta, bval, qval) < _R(d, theta, bval, qval)) {
				b = d;
			}
			else {
				ax = c;
			}
			c = b - (b - ax) / gr;
			d = ax + (b - ax) / gr;
		}
		return (ax + b) / 2;
	}

	static bool checkRup(double rV, double thetaV, double bV, double qV) {
		if (A == 0) return false;
		double min = findMinGoldSec(thetaV, bV, qV, rV, 4 * rV, 0.00001);
		return (_R(min, thetaV, bV, qV) >= 0);
	}

	static double _b0(double r0) {
		return -(sq3(r0) - 3. * sq(r0) + Asq*r0 + Asq) / (A*(r0 - 1.));
	};

	static double _b0diff(double r0) {
		return (Asq + A - 2. * r0*(sq(r0) - 3. * r0 + 3.)) / (A*sq(r0 - 1.));
	};

	static double _q0(double r0) {
		return -sq3(r0)*(sq3(r0) - 6. * sq(r0) + 9. * r0 - 4. * Asq) / (Asq*sq(r0 - 1.));
	};

	static bool checkB_Q(double bV, double qV) {
		double _r1 = 2. * (1. + cos(2. * acos(-A) / 3.));
		double _r2 = 2. * (1. + cos(2. * acos(A) / 3.));
		double error = 0.0000001;
		double r0V = 2.0;
		double bcheck = 100;

		while (fabs(bV - bcheck) > error) {
			bcheck = _b0(r0V);
			double bdiffcheck = _b0diff(r0V);
			double rnew = r0V - (bcheck - bV) / bdiffcheck;
			if (rnew < 1) {
				r0V = 1.0001;
			}
			else {
				r0V = rnew;
			}
		}
		double qb = _q0(r0V);
		double _b1 = _b0(_r2);
		double _b2 = _b0(_r1);
		return ((_b1 >= bV) || (_b2 <= bV) || (qV >= qb));
	}

	inline bool checkCelest(double pRV, double rV, double thetaV, double bV, double qV) {
		bool check1 = checkB_Q(bV, qV);
		bool check2 = !check1 && (pRV < 0);
		bool check3 = check1 && checkRup(rV, thetaV, bV, qV);
		return check2 || check3;
	}

	static void wrapToPi(double& thetaW, double& phiW) {
		thetaW = fmod(thetaW, PI2);
		while (thetaW < 0) thetaW += PI2;
		if (thetaW > PI) {
			thetaW -= 2 * (thetaW - PI);
			phiW += PI;
		}
		while (phiW < 0) phiW += PI2;
		phiW = fmod(phiW, PI2);
	}

	static void wrapToPi(float& thetaW, float& phiW) {
		thetaW = fmod(thetaW, PI2);
		while (thetaW < 0) thetaW += PI2;
		if (thetaW > PI) {
			thetaW -= 2 * (thetaW - PI);
			phiW += PI;
		}
		while (phiW < 0) phiW += PI2;
		phiW = fmod(phiW, PI2);
	}

	static double newPhi(double r, double theta, double b, double q) {
		double cossq = sq(cos(theta));
		double sinsq = sq(sin(theta));
		double delta = _Delta(r);
		double rosq = _rosq(r, theta);
		double P = _P(r, b);
		return (2 * A*P - (2 * A - 2 * b)*delta + (2 * b*cossq * delta) / sinsq) / (rosq * 2 * delta);
	};

	static double newR(double r, double theta, double pr) {
		return _Delta(r) / _rosq(r, theta) * pr;
	};

	static double newTheta(double r, double theta, double ptheta) {
		return 1.0f / _rosq(r, theta) * ptheta;
	};

	static double newPr(double r, double theta, double b, double q, double pr, double ptheta) {
		double prsq = sq(pr);
		double pthetasq = sq(ptheta);
		double delta = _Delta(r);
		double R = _R(r, theta, b, q);
		double partR = (q + sq(A - b));
		double btheta = _BigTheta(r, theta, b, q);
		double rosq = _rosq(r, theta);
		double P = _P(r, b);
		double rosqsq = sq(2 * rosq);

		return ((2 * r - 2)*btheta - (2 * r - 2)*partR + 4 * r*P) / (rosq*(2 * delta)) -
			(prsq*(2 * r - 2)) / (2 * rosq) + (4 * pthetasq*r) /
			rosqsq - ((4 * r - 4)*(btheta*(delta)+R)) /
			(rosq*sq(2 * delta)) + (4 * prsq*r*(delta)) / rosqsq -
			(r*(btheta*delta + R)) / (sq(rosq) * delta);
	};

	static double newPtheta(double r, double theta, double b, double q, double pr, double ptheta) {
		double cosv = cos(theta);
		double sinv = sin(theta);
		double sinsq = sq(sinv);
		double prsq = sq(pr);
		double pthetasq = sq(ptheta);
		double bsq = sq(b);
		double delta = _Delta(r);
		double R = _R(r, theta, b, q);
		double btheta = _BigTheta(r, theta, b, q);
		double rosq = _rosq(r, theta);
		double rosqsq = sq(2 * rosq);

		return ((2 * cosv*sinv*(bsq / sinsq - Asq) + (2 * bsq*sq3(cosv)) /
			sq3(sinv))*delta) / (rosq * 2 * delta) - (4 * Asq*pthetasq*cosv*sinv) /
			rosqsq - (4 * Asq*prsq*cosv*sinv*delta) / rosqsq +
			(Asq*cosv*sinv*(btheta*delta + R)) / (sq(rosq) * delta);
	};

	static void derivs(double* var, double* varOut, double b, double q) {
		double cosv = cos(thetaVar);
		double sinv = sin(thetaVar);
		double cossq = sq(cosv);
		double sinsq = sq(sinv);
		double bsq = sq(b);
		double delta = _Delta(rVar);
		double rosq = _rosq(rVar, thetaVar);
		double P = _P(rVar, b);
		double prsq = sq(pRVar);
		double pthetasq = sq(pThetaVar);
		double R = _R(rVar, thetaVar, b, q);
		double partR = (q + sq(A - b));
		double btheta = _BigTheta(rVar, thetaVar, b, q);
		double rosqsq = sq(2 * rosq);
		double sqrosqdel = (sq(rosq) * delta);
		double asqcossin = Asq*cosv*sinv;
		double rtwo = 2 * rVar - 2;

		drdz = delta / rosq * pRVar;
		dtdz = 1.0f / rosq * pThetaVar;
		dpdz = (2 * A*P - (2 * A - 2 * b)*delta + (2 * b*cossq * delta) / sinsq) / (rosq * 2 * delta);
		dprdz = (rtwo*btheta - rtwo*partR + 4 * rVar*P) / (rosq*(2 * delta)) - (prsq*rtwo) / (2 * rosq)
			+ (4 * pthetasq*rVar) / rosqsq - ((4 * rVar - 4)*(btheta*(delta)+R)) / (rosq*sq(2 * delta))
			+ (4 * prsq*rVar*(delta)) / rosqsq - (rVar*(btheta*delta + R)) / sqrosqdel;
		dptdz = ((2 * cosv*sinv*(bsq / sinsq - Asq) + (2 * bsq*sq3(cosv)) / sq3(sinv))*delta) /
			(rosq * 2 * delta) - (4 * asqcossin*pthetasq) / rosqsq - (4 * asqcossin*prsq*delta) /
			rosqsq + (asqcossin*(btheta*delta + R)) / sqrosqdel;

	}

	static void rkck(const double* var, const double* dvdz, int n, double h,
		double* varOut, double* varErr, double b, double q) {

		int i;
		double* ak2 = new double[n];
		double* ak3 = new double[n];
		double* ak4 = new double[n];
		double* ak5 = new double[n];
		double* ak6 = new double[n];
		double* varTemp = new double[n];

		for (i = 0; i<n; i++)
			varTemp[i] = var[i] + b21*h*dvdz[i];
		derivs(varTemp, ak2, b, q);
		for (i = 0; i<n; i++)
			varTemp[i] = var[i] + h*(b31*dvdz[i] + b32*ak2[i]);
		derivs(varTemp, ak3, b, q);
		for (i = 0; i<n; i++)
			varTemp[i] = var[i] + h*(b41*dvdz[i] + b42*ak2[i] + b43*ak3[i]);
		derivs(varTemp, ak4, b, q);
		for (i = 0; i<n; i++)
			varTemp[i] = var[i] + h*(b51*dvdz[i] + b52*ak2[i] + b53*ak3[i] + b54*ak4[i]);
		derivs(varTemp, ak5, b, q);
		for (i = 0; i<n; i++)
			varTemp[i] = var[i] + h*(b61*dvdz[i] + b62*ak2[i] + b63*ak3[i] + b64*ak4[i] + b65*ak5[i]);
		derivs(varTemp, ak6, b, q);
		for (i = 0; i<n; i++)
			varOut[i] = var[i] + h*(c1*dvdz[i] + c3*ak3[i] + c4*ak4[i] + c6*ak6[i]);
		for (i = 0; i < n; i++)
			varErr[i] = h*(dc1*dvdz[i] + dc3*ak3[i] + dc4*ak4[i] + dc5*ak5[i] + dc6*ak6[i]);

		delete[] ak2;
		delete[] ak3;
		delete[] ak4;
		delete[] ak5;
		delete[] ak6;
		delete[] varTemp;
	}

	static void rkqs(double* var, double* dvdz, int n, double& z, double htry,
		double& hdid, double& hnext, double eps, double* varScal, double b, double q) {
		int i;
		double errmax, h, htemp;
		double* varErr = new double[n];
		double* varTemp = new double[n];
		h = htry;

		for (;;) {
			rkck(var, dvdz, n, h, varTemp, varErr, b, q);
			errmax = 0.0;
			for (i = 0; i<n; i++) errmax = fmax(errmax, abs(varErr[i] / varScal[i]));
			errmax /= eps;
			if (errmax <= 1.0) break;
			htemp = SAFETY*h*pow(errmax, PSHRNK);
			if (h >= 0.0) h = fmax(htemp, 0.1*h);
			else h = fmin(htemp, 0.1*h);
		}
		if (errmax > ERRCON) hnext = SAFETY*h*pow(errmax, PGROW);
		else hnext = ADAPTIVE*h;
		hdid = h;
		z += h;
		for (i = 0; i<n; i++) var[i] = varTemp[i];

		delete[] varErr;
		delete[] varTemp;
	}

	static int sgn(float val) {
		return (0 < val) - (val < 0);
	}

	static void odeint1(double* varStart, int nvar, double zEnd, double eps,
		double h1, double hmin, int& nok, int& nbad, double b, double q) {
		double hnext, hdid;

		double* varScal = new double[nvar];
		double* var = new double[nvar];
		double* dvdz = new double[nvar];

		double z = 0.0;
		double h = h1*sgn(zEnd);
		nok = nbad = 0;

		for (int i = 0; i<nvar; i++) var[i] = varStart[i];

		for (int nstp = 0; nstp<MAXSTP; nstp++) {
			derivs(var, dvdz, b, q);
			for (int i = 0; i<nvar; i++)
				varScal[i] = fabs(var[i]) + fabs(dvdz[i] * h) + TINY;

			rkqs(var, dvdz, nvar, z, h, hdid, hnext, eps, varScal, b, q);

			if (hdid == h) ++nok; else ++nbad;
			if (z <= zEnd) {
				for (int i = 0; i < nvar; i++) varStart[i] = var[i];
				delete[] varScal;
				delete[] var;
				delete[] dvdz;
				return;
			}
			h = hnext;
		}
		delete[] varScal;
		delete[] var;
		delete[] dvdz;
	};

	static void odeint2(double* varStart, int nvar, double zEnd, double eps,
		double h1, double hmin, int& nok, int& nbad, double b, double q, double& hitr, double& hitphi, bool bh) {
		double hnext, hdid;

		double* varScal = new double[nvar];
		double* var = new double[nvar];
		double* dvdz = new double[nvar];

		double z = 0.0;
		double h = h1*sgn(zEnd);
		nok = nbad = 0;

		for (int i = 0; i<nvar; i++) var[i] = varStart[i];

		double thetaprev = var[1];
		double rdiskmin = 9.;
		double rdiskmax = 18.;
		double rprev = var[0];
		double phiprev = var[2];
		bool hit = false;

		for (int nstp = 0; nstp<MAXSTP; nstp++) {
			derivs(var, dvdz, b, q);
			for (int i = 0; i<nvar; i++)
				varScal[i] = fabs(var[i]) + fabs(dvdz[i] * h) + TINY;

			rkqs(var, dvdz, nvar, z, h, hdid, hnext, eps, varScal, b, q);

			if (!hit) {
				double thetanew = var[1];
				double phinew = var[2];
				wrapToPi(thetanew, phinew);
				double rnew = var[0];
				double diffthetanew = thetanew - PI1_2;
				double diffthetaprev = thetaprev - PI1_2;
				if ((diffthetanew * diffthetaprev) < 0. || thetanew == PI1_2) {
					double frac = fabs(diffthetanew / (thetanew-thetaprev));
					double rave = rnew*(1.-frac) + rprev*frac;
					if (rave > rdiskmin && rave < rdiskmax) {
						hit = true;
						hitr = rave;
						vector<double> phicheck = { phinew, phiprev };
						if (phinew > (1 - 1. / 5.)*PI2 || phiprev >(1 - 1. / 5.)*PI2) {
							if (phinew < 1. / 5.*PI2) phinew += PI2;
							if (phiprev < 1. / 5.*PI2) phiprev += PI2;
						}
						hitphi = fmod((1.-frac)*phinew + frac*phiprev, PI2);
					}
				}
				thetaprev = thetanew;
				phiprev = phinew;
				rprev = rnew;
			}

			if (bh && var[0] < 1.) return;

			if (hdid == h) ++nok; else ++nbad;
			if (z <= zEnd) {
				for (int i = 0; i < nvar; i++) varStart[i] = var[i];
				delete[] varScal;
				delete[] var;
				delete[] dvdz;
				return;
			}
			h = hnext;
		}
		delete[] varScal;
		delete[] var;
		delete[] dvdz;
	};


	inline void rkckIntegrate1(double rV, double thetaV, double phiV, double pRV,
		double bV, double qV, double pThetaV, double &thetaOut, double &phiOut) {

		double varStart[] = { rV, thetaV, phiV, pRV, pThetaV };
		double to = -10000000;
		double accuracy = 1e-6;
		double stepGuess = 0.01;
		double minStep = 0.00001;
		int nok, nbad;

		int n = 5;

		odeint1(varStart, n, to, accuracy, stepGuess, minStep, nok, nbad, bV, qV);

		thetaOut = varStart[1];
		phiOut = varStart[2];
		wrapToPi(thetaOut, phiOut);

	}

	inline void rkckIntegrate2(double rV, double thetaV, double phiV, double pRV,
		double bV, double qV, double pThetaV, double &thetaOut, double &phiOut, double &hitr, double &hitphi, bool bh) {

		double varStart[] = { rV, thetaV, phiV, pRV, pThetaV };
		double to = -10000000;
		double accuracy = 1e-6;
		double stepGuess = 0.01;
		double minStep = 0.00001;
		int nok, nbad;

		int n = 5;

		odeint2(varStart, n, to, accuracy, stepGuess, minStep, nok, nbad, bV, qV, hitr, hitphi, bh);

		thetaOut = varStart[1];
		phiOut = varStart[2];
		wrapToPi(thetaOut, phiOut);

	}

	/// <summary>
	/// Checks if a polygon has a high chance of crossing the 2pi border.
	/// </summary>
	/// <param name="poss">The coordinates of the polygon corners.</param>
	/// <param name="factor">The factor to check whether a point is close to the border.</param>
	/// <returns>Boolean indicating if the polygon is a likely 2pi cross candidate.</returns>
	inline bool check2PIcross(const vector<cv::Point2d>& spl, float factor) {
		for (int i = 0; i < spl.size(); i++) {
			if (spl[i]_phi > PI2*(1. - 1. / factor))
				return true;
		}
		return false;
	};

	inline bool check2PIcross(const vector<cv::Point2f>& spl, float factor) {
		for (int i = 0; i < spl.size(); i++) {
			if (spl[i]_phi > PI2*(1. - 1. / factor))
				return true;
		}
		return false;
	};

	/// <summary>
	/// Assumes the polygon crosses 2pi and adds 2pi to every corner value of a polygon
	/// that is close (within 2pi/factor) to 0.
	/// </summary>
	/// <param name="poss">The coordinates of the polygon corners.</param>
	/// <param name="factor">The factor to check whether a point is close to the border.</param>
	inline bool correct2PIcross(vector<cv::Point2d>& spl, float factor) {
		bool check = false;
		for (int i = 0; i < spl.size(); i++) {
			if (spl[i]_phi < PI2*(1. / factor)) {
				spl[i]_phi += PI2;
				check = true;
			}
		}
		return check;
	};

	inline bool correct2PIcross(vector<cv::Point2f>& spl, float factor) {
		bool check = false;
		for (int i = 0; i < spl.size(); i++) {
			if (spl[i]_phi < PI2*(1. / factor)) {
				spl[i]_phi += PI2;
				check = true;
			}
		}
		return check;
	};
}
