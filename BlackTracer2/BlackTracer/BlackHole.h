#pragma once
#include "Metric.h"

class BlackHole
{
public:
	double a;
	BlackHole(double afactor) {
		setA(afactor);
	};

	void setA(double afactor) {
		a = afactor;
		metric::setAngVel(afactor);
	}

	double getAngVel(double mass) {
		return mass*a;
	};
	~BlackHole(){};
};


