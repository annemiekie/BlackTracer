#pragma once

class Star
{
public:
	float phi;
	float theta;
	float magnitude;
	Vec3b color;
	Point2f posInPix;

	Star(float p, float t, float _magnitude, Vec3b _color) {
		phi = p;
		theta = t;
		magnitude = _magnitude;
		color = _color;
	};

	bool operator==(const Star& otherStar) const {
		return (otherStar.phi == phi && otherStar.theta == theta && otherStar.magnitude == magnitude);
	}



	~Star(){};
};

	template<>
	struct hash<Star>
	{
		size_t operator()(const Star& star) const
		{
			// Compute individual hash values for first, second and third
			// http://stackoverflow.com/a/1646913/126995
			size_t res = 17;
			res = res * 31 + hash<float>()(star.theta);
			res = res * 31 + hash<float>()(star.phi);
			res = res * 31 + hash<float>()(star.magnitude);
			return res;
		}
	};