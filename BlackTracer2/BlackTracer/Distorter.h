#pragma once
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdint.h> 
#include "Grid.h"
#include "Viewer.h"
#include "SplineBlock.h"
#include "Metric.h"
#include "Const.h"
#include "Star.h"
#include "StarProcessor.h"
#include "Camera.h"
#include "kernel.cuh"
#include <chrono>

using namespace cv;
using namespace std;

extern int makeImage(float *out, const float2 *thphi, const int *pi, const float *ver, const float *hor,
					const float *stars, const int *starTree, const int starSize, const float *camParam, const float *magnitude, const int treeLevel,
					const bool symmetry, const int M, const int N, const int step, const Mat csImage);

//extern int interpolateKernel(const double *thphiPixels, const double *cornersCam, const double *cornersCel,
//							 int *pi, const int size, const float *stars, const int *starTree, 
//							 const int starSize, const float *camParam, int *out, int* piCheck, int* bh, const float *buff);

/// <summary>
/// Class which handles all the computations that have to do with the distortion
/// of the input image and / or list of stars, given a computed grid and user input.
/// Can be used to produce a distorted output image.
/// </summary>
class Distorter
{
private:

	/** ------------------------------ VARIABLES ------------------------------ **/
	#pragma region variables
	/// <summary>
	/// Grid with computed rays from camera sky to celestial sky.
	/// </summary>
	Grid* grid;

	/// <summary>
	/// View window defined by the user (default: spherical panorama).
	/// </summary>
	Viewer* view;

	/// <summary>
	/// Optional: splines defining the sky distortions.
	/// </summary>
	Splines* splines;

	/// <summary>
	/// Camera defined by the user.
	/// </summary>
	Camera* cam;

	/// <summary>
	/// Star info including binary tree over
	/// list with stars (point light sources) present in the celestial sky.
	/// </summary>
	StarProcessor* starTree;

	/// <summary>
	/// Celestial sky image.
	/// </summary>
	Mat celestSrc;

	bool symmetry = false;

	/// <summary>
	/// Output image
	/// </summary>
	Mat finalImage;

	/// <summary>
	/// Binary matrix with a pixel mask for the black hole location.
	/// </summary>
	Mat blackhole;
	//int* bh;

	/// <summary>
	/// Interpolated theta, phi location on the celestial sky for each
	/// output pixel corner.
	/// </summary>
	Mat thetaPhiCelest;

	/// <summary>
	/// For every output pixel corner: value of the celestial sky image pixel
	/// the corner is mapped to. 
	/// </summary>
	Mat matchPixVal;

	/// <summary>
	/// For every output pixel corner: position of the pixel in the celestial
	/// sky image the corner is mapped to.
	/// </summary>
	Mat matchPixPos;
	int *pixPosArray;

	/// <summary>
	/// For every output pixel corner: orientation of the projection on the 
	/// celestial sky.
	/// </summary>
	Mat orientations;

	/// <summary>
	/// Matrix with possible 2pi crossing positions
	/// </summary>
	Mat pi2Checks;
	//int *pi;

	/// <summary>
	/// Matrix with possible inflection positions
	/// </summary>
	Mat inflections;

	/// <summary>
	/// Matrix containing fraction between solid angle per pixel 
	/// of the output picture and celestial sky projection.
	/// </summary>
	Mat solidAngleFrac;

	/// <summary>
	/// Average colour of whole celestial sky image. Used when so many pixels
	/// need to be averaged, that it won't differ much from using the whole image.
	/// </summary>
	Vec3b allcol;

	/// <summary>
	/// Vectors with the pixel sizes in theta and phi direction
	/// </summary>
	vector<double> thetaDiff, phiDiff;

	/// <summary>
	/// The window to display the output image in
	/// </summary>
	const char* warp_window = "Warp";

	/// <summary>
	/// Hashing function for pixel map.
	/// </summary>
	struct hashing_func {
		size_t operator()(const Point& key) const {
			return key.x * 32 + key.y;
		}
	};

	/// <summary>
	/// Vector with celestial sky pixels per output pixel.
	/// </summary>
	vector<unordered_set<Point, hashing_func>> PixToPix;

	/// <summary>
	/// Vector with stars falling in pixel. Needs better hashing?
	/// </summary>
	vector<unordered_set<Star>> pixToStar;

	/// <summary>
	/// Whether splines have been computed for interpolation.
	/// </summary>
	bool splineInt;

	#pragma endregion

	/** ---------------------------- INTERPOLATING ---------------------------- **/
	#pragma region interpolating

	/// <summary>
	/// Finds the theta phi positions on the celestial sky for every pixel corner.
	/// Finds the pixel postion and colour of the celestial sky image for this position.
	/// Determines which pixels should be black hole pixels.
	/// Stores all the information in matrices.
	/// </summary>
	void findThetaPhiCelest() {
	
		// If there is a symmetry axis, only half needs to be computed.
		int vertical_size = view->ver.size();
		if (!grid->equafactor) {
			vertical_size = (vertical_size - 1) / 2 + 1;
		}
		//#pragma omp parallel for shared(vertical_size, view, grid, thetaPhiCelest, matchPixPos, matchPixVal) schedule(guided, 4)
		for (int t = 0; t < vertical_size; t++) {
			double theta = view->ver[t];
	
			for (int p = 0; p < view->hor.size(); p++) {
				double phi = view->hor[p];
	
				int lvlstart = 0;
				int quarter = findStartingBlock(lvlstart, phi);
				uint64_t ij = findBlock(grid->startblocks[quarter], theta, phi, lvlstart);


				Point2d thphi = interpolate(ij, theta, phi);
				Point pos;
				Vec3b val;
	
				if (thphi != Point2d(-1, -1)) {					
					pi2Checks.at<bool>(t, p) = grid->crossings2pi[ij] ? 1 : 0;
					val = getPixelAtThPhi(thphi, pos);
				}
				else {
					// Fill the black hole matrix.
					fillBlackHole(t, p);
					// Fill the pixelPos and pixelVal matrices.
					pos = Point(-1, -1);
					val = Vec3b(0, 0, 0);
				}
	
				thetaPhiCelest.at<Point2d>(t, p) = thphi;
				matchPixVal.at<Vec3b>(t, p) = val;
				matchPixPos.at<Point>(t, p) = pos;
	
				// If symmetry axis, also fill the values for the bottom half of the image.
				if (!grid->equafactor) {
					int t_down = view->ver.size() - 1 - t;
					if (thphi != Point2d(-1, -1)) {
						thphi_theta = PI - thphi_theta;
						pi2Checks.at<bool>(t_down, p) = pi2Checks.at<bool>(t, p);
						val = getPixelAtThPhi(thphi, pos);
					}
					else {
						fillBlackHole(t_down, p);
					}
					thetaPhiCelest.at<Point2d>(t_down, p) = thphi;
				}
			}
		}
	};

	/// <summary>
	/// Recursively finds the block the specified theta,phi on the camera sky fall into.
	/// </summary>
	/// <param name="ij">The key of the current block the search is in.</param>
	/// <param name="theta">The theta on the camera sky.</param>
	/// <param name="phi">The phi on the camera sky.</param>
	/// <param name="level">The current level to search.</param>
	/// <returns>The key of the block.</returns>
	uint64_t findBlock(uint64_t ij, double theta, double phi, int level) {

		if (grid->blockLevels[ij] <= level) {
			return ij;
		}
		else {
			int gap = (int)pow(2, grid->MAXLEVEL - level);
			uint32_t i = i_32;
			uint32_t j = j_32;
			uint32_t k = i + gap / 2;
			uint32_t l = j + gap / 2;
			double thHalf = PI2*k / grid->M;
			double phHalf = PI2*l / grid->M;
			if (thHalf < theta) i = k;
			if (phHalf < phi) j = l;
			return findBlock(i_j, theta, phi, level + 1);
		}
	};

	/// <summary>
	/// Fill the pixels of the black hole matrix that border on the
	/// given pixel corner with 1.
	/// </summary>
	/// <param name="i">The i.</param>
	/// <param name="j">The j.</param>
	void fillBlackHole(int i, int j) {
		if (i < blackhole.rows) {
			blackhole.at<bool>(i, j) = 1;
			blackhole.at<bool>(i, (j + blackhole.cols - 1) % blackhole.cols) = 1;
		}
		if (i > 0) {
			blackhole.at<bool>(i - 1, j) = 1;
			blackhole.at<bool>(i - 1, (j + blackhole.cols - 1) % blackhole.cols) = 1;
		}
	}

	/// <summary>
	/// Find the starting block of the grid and the starting level of the block.
	/// The starting block number depends on the symmetry and the value of phi.
	/// </summary>
	/// <param name="lvlstart">The lvlstart.</param>
	/// <param name="phi">The phi.</param>
	/// <returns></returns>
	int findStartingBlock(int& lvlstart, double phi) {
		if (!grid->equafactor) {
			lvlstart = 1;
			if (phi > PI) {
				return (phi > 3.f * PI1_2) ? 3 : 2;
			}
			else if (phi > PI1_2) return 1;
		}
		else {
			if (phi > PI) return 1;
		}
		return 0;
	}

	/// <summary>
	/// Interpolates the celestial sky values for the corners of the block with key ij
	/// to find the celestial sky value for the provided theta-phi point that lies within
	/// this block on the camera sky.
	/// </summary>
	/// <param name="ij">The key to the block the pixelcorner is located in.</param>
	/// <param name="thetaCam">Theta value of the pixel corner (on the camera sky).</param>
	/// <param name="phiCam">Phi value of the pixel corner (on the camera sky).</param>
	/// <returns>The Celestial sky position the provided theta-phi value will map to.</returns>
	Point2d interpolate(uint64_t ij, double thetaCam, double phiCam) {
		vector<double> cornersCam(4);
		vector<Point2d> cornersCel(4);
		calcCornerVals(ij, cornersCel, cornersCam);


		double error = 0.00001;
		double thetaUp = cornersCam[0];
		double thetaDown = cornersCam[2];
		double phiLeft = cornersCam[1];
		double phiRight = cornersCam[3];

		double thetaMid = 1000;
		double phiMid = 1000;

		if (grid->crossings2pi[ij] || metric::check2PIcross(cornersCel, 5.)) metric::correct2PIcross(cornersCel, 5.);
		Point2d lu = cornersCel[0];
		Point2d ru = cornersCel[1];
		Point2d ld = cornersCel[2];
		Point2d rd = cornersCel[3];
		if (lu == Point2d(-1, -1) || ru == Point2d(-1, -1) || ld == Point2d(-1, -1) || rd == Point2d(-1, -1)) return Point2d(-1, -1);

		while ((fabs(thetaCam - thetaMid) > error) || (fabs(phiCam - phiMid) > error)) {
			thetaMid = (thetaUp + thetaDown) * HALF;
			phiMid = (phiRight + phiLeft) * HALF;
			Point2d mall = (lu + ld + rd + ru)*(1.f / 4.f);
			if (thetaCam > thetaMid) {
				Point2d md = (rd + ld) * HALF;
				thetaUp = thetaMid;
				if (phiCam > phiMid) {
					phiLeft = phiMid;
					lu = mall;
					ru = (ru + rd)*HALF;
					ld = md;
				}
				else {
					phiRight = phiMid;
					ru = mall;
					rd = md;
					lu = (lu + ld)*HALF;
				}
			}
			else {
				thetaDown = thetaMid;
				Point2d mu = (ru + lu) * HALF;
				if (phiCam > phiMid) {
					phiLeft = phiMid;
					ld = mall;
					lu = mu;
					rd = (ru + rd)*HALF;
				}
				else {
					phiRight = phiMid;
					rd = mall;
					ru = mu;
					ld = (lu + ld)*HALF;
				}
			}
		}
		Point2d meanVal = (ru + lu + rd + ld)*(1. / 4.);
		metric::wrapToPi(meanVal.x, meanVal.y);
		return meanVal;
	}

	/// <summary>
	/// Calculates the camera sky and celestial sky corner values for a given block.
	/// </summary>
	/// <param name="ij">The key to the block.</param>
	/// <param name="cornersCel">Vector to store the celestial sky positions.</param>
	/// <param name="cornersCam">Vector to store the camera sky positions.</param>
	void calcCornerVals(uint64_t ij, vector<Point2d>& cornersCel, vector<double>& cornersCam) {
		int level = grid->blockLevels[ij];
		int gap = (int)pow(2, grid->MAXLEVEL - level);
		uint32_t i = i_32;
		uint32_t j = j_32;
		uint32_t k = i + gap;
		// l should convert to 2PI at end of grid for correct interpolation
		uint32_t l = j + gap;
		double factor = PI2 / grid->M;
		cornersCam = { factor*i, factor*j, factor*k, factor*l };
		// but for lookup l should be the 0-point
		l = l % grid->M;
		cornersCel = { grid->CamToCel[ij], grid->CamToCel[i_l], grid->CamToCel[k_j], grid->CamToCel[k_l] };
	}
	#pragma endregion

	/** ------------------------------- PIXELS -------------------------------- **/
	#pragma region pixels

	/// <summary>
	/// Finds the pixel values for the output image as averaged pixelvalues
	/// of the celestial sky image.
	/// </summary>
	void findPixelValues() {

		for (int q = 0; q < finalImage.rows / 8; q++) {
			cout << "-";
		} cout << "|" << endl;

		PixToPix.resize(finalImage.cols*finalImage.rows);
		cout << "pixpix " << PixToPix.size() << endl;

		#pragma omp parallel for
		for (int i = 0; i < finalImage.rows; i++) {
			if (i % 8 == 0) cout << ".";

			vector<Point> coords(4);
			coords[1] = Point(0, i);
			coords[0] = Point(0, i + 1);

			for (int j = 1; j < finalImage.cols + 1; j++) {
				coords[2] = Point(j, i);
				coords[3] = Point(j, i + 1);
				if (blackhole.at<bool>(i, j - 1) == 1)
					finalImage.at<Vec3b>(i, j - 1) = Vec3b(0, 0, 0);
				else {
					vector<Vec3b> colors;
					unordered_set<Point, hashing_func> pixset;
					findPixels(coords, colors, pixset);
					PixToPix[i * finalImage.cols + (j - 1)] = pixset;
					finalImage.at<Vec3b>(i, j - 1) = averagePix(colors);
				}
				coords[1] = coords[2];
				coords[0] = coords[3];
			}
		}
	}

	/// <summary>
	/// Averages all pixels in the celestial sky image.
	/// </summary>
	void averageAll() {
		int ch0 = 0;
		int ch1 = 0;
		int ch2 = 0;
		for (int i = 0; i < celestSrc.rows; ++i) {
			for (int j = 0; j < celestSrc.cols; ++j) {
				Vec3b col = celestSrc.at<Vec3b>(i, j);
				ch0 += col.val[0] * col.val[0];
				ch1 += col.val[1] * col.val[1];
				ch2 += col.val[2] * col.val[2];
			}
		}
		int numpix = celestSrc.rows*celestSrc.cols;
		allcol = Vec3b(sqrt(ch0 / numpix), sqrt(ch1 / numpix), sqrt(ch2 / numpix));
	}

	/// <summary>
	/// Average pixel values in vector with pixel values.
	/// https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch24.html
	/// </summary>
	/// <param name="pixels">Vector with pixels to be averaged.</param>
	/// <returns></returns>
	Vec3b averagePix(vector<Vec3b>& pixels) {
		int ch0 = 0;
		int ch1 = 0;
		int ch2 = 0;
		for (int q = 0; q < pixels.size(); q++) {
			ch0 += pixels[q].val[0] * pixels[q].val[0];
			ch1 += pixels[q].val[1] * pixels[q].val[1];
			ch2 += pixels[q].val[2] * pixels[q].val[2];
		}
		size_t numpix = pixels.size();
		Vec3b ave = Vec3b(sqrt(ch0 / numpix), sqrt(ch1 / numpix), sqrt(ch2 / numpix));
		return ave;
	}

	/// <summary>
	/// Insertmaxmins the specified maxmin.
	/// </summary>
	/// <param name="maxmin">The maxmin.</param>
	/// <param name="pix">The pix.</param>
	void insertmaxmin(unordered_map<int, vector<int>>& maxmin, Point pix) {
		auto it = maxmin.find(pix.x);
		if (it == maxmin.end())
			maxmin[pix.x] = vector < int > {pix.y, pix.y};
		else if (maxmin[pix.x][0] < pix.y) maxmin[pix.x][0] = pix.y;
		else maxmin[pix.x][1] = pix.y;
	}

	/// <summary>
	/// Finds the celestial sky image pixels that fall into the projected output pixel.
	/// </summary>
	/// <param name="coords">The coordinates of the corners of the projected output pixel.</param>
	/// <param name="pixvals">Vector with colors for the retrieved pixels.</param>
	/// <param name="pixset">Set to save the found pixels in.</param>
	void findPixels(vector<Point>& coords, vector<Vec3b>& pixvals, unordered_set<Point, hashing_func>& pixset) {

		bool check = true;

		Point pix = matchPixPos.at<Point>(coords[0]);
		for (int q = 0; q < 4; q++) {

			auto it = pixset.find(pix);
			if (it == pixset.end()) {
				pixvals.push_back(matchPixVal.at<Vec3b>(coords[q]));
				pixset.insert(pix);
			}

			Point pix2 = matchPixPos.at<Point>(coords[(q + 1) % 4]);
			int dist = abs(pix.x - pix2.x) + abs(pix.y - pix2.y);
			if (dist>1) check = false;
			pix = pix2;
		}
		if (check) return;

		vector<Point2d> thphivals(4);
		for (int q = 0; q < 4; q++)
			thphivals[q] = thetaPhiCelest.at<Point2d>(coords[q]);
		//changed this
		bool check2pi = false;
		if (metric::check2PIcross(thphivals, 5)) {
			metric::correct2PIcross(thphivals, 5);
			check2pi = true;
		}

		unordered_map<int, vector<int>> maxmin;
		pix = matchPixPos.at<Point>(coords[0]);

		for (int q = 0; q < 4; q++) {
			insertmaxmin(maxmin, pix);

			Point pix2 = matchPixPos.at<Point>(coords[(q + 1) % 4]);
			//if (check2pi) if (pix2.y < celestSrc.cols / 5) pix2.y += celestSrc.cols;
			//if (check2pi) if (pix.y < celestSrc.cols / 5) pix.y += celestSrc.cols;


			int dist = abs(pix.x - pix2.x) + abs(pix.y - pix2.y);
			if (dist < 2) continue;
			if (dist > celestSrc.cols / 2) {
				if (check2pi) dist = celestSrc.cols - dist;
				else {
					pixvals = { allcol };
					return;
				}
			}


			Point2d tp = thphivals[q];
			Point2d tp1 = thphivals[(q + 1) % 4];

			double thetadist = tp1.x - tp.x;
			double phidist = tp1.y - tp.y;

			for (int p = 1; p < dist; p++) {
				Point2d pt = Point2d(tp.x + thetadist / dist*p, fmod(tp.y + phidist / dist*p, PI2));
				Point pos;
				//bool def;
				Vec3b pixcol = getPixelAtThPhi(pt, pos);
				//if (def) {
				//	pixvals = { Vec3b(0, 0, 0) };
				//	blackhole.at<bool>(coords[0]) = 1;
				//	return;
				//}
				auto itx = pixset.find(pos);

				if (itx == pixset.end()) {
					pixvals.push_back(pixcol);
					pixset.insert(pos);
					insertmaxmin(maxmin, pos);
				}
			}
			pix = pix2;
		}

		for (auto mm : maxmin) {
			for (int m = mm.second[1] + 1; m < mm.second[0]; m++) {
				pixvals.push_back(celestSrc.at<Vec3b>(Point(mm.first, m)));
				pixset.insert(Point(mm.first, m));
			}
		}
		//drawPixSelection(thphivals, pixset);
	}

	/// <summary>
	/// Gets the pixel colour of the celestial sky image at a certain theta phi position.
	/// </summary>
	/// <param name="thphi">The theta-phi coordinate.</param>
	/// <param name="pos">Stores the pixel position at theta-phi.</param>
	/// <returns></returns>
	Vec3b getPixelAtThPhi(Point2d thphi, Point& pos) {
		double theta = thphi_theta;
		double phi = thphi_phi;
		if (theta < 0 || phi < 0) {
			cout << " aaaaah " << endl;
		}
		metric::wrapToPi(theta, phi);

		pos.x = (int)floor(1. * celestSrc.cols * phi / PI2);
		pos.y = (int)floor(1. * celestSrc.rows * theta / PI);

		return celestSrc.at<Vec3b>(pos);
	};

	#pragma endregion

	/** -------------------------------- STARS -------------------------------- **/
	#pragma region stars

	void cudaTest() {
		// fill arrays with data
		int N = view->pixelheight;
		int M = view->pixelwidth;
		int H = symmetry ? N/2 : N;
		int W = M;
		int H1 = H + 1;
		int W1 = W + 1;
		vector<float2> thphi(H1*W1);
		vector<int> pix(H1*W1);
		vector<int> pi(H*W);
		vector<float> ver(N + 1);
		vector<float> hor(W1);

		//#pragma omp parallel for
		for (int t = 0; t < H1; t++) {
			double theta = view->ver[t];
			for (int p = 0; p < W1; p++) {
				double phi = view->hor[p];
				int lvlstart = 0;
				//int quarter = findStartingBlock(lvlstart, phi);
				//uint64_t ij = findBlock(grid->startblocks[quarter], theta, phi, lvlstart);
				//Point2d thphiInter = interpolate(ij, theta, phi);
				uint32_t k = (uint32_t) t;
				uint32_t j = (uint32_t) p;
				uint64_t ij = (uint64_t)k << 32 | j;
				Point2d thphiInter = grid->CamToCel[ij];
				int i = t*W1 + p;
				if (thphiInter != Point2d(-1, -1)) {
					pix[i] = 0;// grid->crossings2pi[i_j] ? 1 : 0;
				}
				else {
					pix[i] = -4;
				}
				thphi[i].x = thphiInter.x;
				thphi[i].y = thphiInter.y;
			}
		}

		// Fill pi with black hole and picrossing info
		#pragma omp parallel for
		for (int t = 0; t < H; t++) {
			int pi1 = pix[t*W1] + pix[(t + 1)*W1];
			for (int p = 0; p < W; p++) {
				int i = t*W + p;
				int pi2 = pix[t*W1 + p + 1] + pix[(t + 1)*W1 + p + 1];
				pi[i] = pi1 + pi2;
				pi1 = pi2;
			}
		}

		for (int t = 0; t < N+1; t++) {
			ver[t] = view->ver[t];
		}
		for (int t = 0; t < W1; t++) {
			hor[t] = view->hor[t];
		}

		int step = 2;
		vector<float> out(N*M);
		makeImage(&out[0], &thphi[0], &pi[0], &ver[0], &hor[0],
			&(starTree->starPos[0]), &(starTree->binaryStarTree[0]), starTree->starSize, cam->getParamArray(), &(starTree->starMag[0]), starTree->treeLevel,
			symmetry, M, N, step, celestSrc);
	}
	
	/// <summary>
	/// Finds the stars that fall into a particular output pixel.
	/// </summary>
	/// <param name="thphivals">The theta-phi values of the corners of the polygon.</param>
	/// <param name="starSet">The set to save the found stars in.</param>
	/// <param name="sgn">The winding order of the polygon + for CW, - for CCW.</param>
	/// <param name="check2pi">If set to <c>true</c> 2pi crossing is a possibility.</param>
	void findStarsForPixel(vector<Point2d> thphivals, unordered_set<Star>& starSet, int sgn, bool check2pi) {
		for (int i = 0; i < starTree->starVec.size(); ++i) {
			Star star = starTree->starVec[i];
			Point2d starPt = Point2d(star.theta, star.phi);
			bool starInPoly = starInPolygon(starPt, thphivals, sgn);
			if (!starInPoly && check2pi && star.phi < PI2 / 5.) {
				starPt.y += PI2;
				starInPoly = starInPolygon(starPt, thphivals, sgn);
			}
			if (starInPoly) {
				star.posInPix = interpolate2(thphivals, starPt, sgn);
				starSet.insert(star);
			}
		}
	}
	
	/// <summary>
	/// Returns if a (star) location lies within the boundaries of the provided polygon.
	/// </summary>
	/// <param name="starPt">The star location in theta and phi.</param>
	/// <param name="thphivals">The theta-phi values of the corners of the polygon.</param>
	/// <param name="sgn">The winding order of the polygon + for CW, - for CCW.</param>
	/// <returns></returns>
	bool starInPolygon(Point2d& starPt, vector<Point2d>& thphivals, int sgn) {
		for (int q = 0; q < 4; q++) {
			Point2d vecLine = sgn * (thphivals[q] - thphivals[(q + 1) % 4]);
			Point2d vecPoint = sgn ? (starPt - thphivals[(q + 1) % 4]) : (starPt - thphivals[q]);
			if (vecLine.cross(vecPoint) < 0) {
				return false;
			}
		}
		return true;
	}

	/// <summary>
	/// Finds the stars that fall into each pixel of the output image.
	/// Computes solidAngleFractions and 2piCandidates along the way.
	/// </summary>
	void findStars(vector<float>& out) {
		#pragma omp parallel for
		for (int q = 0; q < finalImage.rows / 8; q++) {
			cout << "-";
		} cout << "|" << endl;

		pixToStar.resize(finalImage.total());
		int sum = 0;
		int pixVert = finalImage.rows;// grid->equafactor ? finalImage.rows : finalImage.rows / 2;
		//#pragma omp parallel for
		for (int i = 0; i < pixVert; i++) {
			if (i % 8 == 0) cout << ".";

			bool *pirow0 = pi2Checks.ptr<bool>(i);
			bool *pirow1 = pi2Checks.ptr<bool>(i + 1);
			bool pi2checkLeft = pirow0[0] || pirow1[0];

			vector<Point2d> thphivals(4);
			Point2d *thphiRow0 = thetaPhiCelest.ptr<Point2d>(i);
			Point2d *thphiRow1 = thetaPhiCelest.ptr<Point2d>(i + 1);

			thphivals[1] = thphiRow0[0];
			thphivals[0] = thphiRow1[0];

			double verSize = abs(view->ver[i] - view->ver[(i + 1)]);
			thetaDiff[i] = verSize;

			for (int j = 0; j < finalImage.cols; j++) {
				thphivals[2] = thphiRow0[j + 1];
				thphivals[3] = thphiRow1[j + 1];

				bool pi2checkRight = pirow0[j+1] || pirow1[j+1];

				// Check if pixel is black hole, if yes -> skip
				if (!blackhole.at<bool>(i, j)) {
					// If one of the pixelcorners lies in a 2pi crossing block, check if the 
					// pixel crosses the 2pi border as well.
					bool check2pi = false;
					if (pi2checkLeft || pi2checkRight) {
						if (metric::check2PIcross(thphivals, 5.f)) {
							 check2pi = metric::correct2PIcross(thphivals, 5.f);
						}
						pi2Checks.at<bool>(i, j) = check2pi;
					}

					// Orientation is positive if CW, negative if CCW
					double orient = 0;
					for (int q = 0; q < 4; q++) {
						orient += (thphivals[(q + 1) % 4]_theta - thphivals[q]_theta)
							*(thphivals[(q + 1) % 4]_phi + thphivals[q]_phi);
					}
					int sgn = metric::sgn(orient);

					// save the projected fraction on the celestial sky
					if (i == 0) {
						double horSize = abs(view->hor[j] - view->hor[(j + 1) % finalImage.cols]);
						phiDiff[j] = horSize;
					}
					double solidAnglePix = phiDiff[j] * verSize;
					solidAngleFrac.at<float>(i, j) = solidAnglePix / (fabs(orient) * 0.5f);
					orientations.at<int>(i, j) = sgn;

					// Compute the stars that fall into this pixel.
					unordered_set<Star> starSet;
					findStarsForPixel(thphivals, starSet, sgn, check2pi);
					pixToStar[i * finalImage.cols + j] = starSet;
					//if (starSet.size() != out[i*1024+j]) {
					//	cout << starSet.size() << "\t" << out[i * 1024 + j] << "\t" << i << "\t" << j << "\t" << check2pi << "\t" << sgn << endl;
					//	sum++;
					//}
				}
				if (thphivals[2]_phi > PI2) thphivals[2]_phi -= PI2;
				if (thphivals[3]_phi > PI2) thphivals[3]_phi -= PI2;

				thphivals[1] = thphivals[2];
				thphivals[0] = thphivals[3];

				pi2checkLeft = pi2checkRight;
			}
		}
		//if (!grid->equafactor) {
		//	computeBottomHalf(pixVert);
		//}
		cout << endl << "wrooong " << sum << endl;
	}

	/// <summary>
	/// Computes the bottom half of the image.
	/// </summary>
	/// <param name="pixVert">The pix vert.</param>
	void computeBottomHalf(int pixVert) {
	#pragma omp parallel for
		for (int i = pixVert; i < finalImage.rows; i++) {
			if (i % 8 == 0) cout << ".";
			vector<Point2d> thphivals(4);
			thphivals[1] = thetaPhiCelest.at<Point2d>(i, 0);
			thphivals[0] = thetaPhiCelest.at<Point2d>(i + 1, 0);
			int p = finalImage.rows - 1 - i;

			for (int j = 0; j < finalImage.cols + 1; j++) {
				thphivals[2] = thetaPhiCelest.at<Point2d>(i, j + 1);
				thphivals[3] = thetaPhiCelest.at<Point2d>(i + 1, j + 1);

				// Check if pixel is black hole, if yes -> skip
				if (blackhole.at<bool>(i, j) == 0) {
					int sgn = orientations.at<int>(p, j);
					bool check2pi = pi2Checks.at<bool>(p, j);
					if (check2pi) {
						metric::correct2PIcross(thphivals, 5.f);
					}
					unordered_set<Star> starSet;
					findStarsForPixel(thphivals, starSet, sgn, check2pi);
					pixToStar[i * finalImage.cols + j] = starSet;
					solidAngleFrac.at<float>(i, j) = solidAngleFrac.at<float>(p, j);
				}
				if (thphivals[2]_phi > PI2) thphivals[2]_phi -= PI2;
				if (thphivals[3]_phi > PI2) thphivals[3]_phi -= PI2;
				thphivals[1] = thphivals[2];
				thphivals[0] = thphivals[3];
			}
		}
	}

	/// <summary>
	/// Colors the pixels according to the stars that fall into each pixel.
	/// </summary>
	void colorPixelsWithStars(vector<float>& compare) {
		#pragma omp parallel for
		for (int q = 0; q < finalImage.rows / 8; q++) {
			cout << "-";
		} cout << "|" << endl;

		int step = 1;
		int toEnd = finalImage.rows - step - 1;

		//#pragma omp parallel for
		for (int i = 0; i < 5; i++) {
			if (i % 8 == 0) cout << ".";
			int start = -step;
			int stop = step;
			if (i < step) {
				start += (step - i);
			}
			if (i > toEnd) {
				stop -= (i - toEnd);
			}

			for (int j = 0; j < finalImage.cols; j++) {
				if (blackhole.at<bool>(i, j) == 1) {
					finalImage.at<Vec3b>(i, j) = Vec3b(100, 0, 0);
				}
				else {
					double brightnessPix = 0;
					Point2d pix = Point2d(i + 0.5, j + 0.5);

					for (int p = start; p <= stop; p++) {
						int pi = p + i;
						double theta = view->hor[pi];
						double thetaD = thetaDiff[pi];

						for (int q = -step; q <= step; q++) {

							int qj = (q + j + finalImage.cols) % finalImage.cols;
							double phi = view->ver[qj];
							double phiD = phiDiff[qj];

							unordered_set<Star> pixStars = pixToStar[pi * finalImage.cols + qj];
							//vector<Vec3b> colors;
							if (pixStars.size() > 0) {
								Point2f starPix = Point2f(pi, qj);
								double frac = solidAngleFrac.at<float>(pi, qj);
								for (const Star& star : pixStars) {
									Point2d starPosPix = starPix + star.posInPix;
									double distsq = euclideanDistSq(starPosPix, pix);
									if (distsq > (step+.5)*(step+.5)) continue;

									double thetaCam = theta + thetaD * star.posInPix.x;
									double phiCam = phi + phiD * star.posInPix.y;
									// m2 = m1 - 2.5(log(frac)) where frac = brightnessfraction of b2/b1;
									//colors.push_back(star.color);
									double appMag = star.magnitude + 10 * log10(redshift(thetaCam, phiCam)) - 2.5 * log10(frac*gaussian(distsq));
									brightnessPix += pow(10., -appMag*0.4);
									//printf("%f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \n", star.posInPix.x, star.posInPix.y, appMag, theta, phi, thetaD, phiD, thetaCam, phiCam, frac);
									if (i < 5) printf("%f \t %f \t %d \t %d\n", appMag, distsq, p,q);
								}
							}
						}
					}
					Vec3b color = Vec3b(0, 0, 0);
					if (brightnessPix != 0) {
						double appMagPix = -2.5*log10(brightnessPix);
						if (appMagPix < 8.) {
							//double brightness = brightnessPix*30.;
							double value = min(255., 255. * brightnessPix);
							color = Vec3b(value, value, value);
						}
					}
					finalImage.at<Vec3b>(i, j) = color;//averageStars(stars);
				}
				// Size of image / number of pixels matters a lot in how to choose the gaussian and the brightness to rgb value;
			}
		}
	}

	/// <summary>
	/// Computes the euclidean distance between two point2ds.
	/// </summary>
	double euclideanDistSq(Point2d& p, Point2d& q) {
		Point2d diff = p - q;
		return diff.x*diff.x + diff.y*diff.y;
	}

	/// <summary>
	/// Computes a semigaussian for the specified distance value.
	/// </summary>
	/// <param name="dist">The distance value.</param>
	double gaussian(double dist) {
		return exp(-1.*dist);//0.8*exp(-2*dist);
	}

	/// <summary>
	/// Calculates the redshift for the specified theta-phi on the camera sky.
	/// </summary>
	/// <param name="theta">The theta of the position on the camera sky.</param>
	/// <param name="phi">The phi of the position on the camera sky.</param>
	/// <returns></returns>
	double redshift(double theta, double phi) {
		double yCam = sin(theta) * sin(phi);
		double part = (1. - cam->speed*yCam);
		double betaPart = sqrt(1 - cam->speed * cam->speed) / part;

		double yFido = (-yCam + cam->speed) / part;
		double eF = 1. / (cam->alpha + cam->w * cam->wbar * yFido);
		double b = eF * cam->wbar * yFido;

		return 1. / (betaPart * (1 - b*cam->w) / cam->alpha);
	}

	/// <summary>
	/// Interpolates the corners of a projected pixel on the celestial sky to find the position
	/// of a star in the (normal, unprojected) pixel in the output image.
	/// </summary>
	/// <param name="cornersCel">The projected pixel corners on the celestial sky.</param>
	/// <param name="starPos">The star position.</param>
	/// <param name="sgn">The winding order of the pixel corners.</param>
	/// <returns>Position of star in pixel of output image.</returns>
	Point2f interpolate2(vector<Point2d>& cornersCel, Point2f starPos, int sgn) {

		Point2f cel0 = cornersCel[0];
		Point2f cel1 = cornersCel[1];
		Point2f cel2 = cornersCel[2];
		Point2f cel3 = cornersCel[3];

		float error = 0.00001f;

		Point2f mid = (cel1 + cel2 + cel3 + cel0) / 4.f;
		float starInPixY = 0.5f;
		float starInPixX = 0.5f;

		float perc = 0.5f;
		int count = 0;
		while (((fabs(starPos.x - mid.x) > error) || (fabs(starPos.y - mid.y) > error)) && count < 100) {
			count++;
			Point2f half01 = (cel0 + cel1) / 2.f;
			Point2f half23 = (cel2 + cel3) / 2.f;
			Point2f half12 = (cel2 + cel1) / 2.f;
			Point2f half03 = (cel0 + cel3) / 2.f;

			Point2f line01to23 = half23 - half01;
			Point2f line03to12 = half12 - half03;

			Point2f line01toStar = starPos - half01;
			Point2f line03toStar = starPos - half03;

			int a = metric::sgn(line03to12.cross(line03toStar));
			int b = metric::sgn(line01to23.cross(line01toStar));

			perc *= 0.5f;

			if (sgn*a > 0) {
				if (sgn*b > 0) {
					cel2 = half12;
					cel0 = half01;
					cel3 = mid;
					starInPixX -= perc;
					starInPixY -= perc;
				}
				else {
					cel2 = mid;
					cel1 = half01;
					cel3 = half03;
					starInPixX -= perc;
					starInPixY += perc;
				}
			}
			else {
				if (sgn*b > 0) {
					cel1 = half12;
					cel3 = half23;
					cel0 = mid;
					starInPixX += perc;
					starInPixY -= perc;
				}
				else {
					cel0 = half03;
					cel1 = mid;
					cel2 = half23;
					starInPixX += perc;
					starInPixY += perc;
				}
			}
			mid = (cel1 + cel2 + cel3 + cel0) / 4.f;
		}

		return Point2f(starInPixX, starInPixY);
	}

	#pragma endregion

public:

	/// <summary>
	/// Initializes a new empty instance of the <see cref="Distorter"/> class.
	/// </summary>
	Distorter() {};

	/// <summary>
	/// Initializes a new instance of the <see cref="Distorter"/> class.
	/// </summary>
	/// <param name="filename">The filename for the celestial sky image.</param>
	/// <param name="deformgrid">The grid with mappings from camera sky to celestial sky.</param>
	/// <param name="viewer">The viewer that produces the output image.</param>
	/// <param name="_splines">Optional: Splines.</param>
	/// <param name="_stars">The star info including binary tree with stars to display.</param>
	/// <param name="_splineInt">if set to <c>true</c> interpolate using splines.</param>
	/// <param name="camera">The camera the output image is viewed from.</param>
	Distorter(string filename, Grid* deformgrid, Viewer* viewer, Splines* _splines, StarProcessor* _stars, bool _splineInt, Camera* camera) {
		// Read image
		celestSrc = imread(filename, 1);

		// Load variables
		grid = deformgrid;
		view = viewer;
		splineInt = _splineInt;
		if (splineInt) splines = _splines;
		starTree = _stars;
		cam = camera;
		if (grid->equafactor == 0) symmetry = true;
		// Pixel corners have to be taken into account, so +1 for vertical direction is needed.
		// For horizontal it's not necessary, but easier to use.
		int h = view->pixelheight;
		int w = view->pixelwidth;
		thetaPhiCelest = Mat(h + 1, w + 1, DataType<Point2d>::type);
		matchPixVal = Mat(h + 1, w + 1, DataType<Vec3b>::type);
		matchPixPos = Mat(h + 1, w + 1, DataType<Point>::type);
		orientations = Mat(h + 1, w + 1, DataType<int>::type);
		pi2Checks = Mat::zeros(h + 1, w + 1, CV_8U);

		//pi = new int[h * w];
		//bh = new int[h * w];
		pixPosArray = new int[(h / 2 + 1) * (w + 1) * 2];

		solidAngleFrac = Mat(h, w, DataType<float>::type);
		finalImage = Mat(h, w, DataType<Vec3b>::type);
		blackhole = Mat::zeros(h, w, CV_8U);
		//inflections = Mat::zeros(h, w, CV_8U);

		thetaDiff.resize(finalImage.rows);
		phiDiff.resize(finalImage.cols);

		// Precomputations on the celestial sky image.
		averageAll();
		rayInterpolater();
	};

	/// <summary>
	/// Saves the finalImage matrix as file.
	/// </summary>
	/// <param name="filename">The filename for the image file.</param>
	void saveImg(string filename) {
		imwrite(filename, finalImage);
	}

	/// <summary>
	/// Computes everything the distorter is supposed to compute.
	/// </summary>
	void rayInterpolater() {

		//cout << "Interpolating1..." << endl;
		findThetaPhiCelest();

		auto start_time = std::chrono::high_resolution_clock::now();
		cout << "Interpolating2..." << endl;
		cudaTest();
		auto end_time = std::chrono::high_resolution_clock::now();
		cout << " time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << endl;
		//findPixelValues();
		//findStars(vector<float>());
		//colorPixelsWithStars(out);
		////Mat smoothImage(finalImage.size(), DataType<Vec3b>::type);
		//imshow(warp_window, finalImage);
		//waitKey(10000);

	};


	/**	EXTRA METHODS - NOT NECESSARY **/
	#pragma region unused
	void printpix(Vec3b pix) {
		cout << (int)pix.val[0] << " " << (int)pix.val[1] << " " << (int)pix.val[2] << endl;
	};

	void bhMaskVisualisation() {
		Mat vis = Mat(finalImage.size(), DataType<Vec3b>::type);
		for (int i = 0; i < blackhole.rows; i++) {
			for (int j = 0; j < blackhole.cols; j++) {
				if (!blackhole.at<bool>(i, j)) vis.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
				else vis.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			}
		}
		namedWindow("bh", WINDOW_AUTOSIZE);
		imshow("bh", vis);
		waitKey(0);
	}

	Mat smoothingMask(int s) {
		Mat kernel = getStructuringElement(MORPH_RECT, Size(2 * s + 1, 2 * s + 1));
		dilate(blackhole, blackhole, kernel);
		Mat edge(finalImage.size(), DataType<Vec3b>::type);
		finalImage.copyTo(edge, blackhole);
		namedWindow(warp_window, WINDOW_AUTOSIZE);
		imshow(warp_window, edge);
		waitKey(0);
		GaussianBlur(edge, edge, Size(4 * s + 1, 4 * s + 1), 0, 0);
		namedWindow(warp_window, WINDOW_AUTOSIZE);
		imshow(warp_window, edge);
		waitKey(0);
		Mat rest(finalImage.size(), DataType<Vec3b>::type);
		finalImage.copyTo(rest, Mat::ones(finalImage.size(), CV_8U) - blackhole);

		rest = edge + edge + rest;
		erode(blackhole, blackhole, kernel);
		rest.copyTo(finalImage, Mat::ones(finalImage.size(), CV_8U) - blackhole);
		return rest;
	}

	Point2d interpolateSplines(uint64_t ij, double thetaCam, double phiCam) {
		vector<double> cornersCam;
		vector<Point2d> cornersCel;
		calcCornerVals(ij, cornersCel, cornersCam);

		double thetaUp = cornersCam[0];
		double thetaDown = cornersCam[2];
		double phiLeft = cornersCam[1];
		double phiRight = cornersCam[3];
		double thetaPerc = (thetaCam - thetaUp) / (thetaDown - thetaUp);
		double phiPerc = (phiCam - phiLeft) / (phiRight - phiLeft);
		Point2d cross;
		if (!(splines->CamToSpline[ij].interpolate(thetaPerc, phiPerc, cross))) {
		}
			//return interpolate(ij, thetaCam, phiCam);
		return cross;
	}

	void movieMaker(int speed) {
		Mat img(finalImage.size(), DataType<Vec3b>::type);

		for (int q = 1; q < finalImage.cols; q++) {
			cout << q << endl;
			#pragma omp parallel for
			for (int i = 0; i < finalImage.rows; i++) {
				for (int j = 0; j < finalImage.cols; j++) {
					if (blackhole.at<bool>(i, j) == 1)
						img.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
					else {
						vector<Vec3b> colors;
						for (Point pix : PixToPix[i*finalImage.cols + j]) {
							Point newpoint = Point((pix.x + q*speed) % celestSrc.cols, pix.y);
							if (newpoint.x < 0 || newpoint.y < 0) colors.push_back(Vec3b(0, 0, 0));
							else colors.push_back(celestSrc.at<Vec3b>(newpoint));
						}
						img.at<Vec3b>(i, j) = averagePix(colors);

					}
				}
			}

			namedWindow(warp_window, WINDOW_AUTOSIZE);
			imshow(warp_window, img);
			waitKey(1);
		}
	}

	void drawPixSelection(vector<Point2d>& thphivals, unordered_set<Point, hashing_func>& pixset) {
		cout << "vis" << endl;
		Mat vis = Mat(celestSrc.size(), DataType<Vec3b>::type);
		double sp = 1.*vis.cols / PI2;
		double st = 1.*vis.rows / PI;
		vector<Point2d> corners(4);
		for (int q = 0; q < 4; q++) {
			corners[q].x = fmod(thphivals[q]_phi*sp, vis.cols);
			corners[q].y = thphivals[q]_theta*st;
		}
		for (Point pt : pixset)
			vis.at<Vec3b>(pt) = Vec3b(255, 255, 255);
		for (int q = 0; q < 4; q++)
			line(vis, corners[q], corners[(q + 1) % 4], Vec3b(q * 60, 0, 255 - q * 60), 1, CV_AA);

		namedWindow("pix", WINDOW_NORMAL);
		imshow("pix", vis);
		waitKey(0);
		cout << "vis" << endl;
	}

	void drawBlocks(string filename) {
		Mat gridimg(2 * finalImage.rows + 2, 2 * finalImage.cols + 2, CV_8UC1, Scalar(255));
		double sj = 1.*gridimg.cols / grid->M;
		double si = 1.*gridimg.rows / ((grid->N - 1)*(2 - grid->equafactor) + 1);
		for (auto block : grid->blockLevels) {
			uint64_t ij = block.first;
			int level = block.second;
			int gap = pow(2, grid->MAXLEVEL - level);
			uint32_t i = i_32;
			uint32_t j = j_32;
			uint32_t k = i_32 + gap;
			uint32_t l = j_32 + gap;

			line(gridimg, Point2d(j*sj, i*si), Point2d(j*sj, k*si), Scalar(0), 1, CV_AA);
			line(gridimg, Point2d(l*sj, i*si), Point2d(l*sj, k*si), Scalar(0), 1, CV_AA);
			line(gridimg, Point2d(j*sj, i*si), Point2d(l*sj, i*si), Scalar(0), 1, CV_AA);
			line(gridimg, Point2d(j*sj, k*si), Point2d(l*sj, k*si), Scalar(0), 1, CV_AA);
		}
		namedWindow("blocks", WINDOW_AUTOSIZE);
		imshow("blocks", gridimg);
		imwrite(filename, gridimg);
		waitKey(0);
	}
	#pragma endregion

	/// <summary>
	/// Finalizes an instance of the <see cref="Distorter"/> class.
	/// </summary>
	~Distorter() {
		//delete[] pi;
		//delete[] bh;
	};
	
};