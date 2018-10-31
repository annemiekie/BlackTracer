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
#include "Camera.h"
using namespace cv;
using namespace std;

class Distorter
{
private:
	Grid* grid;
	Viewer* view;
	Splines* splines;
	vector<Star>* stars;
	bool splineInt;
	Mat celestSrc;
	Mat blackhole;
	Mat thetaPhiCelest, matchPixVal, matchPixPos, celSizes;
	Vec3b allcol;
	Camera* cam;
	const char* source_window = "Source image";
	const char* warp_window = "Warp";
	struct hashing_func {
		size_t operator()(const Point& key) const {
			return key.x * 32 + key.y;
		}
	};
	vector<unordered_set<Point, hashing_func>> PixToPix;
	//Better hashing?
	vector<unordered_set<Star>> pixToStar;

public:
	Mat finalImage;
	Distorter() {};

	Distorter(string filename, Grid* deformgrid, Viewer* viewer, Splines* _splines, vector<Star>* _stars, bool _splineInt, Camera* camera) {
		celestSrc = imread(filename, 1);
		grid = deformgrid;
		view = viewer;
		splineInt = _splineInt;
		if (splineInt) splines = _splines;
		stars = _stars;
		cam = camera;
		thetaPhiCelest = Mat(view->pixelheight + 1, view->pixelwidth + 1, DataType<Point2d>::type);
		celSizes = Mat(view->pixelheight, view->pixelwidth, DataType<double>::type);
		matchPixVal = Mat(view->pixelheight + 1, view->pixelwidth + 1, DataType<Vec3b>::type);
		matchPixPos = Mat(view->pixelheight + 1, view->pixelwidth + 1, DataType<Point>::type);
		finalImage = Mat(view->pixelheight, view->pixelwidth, DataType<Vec3b>::type);
		blackhole = Mat::zeros(finalImage.size(), CV_8U);
		fixTvertices();
		averageAll();
	};

	void saveImg(string filename) {
		imwrite(filename, finalImage);
	}

	void fixTvertices() {
		for (auto block : grid->blockLevels) {
			int level = block.second;
			if (level == grid->MAXLEVEL) continue;
			uint64_t ij = block.first;
			if (grid->CamToCel[ij]_phi < 0) continue;

			int gap = pow(2, grid->MAXLEVEL - level);
			uint32_t i = i_32;
			uint32_t j = j_32;
			uint32_t k = i + gap;
			uint32_t l = (j + gap)%grid->M;

			checkAdjacentBlock(ij, k_j, level, 1, 0, gap);
			checkAdjacentBlock(ij, i_l, level, 0, 1, gap);
			checkAdjacentBlock(i_l, k_l,  level, 1, 0, gap);
			checkAdjacentBlock(k_j, k_l, level, 0, 1, gap);

		}
	}

	void checkAdjacentBlock(uint64_t ij, uint64_t ij2, int level, int ud, int lr, int gap) {
		uint32_t i = i_32 + ud * gap / 2;
		uint32_t j = j_32 + lr * gap / 2;
		auto it = grid->CamToCel.find(i_j);
		if (it == grid->CamToCel.end())
			return;
		else {
			grid->CamToCel[i_j] = 1./2.*(grid->CamToCel[ij] + grid->CamToCel[ij2]);
			if (level + 1 == grid->MAXLEVEL) return;
			checkAdjacentBlock(ij, i_j, level + 1, ud, lr, gap / 2);
			checkAdjacentBlock(i_j, ij2, level + 1, ud, lr, gap / 2);
		}		
	}

	void bhMaskVisualisation() {
		Mat vis = Mat(finalImage.size(), DataType<Vec3b>::type);
		for (int i = 0; i < blackhole.rows; i++) {
			for (int j = 0; j < blackhole.cols; j++) {
				if (blackhole.at<bool>(i, j) == 0) vis.at<Vec3b>(Point(i, j)) = Vec3b(0, 0, 0);
				else vis.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			}
		}
		namedWindow("bh", WINDOW_AUTOSIZE);
		imshow("bh", vis);
		waitKey(0);
	}

	void printpix(Vec3b pix) {
		cout << (int)pix.val[0] << " " << (int)pix.val[1] << " " << (int)pix.val[2] << endl;
	};

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

	//https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch24.html
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

	void insertmaxmin(unordered_map<int, vector<int>>& maxmin, Point pix) {
		auto it = maxmin.find(pix.x);
		if (it == maxmin.end())
			maxmin[pix.x] = vector < int > {pix.y, pix.y};
		else if (maxmin[pix.x][0] < pix.y) maxmin[pix.x][0] = pix.y;
		else maxmin[pix.x][1] = pix.y;
	}

	void findStarsForPixel(vector<Point2d> thphivals, unordered_set<Star>& starSet, int i, int j) {

		bool changed2pi = false;
		if (check2PIcross(thphivals, 5)) {
			correct2PIcross(thphivals, 5);
			changed2pi = true;
		}

		// Orientation is positive if CW, negative if CCW
		double orientation = 0;
		for (int q = 0; q < 4; q++) {
			orientation += (thphivals[(q + 1) % 4]_theta - thphivals[q]_theta)
						    *(thphivals[(q + 1) % 4]_phi + thphivals[q]_phi);
		}

		int sgn = metric::sgn(orientation);
		// save the projected size on the celestial sky
		celSizes.at<double>(i, j) = fabs(orientation) * 0.5f;

		for (auto it = stars->begin(); it != stars->end(); it++) {
			Star star = *it;
			Point2d starPt = Point2d(star.theta, star.phi);
			bool checkNormal = starInPolygon(starPt, thphivals, sgn);
			bool check2pi = false;
			if (!checkNormal && star.phi < 0.05 * PI2) {
				starPt.y += PI2;
				check2pi = starInPolygon(starPt, thphivals, sgn);
			}
			if (checkNormal || check2pi) {
				star.posInPix = interpolate2(thphivals, starPt, sgn);
				starSet.insert(star);
			}
		}
	}

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
		if (check2PIcross(thphivals, 5)) {
			correct2PIcross(thphivals, 5);
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
				bool def;
				Vec3b pixcol = getPixelAtThPhi(pt, pos, def);
				if (def) {
					pixvals = { Vec3b(0, 0, 0) };
					blackhole.at<bool>(coords[0]) = 1;
					return;
				}
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

	void drawPixSelection(vector<Point2d>& thphivals, unordered_set<Point, hashing_func>& pixset) {
		cout << "vis" << endl;
		Mat vis = Mat(celestSrc.size(), DataType<Vec3b>::type);
		double sp = 1.*vis.cols / PI2;
		double st = 1.*vis.rows / PI;
		vector<Point2d> corners(4);
		for (int q = 0; q < 4; q++) {
			corners[q].x = fmod(thphivals[q]_phi*sp,vis.cols);
			corners[q].y = thphivals[q]_theta*st;
		}
		for (Point pt : pixset)
			vis.at<Vec3b>(pt) = Vec3b(255, 255, 255);
		for (int q = 0; q < 4; q++)
			line(vis, corners[q], corners[(q + 1) % 4], Vec3b(q*60, 0, 255-q*60), 1, CV_AA);
		
		namedWindow("pix", WINDOW_NORMAL);
		imshow("pix", vis);
		waitKey(0);
		cout << "vis" << endl;
	}

	void findMatchingPixels() {
		for (int i = 0; i < thetaPhiCelest.rows; i++) {
			for (int j = 0; j < thetaPhiCelest.cols; j++) {
				Point pos;
				bool checkbh;
				matchPixVal.at<Vec3b>(i, j) = getPixelAtThPhi(thetaPhiCelest.at<Point2d>(i, j), pos, checkbh);
				matchPixPos.at<Point>(i, j) = pos;

				if (checkbh && i<blackhole.rows) {
					blackhole.at<bool>(i, j) = 1;
					blackhole.at<bool>(i, (j - 1) % blackhole.cols) = 1;
					if (i > 0) {
						blackhole.at<bool>(i - 1, j) = 1;
						blackhole.at<bool>(i - 1, (j - 1) % blackhole.cols) = 1;
					}
				}
			}
		}
		//bhMaskVisualisation();
	}

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
					unordered_set<Point,hashing_func> pixset;
					findPixels(coords, colors, pixset);
					PixToPix[i * finalImage.cols + (j - 1)] = pixset;
					finalImage.at<Vec3b>(i, j - 1) = averagePix(colors);
				}
				coords[1] = coords[2];
				coords[0] = coords[3];
			}
		}
	}

	void findStars() {

		for (int q = 0; q < finalImage.rows / 8; q++) {
			cout << "-";
		} cout << "|" << endl;

		pixToStar.resize(finalImage.total());

		#pragma omp parallel for
		for (int i = 0; i < finalImage.rows; i++) {
			if (i % 8 == 0) cout << ".";

			vector<Point2d> thphivals(4);
			thphivals[1] = thetaPhiCelest.at<Point2d>(i, 0);
			thphivals[0] = thetaPhiCelest.at<Point2d>(i + 1, 0);

			for (int j = 1; j < finalImage.cols + 1; j++) {
				thphivals[2] = thetaPhiCelest.at<Point2d>(i, j);
				thphivals[3] = thetaPhiCelest.at<Point2d>(i + 1, j);

				if (blackhole.at<bool>(i, j-1) == 0) {
					unordered_set<Star> starSet;
					findStarsForPixel(thphivals, starSet, i, j - 1);
					pixToStar[i * finalImage.cols + j - 1] = starSet;
				}
				thphivals[1] = thphivals[2];
				thphivals[0] = thphivals[3];
			}
		}
	}

	void colorPixelsWithStars() {

		Mat solidAngleFrac = Mat(finalImage.rows, finalImage.cols, DataType<double>::type);
		vector<double> thetaDiff, phiDiff;
		thetaDiff.resize(finalImage.rows);
		phiDiff.resize(finalImage.cols);

		#pragma omp parallel for
		for (int i = 0; i < finalImage.rows; i++) {
			double verSize = abs(view->ver[i] - view->ver[(i + 1)]);
			thetaDiff[i] = verSize;
			for (int j = 0; j < finalImage.cols; j++) {
				if (i == 0) {
					double horSize = abs(view->hor[j] - view->hor[(j + 1) % finalImage.cols]);
					phiDiff[j] = horSize;
				}
				double solidAnglePix = phiDiff[j] * verSize;
				solidAngleFrac.at<double>(i, j) = solidAnglePix / celSizes.at<double>(i, j);
			}
		}

		#pragma omp parallel for
		for (int i = 0; i < finalImage.rows; i++) {
			for (int j = 0; j < finalImage.cols; j++) {
				if (blackhole.at<bool>(i, j) == 1) {
					finalImage.at<Vec3b>(i, j) = Vec3b(100, 0, 0);
				}
				else {
					double brightnessPix = 0;
					Point2d pix = Point2d(i + 0.5, j + 0.5);
					int start = -2;
					int stop = 3;
					if (i == 0) { start = 0; }
					if (i == 1) { start = -1; }
					if (i == finalImage.rows - 1) { stop = 1; }
					if (i == finalImage.rows - 2) { stop = 2; }
					for (int p = start; p < stop; p++) {
						int pi = p + i;
						double theta = view->hor[pi];
						double thetaD = thetaDiff[pi];

						for (int q = -2; q < 3; q++) {

							int qj = (q + j + finalImage.cols) % finalImage.cols;
							double phi = view->ver[qj];
							double phiD = phiDiff[qj];

							unordered_set<Star> pixStars = pixToStar[pi * finalImage.cols + qj];
							vector<Vec3b> colors;
							if (pixStars.size() > 0) {								
								Point2d starPix = Point2d(pi, qj);
								double frac = solidAngleFrac.at<double>(pi, qj);
								for (const Star& star : pixStars) {
									Point2d starPosPix = starPix + star.posInPix;
									double distsq = euclideanDistSq(starPosPix, pix);
									if (distsq > 6.25) continue;

									double thetaCam = theta + thetaD * star.posInPix.y;
									double phiCam = phi + phiD * star.posInPix.x;
									// m2 = m1 - 2.5(log(frac)) where frac = brightnessfraction of b2/b1;
									colors.push_back(star.color);
									double appMag = star.magnitude + 4. * log10(redshift(thetaCam, phiCam)) - 2.5 * log10(frac*gaussian(distsq));
									brightnessPix += pow(10., -appMag*0.4);
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
						if (appMagPix < 0){
							cout << "WUT " << appMagPix << endl;
						}
					}
					finalImage.at<Vec3b>(i, j) = color;//averageStars(stars);
				}
			// Size of image / number of pixels matters a lot in how to choose the gaussian and the brightness to rgb value;
			}
		}
	}

	//double calcSolidAngleFrac(int i, int j) {
	//	//probably faster to use after calculating instead of saving to matrix and doing lookup again,
	//	// but for now this is easier and clearer to read.
	//	if (i == 0) {
	//		// calculate also own row: 1
	//		if (j < grid->M - 1)
	//			solidAngleFrac.at<double>(i, (j + 1) % grid->M) = calcSolidAngleFrac(i, (j + 1) % grid->M);
	//		if (j == 0) {
	//			solidAngleFrac.at<double>(i, (j - 1) % grid->M) = calcSolidAngleFrac(i, (j - 1) % grid->M);
	//			solidAngleFrac.at<double>(i, j) = calcSolidAngleFrac(i, j);
	//		}
	//	}
	//	if (j == 0) {
	//		// calculate also first 2,0 and 2,1
	//	}
	//	// calculate 2,2
	//	solidAngleFrac.at<double>(i + 1, (j + 1) % grid->M) = calcSolidAngleFrac(i + 1, j);
	//}

	double euclideanDistSq(Point2d& p, Point2d& q) {
		Point2d diff = p - q;
		return diff.x*diff.x + diff.y*diff.y;
	}

	double gaussian(double dist) {
		return exp(-1.*dist);//0.8*exp(-2*dist);
	}

	double redshift(double theta, double phi) {
		double yCam = sin(theta) * sin(phi);
		double part = (1. - cam->speed*yCam);
		double betaPart = sqrt(1 - cam->speed * cam->speed) / part;

		double yFido = (-yCam + cam->speed) / part;
		double eF = 1. / (cam->alpha + cam->w * cam->wbar * yFido);
		double b = eF * cam->wbar * yFido;

		return 1. / (betaPart * (1 - b*cam->w) / cam->alpha);
	}

	void rayInterpolater() {

		cout << "Interpolating..." << endl;
		time_t start = time(NULL);
		//for (int x=0; x<10; x++) 
		findThetaPhiCelest();
		cout << " time: " << (time(NULL) - start) << endl;
		findMatchingPixels();

		cout << "Filling pixelvalues..." << endl;
		start = time(NULL);
		//findPixelValues();
		findStars();
		cout << " time: " << (time(NULL) - start) << endl;
		colorPixelsWithStars();
		
		namedWindow(warp_window, WINDOW_AUTOSIZE);
		Mat smoothImage(finalImage.size(), DataType<Vec3b>::type);
		imshow(warp_window, finalImage);
		waitKey(0);
	};

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
							if (newpoint.x < 0 || newpoint.y < 0) colors.push_back(Vec3b(0,0,0));
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

	Mat smoothingMask(int s) {
		//Mat mask(finalImage.size(), CV_8U, Scalar(0));
		//Mat inversemask = Mat::ones(finalImage.size(), CV_8U) - blackhole;
		Mat kernel = getStructuringElement(MORPH_RECT, Size(2*s+1,2*s+1));
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

		rest = edge + edge+ rest;
		erode(blackhole, blackhole, kernel);
		rest.copyTo(finalImage, Mat::ones(finalImage.size(), CV_8U) - blackhole);
		
		//GaussianBlur(blackhole, mask, Size(2 * s + 1, 2 * s + 1),0 ,0 );
		//threshold(mask, mask, 0.5,1,THRESH_BINARY);
		return rest;
	}

	int findStartingBlock(int& lvlstart, bool& top, double phi, double& theta) {
		if (!grid->equafactor) {
			lvlstart = 1;
			top = true;
			if (theta > (PI1_2)) {
				top = false;
				theta = PI - theta;
			}
			if (phi > PI) {
				if (phi > 3.f * PI1_2) return 3;
				else return 2;
			}
			else if (phi > PI1_2) return 1;
		}
		else {
			if (phi > PI) return 1;
		}
		return 0;
	}

	void findThetaPhiCelest() {
		#pragma omp parallel for
		for (int q = 0; q < view->ver.size() / 8; q++) {
			cout << "-";
		} cout << "|" << endl;

		#pragma omp parallel for
		for (int t = 0; t < view->ver.size(); t++) {
			if (t % 8 == 0) cout << ".";
			for (int p = 0; p < view->hor.size(); p++) {
				double theta = view->ver[t];
				double phi = view->hor[p];

				int lvlstart = 0;
				bool top;
				int quarter = findStartingBlock(lvlstart, top, phi, theta);
				uint64_t ij = findBlock(grid->startblocks[quarter], theta, phi, lvlstart);
				Point2d thphi = Point2d(-100, -100);

				if (splineInt) {
					auto it = splines->CamToSpline.find(ij);
					if (it != splines->CamToSpline.end())
						thphi = interpolateSplines(ij, theta, phi);
					else
						thphi = interpolate(ij, theta, phi);
				}
				else
					thphi = interpolate(ij, theta, phi);

				if (thphi != Point2d(-1,-1) && !grid->equafactor) {
					if (!top) thphi_theta = PI - thphi_theta;
				}

				thetaPhiCelest.at<Point2d>(t, p) = thphi;
			}
		}
	};

	bool check2PIcross(vector<Point2d>& spl, float factor) {
		for (int i = 0; i < spl.size(); i++) {
			if (spl[i]_phi > PI2*(1. - 1. / factor))
				return true;
		}
		return false;
	};

	void correct2PIcross(vector<Point2d>& spl, float factor) {
		for (int i = 0; i < spl.size(); i++) {
			if (spl[i]_phi < PI2*(1. / factor) && spl[i]_phi >= 0)
				spl[i]_phi += PI2;
		}
	};

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

	Point2d interpolate2(vector<Point2d>& cornersCel, Point2d starPos, int sgn) {

		Point2d cel0 = cornersCel[0];
		Point2d cel1 = cornersCel[1];
		Point2d cel2 = cornersCel[2];
		Point2d cel3 = cornersCel[3];

		double error = 0.00001;

		Point2d mid = (cel1 + cel2 + cel3 + cel0) / 4.f;
		double starInPixY = 0.5f;
		double starInPixX = 0.5f;

		double perc = 0.5f;
		int count = 0;
		while ((fabs(starPos.x - mid.x) > error) || (fabs(starPos.x - mid.x) > error) && count < 100) {
			count++;
			Point2d half01 = (cel0 + cel1) / 2.f;
			Point2d half23 = (cel2 + cel3) / 2.f;
			Point2d half12 = (cel2 + cel1) / 2.f;
			Point2d half03 = (cel0 + cel3) / 2.f;

			Point2d line01to23 = half23 - half01;
			Point2d line03to12 = half12 - half03;

			Point2d line01toStar = starPos - half01;
			Point2d line03toStar = starPos - half03;

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

		return Point2d(starInPixX, starInPixY);
	}

	Point2d interpolate(uint64_t ij, double thetaCam, double phiCam) {
		vector<double> cornersCam;
		vector<Point2d> cornersCel;
		calcCornerVals(ij, cornersCel, cornersCam);

		double error = 0.00001;
		double thetaUp = cornersCam[0];
		double thetaDown = cornersCam[2];
		double phiLeft = cornersCam[1];
		double phiRight = cornersCam[3];

		if (check2PIcross(cornersCel, 5.)) correct2PIcross(cornersCel,5.);

		Point2d lu = cornersCel[0];
		Point2d ru = cornersCel[1];
		Point2d ld = cornersCel[2];
		Point2d rd = cornersCel[3];
		if (lu == Point2d(-1, -1) || ru == Point2d(-1, -1) || ld == Point2d(-1, -1) || rd == Point2d(-1, -1)) return Point2d(-1, -1);

		double thetaMid = 1000;
		double phiMid = 1000;

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
			} else {
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

	uint64_t findBlock(uint64_t ij, double theta, double phi, int level) {
	
		if ( grid->blockLevels[ij] <= level) {
			return ij;
		} else {
			int gap = (int)pow(2, grid->MAXLEVEL - level);
			uint32_t i = i_32;
			uint32_t j = j_32;
			uint32_t k = i + gap / 2;
			uint32_t l = j + gap / 2;
			double thHalf = PI2*k / grid->M;
			double phHalf = PI2*l / grid->M;
			if (thHalf < theta) i = k;
			if (phHalf < phi) j = l;
			return findBlock(i_j, theta, phi, level+1);
		}
	};

	Vec3b getPixelAtThPhi(Point2d thphi, Point& pos, bool& checkbh) {
		double theta = thphi_theta;
		double phi = thphi_phi;
		checkbh = false;
		if (theta < 0 || phi < 0) {
			checkbh = true;
			pos = Point(-1, -1);
			return Vec3b(0, 0, 0);
		}
		// WHY WRAPTOPI NECESSARY??
		metric::wrapToPi(theta, phi);
		
		int posx = (int) floor(1. * celestSrc.cols * phi / (PI2));
		int posy = (int) floor(1. * celestSrc.rows * theta / PI);

		return celestSrc.at<Vec3b>(posy, posx);
	};

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
		if (!(splines->CamToSpline[ij].interpolate(thetaPerc, phiPerc, cross)))
			return interpolate(ij, thetaCam, phiCam);
		return cross;
	}

	~Distorter(){};
	
	/*void undistortImg() {
	for (int i = 0; i < finalImage.rows; i++) {
	Vec3b lu = getPixelAtThPhi(Point2d(view->ver[i], view->hor[0]));
	Vec3b ld = getPixelAtThPhi(Point2d(view->ver[i + 1], view->hor[0]));
	for (int j = 1; j < finalImage.cols + 1; j++) {
	Vec3b ru = getPixelAtThPhi(Point2d(view->ver[i], view->hor[j]));
	Vec3b rd = getPixelAtThPhi(Point2d(view->ver[i + 1], view->hor[j]));
	vector<Vec3b> pixels{ lu, ld, ru, rd };
	finalImage.at<Vec3b>(Point(j - 1, i)) = averagePix(pixels, i, j);

	lu = ru;
	ld = rd;
	}
	}
	namedWindow(warp_window, WINDOW_AUTOSIZE);
	imshow(warp_window, finalImage);
	waitKey(0);
	}*/

	/*void rayInterpolaterx() {
	//cout << "interpolating start" << endl;
	findThetaPhiCelest();
	//cout << "found thetaphicelest" << endl;
	Mat matchPixVal = Mat(view->pixelheight + 1, view->pixelwidth + 1, DataType<Vec3b>::type);
	Mat matchPixPos = Mat(view->pixelheight + 1, view->pixelwidth + 1, DataType<Point>::type);
	findMatchingPixels(matchPixVal, matchPixPos);

	for (int i = 0; i < finalImage.rows; i++) {
	//cout << i << endl;
	vector<Point> coords(4);

	Vec3b lu = getPixelAtThPhi(thetaPhiCelest.at<Point2d>(Point(0,i)));
	Vec3b ld = getPixelAtThPhi(thetaPhiCelest.at<Point2d>(Point(0, i + 1)));
	for (int j = 1; j < finalImage.cols+1; j++) {
	//cout << j << endl;
	Vec3b ru = getPixelAtThPhi(thetaPhiCelest.at<Point2d>(Point(j, i)));
	Vec3b rd = getPixelAtThPhi(thetaPhiCelest.at<Point2d>(Point(j, i + 1)));

	vector<Vec3b> pixels{lu,ld,ru,rd};
	//cvtColor(rgb4, lab4, CV_BGR2Lab);
	finalImage.at<Vec3b>(Point(j - 1, i)) = averagePix(pixels, i, j);

	lu = ru;
	ld = rd;
	}
	}
	Mat mask = smoothingMask(50);
	Mat smoothImage = Mat(finalImage.size(), DataType<Vec3b>::type);

	//cvtColor(finalImage, finalImage, COLOR_Lab2BGR);
	finalImage.copyTo(smoothImage, mask);

	//namedWindow(warp_window, WINDOW_AUTOSIZE);
	//imshow(warp_window, finalImage);
	//waitKey(0);
	};*/
};