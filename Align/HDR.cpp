// Author: Bruno Pop-Stefanov
// Date: 9/15/2013
// Description: Implementation of the following papers:
//  - "Fast, Robust Image Registration for Compositing High
//    Dynamic Range Photographs from Handheld Exposures" by Greg Ward
//    (Journal of graphics tools 8.2, 2003).
//  - "Recovering High Dynamic Range Radiance Maps from Photographs"
//    by Paul E. Debevec and Jitendra Malik (SIGGRAPH 1997).
//  - "Photographic Tone Reproduction for Digital Images" by
//    Erik Reinhard et al. (ACM Transactions on Graphics 2002).
// Using OpenCV, LAPACK, and jhead.

#include "HDR.h"
#include <iostream>
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "lapacke.h"

namespace bps
{
	Bitmap::Bitmap()
	{
		height = 0;
		width = 0;
	};

	Bitmap::Bitmap(int height, int width)
	{
		this->height = height;
		this->width = width;

		data.resize(height);
		for (int i=0; i<height; i++)
			data[i].resize(width, false);
	};

	Bitmap::~Bitmap(void)
	{
	};

	void Bitmap::set(int x, int y, bool value)
	{
		if (x >= 0 && x < height && y >= 0 && y < width)
			data[x][y] = value;
	};

	bool Bitmap::get(int x, int y) const
	{
		if (x >= 0 && x < height && y >= 0 && y < width)
			return data[x][y];
		else
			return false; // return false if we are outside the bitmap's boundaries
	};

	int Bitmap::getHeight(void) const
	{
		return height;
	};

	int Bitmap::getWidth(void) const
	{
		return width;
	};

	void Bitmap::imwrite(const std::string& path)
	{
		cv::Mat tmp(height, width, CV_8U);
		for (int i=0; i<height; i++)
		{
			unsigned char *p = tmp.ptr<unsigned char>(i);
			for (int j=0; j<width; j++)
			{
				if (data[i][j])
					p[j] = 255;
				else
					p[j] = 0;
			}
		}
		cv::imwrite(path, tmp);
	};

	int Bitmap::total(void) const
	{
		int sum = 0;
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				if (get(i,j)) sum++;
			}
		}
		return sum;
	};

	void Image::computeThreshold(void)
	{
		if (empty())
			throw std::exception("Error in Image::computeThreshold(): no matrix data.");

		int chan = channels();
		int typ = type() - ((chan-1) << CV_CN_SHIFT);
		if (typ != CV_8U && typ != CV_8S)
			throw std::exception("Error in Image::computeThreshold(): image must be 8-bit.");

		// Compute histogram
		int histogram[256] = {0};
		for (int i=0; i<rows; i++)
		{
			const uchar *p = this->ptr<const uchar>(i);
			for (int j=0; j<cols; j++)
			{
				int value = 0;
				switch (chan)
				{
				case 1:
					histogram[p[j]]++;
					break;
				case 3:
					//histogram[p[3*j + 1]]++; // take the green channel
					value = (54*p[3*j + 2] + 183*p[3*j + 1]+19*p[3*j + 0])/256;
					histogram[value]++;
					break;
				default:
					throw std::exception("Error in Image::computeThreshold(): image must be either grayscale or RGB.");
				}
			}
		}

		// Transform histogram into ordered array of vaues
		int* values = new int[rows*cols];
		int k = 0;
		for (int i=0; i<256; i++)
		{
			for (int j=0; j<histogram[i]; j++)
			{
				values[k++] = i;
			}
		}

		// Read median value from ordered array
		double position = double(rows*cols+1)/2.0;
		int _position_ = (int)std::floor(position);
		if (position == (double)_position_)
		{
			// odd number of pixels
			threshold = values[_position_];
		}
		else
		{
			// even number of pixels
			threshold = double(values[_position_] + values[_position_+1])/2.0;
		}
	}

	Image::Image(int tolerance) : cv::Mat()
	{
		this->tolerance = tolerance;
		this->threshold = -1;
		exposureTime = -1;
	};

	Image::Image(int height, int width, int type, int tolerance)
	{
		cv::Mat::Mat(height, width, type);
		this->tolerance = tolerance;
		this->tolerance = -1;
		exposureTime = -1;
	};

	Image::Image(cv::Mat& mat, int tolerance) : cv::Mat(mat)
	{
		this->tolerance = tolerance;
		computeThreshold();
		exposureTime = -1;
	};

	Image::Image(const std::string& path, int flags, int tolerance)
	{
		cv::Mat mat = cv::imread(path, flags);
		mat.copyTo(*this);
		this->tolerance = tolerance;
		computeThreshold();
		exposureTime = -1;
	};

	Image::~Image(void)
	{
	};

	void Image::setExposureTime(float exposureTime)
	{
		this->exposureTime = exposureTime;
	};

	float Image::getExposureTime(void) const
	{
		return exposureTime;
	};

	void Image::setTolerance(int tolerance)
	{
		this->tolerance = tolerance;
	};

	int Image::getTolerance(void) const
	{
		return tolerance;
	};

	double Image::getThreshold(void) const
	{
		return threshold;
	};

	Image Image::shrink2(void) const
	{
		Image sub;
		cv::pyrDown(*this, sub);
		sub.computeThreshold();
		sub.setTolerance(this->tolerance);
		return sub;
	};

	void Image::crop(int x, int y, int height, int width)
	{
		if (height+x > rows || width+y > cols)
			throw std::exception("Error in Image::crop(): crop area must be within image borders.");

		Image tmp(height, width, type(), tolerance);
		(*this)(cv::Range(x,x+height), cv::Range(y,y+width)).copyTo(tmp);
		tmp.copyTo(*this);
	};

	void Image::computeThresholdBitmap(Bitmap& bmp) const
	{
		bmp = Bitmap(rows, cols);
		if (threshold < 0)
			throw std::exception("Error in Image::computeThresholdBitmap(): no threshold value in image.");

#pragma omp parallel for
		for (int i=0; i<bmp.getHeight(); i++)
		{
			const uchar *p = ptr<const uchar>(i);
			for (int j=0; j<bmp.getWidth(); j++)
			{
				int value;
				switch (channels())
				{
				case 1:
					value = p[j];
					break;
				case 3:
					value = (54*p[3*j + 2] + 183*p[3*j + 1]+19*p[3*j + 0])/256;
					break;
				default:
					throw std::exception("Error in Image::computeThresholdBitmap(): input image must be either grayscale or RGB.");
				}
				// The bitmap is already initialized to 0
				// so we only need to set 1 values.
				if (value > threshold) bmp.set(i, j, true);
			}
		}
	};

	void Image::computeExclusionBitmap(Bitmap& bmp) const
	{
		bmp = Bitmap(rows, cols);
		if (threshold < 0)
			throw std::exception("Error in Image::computeExclusionBitmap(): no threshold value in image.");

#pragma omp parallel for
		for (int i=0; i<bmp.getHeight(); i++)
		{
			const uchar *p = ptr<const uchar>(i);
			for (int j=0; j<bmp.getWidth(); j++)
			{
				int value;
				switch (channels())
				{
				case 1:
					value = p[j];
					break;
				case 3:
					value = (54*p[3*j + 2] + 183*p[3*j + 1]+19*p[3*j + 0])/256;
					break;
				default:
					throw std::exception("Error in Image::computeExclusionBitmap(): input image must be either grayscale or RGB.");
				}
				// The bitmap is already initialized to 0
				// so we only need to set 1 values.
				if (value > threshold+tolerance || value < threshold-tolerance) bmp.set(i, j, true);
			}
		}
	};

	void Align::getExpShift(const Image& img1, const Image& img2, int shift_bits, std::vector<int>& shift_ret)
	{
		std::vector<int> cur_shift(2, 0);
		int i, j;
		if (shift_bits > 0)
		{
			Image sml_img1 = img1.shrink2();
			Image sml_img2 = img2.shrink2();
			getExpShift(sml_img1, sml_img2, shift_bits-1, cur_shift);
			cur_shift[0] *= 2;
			cur_shift[1] *= 2;
		}
		else
		{
			cur_shift[0] = cur_shift[1] = 0;
		}
		Bitmap tb1; img1.computeThresholdBitmap(tb1);
		Bitmap eb1; img1.computeExclusionBitmap(eb1);
		Bitmap tb2; img2.computeThresholdBitmap(tb2);
		Bitmap eb2; img2.computeExclusionBitmap(eb2);
		int min_err = img1.rows * img1.cols;
		for (i=-1; i<=1; i++)
		{
			for (j=-1; j<=1; j++)
			{
				if (min_err > 0)
				{
					int xs = cur_shift[0] + i;
					int ys = cur_shift[1] + j;
					Bitmap shifted_tb2(img1.rows, img1.cols);
					Bitmap shifted_eb2(img1.rows, img1.cols);
					Bitmap diff_b(img1.rows, img1.cols);
					bitmapShift(tb2, xs, ys, shifted_tb2);
					bitmapShift(eb2, xs, ys, shifted_eb2);
					bitmapXOR(tb1, shifted_tb2, diff_b);
					bitmapAND(diff_b, eb1, diff_b);
					bitmapAND(diff_b, shifted_eb2, diff_b);
					int err = diff_b.total();
					if (err < min_err)
					{
						shift_ret[0] = xs;
						shift_ret[1] = ys;
						min_err = err;
					}
				}
			} // end for j
		} // end for i
	};

	void Align::bitmapShift(Bitmap const& in, int x, int y, Bitmap& out)
	{
		if (in.getHeight() != out.getHeight() || in.getWidth() != out.getWidth())
			throw std::exception("Error in Align::bitmapShift(): input and output bitmaps must be pre-defined and have the same size.");

#pragma omp parallel for
		for (int i=0; i<out.getHeight(); i++)
		{
			for (int j=0; j<out.getWidth(); j++)
			{
				// We don't need to check for boundaries because Bitmap::get()
				// already returns false if we are out of them.
				out.set(i, j, in.get(i-x,j-y));
			}
		}
	};

	void Align::bitmapXOR(Bitmap const& bm1, Bitmap const& bm2, Bitmap& bm_ret)
	{
		if (bm1.getHeight() == bm2.getHeight() &&
			bm1.getWidth() == bm2.getWidth() &&
			bm2.getHeight() == bm_ret.getHeight() &&
			bm2.getWidth() == bm_ret.getWidth())
		{
#pragma omp parallel for
			for (int i=0; i<bm_ret.getHeight(); i++)
			{
				for (int j=0; j<bm_ret.getWidth(); j++)
				{
					bm_ret.set(i, j, bm1.get(i,j) ^ bm2.get(i,j) );
				}
			}
		}
		else
		{
			throw std::exception("Error in Align::bitmapXOR(): input bitmaps must have the same size.");
		}
	};

	void Align::bitmapAND(Bitmap const& bm1, Bitmap const& bm2, Bitmap& bm_ret)
	{
		if (bm1.getHeight() == bm2.getHeight() &&
			bm1.getWidth() == bm2.getWidth() &&
			bm2.getHeight() == bm_ret.getHeight() &&
			bm2.getWidth() == bm_ret.getWidth())
		{
#pragma omp parallel for
			for (int i=0; i<bm_ret.getHeight(); i++)
			{
				for (int j=0; j<bm_ret.getWidth(); j++)
				{
					bm_ret.set(i, j, bm1.get(i,j) & bm2.get(i,j) );
				}
			}
		}
		else
		{
			throw std::exception("Error in Align::bitmapAND(): input bitmaps must have the same size.");
		}
	};

	void Align::updateRelativeShifts(std::vector<std::vector<int>>& consecutiveShifts)
	{
		if (images == nullptr || images->size() == 0)
			throw std::exception("Error in Align::updateRelativeShifts(): empty image set.");

		relativeShiftsX = new int*[images->size()];
		relativeShiftsY = new int*[images->size()];
		for (unsigned int i=0; i<images->size(); i++)
		{
			relativeShiftsX[i] = new int[images->size()];
			relativeShiftsY[i] = new int[images->size()];
		}

		// Fill upper triangle and diagonal values
		for (unsigned int i=0; i<images->size(); i++)
		{
			for (unsigned int j=i; j<images->size(); j++)
			{
				if (i == j)
				{
					// Translation between current image and itself is 0
					relativeShiftsX[i][j] = 0;
					relativeShiftsY[i][j] = 0;
				}
				else
				{
					// Sum of previous value and shift between previous image and current image
					relativeShiftsX[i][j] = relativeShiftsX[i][j-1] + consecutiveShifts[j-1][0];
					relativeShiftsY[i][j] = relativeShiftsY[i][j-1] + consecutiveShifts[j-1][1];
				}
			}
		}

		// Fill lower triangle
		for (unsigned int i=1; i<images->size(); i++)
		{
			for (unsigned int j=0; j<i; j++)
			{
				// Antisymmetric matrix
				relativeShiftsX[i][j] = -relativeShiftsX[j][i];
				relativeShiftsY[i][j] = -relativeShiftsY[j][i];
			}
		}

#ifdef DEBUG
		std::cout << "Matrix of relative shifts:" << std::endl;
		for (unsigned int i=0; i<images->size(); i++)
		{
			std::cout << "row " << i << ":";
			for (unsigned int j=0; j<images->size(); j++)
			{
				std::cout << "\t(" << relativeShiftsX[i][j] << "," << relativeShiftsX[i][j] << ")";
			}
			std::cout << std::endl;
		}
#endif
		relativeShiftsUpdated = true;
	};

	void Align::cropImages(void)
	{
		if (!relativeShiftsUpdated)
			return;

		// Crop each image according to shift values
		for (unsigned int i=0; i<images->size(); i++)
		{
			// Min and max values of translations between
			// current image and all others define the cropping area
			int minX = 0, minY = 0, maxX = 0, maxY = 0;
			for (unsigned int j=0; j<images->size(); j++)
			{
				if (relativeShiftsX[i][j] < minX)
				{
					minX = relativeShiftsX[i][j];
				}
				if (relativeShiftsX[i][j] > maxX)
				{
					maxX = relativeShiftsX[i][j];
				}
				if (relativeShiftsY[i][j] < minY)
				{
					minY = relativeShiftsY[i][j];
				}
				if (relativeShiftsY[i][j] > maxY)
				{
					maxY = relativeShiftsY[i][j];
				}
			}

			// Cropping area
			int x, height;
			if (maxX > 0)
			{
				x = maxX;
			}
			else
			{
				x = 0;
			}
			if (minX < 0)
			{
				height = images->at(i).rows - x + minX;
			}
			else
			{
				height = images->at(i).rows - x;
			}

			int y, width;
			if (maxY > 0)
			{
				y = maxY;
			}
			else
			{
				y = 0;
			}
			if (minY < 0)
			{
				width = images->at(i).cols - y + minY;
			}
			else
			{
				width = images->at(i).cols - y;
			}

#ifdef DEBUG
			std::cout << "image " << i << " (" << (*images)[i].rows << "x" << (*images)[i].cols << "):"
				<< "\n\tx min: " << minX << "\tx max: " << maxX
				<< "\n\ty min: " << minY << "\ty max: " << maxY
				<< "\n\tcrop area: x = " << x
				<< "\n\t           height = " << height
				<< "\n\t           y = " << y
				<< "\n\t           width = " << width << std::endl;
#endif

			// Crop image
			(*images)[i].crop(x, y, height, width);
		} // end for all images
	};

	Align::Align(std::vector<Image>* images, int shift_bits)
	{
		this->images = images;
		this->shift_bits = shift_bits;
		relativeShiftsUpdated = false;
	};

	Align::~Align(void)
	{
	};

	void Align::align(void)
	{
		// Vector containing the shift values in x and y between images i and i+1
		std::vector<std::vector<int>> consecutiveShifts(images->size()-1, std::vector<int>(2,0));

		// Compute the offsets between one image and the next one in the list
		for (unsigned int i=0; i<images->size()-1; i++)
		{
			// Calculate the consecutive shifts in x and y between the current image and the next one
			std::cout << "Calculating shift between image " << i << " and image " << i+1 << "..." << std::endl;
			getExpShift((*images)[i], (*images)[i+1], shift_bits, consecutiveShifts[i]);
#ifdef DEBUG
			std::cout << "Shift between image " << i << " and image " << i+1 << ":\n"
				<< "\tx:  " << consecutiveShifts[i][0] << "\n"
				<< "\ty:  " << consecutiveShifts[i][1] << std::endl;
#endif
		} // end for all image pairs

		// Build a matrix of offsets from any image to any other one.
		updateRelativeShifts(consecutiveShifts);

		// Use this matrix to crop the images so that they
		// have the same size and are correctly registered
		cropImages();
	};

	void HDR::gsolve(std::vector<std::vector<uchar>>& z, std::vector<double>& g, std::vector<double>& lE)
	{
		if (z.size() != P || z[0].size() != N)
			throw std::exception("Error in HDR::gsolve(): input matrix must be (NxP).");

		int n = 256;
		//lapack_int nlines = N*P+n+1, ncols = n+N;
		int nlins = N*P+n+1, ncols = n+N;

		double* b = new double[nlins];
		double** A = new double*[nlins];
		for (int k=0; k<nlins; k++)
		{
			b[k] = 0;
			A[k] = new double[ncols];
			for (int l=0; l<ncols; l++)
			{
				A[k][l] = 0;
			}
		}

		// Include the data-fitting equations
		int k = 0;
		// For all pixel locations
		for (int i=0; i<N; i++)
		{
			// For all images
			for (int j=0; j<P; j++)
			{
				int wij = weight(z[j][i] + 1);
				A[k][z[j][i]+1] = wij;
				A[k][n+i] = -wij;
				b[k] = wij * B[j];
				k++;
			}
		}

		// Fix the curve by setting its middle value to 0
		A[k][128] = 1;
		k++;

		// Include the smoothness equations
		for (int i=0; i<n-2; i++)
		{
			A[k][i] = lambda * weight(i+1);
			A[k][i+1] = -2.0*lambda*weight(i+1);
			A[k][i+2] = lambda * weight(i+1);
			k++;
		}

		// Reshape matrix (in row order) so that it can be used with lapack
		double* system = new double[nlins*ncols];
		for (int k=0; k<nlins*ncols; k++)
		{
			int i = k / ncols;
			int j = k % ncols;
			system[k] = A[i][j];
		}

		// Solve the system using SVD
		int sz = std::min(nlins, ncols);
		double* s = new double[sz];
		lapack_int rank;
		LAPACKE_dgelsd(LAPACK_ROW_MAJOR, nlins, ncols, 1, system, ncols, b, 1, s, -1.0, &rank);

		// Log exposure for pixel values 0 through 255
		g.resize(n, 0.0);
		for (int i=0; i<n; i++)
		{
			g[i] = b[i];
		}

		// Log film irradiance for every sample pixel
		lE.resize(nlins-n, 0.0);
		for (int i=0; i<nlins-n; i++)
		{
			lE[i] = b[n+i];
		}
	};

	/// <summary>
	/// Weighting function <c>w(z)</c> to emphasize the smoothness
	/// and fitting terms toward the middle of the curve
	/// </summary>
	int HDR::weight(int z)
	{
		if (double(z) > .5*(Zmin + Zmax))
		{
			return Zmax-z;
		}
		else
		{
			return z-Zmin;
		}
	};

	void HDR::sample(std::vector<std::vector<uchar>>& zRed,
		std::vector<std::vector<uchar>>& zGreen,
		std::vector<std::vector<uchar>>& zBlue)
	{
		zRed.resize(P); zGreen.resize(P); zBlue.resize(P);

		// Sample points every step
		int step = height*width / N;

		// For all exposures
		for (int p=0; p<P; p++)
		{
			// For all pixels
			int n = 0;
			for (int k=0; k<height*width; k++)
			{
				// Sample a pixel every step
				if (k % step == 0)
				{
					// Pixel coordinates
					int i = k / width;
					int j = k % width;

					// Pointer to the corresponding row
					uchar* img = (*images)[p].ptr<uchar>(i);

					// OpenCV stores images as BGR
					zBlue[p].push_back(img[3*j + 0]);
					zGreen[p].push_back(img[3*j + 1]);
					zRed[p].push_back(img[3*j + 2]);
					n++;
				}
			} // end for k
#ifdef DEBUG
			std::cout << "HDR::sample(): image num " << p << ": number of samples is N=" << n << std::endl;
#endif
		} // end for all images
	};

	HDR::HDR(std::vector<Image>* images, double lambda)
	{
		this->images = images;
		this->lambda = lambda;
		P = images->size();
		if (P <= 0) throw std::exception("Error in HDR::HDR(): there must be at least 2 photos.");

		height = (*images)[0].rows;
		width = (*images)[0].cols;

		// 8-bit images
		Zmin = 0; Zmax = 255;

		// Given measurements of N pixels in P photographs,
		// we have to solve for N values of ln(Ei) and (Zmax-Zmin) samples
		// of g. To ensure a sufficiently overdetermined system, we want
		// N*(P-1) > (Zmax-Zmin).
		N = (int) ceil( 2.0*double(Zmax - Zmin)/double(P-1) );

		// B is the log delta t, or log shutter speed, for each image in the set
		for (int j=0; j<P; j++)
		{
			B.push_back(std::log((double) images->at(j).getExposureTime()));
		}
	};

	HDR::~HDR(void)
	{
	};

	void HDR::responseFunction(void)
	{
		// Sample points
		std::vector<std::vector<uchar>> zRed, zGreen, zBlue;
		sample(zRed, zGreen, zBlue);

		// Solve the objective function for the red channel
		std::vector<double> lERed;
		gsolve(zRed, gRed, lERed);

		// Solve the objective function for the green channel
		std::vector<double> lEGreen;
		gsolve(zGreen, gGreen, lEGreen);

		// Solve the objective function for the blue channel
		std::vector<double> lEBlue;
		gsolve(zBlue, gBlue, lEBlue);
	};

	void HDR::radianceMap(void)
	{
		// For all exposures
		for (int i=0; i<P; i++)
		{
		}
	};

}
