/*
	C++ HDR Copyright (C) 2013 Bruno Pop-Stefanov
	----------------------------------------------------------------------

	Disclaimer:
	This file is part of C++ HDR.

	C++ HDR is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.
	
	C++ HDR is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.
	
	You should have received a copy of the GNU General Public License
	along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
	----------------------------------------------------------------------

	Date: 10/20/2013
	Author: Bruno Pop-Stefanov (brunopop@gatech.edu)
	----------------------------------------------------------------------

	Description:
	C++ HDR is a simple library for high dynamic range imaging. It
	implements image registration (classes Image, Bitmap, and Align) [1],
	HDR computation (classes Image and HDR) [2], and tone mapping (classes
	Image and Tonemap) [3]. It uses code from OpenCV <http://opencv.org/>,
	LAPACK <http://www.netlib.org/lapack/> and jhead
	<http://www.sentex.net/~mwandel/jhead/>.
	----------------------------------------------------------------------

	References:
	[1] Ward, Greg. "Fast, robust image registration for compositing
	high dynamic range photographs from hand-held exposures." Journal
	of graphics tools 8.2 (2003): 17-30.
	[2] Debevec, Paul E., and Jitendra Malik. "Recovering high dynamic
	range radiance maps from photographs." ACM SIGGRAPH 2008 classes.
	ACM, 2008.
	[3] Reinhard, Erik, et al. "Photographic tone reproduction for digital
	images." ACM Transactions on Graphics (TOG). Vol. 21. No. 3. ACM, 2002.
	[4] Reinhard, Erik, et al. High dynamic range imaging: acquisition,
	display, and image-based lighting. Morgan Kaufmann, 2010.
	----------------------------------------------------------------------
*/

#include "Image.h"
#include "Bitmap.h"
#include "opencv2/imgproc/imgproc.hpp"

namespace bps
{
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
		cv::Mat mat = cv::Mat(height, width, type);
		mat.copyTo(*this);
		this->tolerance = tolerance;
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

	bool Image::write(const std::string& filename)
	{
		return cv::imwrite(filename, *this);
	};

	void Image::convert32Fto8U(Image& mat8UC)
	{
		if (this->type() != CV_32F)
			return;

		// First, get min and max values
		double minVal, maxVal;
		cv::minMaxLoc(*this, &minVal, &maxVal);

		double alpha = 255.0/(maxVal-minVal);
		double beta = -alpha*minVal;

		this->convertTo(mat8UC,CV_8UC1,alpha,beta);
	};
}