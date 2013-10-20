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

#include "Bitmap.h"
#include "opencv2/highgui/highgui.hpp"

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
		// Nothing to free
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
}
