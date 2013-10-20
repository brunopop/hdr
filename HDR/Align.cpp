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

#include "Align.h"
#include "Bitmap.h"
#include "Image.h"
#include <iostream>

namespace bps
{
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
		std::cout << "Registering input images..." << std::endl;

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
}
