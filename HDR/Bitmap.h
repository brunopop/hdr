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

#pragma once

#include "opencv2/core/core.hpp"

#ifdef _DEBUG
#	define DEBUG
#endif

#define DEBUG

namespace bps
{
	class Bitmap
	{
	private:
		std::vector<std::vector<bool>> data;
		int height;
		int width;

	public:
		/// <summary>
		/// Default constructor. Set height and width to zero.
		/// </summary>
		Bitmap();

		/// <summary>
		/// Construct a Bitmap with the specified size and set
		/// all pixels to false.
		/// </summary>
		/// <param name="height">[in] Height of the Bitmap</param>
		/// <param name="width">[in] Width of the Bitmap</param>
		Bitmap(int height, int width);

		~Bitmap(void);

		/// <summary>
		/// Set the value of pixel (x,y).
		/// </summary>
		/// <param name="x">[in] Row index of the pixel</param>
		/// <param name="y">[in] Column index of the pixel</param>
		/// <param name="value">[in] Boolean value to set</param>
		void set(int x, int y, bool value);

		/// <summary>
		/// Get the value of pixel (x,y).
		/// </summary>
		/// <param name="x">[in] Row index of the pixel</param>
		/// <param name="y">[in] Column index of the pixel</param>
		/// <return>The boolean value of that pixel</return>
		bool get(int x, int y) const;

		/// <summary>
		/// Get the height of the Bitmap.
		/// </summary>
		/// <return>The height of the Bitmap</return>
		int getHeight(void) const;

		/// <summary>
		/// Get the width of the Bitmap.
		/// </summary>
		/// <return>The width of the Bitmap</return>
		int getWidth(void) const;

		/// <summary>
		/// Write Bitmap to image file. Wraps OpenCV's imwrite.
		/// </summary>
		/// <param name="path">[in] File path</param>
		void imwrite(const std::string& path);

		/// <summary>
		/// Compute the sum of all 1 bits in the Bitmap.
		/// </summary>
		/// <return>Sum of all 1 bits</return>
		int total(void) const;
	};
}
