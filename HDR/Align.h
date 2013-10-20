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

#include <vector>

#define SHIFT_BITS 6

#ifdef _DEBUG
#	define DEBUG
#endif

#define DEBUG

namespace bps
{
	class Bitmap;
	class Image;

	/// <summary>
	/// Implementation of Greg Ward's "Fast, Robust Image Registration for
	/// Compositing High Dynamic Range Photographs from Handheld Exposures."
	/// In Journal of Graphics Tools. 2003.
	/// </summary>
	class Align
	{
	private:
		int shift_bits;				///< Maximum final offset in bits.
		std::vector<Image>* images;	///< Pointer to a vector containing the images to register.
		int **relativeShiftsX;		///< Antisymmetric matrix of relative translations in x from one image to all others in the set.
		int **relativeShiftsY;		///< Antisymmetric matrix of relative translations in y from one image to all others in the set.
		// Example:
		//   (0,0)    (1,1)    (2,2)
		//  (-1,-1)   (0,0)    (1,1)
		//  (-2,-2)  (-1,-1)   (0,0)
		bool relativeShiftsUpdated;	///< This flag avoids using <c>relativeShiftsX</c> and <c>relativeShiftsY</c> if they haven't been initialized.

		/// <summary>
		/// Compute the alignment offset between two exposure images recursively.
		/// </summary>
		/// <param name="img1">[in] First exposure image.</param>
		/// <param name="img2">[in] Second exposure image.</param>
		/// <param name="shift_bits">[in] Maximum number of bits in the final offsets.</param>
		/// <param name="shift_ret">[out] Offsets for a given level in the pyramid.</param>
		void getExpShift(const Image& img1, const Image& img2, int shift_bits, std::vector<int>& shift_ret);

		/// <summary>
		/// Shift a bitmap by <c>x</c> and <c>y</c> and put the result
		/// into the preallocated bitmap <c>out</c>.
		/// </summary>
		/// <param name="in">[in] Input bitmap to be shifted.</param>
		/// <param name="x">[in] Vertical offset.</param>
		/// <param name="y">[in] Horizontal offset.</param>
		/// <param name="out">[out] Shifted bitmap.</param>
		void bitmapShift(Bitmap const& in, int x, int y, Bitmap& out);

		/// <summary>
		/// Compute the "exclusive-or" of bm1 and bm2 and put the result into bm_ret.
		/// </summary>
		/// <param name="bm1">[in] First Bitmap.</param>
		/// <param name="bm2">[in] Second Bitmap.</param>
		/// <param name="bm_ret">[out] Output.</param>
		void bitmapXOR(Bitmap const& bm1, Bitmap const& bm2, Bitmap& bm_ret);

		/// <summary>
		/// Compute the "and" of bm1 and bm2 and put the result into bm_ret.
		/// </summary>
		/// <param name="bm1">[in] First bitmap.</param>
		/// <param name="bm2">[in] Second bitmap.</param>
		/// <param name="bm_ret">[out] Output.</param>
		void bitmapAND(Bitmap const& bm1, Bitmap const& bm2, Bitmap& bm_ret);

		/// <summary>
		/// Update the matrix of relative offsets between images. This antisymmetric
		/// matrix contains relative translations in x and in y from one image to
		/// all others in the set.
		/// Example:
		///   (0,0)    (1,1)    (2,2)
		///  (-1,-1)   (0,0)    (1,1)
		///  (-2,-2)  (-1,-1)   (0,0)
		/// This method transforms offsets between adjacent images into a matrix
		/// containing offsets between all images. Having such a matrix simplifies
		/// the cropping process.
		/// </summary>
		/// <param name="consecutiveShifts">[in] Offsets between one image and the adjacent one.</param>
		void updateRelativeShifts(std::vector<std::vector<int>>& consecutiveShifts);

		/// <summary>
		/// Determine a cropping area and crop each image in the set
		/// using the matrix of relative shifts.
		/// </summary>
		void cropImages(void);

	public:
		Align(std::vector<Image>* images, int shift_bits = SHIFT_BITS);

		~Align(void);

		/// <summary>
		/// Call Ward's registration algorithm and register the images.
		/// </summary>
		void align(void);
	};
}
