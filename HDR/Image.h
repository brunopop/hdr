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

#include <opencv2/highgui/highgui.hpp>

#define TOLERANCE 6

#ifdef _DEBUG
#	define DEBUG
#endif

#define DEBUG

namespace bps
{
	class Bitmap;

	class Image : public cv::Mat
	{
	private:
		float exposureTime;	///< Exposure time. Should be extracted from the image file.
		int tolerance;		///< Tolerance value used in Greg Ward's registration algorithm.
		double threshold;	///< Threshold value computed using Greg Ward's registration algorithm.

		/// <summary>
		/// Compute the ideal threshold from the histogram.
		/// </summary>
		void computeThreshold(void);

	public:
		/// <summary>
		/// Default constructor.
		/// <summary>
		//// <param name="tolerance">[in] Tolerance value</param>
		Image(int tolerance = TOLERANCE);

		/// <summary>
		/// Construct an empty image with the specified size and type.
		/// </summary>
		/// <param name="height">[in] Height of the Image</param>
		/// <param name="width">[in] Width of the Image</param>
		/// <param name="type">[in] Type of the Image as defined in OpenCV</param>
		/// <param name="tolerance">[in] Tolerance value</param>
		Image(int height, int width, int type, int tolerance = TOLERANCE);

		/// <summary>
		/// Construct an Image from a cv::Mat.
		/// </summary>
		/// <param name="mat">[in] Mat to copy</param>
		/// <param name="tolerance">[in] Tolerance value</param>
		Image(const cv::Mat& mat, int tolerance = TOLERANCE);

		/// <summary>
		/// Construct an Image from a file. Wraps OpenCV's imread function.
		/// </summary>
		/// <param name="path">[in] Path to the file</param>
		/// <param name="flags">[in] Specifies the color type of a loaded image:
		/// CV_LOAD_IMAGE_ANYDEPTH - If set, return 16-bit/32-bit image when the input has the corresponding depth, otherwise convert it to 8-bit.
		/// CV_LOAD_IMAGE_COLOR - If set, always convert image to the color one
		/// CV_LOAD_IMAGE_GRAYSCALE - If set, always convert image to the grayscale one
		/// </param>
		/// <param name="tolerance">[in] Tolerance value</param>
		Image(const std::string& path, int flags = 1, int tolerance = TOLERANCE);

		/// <summary>
		/// Copy constructor.
		/// </summary>
		Image(const Image& img);

		/// <summary>
		/// Move constructor.
		/// </summary>
		Image(Image&& img);

		~Image(void);

		/// <summary>
		/// Set the exposure time for that image.
		/// </summary>
		/// <param name="exposureTime">Value of the exposure time</param>
		void setExposureTime(float exposureTime);

		/// <summary>
		/// Get the exposure time of that image.
		/// </summary>
		/// <return>The exposure time</return>
		float getExposureTime(void) const;

		/// <summary>
		/// Set the tolerance for that image.
		/// </summary>
		/// <param name="tolerance">Value of the tolerance</param>
		void setTolerance(int tolerance = TOLERANCE);

		/// <summary>
		/// Get the tolerance set for that image.
		/// </summary>
		/// <return>The tolerance set for that image</return>
		int getTolerance(void) const;

		/// <summary>
		/// Get the threshold computed for that image.
		/// </summary>
		/// <return>The computed threshold</return>
		double getThreshold(void) const;

		/// <summary>
		/// Subsample the image by a factor of two in each dimension
		/// (after Gaussian filtering) and put the result into
		/// a newly allocated image.
		/// </summary>
		/// <return>New image subsampled by two.</return>
		Image shrink2(void) const;

		/// <summary>
		/// Crop the image from top left corner (x,y) and with
		/// size (height x width).
		/// </summary>
		/// <param name="x">[in] Row index of the top left corner</param>
		/// <param name="y">[in] Column index of the top left corner</param>
		/// <param name="height">[in] Height of the image after cropping</param>
		/// <param name="width">[in] Width of the image after cropping</param>
		void crop(int x, int y, int height, int width);

		/// <summary>
		/// Compute the threshold bitmap associated with the
		/// image as explained in Ward (2003).
		/// </summary>
		/// <param name="bmp">[out] Output bitmap</param>
		void computeThresholdBitmap(Bitmap& bmp) const;

		/// <summary>
		/// Compute the exclusion bitmap associated with the
		/// image as explained in Ward (2003).
		/// </summary>
		/// <param name="bmp">[out] Output bitmap</param>
		void computeExclusionBitmap(Bitmap& bmp) const;

		/// <summary>
		/// Write to image file. Wraps OpenCV's imwrite.
		/// </summary>
		/// <param name="filename">[in] Path to the file</param>
		/// <return>True is operation is successful, false otherwise</return>
		bool write(const std::string& filename) const;

		/// <summary>
		/// Converts the image into an 8-bit equivalent,
		/// if the image is 16-bit float and 1 channel.
		/// </summary>
		/// <param name="mat8UC">[out] 8-bit 1-channel converted image</param>
		void convert32Fto8U(Image& mat8UC) const;
	};
}
