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

#include "Tonemap.h"
#include <iostream>

namespace bps
{
	inline int round(float x)
	{
		return int(x < 0.0 ? std::ceil(x - 0.5f) : std::floor(x + 0.5f));
	};

	void Tonemap::computeLuminance(void)
	{
		luminanceMap = Image(height, width, CV_32F);
		// Take luminance of first pixel as min value
		minLum = 0.2126f * radianceMap.ptr<float>(0)[2] + 0.7152f * radianceMap.ptr<float>(0)[1] + 0.0722f * radianceMap.ptr<float>(0)[0];
		// Set all other values to zero
		maxLum = 0;
		avgLum = 0;
		logAvgLum = 0;
		int numNotNaNPx = 0;	// number of pixels that are not NaN
		int numLogPx = 0; // number of non-zero pixels

		// For all pixels
		for (int i=0; i<height; i++)
		{
			// Pointer to the ith row of the luminance map
			float* p = luminanceMap.ptr<float>(i);
			// Pointer to the ith row of the radiance map
			float* e = radianceMap.ptr<float>(i);
			for (int j=0; j<width; j++)
			{
				// Compute luminance value
				p[j] = 0.2126f * e[3*j + 2] + 0.7152f * e[3*j + 1] + 0.0722f * e[3*j + 0];
				// Compute min and max luminance values
				if (p[j] > maxLum) maxLum = p[j];
				if (p[j] < minLum) minLum = p[j];
				// Compute average luminance avoiding NaN values
				if (cvIsNaN(p[j]) == 0)
				{
					avgLum += long double(p[j]);
					numNotNaNPx++;
				}
				else
				{
#ifdef DEBUG
					std::cout << "Warning! Pixel location (" << i << "," << j << ") in radiance map is NaN."
						<< "\n\tRed channel value is " << e[3*j + 2]
					<< "\n\tGreen channel value is " << e[3*j + 1]
					<< "\n\tBlue channel value is " << e[3*j + 0]
					<< std::endl;
#endif
				}
				// Compute log average for non-zero pixels
				if (p[j] > 0)
				{
					logAvgLum += log(p[j]);
					numLogPx++;
				}
			}
		}
		// Divide by total number of non-NaN pixels
		avgLum /= double(numNotNaNPx);
		// Divide by total number of non-zero pixels
		logAvgLum = exp(logAvgLum/long double(numLogPx));
	};

	Tonemap::Tonemap(Image& radianceMap)
	{
		this->radianceMap = radianceMap;
		height = radianceMap.rows;
		width = radianceMap.cols;
		computeLuminance();

#ifdef DEBUG
		Image luminance8U;
		luminanceMap.convert32Fto8U(luminance8U);
		luminance8U.write("debug_world_luminance.jpg");
#endif
	};

	Tonemap::~Tonemap(void)
	{
	};

	void Tonemap::linearDisplayLuminance(Image& dispLum)
	{
		dispLum = Image(height, width, CV_32F);
		for (int i=0; i<height; i++)
		{
			// Pointer to the display luminance
			float* Ld = dispLum.ptr<float>(i);
			// Pointer to the world luminance
			float* Lw = luminanceMap.ptr<float>(i);
			for (int j=0; j<width; j++)
			{
				Ld[j] = (Lw[j] - minLum)/(maxLum - minLum);
			}
		}
	};

	void Tonemap::logarithmicDisplayLuminance(Image& dispLum)
	{
		dispLum = Image(height, width, CV_32F);
		for (int i=0; i<height; i++)
		{
			// Pointer to the display luminance
			float* Ld = dispLum.ptr<float>(i);
			// Pointer to the world luminance
			float* Lw = luminanceMap.ptr<float>(i);
			for (int j=0; j<width; j++)
			{
				Ld[j] = log10(1 + Lw[j])/log10(1 + maxLum);
			}
		}
	};

	void Tonemap::exponentialDisplayLuminance(Image& dispLum)
	{
		if (avgLum == 0 || cvIsNaN(avgLum) != 0)
		{
			std::stringstream ss;
			ss << "Error in Tonemap::exponentialDisplayLuminance(): The arithmetic average luminance cannot be zero or NaN. Average luminance is " << avgLum << ".";
			throw std::exception(ss.str().c_str());
		}

		dispLum = Image(height, width, CV_32F);
		for (int i=0; i<height; i++)
		{
			// Pointer to the display luminance
			float* Ld = dispLum.ptr<float>(i);
			// Pointer to the world luminance
			float* Lw = luminanceMap.ptr<float>(i);
			for (int j=0; j<width; j++)
			{
				Ld[j] = (float) 1 - exp(-double(Lw[j])/avgLum);	// we use the arithmetic average instead of the log average luminance
			}
		}
	};

	void Tonemap::reinhardGlobalDisplayLuminance(float a, Image& dispLum)
	{
		if (logAvgLum == 0 || cvIsNaN(logAvgLum) != 0)
		{
			std::stringstream ss;
			ss << "Error in Tonemap::reinhardGlobalDisplayLuminance(): The log average luminance cannot be zero or NaN. Log average luminance is " << logAvgLum << ".";
			throw std::exception(ss.str().c_str());
		}

		dispLum = Image(height, width, CV_32F);
		for (int i=0; i<height; i++)
		{
			// Pointer to the display luminance
			float* Ld = dispLum.ptr<float>(i);
			// Pointer to the world luminance
			float* Lw = luminanceMap.ptr<float>(i);
			for (int j=0; j<width; j++)
			{
				Ld[j] = ( a/logAvgLum * Lw[j] )/( 1 + a/logAvgLum * Lw[j] );
			}
		}
	};

	void Tonemap::mapToDisplayBGR(Image& displayLum, Image& displayBGR, float sat_r, float sat_g, float sat_b)
	{
		for (int i=0; i<height; i++)
		{
			// Pointer to the tone mapped image
			uchar* p = displayBGR.ptr<uchar>(i);
			// Pointer to the display luminance
			float* Ld = displayLum.ptr<float>(i);
			// Pointer to the world luminance
			float* Lw = luminanceMap.ptr<float>(i);
			// Pointer to the radiance map
			float* hdr = radianceMap.ptr<float>(i);
			for (int j=0; j<width; j++)
			{
				// Map to [0,1]: Id = Ld * ( Iw / Lw )^saturation
				float blue = Ld[j] * pow( (hdr[3*j + 0])/(Lw[j]), sat_b );
				float green = Ld[j] * pow( (hdr[3*j + 1])/(Lw[j]), sat_g );
				float red = Ld[j] * pow( (hdr[3*j + 2])/(Lw[j]), sat_r );

				// Convert to 8-bit values
				p[3*j + 0] = round(255*blue);
				p[3*j + 1] = round(255*green);
				p[3*j + 2] = round(255*red);
			}
		}
	};

	Image Tonemap::linear(float sat_r, float sat_g, float sat_b)
	{
		if (sat_r < 0 || sat_r > 1 || sat_g < 0 || sat_g > 1 || sat_b < 0 || sat_b > 1)
			throw std::exception("Error in Tonemap::linear(): Saturation parameter should be 0 <= s <= 1.");

		std::cout << "Computing linear tone map with saturation " << sat_r << ", " << sat_g << ", and " << sat_b << "..." << std::endl;

		Image displayMap = Image(height, width, CV_8UC3);

		// Compute display luminance
		Image displayLum;
		linearDisplayLuminance(displayLum);

#ifdef DEBUG
		Image luminance8U;
		displayLum.convert32Fto8U(luminance8U);
		luminance8U.write("debug_lin_luminance.jpg");
#endif

		// Use the luminance to map back into displayable values
		mapToDisplayBGR(displayLum, displayMap, sat_r, sat_g, sat_b);

		return displayMap;
	};

	Image Tonemap::logarithmic(float sat_r, float sat_g, float sat_b)
	{
		if (sat_r < 0 || sat_r > 1 || sat_g < 0 || sat_g > 1 || sat_b < 0 || sat_b > 1)
			throw std::exception("Error in Tonemap::logarithmic(): Saturation parameter should be 0 <= s <= 1.");

		std::cout << "Computing logarithmic tone map with saturation " << sat_r << ", " << sat_g << ", and " << sat_b << "..." << std::endl;

		Image displayMap = Image(height, width, CV_8UC3);

		// Compute display luminance
		Image displayLum;
		logarithmicDisplayLuminance(displayLum);

#ifdef DEBUG
		Image luminance8U;
		displayLum.convert32Fto8U(luminance8U);
		luminance8U.write("debug_log_luminance.jpg");
#endif

		// Use the luminance to map back into displayable values
		mapToDisplayBGR(displayLum, displayMap, sat_r, sat_g, sat_b);

		return displayMap;
	};

	Image Tonemap::exponential(float sat_r, float sat_g, float sat_b)
	{
		if (sat_r < 0 || sat_r > 1 || sat_g < 0 || sat_g > 1 || sat_b < 0 || sat_b > 1)
			throw std::exception("Error in Tonemap::exponential(): Saturation parameter should be 0 <= s <= 1.");

		std::cout << "Computing exponential tone map with saturations " << sat_r << ", " << sat_g << ", and " << sat_b << "..." << std::endl;

		Image displayMap = Image(height, width, CV_8UC3);

		// Compute display luminance
		Image displayLum;
		exponentialDisplayLuminance(displayLum);

#ifdef DEBUG
		Image luminance8U;
		displayLum.convert32Fto8U(luminance8U);
		luminance8U.write("debug_exp_luminance.jpg");
#endif

		// Use the luminance to map back into displayable values
		mapToDisplayBGR(displayLum, displayMap, sat_r, sat_g, sat_b);

		return displayMap;
	};

	Image Tonemap::reinhardGlobal(float a, float sat_r, float sat_g, float sat_b)
	{
		if (sat_r < 0 || sat_r > 1 || sat_g < 0 || sat_g > 1 || sat_b < 0 || sat_b > 1)
			throw std::exception("Error in Tonemap::reinhardGlobal(): Saturation parameter should be 0 <= s <= 1.");

		std::cout << "Computing Reinhard's global tone map with key value " << a << " and saturations " << sat_r << ", " << sat_g << ", and " << sat_b << "..." << std::endl;

		Image displayMap = Image(height, width, CV_8UC3);

		// Compute display luminance
		Image displayLum;
		reinhardGlobalDisplayLuminance(a, displayLum);

#ifdef DEBUG
		Image luminance8U;
		displayLum.convert32Fto8U(luminance8U);
		luminance8U.write("debug_reinhard_global_luminance.jpg");
#endif

		// Use the luminance to map back into displayable values
		mapToDisplayBGR(displayLum, displayMap, sat_r, sat_g, sat_b);

		return displayMap;
	};

	Image Tonemap::reinhardLocal(void)
	{
		return Image();
	};
}
