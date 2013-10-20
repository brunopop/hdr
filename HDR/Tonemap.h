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

#include "Image.h"

#ifdef _DEBUG
#	define DEBUG
#endif

#define DEBUG

namespace bps
{
	/// <summary>
	/// Implementation of different tonemapping algorithms.
	/// Each algorithm maps a high dynamic range image to
	/// a low dynamic range image (8-bit) that can be displayed.
	/// </summary>
	class Tonemap
	{
	private:
		Image radianceMap;	///< Radiance map to tone map
		Image luminanceMap;	///< World luminance map of the radiance map
		int height;			///< Dimensions of the radiance map
		int width;			///< Dimensions of the radiance map
		float minLum;		///< Min luminance value
		float maxLum;		///< Max luminance value
		double avgLum;		///< Arithmetic average luminance
		double logAvgLum;	///< Log average luminance

		/// <summary>
		/// Compute the luminance map from the radiance map,
		/// as well as the min, max, arithmetic average,
		/// and log average luminance values.
		/// </summary>
		void computeLuminance(void);

		/// <summary>
		/// Compute the linear display luminance.
		/// World luminance values are linearly mapped to display
		/// luminance values in [0,1].
		/// </summary>
		/// <param name="dispLum">[out] Display luminance to be calculated.</param>
		void linearDisplayLuminance(Image& dispLum);

		/// <summary>
		/// Compute the logarithmic display luminance.
		/// World luminance values are mapped to display
		/// luminance values in [0,1] as
		///     Ld(x,y) = log10(1 + Lw(x,y))/log10(1 + maxLum)
		/// where maxLum is the maximum world luminance value.
		/// </summary>
		/// <param name="dispLum">[out] Display luminance to be calculated.</param>
		void logarithmicDisplayLuminance(Image& dispLum);

		/// <summary>
		/// Compute the exponential display luminance.
		/// World luminance values are mapped to display
		/// luminance values in [0,1] as
		///     Ld(x,y) = 1 - exp(-Lw(x,y)/avgLum)
		/// where avgLum is the arithmetic average world luminance value.
		/// </summary>
		/// <param name="dispLum">[out] Display luminance to be calculated.</param>
		void exponentialDisplayLuminance(Image& dispLum);

		/// <summary>
		/// Global Reinhard operator (see Reinhard, Erik, et al.
		/// "Photographic tone reproduction for digital images."
		/// ACM Transactions on Graphics (TOG). Vol. 21. No. 3. ACM, 2002.)
		/// The log average luminance is used as an approximation
		/// of the key of the scene. The luminance is then scaled
		/// using this key and a user-controlled parameter a such as:
		///     L(x,y) = a/key * Lw(x,y)
		/// where Lw(x,y) is the world luminance at pixel (x,y).
		/// The display luminance is then obtained as
		///     Ld(x,y) = L(x,y) / ( 1 + L(x,y) )
		/// </summary>
		/// <param name="a">[in] Key value.</param>
		/// <param name="displayLum">[out] Display luminance map.</param>
		void reinhardGlobalDisplayLuminance(float a, Image& displayLum);

		/// <summary>
		/// Reconstruct the red, green, and blue channels of the
		/// low dynamic range, display image from the display luminance.
		/// Each channel is reconstructed according to the formula
		///     ldr(x,y) = Ld(x,y) * pow( hdr(x,y)/Lw(x,y) , saturation )
		/// where Ld(x,y) is the display luminance at pixel (x,y)
		///       Lw(x,y) is the world luminance at pixel (x,y)
		///       hdr(x,y) is the radiance map at pixel (x,y)
		///       saturation is a gamma correction applied to that channel.
		/// </summary>
		/// <param name="displayLum">[in] Display luminance.</param>
		/// <param name="displayBGR">[out] Low dynamic range image.</param>
		/// <param name="sat_r">[in] Saturation for the red channel.</param>
		/// <param name="sat_g">[in] Saturation for the green channel.</param>
		/// <param name="sat_b">[in] Saturation for the blue channel.</param>
		void mapToDisplayBGR(Image& displayLum, Image& displayBGR, float sat_r, float sat_g, float sat_b);

	public:
		/// <summary>
		/// Unique constructor. Takes a high dynamic range image as input
		/// and construct the world luminance map, which will be used
		/// by any tone mapping operator the user chooses to use.
		/// </summary>
		/// <param name="radianceMap">[in] High dynamic range image.</param>
		Tonemap(Image& radianceMap);

		/// <summary>
		/// Nothing to free in the destructor.
		/// </summary>
		~Tonemap(void);

		/// <summary>
		/// Linear tone mapping operator
		/// </summary>
		/// <param name="sat_r">[in] Saturation of the red channel. Must be between 0 and 1.</param>
		/// <param name="sat_g">[in] Saturation of the green channel. Must be between 0 and 1.</param>
		/// <param name="sat_b">[in] Saturation of the blue channel. Must be between 0 and 1.</param>
		Image linear(float sat_r = 1.0f, float sat_g = 1.0f, float sat_b = 1.0f);

		/// <summary>
		/// Logarithmic tone mapping operator
		/// </summary>
		/// <param name="sat_r">[in] Saturation of the red channel. Must be between 0 and 1.</param>
		/// <param name="sat_g">[in] Saturation of the green channel. Must be between 0 and 1.</param>
		/// <param name="sat_b">[in] Saturation of the blue channel. Must be between 0 and 1.</param>
		Image logarithmic(float sat_r = 1.0f, float sat_g = 1.0f, float sat_b = 1.0f);

		/// <summary>
		/// Exponential tone mapping operator
		/// </summary>
		/// <param name="sat_r">[in] Saturation of the red channel. Must be between 0 and 1.</param>
		/// <param name="sat_g">[in] Saturation of the green channel. Must be between 0 and 1.</param>
		/// <param name="sat_b">[in] Saturation of the blue channel. Must be between 0 and 1.</param>
		Image exponential(float sat_r = 1.0f, float sat_g = 1.0f, float sat_b = 1.0f);

		/// <summary>
		/// Global Reinhard operator (see Reinhard, Erik, et al.
		/// "Photographic tone reproduction for digital images."
		/// ACM Transactions on Graphics (TOG). Vol. 21. No. 3. ACM, 2002.)
		/// The log average luminance is used as an approximation
		/// of the key of the scene. The luminance is then scaled
		/// using this key and a user-controlled parameter a such as:
		///     L(x,y) = a/key * Lw(x,y)
		/// where Lw(x,y) is the world luminance at pixel (x,y).
		/// The display luminance is then obtained as
		///     Ld(x,y) = L(x,y) / ( 1 + L(x,y) )
		/// </summary>
		/// <param name="a">[in] Key value.</param>
		/// <param name="sat_r">[in] Saturation of the red channel. Must be between 0 and 1.</param>
		/// <param name="sat_g">[in] Saturation of the green channel. Must be between 0 and 1.</param>
		/// <param name="sat_b">[in] Saturation of the blue channel. Must be between 0 and 1.</param>
		Image reinhardGlobal(float a = 0.18f, float sat_r = 1.0f, float sat_g = 1.0f, float sat_b = 1.0f);

		Image reinhardLocal(void);
	};
}
