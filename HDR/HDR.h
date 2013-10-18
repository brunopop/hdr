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

#pragma once

#include "opencv2\core\core.hpp"


#define TOLERANCE 6
#define SHIFT_BITS 6
#define LAMBDA 50

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
		Image(cv::Mat& mat, int tolerance = TOLERANCE);

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
		bool write(const std::string& filename);

		/// <summary>
		/// Converts the image into an 8-bit equivalent,
		/// if the image is 16-bit float and 1 channel.
		/// </summary>
		/// <param name="mat8UC">[out] 8-bit 1-channel converted image</param>
		void convert32Fto8U(Image& mat8UC);

	};

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

	/// <summary>
	/// Implementation of Debevec and Malik's "Recovering High Dynamic
	/// Range Radiance Maps from Photographs." In SIGGRAPH 97, 1997.
	/// </summary>
	class HDR
	{
	private:
		std::vector<Image>* images;		///< Set of exposure images.
		int height;						///< Height of an image.
		int width;						///< Width of an image.
		double lambda;					///< Smoothing factor
		int N;							///< Number of pixels/equations in the objective function
		int P;							///< Number of photographs
		int Zmax;						///< Maximum pixel value
		int Zmin;						///< Minimum pixel value
		std::vector<float> B;			///< Log delta t, or log shutter speed, for each image in the set
		std::vector<float> gRed;		///< Response function for the red channel
		std::vector<float> gGreen;		///< Response function for the green channel
		std::vector<float> gBlue;		///< Response function for the blue channel
		std::vector<float> lERed;		///< Log irradiance map for the red channel
		std::vector<float> lEGreen;	///< Log irradiance map for the green channel
		std::vector<float> lEBlue;		///< Log irradiance map for the blue channel
		bool responseFunctionCalculated;///< Sentinel variable that is true if the response function g and the log irradiance log(E) have been calculated
		bool radianceMapCalculated;		///< Sentinel variable that is true if the radiance map E has been calculated

		/// <summary>
		/// Solve the linear system that minimize the objective function O:
		///
		/// O = sum_{i=1}^{N} sum_{j=1}^{P} { weight(Z_{ij}) [g(Z_{ij}) - ln(E_i) - ln(\Deltat_j)] }^2
		///     + \lambda sum_{z=Zmin+1}^{Zmax-1} [ weight(z) g''(z) ]^2
		///
		/// Given a set of observed pixel values in a set of images with
		/// known exposures, this routine reconstructs the imaging response
		/// curve and the radiance values for the given pixels.
		/// </summary>
		/// <param name="z">[in] Pixel values of pixel location number i in image j.</param>
		/// <param name="g">[out] Log exposure corresponding to a pixel value z.</param>
		/// <param name="lE">[out] Log film irradiance at pixel location i.</param>
		void gsolve(std::vector<std::vector<uchar>>& z, std::vector<float>& g, std::vector<float>& lE);

		/// <summary>
		/// Weighting function <c>w(z)</c> to emphasize the smoothness
		/// and fitting terms toward the middle of the curve
		/// </summary>
		inline int weight(int z);

		/// <summary>
		/// Samples pixel locations in the three channels of an image
		/// across all exposures.
		/// </summary>
		void sample(std::vector<std::vector<uchar>>& zRed,
					std::vector<std::vector<uchar>>& zGreen,
					std::vector<std::vector<uchar>>& zBlue);

		/// <summary>
		/// Recover the response function from the set
		/// of images.
		/// </summary>
		void responseFunction(void);

		/// <summary>
		/// Construct the radiance map.
		/// </summary>
		void radianceMap(Image& radianceMap);

	public:
		HDR(std::vector<Image>* images, float lambda = LAMBDA);

		~HDR(void);

		/// <summary>
		/// The user can either successively call responseFunction() and radianceMap(),
		/// or directly call computeRadianceMap() to call the intermediate methods.
		/// </summary>
		void computeRadianceMap(Image& radianceMap);
	};

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
		long double avgLum;		///< Arithmetic average luminance
		long double logAvgLum;	///< Log average luminance

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
