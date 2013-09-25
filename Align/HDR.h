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
		Bitmap();

		Bitmap(int height, int width);

		~Bitmap(void);

		void set(int x, int y, bool value);

		bool get(int x, int y) const;

		int getHeight(void) const;

		int getWidth(void) const;

		void imwrite(const std::string& path);

		/// <summary>Compute the sum of all 1 bits in the input bitmap.</summary>
		/// <param name="bm">[in] Input bitmap.</param>
		/// <return>Sum of all 1 bits.</return>
		int total(void) const;
	};

	class Image : public cv::Mat
	{
	private:
		float exposureTime;
		int tolerance;
		double threshold;

		void computeThreshold(void);

	public:
		Image(int tolerance = TOLERANCE);

		Image(int height, int width, int type, int tolerance = TOLERANCE);

		Image(cv::Mat& mat, int tolerance = TOLERANCE);

		Image(const std::string& path, int flags = 1, int tolerance = TOLERANCE);

		~Image(void);

		void setExposureTime(float exposureTime);

		float getExposureTime(void) const;

		void setTolerance(int tolerance = TOLERANCE);

		int getTolerance(void) const;

		double getThreshold(void) const;

		/// <summary>Subsample the image by a factor of two in each dimension
		/// and put the result into a newly allocated image.</summary>
		/// <return>A new image subsampled by two.</return>
		Image shrink2(void) const;

		void crop(int x, int y, int height, int width);

		void computeThresholdBitmap(Bitmap& bmp) const;

		void computeExclusionBitmap(Bitmap& bmp) const;

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

		/// <summary>Compute the alignment offset between two exposure images recursively.</summary>
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

		/// <summary>Call Ward's registration algorithm and register the images.</summary>
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
		std::vector<double> B;			///< Log delta t, or log shutter speed, for each image in the set
		std::vector<double> gRed;		///< Response function for the red channel
		std::vector<double> gGreen;		///< Response function for the green channel
		std::vector<double> gBlue;		///< Response function for the blue channel

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
		void gsolve(std::vector<std::vector<uchar>>& z, std::vector<double>& g, std::vector<double>& lE);

		/// <summary>
		/// Weighting function <c>w(z)</c> to emphasize the smoothness
		/// and fitting terms toward the middle of the curve
		/// </summary>
		int weight(int z);

		void sample(std::vector<std::vector<uchar>>& zRed,
			std::vector<std::vector<uchar>>& zGreen,
			std::vector<std::vector<uchar>>& zBlue);

	public:
		HDR(std::vector<Image>* images, double lambda = LAMBDA);

		~HDR(void);

		/// <summary>
		/// Recover the response function from the set
		/// of images.
		/// </summary>
		void responseFunction(void);

		/// <summary>
		/// Construct the radiance map.
		/// </summary>
		void radianceMap(void);
	};

}
