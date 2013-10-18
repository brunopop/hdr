C++ HDR
===

High Dynamic Range Imaging app project written in C++ and using OpenCV, LAPACK and jhead.

The goal of this project is to write the core functionality of an HDR app for Android (using Android NDK). Read more on http://brunopop.com/prjcts/hdr/.

The current version of this (on-going) project contains classes that implement image registration using "Fast, Robust Image Registration for Compositing High Dynamic Range Photographs from Handheld Exposures" by Greg Ward (Journal of graphics tools 8.2, 2003) and high dynamic range imaging using "Recovering High Dynamic Range Radiance Maps from Photographs" by Paul E. Debevec and Jitendra Malik (SIGGRAPH 1997). The radiance map is then mapped to a displayable gamut using one of the methods written in the Tonemap class. So far, the available tonemapping algorithms are:
 - linear tonemapping
 - logarithmic tonemapping
 - exponential tonemapping
 - global Reinhard operator from "Photographic Tone Reproduction for Digital Images" by Erik Reinhard et al. (ACM Transactions on Graphics 2002).

The project includes a basic Win32 console application to test the algorithms. It is written using Visual Studio 2012 and compiled for Win32 (because LAPACKE is only available in 32-bit). The required DLLs to run the application are in the Release folder.

I am currently working on adding Reinhard's local operator and writing unit tests for each class. In the future, a ghost detection and removal feature will be added.