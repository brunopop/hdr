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

// Win32 libraries for parsing folders
#include <windows.h>
#include <Shlwapi.h>
#include <tchar.h>
#include <strsafe.h>

// STL libraries
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <time.h>

// Other libraries
#include "jhead.h"
#include "HDR.h"
#include "Image.h"
#include "Align.h"
#include "Tonemap.h"

#ifdef _DEBUG
#	define DEBUG
#endif

#define DEBUG

using namespace bps;
using namespace std;

void Error(const string& message);
void Warning(const string& message);
void Info(const string& message);
void Debug(const string& message);

extern "C" ImageInfo_t GetExifWithJhead(const char * FileName);

void Usage();
DWORD ListFilesInDirectory(_TCHAR* directory_path, vector<string>& files, wstring desired_extension);
void DisplayErrorBox(LPTSTR lpszFunction);
string ConvertTCHARToString(_TCHAR* lpsztr);
void ConvertSecToHMS(double input, int* days, int* hours, int* minutes, double* seconds);
void DisplayTime(double totalTime);

int _tmain(int argc, _TCHAR* argv[])
{
	if (argc != 2) Usage();

	time_t startTime = clock();

	// List files in directory
	vector<string> files;
	ListFilesInDirectory(argv[argc-1], files, wstring(_TEXT("jpg")));
	if (files.size() == 0)
	{
		ListFilesInDirectory(argv[argc-1], files, wstring(_TEXT("JPG")));
		if (files.size() == 0)
		{
			Error("No JPEG file found.");
		}
		else
		{
			cout << files.size() << " files with the extension \'JPG\' found." << endl;
		}
	}
	else
	{
		cout << files.size() << " files with the extension \'jpg\' found." << endl;
	}

	// Open images and read EXIF data
	vector<Image> images;
	for (unsigned int i=0; i<files.size(); i++)
	{
		try
		{
			images.push_back(Image(files[i].c_str()));
			ImageInfo_t exifData = GetExifWithJhead(files[i].c_str());
			if (exifData.ExposureTime <= 0)
			{
				stringstream ss;
				ss << "Image " << i << " has invalid exposure time.";
				Error(ss.str());
			}
			images[i].setExposureTime(exifData.ExposureTime);
#ifdef DEBUG
			cout << "Image " << i << " has exposure time " << exifData.ExposureTime << endl;
#endif
		}
		catch (exception& e)
		{
			Warning(e.what());
		}
	}

	// Register the images
	try
	{
		Align align(&images);
		align.align();
		for (unsigned int i=0; i<images.size(); i++)
		{
			stringstream ss;
			ss << "image_" << i << ".JPG";
			images[i].write(ss.str());
		}
	}
	catch (exception& e)
	{
		Warning(e.what());
	}

	// Combine them to make a high dynamic range image
	HDR hdr(&images);
	Image radianceMap;
	try
	{
		hdr.computeRadianceMap(radianceMap);
	}
	catch (exception& e)
	{
		Warning(e.what());
	}

	// Tonemap
	Tonemap tonemapper(radianceMap);
	try
	{
		for (int i=0; i<10; i++)
		{
			// Varying saturation
			float s = (i+1)*0.1f;
			
			// Linear tonemapping
			Image linear = tonemapper.linear(s, s, s);
			// Save
			stringstream ss_lin;
			ss_lin << "output_lin_" << i+1 << ".jpg";
			linear.write(ss_lin.str());

			// Logarithmic tonemapping
			Image logarithmic = tonemapper.logarithmic(s, s, s);
			// Save
			stringstream ss_log;
			ss_log << "output_log_" << i+1 << ".jpg";
			logarithmic.write(ss_log.str());

			// exponential tonemapping
			Image exponential = tonemapper.exponential(s, s, s);
			// Save
			stringstream ss_exp;
			ss_exp << "output_exp_" << i+1 << ".jpg";
			exponential.write(ss_exp.str());

			// Varying key value
			float a = (i+1)*0.1f;

			// Global Reinhard
			Image reinhardGlobal = tonemapper.reinhardGlobal(a);
			// Save
			stringstream ss_global;
			ss_global << "output_reinhard_global_" << i+1 << ".jpg";
			reinhardGlobal.write(ss_global.str());
		}
	}
	catch (exception& e)
	{
		Warning(e.what());
	}

	double totalTime = (double) (clock() - startTime)/CLOCKS_PER_SEC;
	DisplayTime(totalTime);
	
	return 0;
}

void Usage()
{
	cerr << "Usage: HDR.exe <input directory path>" << endl;
	exit(1);
}

void Error(const string& message)
{
	cerr << message << endl;
	exit(1);
}

void Warning(const string& message)
{
	cout << "Warning!  " << message << endl;
}

void Info(const string& message)
{
	cout << message << endl;
}

void Debug(const string& message)
{
#ifdef DEBUG
	cout << message << endl;
#endif
}

DWORD ListFilesInDirectory(_TCHAR* directory_path, vector<string>& files, wstring desired_extension)
{
	WIN32_FIND_DATA ffd;
	TCHAR szDir[MAX_PATH];
	HANDLE hFind = INVALID_HANDLE_VALUE;
	DWORD dwError = 0;

	_tprintf(TEXT("Listing files in directory %s...\n"), directory_path);

	// Prepare string for use with FindFile functions.  First, copy the
	// string to a buffer, then append '\*' to the directory name.

	StringCchCopy(szDir, MAX_PATH, directory_path);
	StringCchCat(szDir, MAX_PATH, TEXT("\\*"));

	// Find the first file in the directory.

	hFind = FindFirstFile(szDir, &ffd);

	if (INVALID_HANDLE_VALUE == hFind) 
	{
		DisplayErrorBox(TEXT("FindFirstFile"));
		return dwError;
	}

	// List all the files in the directory

	do
	{
		// Concatenate file name with directory path
		TCHAR file_path[MAX_PATH];
		StringCchCopy(file_path, MAX_PATH, directory_path);
		StringCchCat(file_path, MAX_PATH, TEXT("\\"));
		StringCchCat(file_path, MAX_PATH, ffd.cFileName);

		// String equivalent of the file name
		wstring file_name(ffd.cFileName);

#ifdef DEBUG_LIST
		_tprintf(TEXT("\nDEBUG--------------------\n\tCurrent file: %s\nDEBUG--------------------\n"), file_path);
#endif // DEBUG_LIST

		if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
		{
			if ( file_name.compare(wstring(_T("."))) != 0 && file_name.compare(wstring(_T(".."))) != 0 )
			{
				// If it is a directory, then list files in that directory
				// after concatenating the sub directory name with the directory path
				ListFilesInDirectory(file_path, files, desired_extension);
			}
		}
		else
		{
			// If it is a file, add it to the appropriate list with its full path
			wstring extension = file_name.substr(file_name.length()-3, file_name.length());
			if (extension.compare(desired_extension) == 0)
			{
				files.push_back(ConvertTCHARToString(file_path));
			}
		}
	}
	while (FindNextFile(hFind, &ffd) != 0);

	dwError = GetLastError();
	if (dwError != ERROR_NO_MORE_FILES) 
	{
		DisplayErrorBox(TEXT("FindFirstFile"));
	}

	FindClose(hFind);
	return dwError;
}

void DisplayErrorBox(LPTSTR lpszFunction) 
{ 
	// Retrieve the system error message for the last-error code

	LPVOID lpMsgBuf;
	LPVOID lpDisplayBuf;
	DWORD dw = GetLastError(); 

	FormatMessage(
		FORMAT_MESSAGE_ALLOCATE_BUFFER | 
		FORMAT_MESSAGE_FROM_SYSTEM |
		FORMAT_MESSAGE_IGNORE_INSERTS,
		NULL,
		dw,
		MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
		(LPTSTR) &lpMsgBuf,
		0, NULL );

	// Display the error message and clean up

	lpDisplayBuf = (LPVOID)LocalAlloc(LMEM_ZEROINIT, 
		(lstrlen((LPCTSTR)lpMsgBuf)+lstrlen((LPCTSTR)lpszFunction)+40)*sizeof(TCHAR)); 
	StringCchPrintf((LPTSTR)lpDisplayBuf, 
		LocalSize(lpDisplayBuf) / sizeof(TCHAR),
		TEXT("%s failed with error %d: %s"), 
		lpszFunction, dw, lpMsgBuf); 
	MessageBox(NULL, (LPCTSTR)lpDisplayBuf, TEXT("Error"), MB_OK); 

	LocalFree(lpMsgBuf);
	LocalFree(lpDisplayBuf);
}

string ConvertTCHARToString(_TCHAR* lpsztr)
{
	size_t dummy;
	char *cstring = (char *)malloc(MAX_PATH);
	size_t sizeInBytes = MAX_PATH;
	wcstombs_s(&dummy, cstring, sizeInBytes, lpsztr, sizeInBytes);
	string stdstring(cstring);
	return stdstring;
}

void ConvertSecToHMS(double input, int* days, int* hours, int* minutes, double* seconds)
{
	*days = (int) std::floor(input/(3600.0*24.0));
	*hours = (int) std::floor(input/3600.0 - (*days)*24.0);
	*minutes = (int) std::floor( input/60.0 - ((*days)*24.0 + (*hours))*60.0);
	*seconds = input - ((*days)*3600.0*24.0 + (*hours)*3600.0 + (*minutes)*60.0);
}

void DisplayTime(double totalTime)
{
	int days, hours, minutes;
	double seconds;
	ConvertSecToHMS(totalTime, &days, &hours, &minutes, &seconds);
	cout << "Done in ";
	if (days > 0)
	{
		cout << days << "d " << hours << "h " << minutes << "m " << seconds << "s." << endl;
	}
	else if (hours > 0)
	{
		cout << hours << "h " << minutes << "m " << seconds << "s." << endl;
	}
	else if (minutes > 0)
	{
		cout << minutes << "m " << seconds << "s." << endl;
	}
	else
	{
		cout << totalTime << " seconds." << endl;
	}
}

