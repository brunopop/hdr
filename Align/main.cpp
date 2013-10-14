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

#include "jhead.h"
#include "HDR.h"
#include "opencv2\highgui\highgui.hpp"

#ifdef _DEBUG
#	define DEBUG
#endif

#define DEBUG

using namespace bps;

void Error(const std::string& message);
void Warning(const std::string& message);
void Info(const std::string& message);
void Debug(const std::string& message);

extern "C" ImageInfo_t GetExifWithJhead(const char * FileName);

void Usage();
DWORD ListFilesInDirectory(_TCHAR* directory_path, std::vector<std::string>& files, std::wstring desired_extension);
void DisplayErrorBox(LPTSTR lpszFunction);
std::string ConvertTCHARToString(_TCHAR* lpsztr);
void ConvertSecToHMS(double input, int* days, int* hours, int* minutes, double* seconds);
void DisplayTime(double totalTime);

int _tmain(int argc, _TCHAR* argv[])
{
	if (argc != 2) Usage();

	time_t startTime = clock();

	// List files in directory
	std::vector<std::string> files;
	ListFilesInDirectory(argv[argc-1], files, std::wstring(_TEXT("jpg")));
	if (files.size() == 0)
	{
		ListFilesInDirectory(argv[argc-1], files, std::wstring(_TEXT("JPG")));
		if (files.size() == 0)
		{
			Error("No JPEG file found.");
		}
		else
		{
			std::cout << files.size() << " files found." << std::endl;
		}
	}
	else
	{
		std::cout << files.size() << " files found." << std::endl;
	}

	// Open images and read EXIF data
	std::vector<Image> images;
	std::vector<ImageInfo_t> exifData;
	for (unsigned int i=0; i<files.size(); i++)
	{
		try
		{
			images.push_back(Image(files[i].c_str()));
			exifData.push_back(GetExifWithJhead(files[i].c_str()));
			images[i].setExposureTime(exifData[i].ExposureTime);
#ifdef DEBUG
			std::cout << "Image " << i << " has exposure time " << exifData[i].ExposureTime << std::endl;
#endif
		}
		catch (std::exception& e)
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
			std::stringstream ss;
			ss << "image_" << i << ".JPG";
			images[i].write(ss.str());
		}
	}
	catch (std::exception& e)
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
	catch (std::exception& e)
	{
		Warning(e.what());
	}

	// Tonemap
	Tonemap tonemapper(radianceMap);
	std::vector<Image> linMap, logMap, expMap, reinhardGlobal;
	bool err = false;
	try
	{
		for (int i=0; i<10; i++)
		{
			// Varying saturation
			float s = (i+1)*0.1f;
			linMap.push_back(tonemapper.linear(s, s, s));
			logMap.push_back(tonemapper.logarithmic(s, s, s));
			expMap.push_back(tonemapper.exponential(s, s, s));
			// Varying key value
			float a = (i+1)*0.1f;
			reinhardGlobal.push_back(tonemapper.reinhardGlobal(a));
		}
	}
	catch (std::exception& e)
	{
		err = true;
		Warning(e.what());
	}

	// Save
	if (!err)
	{
		for (int i=0; i<linMap.size(); i++)
		{
			std::stringstream ss;
			ss << "output_lin_" << i+1 << ".jpg";
			linMap[i].write(ss.str());
		}
		for (int i=0; i<logMap.size(); i++)
		{
			std::stringstream ss;
			ss << "output_log_" << i+1 << ".jpg";
			logMap[i].write(ss.str());
		}
		for (int i=0; i<expMap.size(); i++)
		{
			std::stringstream ss;
			ss << "output_exp_" << i+1 << ".jpg";
			expMap[i].write(ss.str());
		}
		for (int i=0; i<reinhardGlobal.size(); i++)
		{
			std::stringstream ss;
			ss << "output_reinhard_global_" << i+1 << ".jpg";
			reinhardGlobal[i].write(ss.str());
		}
	}

	double totalTime = (double) (clock() - startTime)/CLOCKS_PER_SEC;
	DisplayTime(totalTime);
	
	return 0;
}

void Usage()
{
	std::cerr << "Usage: Align.exe <directory path>" << std::endl;
	exit(1);
}

void Error(const std::string& message)
{
	std::cerr << message << std::endl;
	exit(1);
}

void Warning(const std::string& message)
{
	std::cout << "Warning!  " << message << std::endl;
}

void Info(const std::string& message)
{
	std::cout << message << std::endl;
}

void Debug(const std::string& message)
{
#ifdef DEBUG
	std::cout << message << std::endl;
#endif
}

DWORD ListFilesInDirectory(_TCHAR* directory_path, std::vector<std::string>& files, std::wstring desired_extension)
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
		std::wstring file_name(ffd.cFileName);

#ifdef DEBUG_LIST
		_tprintf(TEXT("\nDEBUG--------------------\n\tCurrent file: %s\nDEBUG--------------------\n"), file_path);
#endif // DEBUG_LIST

		if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
		{
			if ( file_name.compare(std::wstring(_T("."))) != 0 && file_name.compare(std::wstring(_T(".."))) != 0 )
			{
				// If it is a directory, then list files in that directory
				// after concatenating the sub directory name with the directory path
				ListFilesInDirectory(file_path, files, desired_extension);
			}
		}
		else
		{
			// If it is a file, add it to the appropriate list with its full path
			std::wstring extension = file_name.substr(file_name.length()-3, file_name.length());
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

std::string ConvertTCHARToString(_TCHAR* lpsztr)
{
	size_t dummy;
	char *cstring = (char *)malloc(MAX_PATH);
	size_t sizeInBytes = MAX_PATH;
	wcstombs_s(&dummy, cstring, sizeInBytes, lpsztr, sizeInBytes);
	std::string stdstring(cstring);
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
	std::cout << "Done in ";
	if (days > 0)
	{
		std::cout << days << "d " << hours << "h " << minutes << "m " << seconds << "s." << std::endl;
	}
	else if (hours > 0)
	{
		std::cout << hours << "h " << minutes << "m " << seconds << "s." << std::endl;
	}
	else if (minutes > 0)
	{
		std::cout << minutes << "m " << seconds << "s." << std::endl;
	}
	else
	{
		std::cout << totalTime << " seconds." << std::endl;
	}
}

