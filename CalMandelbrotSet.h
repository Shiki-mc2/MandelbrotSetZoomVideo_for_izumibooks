#pragma once
#ifdef CALMANDELBROTSET_EXPORTS
#define CALMANDELBROTSET_API extern "C" __declspec(dllexport)
#else
#define CALMANDELBROTSET_API extern "C" __declspec(dllimport)
#endif

CALMANDELBROTSET_API void CalMandelbrotSet(
	int, int, int, char*, char*, char*, int, float, int, int[]);