#ifndef _MAIN_H
#define _MAIN_H

#include<vector>
#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<time.h>
#include<string.h>
#include<ctype.h>
#include<cuda_runtime.h>
#include<string>
#include<fstream>
using namespace std;

typedef long long int64;
typedef unsigned long long uint64;

static unsigned char wordbits[65536];

#define MarginalDistrSNP_Y_DimensionX 4
#define MarginalDistrSNP_Y_DimensionY 3

#define THREAD_NUM 256
#define BLOCK_NUM 10000

static int popcount(uint64 i)
{
	return(wordbits[i & 0xFFFF] + wordbits[(i >> 16) & 0xFFFF] + wordbits[(i >> 32) & 0xFFFF] + wordbits[i >> 48]);
}

int bitCount(uint64 i);

double Abs(double a);

void CalculateMarginalDistr(uint64* genoCtrl_F, uint64* genoCtrl_M, uint64* genoCase_F, uint64* genoCase_M, int* nlongintCase_Gender, int nsnps, int nsamples, int* pMarginalDistrSNP, int* pMarginalDistrSNP_Y);

class DeviceProperties{
public:
	DeviceProperties();
	~DeviceProperties();
	int getDeviceCount();
	cudaDeviceProp getDeviceProp(int i);
	void printDevProp(int i);
private:
	cudaDeviceProp* devPropArray;
	int devCount;
};

extern "C" void cuda_GetInteractionPairs(vector<double>&InteractionMeasure, vector<pair<int, int>>& InteractionPair, uint64* genoCtrl_F, uint64* genoCtrl_M, uint64* genoCase_F, uint64* genoCase_M,
	int nsnps, int nsamples, int* nlongintCase_Gender, int* pMarginalDistrSNP, int* pMarginalDistrSNP_Y,const unsigned char* wordbits, int wordBitCount);


#endif