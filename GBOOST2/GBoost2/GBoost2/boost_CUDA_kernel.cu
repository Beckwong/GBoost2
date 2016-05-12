#include<cuda_runtime.h>
#include "main.h"
#include "device_launch_parameters.h"
texture<uint2, 1, cudaReadModeElementType> genoCtrl_F_Texture;
texture<uint2, 1, cudaReadModeElementType> genoCtrl_M_Texture;
texture<uint2, 1, cudaReadModeElementType> genoCase_F_Texture;
texture<uint2, 1, cudaReadModeElementType> genoCase_M_Texture;
texture<unsigned char, 1, cudaReadModeElementType> wordbits_Texture;

long long iDivUp(long long a, long long b) {
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

inline __device__ int dev_count_bit(__int64 i) {
	i = i - ((i >> 1) & 0x5555555555555555);
	i = (i & 0x3333333333333333) + ((i >> 2) & 0x3333333333333333);
	return (((i + (i >> 4)) & 0xF0F0F0F0F0F0F0F) * 0x101010101010101) >> 56;
}

inline __device__ int dev_count_bit_slow_mult(__int64 x) {
	x -= (x >> 1) & 0x5555555555555555;								//put count of each 2 bits into those 2 bits
	x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333); //put count of each 4 bits into those 4 bits 
	x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0f;						//put count of each 8 bits into those 8 bits 
	x += x >> 8;													//put count of each 16 bits into their lowest 8 bits
	x += x >> 16;													//put count of each 32 bits into their lowest 8 bits
	x += x >> 32;													//put count of each 64 bits into their lowest 8 bits
	return x & 0x7f;
}

__global__ void Screening_kernel(uint64* genoCtrl_F, uint64* genoCtrl_M, uint64* genoCase_F, uint64* genoCase_M, int nsnps, int nsamples, int nlongintCtrl_F, int nlongintCtrl_M,
	int nlongintCase_F, int nlongintCase_M, int* interactionInputOffsetJ1, int* interactionInputOffsetJ2, int *interactionPairOffsetJ1, int*interactionPairOffsetJ2,
	int* pMarginalDistrSNP, int* pMarginalDistrSNP_Y, unsigned char* wordbits)
{
	__int64 andResult = 0;
	
	int outIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int snp1 = interactionInputOffsetJ1[outIndex];
	int snp2 = interactionInputOffsetJ2[outIndex];

	int count;
	int localGenoDistr[36];
	float tao = 0;
	float InteractionMeasure = 0;
	float ptmp1, ptmp2;

	if ((snp1 >= snp2) || (snp1 >= nsnps - 1) || (snp2 >= nsnps))
	{
		return;
	}

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			count = 0;
			for (int k = 0; k < nlongintCtrl_F; k++)
			{
				andResult = genoCtrl_F[k * 3 * nsnps + i*nsnps + snp1] & genoCtrl_F[k * 3 * nsnps + j*nsnps + snp2];
				count += dev_count_bit(andResult);
			}
			localGenoDistr[i * 3 + j] = count;
		}
	}

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			count = 0;
			for (int k = 0; k < nlongintCase_M; k++)
			{
				andResult = genoCtrl_M[k * 3 * nsnps + i*nsnps + snp1] & genoCtrl_M[k * 3 * nsnps + j*nsnps + snp2];
				count += dev_count_bit(andResult);
			}
			localGenoDistr[9 + i * 3 + j] = count;
		}
	}

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			count = 0;
			for (int k = 0; k < nlongintCase_F; k++)
			{
				andResult = genoCase_F[k * 3 * nsnps + i*nsnps + snp1] & genoCase_F[k * 3 * nsnps + j*nsnps + snp2];
				count += dev_count_bit(andResult);
			}
			localGenoDistr[18 + i * 3 + j] = count;
		}
	}

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			count = 0;
			for (int k = 0; k < nlongintCase_M; k++)
			{
				andResult = genoCase_M[k * 3 * nsnps + i*nsnps + snp1] & genoCase_M[k * 3 * nsnps + j*nsnps + snp2];
				count += dev_count_bit(andResult);
			}
			localGenoDistr[27 + i * 3 + j] = count;
		}
	}

	//calculate other cells in localGenoDistr
	//control and female
	localGenoDistr[2] = pMarginalDistrSNP_Y[(0 * MarginalDistrSNP_Y_DimensionX + 0)*nsnps + snp1] - localGenoDistr[0] - localGenoDistr[1];
	localGenoDistr[5] = pMarginalDistrSNP_Y[(1 * MarginalDistrSNP_Y_DimensionX + 0)*nsnps + snp1] - localGenoDistr[3] - localGenoDistr[4];

	localGenoDistr[6] = pMarginalDistrSNP_Y[(0 * MarginalDistrSNP_Y_DimensionX + 0)*nsnps + snp2] - localGenoDistr[0] - localGenoDistr[3];
	localGenoDistr[7] = pMarginalDistrSNP_Y[(1 * MarginalDistrSNP_Y_DimensionX + 0)*nsnps + snp2] - localGenoDistr[1] - localGenoDistr[4];
	localGenoDistr[8] = pMarginalDistrSNP_Y[(2 * MarginalDistrSNP_Y_DimensionX + 0)*nsnps + snp2] - localGenoDistr[2] - localGenoDistr[5];
	
	//control and male
	localGenoDistr[11] = pMarginalDistrSNP_Y[(0 * MarginalDistrSNP_Y_DimensionX + 1)*nsnps + snp1] - localGenoDistr[9] - localGenoDistr[10];
	localGenoDistr[14] = pMarginalDistrSNP_Y[(1 * MarginalDistrSNP_Y_DimensionX + 1)*nsnps + snp1] - localGenoDistr[12] - localGenoDistr[13];
	
	localGenoDistr[15] = pMarginalDistrSNP_Y[(0 * MarginalDistrSNP_Y_DimensionX + 1)*nsnps + snp2] - localGenoDistr[9] - localGenoDistr[12];
	localGenoDistr[16] = pMarginalDistrSNP_Y[(1 * MarginalDistrSNP_Y_DimensionX + 1)*nsnps + snp2] - localGenoDistr[10] - localGenoDistr[13];
	localGenoDistr[17] = pMarginalDistrSNP_Y[(2 * MarginalDistrSNP_Y_DimensionX + 1)*nsnps + snp2] - localGenoDistr[11] - localGenoDistr[14];

	//case and female
	localGenoDistr[20] = pMarginalDistrSNP_Y[(0 * MarginalDistrSNP_Y_DimensionX + 2)*nsnps + snp1] - localGenoDistr[18] - localGenoDistr[19];
	localGenoDistr[23] = pMarginalDistrSNP_Y[(1 * MarginalDistrSNP_Y_DimensionX + 2)*nsnps + snp1] - localGenoDistr[21] - localGenoDistr[22];
	
	localGenoDistr[24] = pMarginalDistrSNP_Y[(0 * MarginalDistrSNP_Y_DimensionX + 2)*nsnps + snp2] - localGenoDistr[18] - localGenoDistr[21];
	localGenoDistr[25] = pMarginalDistrSNP_Y[(1 * MarginalDistrSNP_Y_DimensionX + 2)*nsnps + snp2] - localGenoDistr[19] - localGenoDistr[22];
	localGenoDistr[26] = pMarginalDistrSNP_Y[(2 * MarginalDistrSNP_Y_DimensionX + 2)*nsnps + snp2] - localGenoDistr[20] - localGenoDistr[23];

	//case and male
	localGenoDistr[29] = pMarginalDistrSNP_Y[(0 * MarginalDistrSNP_Y_DimensionX + 3)*nsnps + snp1] - localGenoDistr[27] - localGenoDistr[28];
	localGenoDistr[32] = pMarginalDistrSNP_Y[(1 * MarginalDistrSNP_Y_DimensionX + 3)*nsnps + snp1] - localGenoDistr[30] - localGenoDistr[31];

	localGenoDistr[33] = pMarginalDistrSNP_Y[(0 * MarginalDistrSNP_Y_DimensionX + 3)*nsnps + snp2] - localGenoDistr[27] - localGenoDistr[30];
	localGenoDistr[34] = pMarginalDistrSNP_Y[(1 * MarginalDistrSNP_Y_DimensionX + 3)*nsnps + snp2] - localGenoDistr[28] - localGenoDistr[31];
	localGenoDistr[35] = pMarginalDistrSNP_Y[(2 * MarginalDistrSNP_Y_DimensionX + 3)*nsnps + snp2] - localGenoDistr[29] - localGenoDistr[32];

	tao = 0;
	InteractionMeasure = 0;



}





extern "C" void cuda_GetInteractionPairs(vector<double>&InteractionMeasure, vector<pair<int, int>>& InteractionPair, uint64* genoCtrl_F, uint64* genoCtrl_M, uint64* genoCase_F, uint64* genoCase_M,
	int nsnps, int nsamples, int* nlongintCase_Gender, int* pMarginalDistrSNP, int* pMarginalDistrSNP_Y, const unsigned char* wordbits, int wordBitCount)
{
	printf("\nStarting screening ...\n");
	float timeInMs;
	cudaEvent_t evStart, evStop;
	cudaEventCreate(&evStart);
	cudaEventCreate(&evStop);

	cudaEventRecord(evStart, 0);

	uint64 *gpu_genoCtrl_F,*gpu_genoCtrl_M,*gpu_genoCase_F,*gpu_genoCase_M;
	int* gpu_pMarginalDistrSNP;
	int* gpu_pMarginalDistrSNP_Y;
	int *gpu_inputOffsetJ1;
	int *gpu_inputOffsetJ2;


	unsigned char* gpu_wordBits;
	cudaMalloc((void**)&gpu_wordBits, sizeof(unsigned char)*wordBitCount);
	cudaMemcpy(gpu_wordBits, wordbits, sizeof(unsigned char)*wordBitCount, cudaMemcpyHostToDevice);
	cudaBindTexture(0, wordbits_Texture, gpu_wordBits, sizeof(unsigned char)*wordBitCount);

	int snp1 = 0, snp2 = snp1 + 1;
	bool firstLoop = true;
	int shiftOffset = 0;
	long long totaltasks = ((long long)nsnps*(nsnps - 1)) / 2;
	long long offset = 0;

	int threadNum = THREAD_NUM;
	int blockNum = BLOCK_NUM;
	int totalNumberOfThreadBlock = iDivUp(totaltasks, (long long)threadNum);
	int totalNumberOfGridBlock = iDivUp(totalNumberOfThreadBlock, (long long)blockNum);

	int* interactionInputOffsetJ1;
	int* interactionInputOffsetJ2;
	int *gpu_InteractionPairOffsetJ1;
	int *gpu_InteractionPairOffsetJ2;
	dim3 threads(threadNum, 1, 1);
	dim3 grids(blockNum, 1, 1);

	float* gpu_floatArray;

	// normal host memory allocation
	int* interactionPairOffsetJ1 = (int *)calloc(threadNum*blockNum, sizeof(int));
	int* interactionPairOffsetJ2 = (int *)calloc(threadNum*blockNum, sizeof(int));

	
	cudaHostAlloc((void**)&interactionInputOffsetJ1, sizeof(int)*blockNum*threadNum, cudaHostAllocMapped);
	cudaHostAlloc((void**)&interactionInputOffsetJ2, sizeof(int)*blockNum*threadNum, cudaHostAllocMapped);

	//pass back the device pointer and map with host
	cudaHostGetDevicePointer((void**)&gpu_inputOffsetJ1, (void*)interactionInputOffsetJ1, 0);
	cudaHostGetDevicePointer((void**)&gpu_inputOffsetJ2, (void*)interactionInputOffsetJ2, 0);

	//alocate GPU memory
	cudaMalloc((void**)&gpu_genoCtrl_F, sizeof(uint64)*nlongintCase_Gender[0] * 3 * nsnps);
	cudaMalloc((void**)&gpu_genoCtrl_M, sizeof(uint64)*nlongintCase_Gender[1] * 3 * nsnps);
	cudaMalloc((void**)&gpu_genoCase_F, sizeof(uint64)*nlongintCase_Gender[2] * 3 * nsnps);
	cudaMalloc((void**)&gpu_genoCase_M, sizeof(uint64)*nlongintCase_Gender[3] * 3 * nsnps);

	cudaMalloc((void**)&gpu_pMarginalDistrSNP, sizeof(int)*MarginalDistrSNP_Y_DimensionY*nsnps);
	cudaMalloc((void**)&gpu_pMarginalDistrSNP_Y, sizeof(int)*MarginalDistrSNP_Y_DimensionY*MarginalDistrSNP_Y_DimensionX*nsnps);

	cudaMalloc((void**)&gpu_InteractionPairOffsetJ1, sizeof(int)*threadNum*blockNum);
	cudaMalloc((void**)&gpu_InteractionPairOffsetJ2, sizeof(int)*threadNum*blockNum);

	//copy geno data to GPU device and bind as texture
	cudaMemcpy(gpu_genoCtrl_F, genoCtrl_F, sizeof(uint64)*nlongintCase_Gender[0] * 3 * nsnps, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_genoCtrl_M, genoCtrl_M, sizeof(uint64)*nlongintCase_Gender[1] * 3 * nsnps, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_genoCase_F, genoCase_F, sizeof(uint64)*nlongintCase_Gender[2] * 3 * nsnps, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_genoCase_M, genoCase_M, sizeof(uint64)*nlongintCase_Gender[3] * 3 * nsnps, cudaMemcpyHostToDevice);

	cudaMemcpy(gpu_pMarginalDistrSNP, pMarginalDistrSNP, sizeof(int)*MarginalDistrSNP_Y_DimensionY*nsnps,cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_pMarginalDistrSNP_Y, pMarginalDistrSNP_Y, sizeof(int)*MarginalDistrSNP_Y_DimensionY*MarginalDistrSNP_Y_DimensionX*nsnps, cudaMemcpyHostToDevice);

	cudaBindTexture(0, genoCtrl_F_Texture, gpu_genoCtrl_F, sizeof(uint64)*nlongintCase_Gender[0] * 3 * nsnps);
	cudaBindTexture(0, genoCtrl_M_Texture, gpu_genoCtrl_M, sizeof(uint64)*nlongintCase_Gender[1] * 3 * nsnps);
	cudaBindTexture(0, genoCase_F_Texture, gpu_genoCase_F, sizeof(uint64)*nlongintCase_Gender[2] * 3 * nsnps);
	cudaBindTexture(0, genoCase_M_Texture, gpu_genoCase_M, sizeof(uint64)*nlongintCase_Gender[3] * 3 * nsnps);


	for (int i = 0, offset = 0; i <= totalNumberOfGridBlock; i++, offset = offset + blockNum*threadNum)
	{
		if (i % 100 == 0)
		{
			printf("\rProgress:%d%%", (int)floor(((float)i / totalNumberOfGridBlock) * 100));
			fflush(stdout);
		}
		//snp2 = snp1 + 1;
		for (; snp1 < nsnps - 1; snp1++)
		{
			if (firstLoop)
			{
				firstLoop = false;
			}
			else
			{
				snp2 = snp1 + 1;
			}

			for (; snp2 < nsnps; snp2++)
			{
				interactionInputOffsetJ1[shiftOffset] = snp1;
				interactionInputOffsetJ2[shiftOffset] = snp2;
				shiftOffset++;

				if (shiftOffset == blockNum*threadNum)
				{
					snp2++;
					break;
				}
			}

			if (shiftOffset == blockNum*threadNum)
			{
				break;
			}
		}
		firstLoop = true;
		shiftOffset = 0;
		cudaMemset(gpu_InteractionPairOffsetJ1, 0, sizeof(int)*blockNum*threadNum);
		cudaMemset(gpu_InteractionPairOffsetJ2, 0, sizeof(int)*blockNum*threadNum);


	}








}