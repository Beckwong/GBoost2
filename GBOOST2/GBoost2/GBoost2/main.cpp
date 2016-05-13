#include "main.h"


int bitCount(uint64 i)
{
	i = i - ((i >> 1) & 0x5555555555555555);
	i = (i & 0x3333333333333333) + ((i >> 2) & 0x3333333333333333);
	i = (i + (i >> 4)) & 0x0f0f0f0f0f0f0f0f;
	i = i + (i >> 8);
	i = i + (i >> 16);
	i = i + (i >> 32);
	return (int)i & 0x7f;
}

//absolute value
double Abs(double a)
{
	return (a < 0) ? -a : a;
}

int GetDataSize(char *filename, int  **DataSize)
{
	FILE *fp,*fp_i;
	int c, ndataset;
	time_t st, ed;
	int nsamples, nsnps, i, flag, ii;
	char filename_i[100];
	
	fp = fopen(filename, "r");
	if (fp == NULL)
	{
		printf("Can't open file %s\n", filename);
		exit(1);
	}

	ndataset = 0;
	while (!feof(fp))
	{
		ndataset++;
		fscanf(fp, "%s\n", &filename_i);
	}

	*DataSize = (int *)calloc(ndataset * 2, sizeof(int));

	ii = 0;
	rewind(fp);
	while (!feof(fp))
	{
		ii++;
		fscanf(fp, "%s\n", &filename_i);

		fp_i = fopen(filename_i, "r");
		if (fp_i == NULL)
		{
			printf("Can't open file %s\n", filename_i);
			exit(1);
		}
		printf("start getting data size of file %d: %s\n", ii, filename_i);
		time(&st);

		if (ii == 1)
		{
			nsamples = 0;
			bool tmp_flag = 0;
			while (1)
			{
				int c = fgetc(fp_i);
				switch (c)
				{
				case '\n':
					nsamples++;
					break;
				case  EOF:
					tmp_flag = 1;
					break;
				default:
					;
				}
				if (tmp_flag)
					break;
			}
		}

		rewind(fp_i);
		
		nsnps = 0;
		i = 0;
		flag = 1;
		while (1)
		{
			c = getc(fp_i);
			if (c == '\n')break;
			if (isspace(c))
			{
				flag = 1;
			}
			if (!isspace(c) && (flag == 1))
			{
				nsnps++;
				flag = 0;
			}
		}

		fclose(fp_i);
		time(&ed);

		(*DataSize)[ndataset * 0 + ii - 1] = nsamples;
		(*DataSize)[ndataset * 1 + ii - 1] = nsnps - 1;

	}

	fclose(fp);

	printf("cputime for getting data size: %d seconds.\n", (int)ed - st);
	return ndataset;
}

int initCUDA()
{
	int count,i;
	struct cudaDeviceProp prop;
	
	cudaGetDeviceCount(&count);
	if (count == 0)
	{
		fprintf(stderr, "There is no device.\n");
		return -1;
	}
	
	for (i = 0; i < count; i++)
	{
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess)
		{
			if (prop.major >= 1)
				break;
		}
	}

	if (i == count)
	{
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return -1;
	}

	cudaSetDevice(i);

	return 0;
}

void CalculateMarginalDistr(uint64* genoCtrl_F, uint64* genoCtrl_M, uint64* genoCase_F, uint64* genoCase_M, int* nlongintCase_Gender, int nsnps, int nsamples, int* pMarginalDistrSNP, int* pMarginalDistrSNP_Y)
{
	
	int count;

	for (int i1 = 0; i1 < nsnps; i1++)
	{
		for (int i2 = 0; i2 < 3; i2++)
		{
			count = 0;
			for (int i3 = 0; i3 < nlongintCase_Gender[0]; i3++)
			{
				count += bitCount(genoCtrl_F[i3 * 3 * nsnps + i2*nsnps + i1]);
			}
			pMarginalDistrSNP_Y[(i2*MarginalDistrSNP_Y_DimensionX + 0)*nsnps + i1] = count;

			count = 0;
			for (int i3 = 0; i3 < nlongintCase_Gender[1]; i3++)
			{
				count += bitCount(genoCtrl_M[i3 * 3 * nsnps + i2*nsnps + i1]);
			}
			pMarginalDistrSNP_Y[(i2*MarginalDistrSNP_Y_DimensionX + 1)*nsnps + i1] = count;

			count = 0;
			for (int i3 = 0; i3 < nlongintCase_Gender[2]; i3++)
			{
				count += bitCount(genoCase_F[i3 * 3 * nsnps + i2*nsnps + i1]);
			}
			pMarginalDistrSNP_Y[(i2*MarginalDistrSNP_Y_DimensionX + 2)*nsnps + i1] = count;

			count = 0;
			for (int i3 = 0; i3 < nlongintCase_Gender[3]; i3++)
			{
				count += bitCount(genoCase_F[i3 * 3 * nsnps + i2*nsnps + i1]);
			}
			pMarginalDistrSNP_Y[(i2*MarginalDistrSNP_Y_DimensionX + 2)*nsnps + i1] = count;

			pMarginalDistrSNP[i2*nsnps+i1] = 
				pMarginalDistrSNP_Y[(i2*MarginalDistrSNP_Y_DimensionX + 0)*nsnps + i1] +
				pMarginalDistrSNP_Y[(i2*MarginalDistrSNP_Y_DimensionX + 1)*nsnps + i1] +
				pMarginalDistrSNP_Y[(i2*MarginalDistrSNP_Y_DimensionX + 2)*nsnps + i1] +
				pMarginalDistrSNP_Y[(i2*MarginalDistrSNP_Y_DimensionX + 3)*nsnps + i1];
		}
	}
}

void CalculateGenoJointDistr(uint64* genoCtrl_F, uint64* genoCtrl_M, uint64* genoCase_F, uint64* genoCase_M, int* nlongintCase_Gender,int nsnps, int *GenoDistr, int snp1, int snp2, int* pMarginalDistrSNP_Y)
{
	int i1, i2, i3;
	register int count;

	for (i1 = 0; i1 < 2; i1++)
	{
		for (i2 = 0; i2 < 2; i2++)
		{
			count = 0;
			for (i3 = 0; i3 < nlongintCase_Gender[0]; i3++)
			{
				count += popcount(genoCtrl_F[i3*nsnps * 3 + i1*nsnps + snp1] & genoCtrl_F[i3 * 3 * nsnps + i2*nsnps + snp2]);
			}
			GenoDistr[i1 * 3 + i2] = count;

			count = 0;
			for (i3 = 0; i3 < nlongintCase_Gender[1]; i3++)
			{
				count += popcount(genoCtrl_M[i3*nsnps * 3 + i1*nsnps + snp1] & genoCtrl_M[i3 * 3 * nsnps + i2*nsnps + snp2]);
			}
			GenoDistr[9+i1*3+i2]=count;

			count = 0;
			for (i3 = 0; i3 < nlongintCase_Gender[2]; i3++)
			{
				count += popcount(genoCase_F[i3*nsnps * 3 + i1*nsnps + snp1] & genoCase_M[i3 * 3 * nsnps + i2*nsnps + snp2]);
			}
			GenoDistr[18 + i1 * 3 + i2] = count;
			count = 0;
			for (i3 = 0; i3 < nlongintCase_Gender[3]; i3++)
			{
				count += popcount(genoCase_M[i3*nsnps * 3 + i1*nsnps + snp1] & genoCase_M[i3 * 3 * nsnps + i2*nsnps + snp2]);
			}
			GenoDistr[27 + i1 * 3 + i2] = count;
		}
	}

	GenoDistr[2] = pMarginalDistrSNP_Y[(0 * MarginalDistrSNP_Y_DimensionX + 0)*nsnps + snp1] - GenoDistr[0] - GenoDistr[1];
	GenoDistr[5] = pMarginalDistrSNP_Y[(1 * MarginalDistrSNP_Y_DimensionX + 0)*nsnps + snp1] - GenoDistr[3] - GenoDistr[4];

	GenoDistr[6] = pMarginalDistrSNP_Y[(0 * MarginalDistrSNP_Y_DimensionX + 0)*nsnps + snp2] - GenoDistr[0] - GenoDistr[3];
	GenoDistr[7] = pMarginalDistrSNP_Y[(1 * MarginalDistrSNP_Y_DimensionX + 0)*nsnps + snp2] - GenoDistr[1] - GenoDistr[4];
	GenoDistr[8] = pMarginalDistrSNP_Y[(2 * MarginalDistrSNP_Y_DimensionX + 0)*nsnps + snp2] - GenoDistr[2] - GenoDistr[5];

	//control and male
	GenoDistr[11] = pMarginalDistrSNP_Y[(0 * MarginalDistrSNP_Y_DimensionX + 1)*nsnps + snp1] - GenoDistr[9] - GenoDistr[10];
	GenoDistr[14] = pMarginalDistrSNP_Y[(1 * MarginalDistrSNP_Y_DimensionX + 1)*nsnps + snp1] - GenoDistr[12] - GenoDistr[13];

	GenoDistr[15] = pMarginalDistrSNP_Y[(0 * MarginalDistrSNP_Y_DimensionX + 1)*nsnps + snp2] - GenoDistr[9] - GenoDistr[12];
	GenoDistr[16] = pMarginalDistrSNP_Y[(1 * MarginalDistrSNP_Y_DimensionX + 1)*nsnps + snp2] - GenoDistr[10] - GenoDistr[13];
	GenoDistr[17] = pMarginalDistrSNP_Y[(2 * MarginalDistrSNP_Y_DimensionX + 1)*nsnps + snp2] - GenoDistr[11] - GenoDistr[14];

	//case and female
	GenoDistr[20] = pMarginalDistrSNP_Y[(0 * MarginalDistrSNP_Y_DimensionX + 2)*nsnps + snp1] - GenoDistr[18] - GenoDistr[19];
	GenoDistr[23] = pMarginalDistrSNP_Y[(1 * MarginalDistrSNP_Y_DimensionX + 2)*nsnps + snp1] - GenoDistr[21] - GenoDistr[22];

	GenoDistr[24] = pMarginalDistrSNP_Y[(0 * MarginalDistrSNP_Y_DimensionX + 2)*nsnps + snp2] - GenoDistr[18] - GenoDistr[21];
	GenoDistr[25] = pMarginalDistrSNP_Y[(1 * MarginalDistrSNP_Y_DimensionX + 2)*nsnps + snp2] - GenoDistr[19] - GenoDistr[22];
	GenoDistr[26] = pMarginalDistrSNP_Y[(2 * MarginalDistrSNP_Y_DimensionX + 2)*nsnps + snp2] - GenoDistr[20] - GenoDistr[23];

	//case and male
	GenoDistr[29] = pMarginalDistrSNP_Y[(0 * MarginalDistrSNP_Y_DimensionX + 3)*nsnps + snp1] - GenoDistr[27] - GenoDistr[28];
	GenoDistr[32] = pMarginalDistrSNP_Y[(1 * MarginalDistrSNP_Y_DimensionX + 3)*nsnps + snp1] - GenoDistr[30] - GenoDistr[31];

	GenoDistr[33] = pMarginalDistrSNP_Y[(0 * MarginalDistrSNP_Y_DimensionX + 3)*nsnps + snp2] - GenoDistr[27] - GenoDistr[30];
	GenoDistr[34] = pMarginalDistrSNP_Y[(1 * MarginalDistrSNP_Y_DimensionX + 3)*nsnps + snp2] - GenoDistr[28] - GenoDistr[31];
	GenoDistr[35] = pMarginalDistrSNP_Y[(2 * MarginalDistrSNP_Y_DimensionX + 3)*nsnps + snp2] - GenoDistr[29] - GenoDistr[32];
}

DeviceProperties::DeviceProperties() {
	// Find the number of devices
	cudaGetDeviceCount(&devCount);

	if (devCount == 0) {
		return;
	}

	devPropArray = (cudaDeviceProp*)malloc(sizeof(struct cudaDeviceProp)*devCount);

	for (int i = 0; i < devCount; ++i)
	{
		cudaGetDeviceProperties(&devPropArray[i], i);
	}
}

DeviceProperties::~DeviceProperties() {
	free(devPropArray);
}

int DeviceProperties::getDeviceCount() {
	return devCount;
}

cudaDeviceProp DeviceProperties::getDeviceProp(int i) {
	return devPropArray[i];
}

void DeviceProperties::printDevProp(int i)
{
	cudaDeviceProp devProp = devPropArray[i];
	printf("Major revision number:         %d\n", devProp.major);
	printf("Minor revision number:         %d\n", devProp.minor);
	printf("Name:                          %s\n", devProp.name);
	printf("Total global memory:           %u\n", devProp.totalGlobalMem);
	printf("Total shared memory per block: %u\n", devProp.sharedMemPerBlock);
	printf("Total registers per block:     %d\n", devProp.regsPerBlock);
	printf("Warp size:                     %d\n", devProp.warpSize);
	printf("Maximum memory pitch:          %u\n", devProp.memPitch);
	printf("Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
	printf("Clock rate:                    %d\n", devProp.clockRate);
	printf("Total constant memory:         %u\n", devProp.totalConstMem);
	printf("Texture alignment:             %u\n", devProp.textureAlignment);
	printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
	printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
	printf("Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
	return;
}



int main()
{
	time_t st, ed;
	int *pMarginalDistrSNP;
	int *pMarginalDistrSNP_Y; 

	int *DataSize;
	int ndataset;

	int nsamples, nsnps;
	int *nCase_Gender, *nlongintCase_Gender;
	int *GenoJointDistr;

	int offset;
	 
	int LengthLongType = 64;
	uint64 mask1 = 0x0000000000000001;

	int buffersize = 50000;
	int buffersizeAssociation = 50000;

	FILE *fp, *fp_i;
	char filename_i[100];
	char *inputfilename = "filenamelist.txt";
	uint64 *genoCtrl_F, *genoCtrl_M, *genoCase_F, *genoCase_M;
	

	if (initCUDA() == -1)
	{
		printf("Unable to initialize CUDA\n");
		return -1;
	}

	printf("CUDA initialized\n");

	fp = fopen("filenamelist.txt", "r");

	if (fp == NULL)
	{
		printf("Can't open filenamelist.txt");
		return -1;
	}

	//computing the wordbits
	for (int i = 0; i < 65536; i++)
	{
		wordbits[i] = bitCount(i);
	}

	printf("Start loading data...\n");
	fflush(stdout);

	ndataset = GetDataSize(inputfilename, &DataSize);
	nsamples = DataSize[0];
	nsnps = 0;
	printf("Number of samples = %d\n", nsamples);

	for (int i = 0; i < ndataset; i++)
	{
		nsnps += DataSize[ndataset * 1 + i];
		printf("DataSize %d-th file: p[%d] = %d \n", i + 1, i + 1, DataSize[ndataset * 1 + i]);
	}

	printf("p = %d\n", nsnps);

	//get nCase_gender
	nCase_Gender = (int *)calloc(4, sizeof(int));
	nlongintCase_Gender = (int *)calloc(4, sizeof(int));

	int ipheno=0, igender = 0,itmp=0;
	int col = 0, row = 0;
	rewind(fp);

	fscanf(fp, "%s\n", &filename_i);
	printf("%s\n", filename_i);
	fp_i = fopen(filename_i, "r");


	while (!feof(fp_i))
	{
		if (col == 0)
		{
			fscanf(fp_i, "%d", &ipheno);
			col++;
			fscanf(fp_i, "%d", &igender);
			col++;
			nCase_Gender[ipheno * 2 + igender]++;
		}
		else
		{
			fscanf(fp_i, "%d", &itmp);
			col++;
			if (col == (DataSize[ndataset] + 1))
			{
				col = 0;
				row++;
			}
		}
		if (row >= nsamples)
		{
			break;
		}

	}

	printf("total sample: %d (nCtrl_F = %d; nCtrl_M = %d; nCase_F = %d; nCase_M = %d).\n", nsamples, nCase_Gender[0], nCase_Gender[1], nCase_Gender[2], nCase_Gender[3]);
	for (int i = 0; i < 4; i++)
	{
		nlongintCase_Gender[i] = ceil(((double)nCase_Gender[i]) / LengthLongType);
		printf("nlongintCase_Gender[%d] = %d\n", i, nlongintCase_Gender[i]);
	}
	fflush(stdout);

	genoCtrl_F = (uint64 *)calloc(3 * nsnps*nlongintCase_Gender[0], sizeof(uint64));
	genoCtrl_M = (uint64 *)calloc(3 * nsnps*nlongintCase_Gender[1], sizeof(uint64));
	genoCase_F = (uint64 *)calloc(3 * nsnps*nlongintCase_Gender[2], sizeof(uint64));
	genoCase_M = (uint64 *)calloc(3 * nsnps*nlongintCase_Gender[3], sizeof(uint64));

	//using cuda
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	if (!prop.canMapHostMemory) {
		printf("No map memory support");
		exit(1);
	}

	cudaSetDeviceFlags(cudaDeviceMapHost);

	rewind(fp);

	time(&st);
	col = 0;
	int ii = 0;
	int k = 0;   //k denotes the total number of snps in previouly processed files  


	while (!feof(fp))
	{
		ii++;
		fscanf(fp, "%s\n", &filename_i);

		fp_i = fopen(filename_i, "r");

		if (fp_i == NULL)
		{
			printf("Can't open file %s", filename_i);
			exit(1);
		}

		int row = 0;   
		int iCase_M=-1, iCase_F=-1, iCtrl_M=-1, iCtrl_F=-1;

		printf("Loading data in file %d: %s\n", ii, filename_i);
		fflush(stdout);

		while (!feof(fp_i))
		{
			if (col == 0)
			{
				fscanf(fp_i, "%d", &ipheno);
				col++;
				fscanf(fp_i, "%d", &igender);
				col++;

				if ((ipheno == 0) && (igender == 0))
				{
					iCtrl_F++;
				}
				else if ((ipheno == 0) && (igender == 1))
				{
					iCtrl_M++;
				}
				else if ((ipheno == 1) && (igender == 0))
				{
					iCase_F++;
				}
				else
				{
					iCase_M++;
				}
			}
			else
			{
				int tmp;
				fscanf(fp_i, "%d", &tmp);
				if ((ipheno == 0) && (igender == 0))
				{
					genoCtrl_F[((iCtrl_F / LengthLongType) * 3 + tmp)*nsnps + (col + k - 1)] |= (mask1 << (iCtrl_F%LengthLongType));
				}
				else if ((ipheno == 0) && (igender == 1))
				{
					genoCtrl_M[((iCtrl_M / LengthLongType) * 3 + tmp)*nsnps + (col + k - 1)] |= (mask1 << (iCtrl_M%LengthLongType));
				}
				else if ((ipheno == 1) && (igender == 0))
				{
					genoCase_F[((iCase_F / LengthLongType) * 3 + tmp)*nsnps + (col + k - 1)] |= (mask1 << (iCase_F%LengthLongType));
				}
				else
				{
					genoCase_M[((iCase_M / LengthLongType) * 3 + tmp)*nsnps + (col + k - 1)] |= (mask1 << (iCase_M%LengthLongType));
				}
				col++;

				if (col == DataSize[ndataset + ii - 1] + 1)
				{
					col = 0;
					row++;
				}
			}

			if (row >= nsamples)
			{
				break;
			}
		}

		fclose(fp_i);
		k += DataSize[ndataset + ii - 1];

	}

	fclose(fp);
	time(&ed);
	printf("cputime used for loading data: %d seconds", (int)ed - st);

	fflush(stdout);
	free(DataSize);

	//calculate marginal distribution


	pMarginalDistrSNP = (int *)malloc(MarginalDistrSNP_Y_DimensionY*nsnps*sizeof(int));
	pMarginalDistrSNP_Y = (int *)malloc(MarginalDistrSNP_Y_DimensionX*MarginalDistrSNP_Y_DimensionY*nsnps*sizeof(int));
	CalculateMarginalDistr(genoCtrl_F, genoCtrl_M, genoCase_F, genoCase_M, nlongintCase_Gender, nsnps, nsamples, pMarginalDistrSNP, pMarginalDistrSNP_Y);

	time(&st);

	GenoJointDistr = (int *)calloc(4 * 9, sizeof(int));
	
	list<int>offsetListJ1;
	list<int>offsetListJ2;
	list<int>::iterator iterJ1;
	list<int>::iterator iterJ2;

	vector<pair<int,int>>selectedpairs;
	vector<double>selectedpairsMeasure;

	static double mu[9][4] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
	static double mu0[9][4] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	static double mutmp[9][4];
	static double mu0tmp[9][4];
	double muError = 0;

	cuda_GetInteractionPairs(genoCtrl_F, genoCtrl_M, genoCase_F, genoCase_M, 
							nsnps, nsamples, nlongintCase_Gender, pMarginalDistrSNP, pMarginalDistrSNP_Y, wordbits, 65536,offsetListJ1,offsetListJ2);

	printf("Number of SNPs pairs passed screening step: %d\n",offsetListJ1.size());

	//test step post_correction
	iterJ1 = offsetListJ1.begin();
	iterJ2 = offsetListJ2.begin();

	int passedCount = 0;
	printf("\n Start post-correction...");
	fflush(stdout);

	for (int ii; ii < offsetListJ1.size(); ii++,iterJ1++,iterJ2++)
	{
		int snp1 = *iterJ1;
		int snp2 = *iterJ2;

		CalculateGenoJointDistr(genoCtrl_F, genoCtrl_M, genoCase_F, genoCase_M, nlongintCase_Gender, nsnps, GenoJointDistr, snp1, snp2, pMarginalDistrSNP_Y);

		
		

	}
		









}