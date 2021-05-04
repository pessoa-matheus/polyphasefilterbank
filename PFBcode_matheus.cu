//nvcc -o pfb_cuda pfb_cuda.cu -lgomp -lfftw3f -lm -lcufft -lcufftw

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <fftw3.h>
#include <omp.h>
#include <cufft.h>
#include <math.h>
#include <string.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <array>   
#include <time.h>
// System includes
#include <assert.h>
#include <iomanip>
// CUDA runtime
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#ifndef M_PI
	#define M_PI 3.141592
#endif

#define NTHREAD_PFB 128

#include <iostream>
using std::cerr;
using std::endl;
#include <fstream>
using std::ofstream;
#include <cstdlib> // for exit function

using namespace std;
//nvcc -o pfb_cuda pfb_cuda.cu -lgomp -lfftw3f -lm -lcufft -lcufftw

/*--------------------------------------------------------------------------------*/
struct PFB_GPU_PLAN {
	int n;
	int nchan;
	int nchunk;
	int ntap;
	int nthread;

	float* win_gpu;
	float* dat_gpu;
	float* dat_tapered_gpu;
	cufftComplex* dat_trans_gpu;
	float* pfb_gpu;
	cufftHandle cuplan;
};

/*-------------------------------------------------------------------------------*/

int get_nchunk(int n, int nchan, int ntap)
{
	return n / nchan - ntap;
}

/*--------------------------------------------------------------------------------*/

void coswin(float* vec, int n)
{
	for (int i = 0; i < n; i++) {
		float xx = 2.0 * (i - n / 2) / (n - 1);
		vec[i] = 0.5 + 0.5 * cos(xx * M_PI);
	}
}

/*--------------------------------------------------------------------------------*/
void mul_sinc(float* vec, int n, int ntap)
{
	for (int i = 0; i < n; i++) {
		float xx = ntap * 1.0 * (i - n / 2) / (n - 1);
		if (xx != 0)
			vec[i] = vec[i] * sin(M_PI * xx) / (M_PI * xx);
	}
}
/*--------------------------------------------------------------------------------*/
struct PFB_GPU_PLAN* setup_pfb_plan(int n, int nchan, int ntap)
{
	struct PFB_GPU_PLAN* tmp = (struct PFB_GPU_PLAN*)malloc(sizeof(struct PFB_GPU_PLAN));
	int nn = nchan * ntap;
	float* win = (float*)malloc(sizeof(float) * nn); //window needs to be the size of channels * taps 
	coswin(win, nn); //multiplies by a cosine
	mul_sinc(win, nn, ntap); //multiplies by the sinc function

	int nchunk = get_nchunk(n, nchan, ntap);
	//Here we use -> because we are accessing from a struct	
	tmp->n = n;
	tmp->nchan = nchan;
	tmp->nchunk = nchunk;
	tmp->ntap = ntap;
	tmp->nthread = NTHREAD_PFB;
	if (cudaMalloc((void**) & (tmp->dat_gpu), sizeof(float) * n) != cudaSuccess)
		printf("Malloc error on dat_gpu.\n");

	if (cudaMalloc((void**) & (tmp->dat_tapered_gpu), sizeof(float) * nchunk * nchan) != cudaSuccess)
		printf("Malloc error on dat_tapered_gpu.\n");

	if (cudaMalloc((void**) & (tmp->win_gpu), sizeof(float) * nn) != cudaSuccess)
		printf("Malloc error on win_gpu.\n");

	if (cudaMalloc((void**) & (tmp->dat_trans_gpu), sizeof(cufftComplex) * nchan * nchunk) != cudaSuccess)
		printf("Malloc error on dat_trans_gpu.\n");
	if (cudaMalloc((void**) & (tmp->pfb_gpu), sizeof(float) * nchan) != cudaSuccess)
		printf("Malloc error on pfb_gpu.\n");

	if (cudaMemcpy(tmp->win_gpu, win, nn * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
		printf("Copy error on win_gpu.\n");


	if (cufftPlan1d(&(tmp->cuplan), nchan, CUFFT_R2C, nchunk) != CUFFT_SUCCESS)
		printf("we had an issue creating plan.\n");


	return tmp;
}
/*--------------------------------------------------------------------------------*/
void destroy_pfb_gpu_plan(struct PFB_GPU_PLAN* plan)
{
	cufftDestroy(plan->cuplan);
	cudaFree(plan->dat_gpu);
	cudaFree(plan->win_gpu);
	cudaFree(plan->dat_tapered_gpu);
	cudaFree(plan->dat_trans_gpu);
	cudaFree(plan->pfb_gpu);

	free(plan);
	//we need to free memory space after we're done with the functions
}
/*--------------------------------------------------------------------------------*/

void format_data(float* dat, int n, int nchan, int ntap, float* win, float** dat_out, int* nchunk)
{
	int nn = n / nchan - ntap;
	float* dd = (float*)malloc(sizeof(float) * nn * nchan);
	memset(dd, 0, sizeof(float) * nn * nchan);
	for (int i = 0; i < nn; i++)
		for (int j = 0; j < ntap; j++)
			for (int k = 0; k < nchan; k++)
				dd[i * nchan + k] += dat[(i + j) * nchan + k] * win[j * nchan + k];
	*nchunk = nn;
	*dat_out = dd;
}

/*--------------------------------------------------------------------------------*/
__global__
void gpu_int162float32(short* in, float* out, int n)
{
	int myi = blockIdx.x * blockDim.x + threadIdx.x;
	int nthread = gridDim.x * blockDim.x;
	for (int i = 0; i < n; i += nthread)
		if (myi + i < n)
			out[myi + i] = in[myi + i];

}
/*--------------------------------------------------------------------------------*/
__global__
void format_data_gpu(float* dat, int nchunk, int nchan, int ntap, float* win, float* dat_out)
{
	int myi = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = 0; i < nchunk; i++) {
		float tot = 0;
		for (int j = 0; j < ntap; j++) {
			tot += dat[(i + j) * nchan + myi] * win[j * nchan + myi];
		}
		dat_out[i * nchan + myi] = tot;
	}
}
/*--------------------------------------------------------------------------------*/
__global__
void sum_pfb_gpu(cufftComplex* dat_trans, int nchan, int nchunk, float* pfb_out)
{
	int myi = blockIdx.x * blockDim.x + threadIdx.x;
	float tot = 0;
	for (int i = 0; i < nchunk; i++) {
		//cufftComplex tmp = dat_trans[myi + i * nchan];
		cufftComplex tmp = dat_trans[myi + i * (nchan/2+1)];
		tot += tmp.x * tmp.x + tmp.y * tmp.y;
	}
	pfb_out[myi] = tot;

	}
/*--------------------------------------------------------------------------------*/
void pfb_gpu(float* dat, float* pfb, struct PFB_GPU_PLAN* pfbplan)
{
	if (cudaMemcpy(pfbplan->dat_gpu, dat, pfbplan->n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
		printf("Copy error on dat_gpu.\n");

	cout << "copied data" << endl;
	cout << "nchunk is " << pfbplan->nchunk << endl;
	cout << "nthread is " << pfbplan->nthread << endl;
	cout << "nchan is " << pfbplan->nchan << endl;
	cout << "n is " << pfbplan->n << endl;
		format_data_gpu << <pfbplan->nchan / pfbplan->nthread, pfbplan->nthread >> > (pfbplan->dat_gpu, pfbplan->nchunk, pfbplan->nchan, pfbplan->ntap, pfbplan->win_gpu, pfbplan->dat_tapered_gpu);
		float *dat2 = (float*)malloc(pfbplan->n * sizeof(float));//Matheus created a new array dat2 on the cpu
		cudaMemcpy(dat2, pfbplan->dat_tapered_gpu, pfbplan->nchan * sizeof(float), cudaMemcpyDeviceToHost);//Matheus added to come back to the CPU
		for (int i = 0; i < 10; i++) {
			cout << "copied back from GPU " << dat2[i] << endl;//this is giving -4.31E8 
		}
		if (cufftExecR2C(pfbplan->cuplan, pfbplan->dat_tapered_gpu, pfbplan->dat_trans_gpu) != CUFFT_SUCCESS)
			printf("Error executing FFT on GPU.\n");
					
		int nn = 1;
		cudaMemcpy(dat2, pfbplan->dat_trans_gpu, 10*pfbplan->nchan * sizeof(float), cudaMemcpyDeviceToHost);//Matheus added to come back to the CPU
		for (int i = pfbplan-> nchan-10; i < pfbplan->nchan; i++) {
			cout << "copied back from GPU after fft  " << dat2[2*i+nn*(pfbplan->nchan+2)] << "  " << dat2[2*i+nn*(pfbplan->nchan+2)+1] << endl;//this is giving -4.31E8 
		}
		int tmp = pfbplan->nchunk;
		//pfbplan->nchunk = 1;
		sum_pfb_gpu << <pfbplan->nchan / pfbplan->nthread, pfbplan->nthread >> > (pfbplan->dat_trans_gpu, pfbplan->nchan, pfbplan->nchunk, pfbplan->pfb_gpu);
	
		pfbplan->nchunk=tmp;
		//}
	if (cudaMemcpy(pfb, pfbplan->pfb_gpu, sizeof(float) * pfbplan->nchan, cudaMemcpyDeviceToHost) != cudaSuccess)
		printf("Error copying PFB to cpu.\n");
	for (int i = 0; i < 10; i++)
		cout << "PFB(" << i << ") is " << pfb[-i] << endl;
}

/*--------------------------------------------------------------------------------*/
void pfb_gpu16(short int* dat, float* pfb, struct PFB_GPU_PLAN* pfbplan)
{
	if (cudaMemcpy(pfbplan->dat_tapered_gpu, dat, pfbplan->n * sizeof(short int), cudaMemcpyHostToDevice) != cudaSuccess)
		printf("Copy error on dat_gpu.\n");
	gpu_int162float32 << <8 * pfbplan->nchan / pfbplan->nthread, pfbplan->nthread >> > ((short int*)pfbplan->dat_tapered_gpu, pfbplan->dat_gpu, pfbplan->n);

	format_data_gpu << <pfbplan->nchan / pfbplan->nthread, pfbplan->nthread >> > (pfbplan->dat_gpu, pfbplan->nchunk, pfbplan->nchan, pfbplan->ntap, pfbplan->win_gpu, pfbplan->dat_tapered_gpu);
	if (cufftExecR2C(pfbplan->cuplan, pfbplan->dat_tapered_gpu, pfbplan->dat_trans_gpu) != CUFFT_SUCCESS)
		printf("Error executing FFT on GPU.\n");

	sum_pfb_gpu << <pfbplan->nchan / pfbplan->nthread, pfbplan->nthread >> > (pfbplan->dat_trans_gpu, pfbplan->nchan, pfbplan->nchunk, pfbplan->pfb_gpu);

	if (cudaMemcpy(pfb, pfbplan->pfb_gpu, sizeof(float) * pfbplan->nchan, cudaMemcpyDeviceToHost) != cudaSuccess)
		printf("Error copying PFB to cpu.\n");

}

void sinwave(float* out, float n, float fs)
{
	float M = -1.0;
	float N = 1.0;
	for (int i = 0; i < n; i++)
	{
		//cout << i << endl;
		float y;
		y = sin(2 * M_PI * 70.e6 * i / fs)+ sin(2 * M_PI * 71e6 * i / fs)+sin(2 * M_PI * 75e6 * i / fs)+ M + (rand() / (RAND_MAX / (N - M)));
		out[i] = y;
	}
}


/*================================================================================*/

int main(int argc, char* argv[])
{

	float n = 2000000;
	float fs = 2e8;
	float* dat = (float*)malloc(sizeof(float) * n);
	sinwave(dat, n, fs);

	float max = dat[0];
	float min = dat[0];

	cout << "the first element is " << dat[0] << endl;

	for (int i = 1; i < n; i++) {
		if (dat[i] > max)
			max = dat[i];
		if (dat[i] < min)
			min = dat[i];
	}
	cout << "The max value is" << max << endl;
	cout << "The min value is" << min << endl;
	// Dump each 100 values to screen
	//int nchan = 3584 * 4;
	int nchan = 4096 * 4;
	cout << "Number of channels is : " << nchan << endl;
	int ntap = 4;
	cout << "We're executing a " << ntap << " tap PFB " << "with " << nchan << "channels" << endl;

	//int nn=nchan*ntap;
	int niter = 1;

	struct PFB_GPU_PLAN* pfbplan = setup_pfb_plan(n, nchan, ntap);
	float* pfb_sum = (float*)malloc(sizeof(float) * nchan);
	memset(pfb_sum, 0, sizeof(float) * nchan);// array to be filled with values
#if 0
	short int* dd = (short int*)malloc(n * sizeof(short int));
#else
	short int* dd;
	if (cudaMallocHost(&dd, sizeof(short int) * n) != cudaSuccess)
		printf("cuda malloc error on dd.\n");
#endif
	memset(dd, 0, sizeof(short int) * n);

	for (int i = 0; i < n; i++)
		dd[i] = 1000 * dat[i];

	double t1 = omp_get_wtime();

	cout << "calling pfb" << endl;
	pfb_gpu(dat, pfb_sum, pfbplan);// this is the float version
	//pfb_gpu16(dd, pfb_sum, pfbplan);
	
	for (int i = 0; i < nchan; i++)
		if (pfb_sum[i] > 1e20) {
			cout << "Sample " << i << " had value " << pfb_sum[i] << endl;
			return 0;
		}
	//cout << "after sanity check" << endl;
	
	FILE* scan;
	scan = fopen("sine2times.txt", "w+");
	for (int i = 0; i < nchan; i++)
	{
		fprintf(scan,"%f\n", pfb_sum[i]);
		//cout << "written to a file " << pfb_sum[i] << endl;
	}
	fclose(scan);
	cout << "pfb_sum type is : " << typeid(pfb_sum).name() << '\n';

	double t2 = omp_get_wtime();
	double throughput = 1.0 * nchan * pfbplan->nchunk * niter / (t2 - t1) / 1e6;
	printf("pfb[0] is now %12.4g, with time per iteration %12.4e and throughput %12.4f Msamp/s\n", pfb_sum[0], (t2 - t1) / niter, throughput);

	float* tmpv = (float*)malloc(sizeof(float) * n);
	if (cudaMemcpy(tmpv, pfbplan->dat_gpu, n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
		printf("Error copying temp data back to memory.\n");
	printf("vals are %12.4f %d\n", tmpv[0], dd[0]);
	destroy_pfb_gpu_plan(pfbplan);


	/*
	int numero = 0;
	
	for (int i = 0; i < 1000; i++) {
		cout << "fis values read after" << dat3[i] << endl;
		if (dat3[i] > 1e10){
			cout << "problem" << endl;
			numero = numero +1;
			}
	}
	cout << "numero total de erros e " << numero << endl;
	*/
}
#if 0
	int rank = 1;
	int nchunk = get_nchunk(n, nchan, ntap);
	float* dat_out = (float*)malloc(sizeof(float) * n);
	//fftwf_complex *crap=fftwf_malloc(nchunk*(nchan/2+1)*sizeof(fftwf_complex));
	fftwf_complex* crap = fftwf_alloc_complex(nchunk * (nchan / 2 + 1));
	fftwf_plan plan=fftwf_plan_many_dft_r2c(rank,&nchan,nchunk,dat_out,NULL,1,nchan,crap,NULL,1,nchan/2+1,FFTW_ESTIMATE);
	//fftwf_plan plan = fftwf_plan_many_dft_r2c(rank, &nchan, nchunk, dat_out, NULL, 1, nchan, crap, NULL, 1, nchan / 2 + 1, FFTW_ESTIMATE);
	fftwf_execute(plan);
	FILE* outfile;
	outfile = fopen("out_dat.raw", "w");
	fwrite(&nchan, 1, sizeof(int), outfile);
	fwrite(&nchunk, 1, sizeof(int), outfile);
	fwrite(dat_out, nchan * nchunk, sizeof(float), outfile);
	fclose(outfile);
	outfile = fopen("out_trans.raw", "w");
	int asdf = nchan / 2 + 1;
	fwrite(&asdf, 1, sizeof(int), outfile);
	fwrite(&nchunk, 1, sizeof(int), outfile);
	fwrite(crap, asdf * nchunk, sizeof(fftwf_complex), outfile);
	fclose(outfile);
#endif

