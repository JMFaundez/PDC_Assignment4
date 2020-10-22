#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#define TPB 256


double cpuSecond() {
	struct timeval tp;
	gettimeofday(&tp,NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}


void saxpyCPU(float *x, float *y, float a)
{
	for (int i=0; i<ARRAY_SIZE; i++)
	{
		y[i] = y[i] + x[i] * a;
	}
}

__device__ float saxpy(float x, float y, float a)
{
  return y + x*a;
}

__global__ void SAXPYKernel(float *x, float *y, float a)
{
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
	y[i] = saxpy(x[i],y[i],a);
}

int main(int argc, char *argv[])
{
  const float a = 0.5f;
  int Dg = (ARRAY_SIZE + TPB - 1)/TPB; 
  float *h_x = (float*) malloc(ARRAY_SIZE*sizeof(float));
  float *h_y = (float*) malloc(ARRAY_SIZE*sizeof(float));
  float *ycpu = (float*) malloc(ARRAY_SIZE*sizeof(float));
  double timeCPU, timeGPU, istart;
  float dif;
  float tol=1e-5;
  FILE *fp;

  fp = fopen("times_ex2.dat","a");

  int flag1=1;
  srand(923);	
	for (int i=0; i<ARRAY_SIZE; i++)
	{
		h_x[i] = 2*(double)random() / (double)RAND_MAX; //i*1.0;
		h_y[i] = 10*(double)random() / (double)RAND_MAX; //
		ycpu[i] = h_y[i];
	}
	
	// cpu implementation
	printf("Computing SAXPY on the CPU... ");
	istart = cpuSecond();
	saxpyCPU(h_x,ycpu,a);
	timeCPU = cpuSecond() - istart;
	printf(" Done!\n\n");
  
  // Declare a pointer for an array of floats
  float *d_x = 0;
  float *d_y = 0;
  
  // Allocate device memory to store the output array
  cudaMalloc(&d_x, ARRAY_SIZE*sizeof(float));
  cudaMalloc(&d_y, ARRAY_SIZE*sizeof(float));
  
	printf("Computing SAXPY on the GPU... ");
  // Move data from CPU to GPU
  cudaMemcpy(d_x, h_x, ARRAY_SIZE*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, ARRAY_SIZE*sizeof(float),cudaMemcpyHostToDevice);
  
  // Launch kernel to compute and store distance values
	istart = cpuSecond();
  SAXPYKernel<<<Dg, TPB>>>(d_x, d_y, a);
  cudaDeviceSynchronize(); 
	timeGPU = cpuSecond() - istart;
  
  // Move data from CPU to GPU
  cudaMemcpy(h_y, d_y, ARRAY_SIZE*sizeof(float),cudaMemcpyDeviceToHost);
//	timeGPU = cpuSecond() - istart;
  cudaFree(d_x); // Free the memory
  cudaFree(d_y); // Free the memory
	printf(" Done!\n\n");
	
	printf("Comparing the output for each implementation... ");
	for(int i=0; i<ARRAY_SIZE; i++)
	{
		dif = (ycpu[i] - h_y[i]);
		if (abs(dif)>tol)
		{
			printf("Error!,i=%d, dif=%f\n",i,dif);
			flag1=0;
		}
	}
	if (flag1) printf(" Done!\n");

	printf("TCPU= %.10e [s],TGPU= %.10e\n",timeCPU,timeGPU);
	fprintf(fp,"%*d\t%.10e\t%.10e\n",12,ARRAY_SIZE,timeCPU,timeGPU);
	fclose(fp);
  return 0;
}
