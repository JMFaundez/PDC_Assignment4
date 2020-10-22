#include <stdio.h>
#define N 256
#define TPB 256

__global__ void Kernel()
{
  const int myId = blockIdx.x*blockDim.x + threadIdx.x;
  printf("Hello World! My threadId is %2d\n", myId);
}

int main()
{
  // Launch kernel to print Hello World
  Kernel<<<N/TPB, TPB>>>();
  
  cudaDeviceSynchronize();
  
  return 0;
}
