/* Adithep Narula
 * an1375
 * parallel lab 3
 */

#include <stdio.h>
#include <cuda.h>
#include <math.h>

#define THREADS_PER_BLOCK 256
unsigned int getmax(unsigned int num[], unsigned int size);


__global__ void getmaxcu(unsigned int *d_numbers, unsigned int *d_max_array)
{

	//calculate thread id
	unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;


	//declare array to hold the elements that current block will access in shared memory
	__shared__ unsigned int d_sub_array[THREADS_PER_BLOCK];


	//get the corresponding element from global memory and save it in shared memory
	unsigned int tx = threadIdx.x;
	d_sub_array[tx] = d_numbers[global_tid];

	//use reduction 
	unsigned int stride;
	for(stride = THREADS_PER_BLOCK/2; stride > 0; stride >>= 1){
		__syncthreads();
		if(tx < stride){
			
			unsigned int temp1 = d_sub_array[tx];
			unsigned int temp2 = d_sub_array[tx + stride];

			if(temp1 < temp2){
				d_sub_array[tx] = temp2;
			}
			else{
				d_sub_array[tx] = temp1;
			}
		}
	}

	//maxmimum is found
	if(tx == 0){
		d_max_array[blockIdx.x] = d_sub_array[0];
	}
	
}


int main(int argc, char *argv[]){

	//step 0: set up 
	unsigned int h_size = 0;  // The size of the array
  	unsigned int i;  // loop index
   	
    
    if(argc !=2)
    {
       printf("usage: maxseq num\n");
       printf("num = size of the array\n");
       exit(1);
    }
   
    h_size = atol(argv[1]);

    unsigned int blocks_per_grid = (unsigned int) ceil((double)h_size/THREADS_PER_BLOCK);

    //find next multiple of block size 
    unsigned int h_size_adjusted = h_size;
    //if not multiple
    if(h_size_adjusted % THREADS_PER_BLOCK != 0) {
    	h_size_adjusted = h_size + (THREADS_PER_BLOCK - h_size % THREADS_PER_BLOCK); 
    }

    //printf("h_size_adjusted = %ld\n", h_size_adjusted);
 
	//step 1: memory declaration and allocation - allocate memory on both host and device before transfer data between host and device
	//step 1a: declare pointers 
	
	//store random numbers
	unsigned int *d_numbers;
	unsigned int *h_numbers; //pointer to the array
	
	//store max number from each block
	unsigned int *d_max_array;
	unsigned int *h_max_array;

	
	//step 1b: allocate host memory and populate
    h_numbers = (unsigned int *)calloc(h_size_adjusted, sizeof(unsigned int));



    srand(time(NULL)); // setting a seed for the random number generator
    // Fill-up the array with random numbers from 0 to size-1 
    for( i = 0; i < h_size; i++){
    	h_numbers[i] = rand() % h_size;
    }
       

    h_max_array = (unsigned int*)malloc(blocks_per_grid * sizeof(unsigned int));    

	//step 1c: allocate device memory
	cudaError_t err1 = cudaMalloc((void **)&d_numbers, h_size_adjusted * sizeof(unsigned int));
	cudaError_t err2 = cudaMalloc((void **)&d_max_array, blocks_per_grid * sizeof(unsigned int));

	if(err1 != cudaSuccess){
		printf("not success 1\n");
		exit(0);
	}

	if(err2 != cudaSuccess){
		printf("not success 2\n");
		exit(0);
	}

	//step 2: memory transfer - once memory is allocated, we can then transfer data to GPU global memory from the device

	//step 2a: first copy data from device to GPU using cudaMemcpy() function
	cudaError_t err3 = cudaMemcpy(d_numbers, h_numbers, h_size_adjusted * sizeof(unsigned int), cudaMemcpyHostToDevice);

	if(err3 != cudaSuccess){
		printf("not success 3\n");
		exit(0);
	}

	//step 2b: invoke the kernel
	getmaxcu<<<blocks_per_grid, THREADS_PER_BLOCK>>>(d_numbers, d_max_array);

	//step 2c: copy back from device to host
	cudaError_t err4 = cudaMemcpy(h_max_array, d_max_array, blocks_per_grid*sizeof(unsigned int), cudaMemcpyDeviceToHost);

	if(err4 != cudaSuccess){
		printf("not success 4\n");
		exit(0);
	}

	printf("my max: %ld\n", getmax(h_max_array, blocks_per_grid));
	printf("prof max: %ld\n", getmax(h_numbers, h_size));

	//printf("\n");

	//step 3: deallocate memory
	cudaFree(d_numbers);
	cudaFree(d_max_array);
	free(h_numbers);
	free(h_max_array);
	//free(h_size);

}

/*
   input: pointer to an array of long int
          number of elements in the array
   output: the maximum number of the array
*/
unsigned int getmax(unsigned int num[], unsigned int size)
{
  unsigned int i;
  unsigned int max = num[0];

  for(i = 1; i < size; i++)
	if(num[i] > max)
	   max = num[i];

  return( max );

}



