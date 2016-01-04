/* Matrix normalization.
 * Compile with "gcc matrixNorm.c" 
 */

/* ****** ADD YOUR CODE AT THE END OF THIS FILE. ******
 * You need not submit the provided code.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>
#include <cuda.h>
/* Program Parameters */
#define MAXN 8000  /* Max value of N */
#define NUM_THREADS_PER_BLOCK 32  /* Max value of N */

int N;  /* Matrix size */

/* Matrices */
float *A;

//Temporary matrix for testing
float temp[8][8] ={1,2,3,4,5,6,7,8,
							2,3,4,1,7,4,5,6,
							2,3,2,1,2,2,1,1,
							4,5,4,5,5,3,4,2,
							1,4,8,4,3,7,6,6,
							9,7,7,3,2,8,5,4,
							8,6,4,1,1,5,3,3,
							8,3,2,6,4,6,9,7};

float *B;

/* junk */
#define randm() 4|2[uid]&3

/* Prototype */
void matrixNorm();

/* returns a seed for srand based on the time */
unsigned int time_seed() {
  struct timeval t;
  struct timezone tzdummy;

  gettimeofday(&t, &tzdummy);
  return (unsigned int)(t.tv_usec);
}

/* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv) {
  int seed = 0;  /* Random seed */
  char uid[32]; /*User name */

  /* Read command-line arguments */
  srand(time_seed());  /* Randomize */

  if (argc == 3) {
    seed = atoi(argv[2]);
    srand(seed);
    printf("Random seed = %i\n", seed);
  } 
  if (argc >= 2) {
    N = atoi(argv[1]);
    if (N < 1 || N > MAXN) {
      printf("N = %i is out of range.\n", N);
      exit(0);
    }
  }
  else {
    printf("Usage: %s <matrix_dimension> [random seed]\n",
           argv[0]);    
    exit(0);
  }

  /* Print parameters */
  printf("\nMatrix dimension N = %i.\n", N);
}

/* Initialize A and B*/
void initialize_inputs() {
  int row, col;
  A = (float*)malloc(sizeof(float)*N*N);
  B = (float*)malloc(sizeof(float)*N*N);
  printf("\nInitializing...\n");
  for (col = 0; col < N; col++) {
    for (row = 0; row < N; row++) {
      //*(A+(row*N)+col) = temp[row][col];
      *(A+(row*N)+col) = (float)rand() / 32768.0;
      *(B+(row*N)+col) = 0.0;
    }
  }

}

/* Print input matrices */
void print_inputs() {
  int row, col;

  if (N < 10) {
    printf("\nA =\n\t");
    for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
	    printf("%5.2f%s", *(A+(row*N)+col), (col < N-1) ? ", " : ";\n\t");
      }
    }
  }
}

void print_B() {
    int row, col;

    if (N < 10) {
        printf("\nB =\n\t");
        for (row = 0; row < N; row++) {
            for (col = 0; col < N; col++) {
                printf("%1.10f%s",*(B+(row*N)+col), (col < N-1) ? ", " : ";\n\t");
            }
        }
    }
}

int main(int argc, char **argv) {
  /* Timing variables */
  struct timeval etstart, etstop;  /* Elapsed times using gettimeofday() */
  struct timezone tzdummy;
  clock_t etstart2, etstop2;  /* Elapsed times using times() */
  unsigned long long usecstart, usecstop;
  struct tms cputstart, cputstop;  /* CPU times for my processes */

  /* Process program parameters */
  parameters(argc, argv);

  /* Initialize A and B */
  initialize_inputs();

  /* Print input matrices */
  print_inputs();

  /* Start Clock */
  printf("\nStarting clock.\n");
  gettimeofday(&etstart, &tzdummy);
  etstart2 = times(&cputstart);

  /* Gaussian Elimination */
  matrixNorm();

  /* Stop Clock */
  gettimeofday(&etstop, &tzdummy);
  etstop2 = times(&cputstop);
  printf("Stopped clock.\n");
  usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
  usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

  /* Display output */
  print_B();
  
  free(A);
  free(B);
  /* Display timing results */
  printf("\nElapsed time = %g ms.\n",
	 (float)(usecstop - usecstart)/(float)1000);

  printf("(CPU times are accurate to the nearest %g ms)\n",
	 1.0/(float)CLOCKS_PER_SEC * 1000.0);
  printf("My total CPU time for parent = %g ms.\n",
	 (float)( (cputstop.tms_utime + cputstop.tms_stime) -
		  (cputstart.tms_utime + cputstart.tms_stime) ) /
	 (float)CLOCKS_PER_SEC * 1000);
  printf("My system CPU time for parent = %g ms.\n",
	 (float)(cputstop.tms_stime - cputstart.tms_stime) /
	 (float)CLOCKS_PER_SEC * 1000);
  printf("My total CPU time for child processes = %g ms.\n",
	 (float)( (cputstop.tms_cutime + cputstop.tms_cstime) -
		  (cputstart.tms_cutime + cputstart.tms_cstime) ) /
	 (float)CLOCKS_PER_SEC * 1000);
      /* Contrary to the man pages, this appears not to include the parent */
  printf("--------------------------------------------\n");
  
  exit(0);
}

/* ------------------ Above Was Provided --------------------- */

/****** You will replace this routine with your own parallel version *******/
/* Provided global variables are MAXN, N, A[][] and B[][],
 * defined in the beginning of this code.  B[][] is initialized to zeros.
 */
 
/* 
void matrixNorm() {
  int row, col; 
  float mu, sigma; // Mean and Standard Deviation

  printf("Computing Serially.\n");

    for (col=0; col < N; col++) {
        mu = 0.0;
        for (row=0; row < N; row++)
            mu += A[row][col];
        mu /= (float) N;
        sigma = 0.0;
        for (row=0; row < N; row++)
            sigma += powf(A[row][col] - mu, 2.0);
        sigma /= (float) N;
        for (row=0; row < N; row++) {
            if (sigma == 0.0)
                B[row][col] = 0.0;
            else
                B[row][col] = (A[row][col] - mu) / sigma;
        }
    }

}
*/

__global__ void calculateMean(float *d_A, float *d_mean, int *d_N) {
  int  col; 
   
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id<*d_N)
  {		
	float mean = 0.0;
	for(col=0;col<*d_N;col++)
	 {
		 mean+= *(d_A + (*(d_N) * col) +id);
	  }
	mean/=*d_N;
	*(d_mean+id)= mean;
  }

}

__global__ void calculateStandardDev(float *d_A, float *d_mean, int *d_N) {
  int  col; 
   
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id<*d_N)
  {		
	float mean = 0.0;
	for(col=0;col<*d_N;col++)
	 {
		 mean+= *(d_A + (*(d_N) * col) +id);
	  }
	mean/=*d_N;
	//mean = sqrt(mean);
	*(d_mean+id)= mean;
  }

}



__global__ void calculateVarianceSquared(float *d_A, float *d_mean,float *d_B, int *d_N) {
  int  col; 
   
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id<(*d_N)*(*d_N))
  {	
	*(d_B+id)= powf(*(d_A+id) - *(d_mean+(id%*(d_N))), 2.0);	
  }	  

}

__global__ void normalize(float *d_A, float *d_mean,float *d_B, float *d_std_dev, int *d_N) 
{
  int  col; 
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id<(*d_N)*(*d_N))
  {	
	*(d_B+id)= (*(d_A+id) - *(d_mean+(id%*(d_N))))/ *(d_std_dev+(id%*(d_N)));	
  }	  

}



/*This is the parallel version of the MatrixNorm method written in serial code.*/
void matrixNorm()
{
 
 float *d_A,*d_B,*d_mean,*d_std_dev;
 int *d_N;
 
 //lets allocate memory on device
 if(cudaMalloc(&d_A,sizeof(float)*N*N) != cudaSuccess)
 {
	 printf("error allocating memory!!!!");
 }
 if(cudaMalloc(&d_B,sizeof(float)*N*N) != cudaSuccess)
 {
	 printf("error allocating memory!!!!");
 }
  if(cudaMalloc(&d_mean,sizeof(float)*N) != cudaSuccess)
 {
	 printf("error allocating memory!!!!");
 }
  if(cudaMalloc(&d_std_dev,sizeof(float)*N) != cudaSuccess)
 {
	 printf("error allocating memory!!!!");
 }
  if(cudaMalloc(&d_N,sizeof(int)) != cudaSuccess)
 {
	 printf("error allocating memory!!!!");
 }
 


//send data to device memory
if(cudaMemcpy(d_A,A,sizeof(float)*(N*N),cudaMemcpyHostToDevice) != cudaSuccess)
{
	printf("error while sending data to device");
}
 
if(cudaMemcpy(d_N,&N,sizeof(int),cudaMemcpyHostToDevice) != cudaSuccess)
{
	printf("error while sending data to device");
} 
 
 printf("Computing parallely on GPU.\n");


//lets calculate the mean value for every column. The number of threads will be same as the number of columns
calculateMean<<<N/NUM_THREADS_PER_BLOCK+1,NUM_THREADS_PER_BLOCK>>>(d_A, d_mean, d_N);

//This method call calculate the variance squared value for each element.
//Number of threads will be equal to number of elements. 
calculateVarianceSquared<<<(N*N/NUM_THREADS_PER_BLOCK)+1,NUM_THREADS_PER_BLOCK>>>(d_A, d_mean,d_B, d_N);

///////lets find the standard deviation for each column/////////
calculateStandardDev<<<N/NUM_THREADS_PER_BLOCK+1,NUM_THREADS_PER_BLOCK>>>(d_B, d_std_dev, d_N);

//finally normalize the elements of the matrix//////////////////
normalize<<<(N*N/NUM_THREADS_PER_BLOCK)+1,NUM_THREADS_PER_BLOCK>>>(d_A, d_mean,d_B,d_std_dev, d_N);


//We need to copy the result from device back to host///////////////
if(cudaMemcpy(B,d_B,sizeof(float)*N*N,cudaMemcpyDeviceToHost) != cudaSuccess)
{
	printf("error while sending data to device");
}


//lets free the allocated memory on device
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_N);
cudaFree(d_mean);
cudaFree(d_std_dev);

}
