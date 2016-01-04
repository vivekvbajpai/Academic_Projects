/*
 * gauss.c
 *
 *  Created on: 10 Oct 2015
 *      Author: osboxes
 */


/* Gaussian elimination without pivoting.
 * Compile with "gcc gauss.c"
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
#include <pthread.h>
#include <math.h>
/* Program Parameters */
#define MAXN 6000  /* Max value of N */
#define NO_OF_THREADS 4
int N;  /* Matrix size */

/* Matrices and vectors */
volatile float A[MAXN][MAXN]; /*= {{1,2,3,4,5,6,7,8},
							{2,3,4,1,7,4,5,6},
							{2,3,2,1,2,2,1,1},
							{4,5,4,5,5,3,4,2},
							{1,4,8,4,3,7,6,6},
							{9,7,7,3,2,8,5,4},
							{8,6,4,1,1,5,3,3},
							{8,3,2,6,4,6,9,7}};*/
volatile float B[MAXN];// = {5,6,7,3,5,2,9,5};
volatile float X[MAXN];// = {0,0,0,0,0,0,0,0};
/* A * X = B, solve for X */

/* junk */
#define randm() 4|2[uid]&3

/* Prototype */
void gauss();  /* The function you will provide.
		* It is this routine that is timed.
		* It is called only on the parent.
		*/
void eliminate(int data[]);

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

/* Initialize A and B (and X to 0.0s) */
void initialize_inputs() {
  int row, col;

  printf("\nInitializing...\n");
  for (col = 0; col < N; col++) {
    for (row = 0; row < N; row++) {
      A[row][col] = (float)rand() / 32768.0;
    }
    B[col] = (float)rand() / 32768.0;
    X[col] = 0.0;
  }

}

/* Print input matrices */
void print_inputs() {
  int row, col;

  if (N < 10) {
    printf("\nA =\n\t");
    for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
	printf("%5.2f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
      }
    }
    printf("\nB = [");
    for (col = 0; col < N; col++) {
      printf("%5.2f%s", B[col], (col < N-1) ? "; " : "]\n");
    }
  }
}

void print_X() {
  int row;

  if (N < 100) {
    printf("\nX = [");
    for (row = 0; row < N; row++) {
      printf("%5.2f%s", X[row], (row < N-1) ? "; " : "]\n");
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
  gauss();

  /* Stop Clock */
  gettimeofday(&etstop, &tzdummy);
  etstop2 = times(&cputstop);
  printf("Stopped clock.\n");
  usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
  usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

  /* Display output */
  print_X();

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

/******This method calculates parallely, using 4 threads,
 * the X matrix. The row operations required for calculation
 * is broken down, and shared between 4 threads, using data parallelism.
 */
void gauss() {
  int norm, row, col;  /* Normalization row, and zeroing
			* element row and col */
  float multiplier;
  pthread_t thread1,thread2,thread3,thread4;

  printf("Computing parallely.\n");

  /* Gaussian elimination */
  for (norm = 0; norm < N - 1; norm++) {

	  /*This section creates 4 threads, amongst which, all the rows
	  *of the matrix are almost equally divided. The row operation on first
	  *1/4 th rows are performed in 1st thread, the next 1/4 th by the second
	  *thread and so on. Please note that there is no data dependency between these
	  *threads, and hence no synchronisation is required.
	  */

	  /*We have taken separate "data" arrays for each thread, as they are needed to
	   * private to each thread.
	   * data[0]= start index of the row to be processed (inclusive).
	   * data[1]= end index of the row to be processed (inclusive).
	   * data[2]= row number for which the normalisation is taking place.
	   * */
	  //THREAD 1
	  int data1[3];
	  data1[0]= norm+1;
	  data1[1] = data1[0]+((N-norm)/NO_OF_THREADS)-1;
	  data1[2] = norm;
	  //printf("\n\nstart=%d end=%d norm=%d",data1[0],data1[1],data1[2]);
	  pthread_create(&thread1,NULL,(void*)&eliminate,(void*)&data1);

	  //THREAD 2
	  int data2[3];
	  data2[0]= data1[1]+1;
	  data2[1] = data2[0]+((N-norm)/NO_OF_THREADS)-1;
	  data2[2] = norm;
	  //printf("\nstart=%d end=%d norm=%d",data2[0],data2[1],data2[2]);
	  pthread_create(&thread2,NULL,(void*)&eliminate,(void*)&data2);

	  //THREAD 3
	  int data3[3];
	  data3[0]= data2[1]+1;
	  data3[1] = data3[0]+((N-norm)/NO_OF_THREADS)-1;
	  data3[2] = norm;
	  //printf("\nstart=%d end=%d norm=%d",data3[0],data3[1],data3[2]);
	  pthread_create(&thread3,NULL,(void*)&eliminate,(void*)&data3);

	  //THREAD 4
	  int data4[3];
	  data4[0]= data3[1]+1;
	  data4[1] = N-1;
	  data4[2] = norm;
	  //printf("\nstart=%d end=%d norm=%d",data4[0],data4[1],data4[2]);
	  pthread_create(&thread4,NULL,(void*)&eliminate,(void*)&data4);

	  pthread_join(thread1,NULL);
	  pthread_join(thread2,NULL);
	  pthread_join(thread3,NULL);
	  pthread_join(thread4,NULL);
  }
  /* (Diagonal elements are not normalized to 1.  This is treated in back
   * substitution.)
   */


  /* Back substitution */
  for (row = N - 1; row >= 0; row--) {
    X[row] = B[row];
    for (col = N-1; col > row; col--) {
      X[row] -= A[row][col] * X[col];
    }
    X[row] /= A[row][row];
  }
}

/**
 * This method does the normalisation operation for a single row.
 * It is executed by each thread, and iteratively normalise the rows
 * which belong to that particular thread.
 *
 * params:
 * data[0]= start index of the row to be processed (inclusive).
 * data[1]= end index of the row to be processed (inclusive).
 * data[2]= row number for which the normalisation is taking place.
 */
void eliminate(int data[3])
{
	int row,col;
	int norm = data[2];
	for (row = data[0]; row <= data[1]; row++) {
		float multiplier = A[row][norm] / A[norm][norm];
	      for (col = norm; col < N; col++) {
		A[row][col] -= A[norm][col] * multiplier;
	      }
	      B[row] -= B[norm] * multiplier;
	}
}
