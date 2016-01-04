/*
 * gauss.c
 *
 *  Created on: 10 Oct 2015
 *  Author: vbajpai
 */


/* Gaussian elimination without pivoting using MPI.
 * Compile with "mpicc -c gauss.c"
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>
#include <mpi.h>

/* Program Parameters */
#define MAXN 2000  /* Max value of N */
/* A * X = B, solve for X */

/* junk */
#define randm() 4|2[uid]&3

/* Prototype */


/*This method does the job of scattering the data accross all processes.
*/
void scatter_data(

         int  norm_row    /* out */, 

         int     my_rank  /* in  */, 

         int     p        /* in  */,
	 float *A ,
	 int N,
	 int* local_no_of_rows,
	 int* local_matrix_size,
	 float* local_norm_row,
	float* local_matrix_A,
	int* rows_per_process);


/*This method eliminates the lower half elements of the matrix
for gaussian elimination.
*/
void eliminate(int local_matrix_size,int local_no_of_rows, 
		float* local_norm_row, float* local_matrix_A,
		int norm,float* local_matrix_B,float local_norm_B);



/* returns a seed for srand based on the time */
unsigned int time_seed() {
  struct timeval t;
  struct timezone tzdummy;

  gettimeofday(&t, &tzdummy);
  return (unsigned int)(t.tv_usec);
}

/* Set the program parameters from the command-line arguments */
int parameters(int argc, char **argv) {
  int seed = 0;  /* Random seed */
  char uid[32]; /*User name */
  int N=0;
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
  return N;
}

/* Initialize A and B (and X to 0.0s) */
void initialize_inputs(float *A, float *B, float *X, int N) {
  int row, col;

  printf("\nInitializing...\n");
  for (col = 0; col < N; col++) {
    for (row = 0; row < N; row++) {
      *(A+(N*row)+col) = (float)rand() / 32768.0;
    }
    *(B+col) = (float)rand() / 32768.0;
    *(X+col) = 0.0;
  }

}

/* Print input matrices */
void print_inputs(float *A, float *B, int N) {
  int row, col;

  if (N < 10) {
    printf("\nA =\n\t");
    for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
	printf("%5.2f%s", *(A+(N*row)+col), (col < N-1) ? ", " : ";\n\t");
      }
    }
    printf("\nB = [");
    for (col = 0; col < N; col++) {
      printf("%5.2f%s", *(B+col), (col < N-1) ? "; " : "]\n");
    }
  }
}


main(int argc, char **argv) {

//declare the required data structures

  int N =32;  /* Matrix size */

  /* Matrices and vectors */
  float *A= malloc(MAXN*MAXN);
  int i,j;
   
   //code commented. was used for testing.
  /*
  float temp[64] = {1,2,3,4,5,6,7,8,
		  2,3,4,1,7,4,5,6,
		  2,3,2,1,2,2,1,1,
		  4,5,4,5,5,3,4,2,
		  1,4,8,4,3,7,6,6,
		  9,7,7,3,2,8,5,4,
		  8,6,4,1,1,5,3,3,
		  8,3,2,6,4,6,9,7};

 for(i=0;i<N;i++){
	for(j=0;j<N;j++)
		{
		*(A+((N*i)+j))=temp[i*N+j];
		//printf(" %f",*(A+((8*i)+j)));
		}
	//printf("\n");	
	}

*/
float B[MAXN];// = {5,6,7,3,5,2,9,5};
float X[MAXN];// = {0,0,0,0,0,0,0,0};


  int my_rank=0;   /* My process rank           */
  int p;         /* The number of processes   */
  
  //clock time recording variables
  double      start_time,end_time=0.0; 

///////////////////MPI code starts////////////////////


  //status variable used to check status of communication operation.	 
  MPI_Status  status;

  /* Let the system do what it needs to start up MPI */
   MPI_Init(&argc, &argv);

  /* Get my process rank */
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  /* Find out how many processes are being used */
   MPI_Comm_size(MPI_COMM_WORLD, &p);
     
  if(my_rank==0)
  {
  /* Process program parameters */
  N = parameters(argc, argv);
	
  /* Initialize A and B */

  initialize_inputs(A, B, X,N);

  /* Print input matrices */
  print_inputs(A, B,N);


  //Start clock and record the start time.
  start_time = MPI_Wtime();

} 

  //broadcast the size of the matrix read by the to all processes.
  MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

  //we need all processes to wait here until all others arrive. 
  //we need to make sure that the input matrix has been initialized
  //by process 0 and the marix size has been propogated to all processes.
  MPI_Barrier(MPI_COMM_WORLD);


  //declare the local variables
   int local_no_of_rows;  //number of rows to be processesd by each process
   int local_matrix_size; //size of the matrix
   float local_norm_row[N]; //the current normaization row
   float local_matrix_A[N][N]; //the part of A matrix on which each process will work
   float local_matrix_B[N];  //the part of B matrix on which each process will work
   int rows_per_process[p];  //the number of rows distributed to each process
   float local_norm_B;      //the element on which B will be normalized
   int displ[p];	    //displacement variable
   int norm=0;              //the index of the current normalizing row
 
 //lets begin. The loop is outermost loop of Gaussian elimination operation.
 for (norm = 0; norm < N - 1; norm++) {

   //lets scatter the data accross all processes.   	
   //This method scatters the matrix A, and broadcasts the current normalizing row,
   // number of rows each process will work on.
   scatter_data(norm,
		my_rank,
		p,
		A,
		N,
		&local_no_of_rows,
		&local_matrix_size,
		local_norm_row,
		&(local_matrix_A[0][0]),
		&rows_per_process[0]);
   
   //lets calculate the send counts and displacement vector for scatter of B matrix.
   if(my_rank==0)
   {
	//printf(" %d", *(rows_per_process));
     *(displ)=0;
     for(j=1;j<p;j++)
		{
		 *(displ+j) = rows_per_process[j-1]+ *(displ+j-1);
		 //printf(" %d", *(rows_per_process+j));	
		}
   }
   
   //This method call scatter the matrix B. Different processes may have different
   //number of elements to work on, when the size of matrix is not completely divisible
   //by number of processes. Hence we have used MPI_Scatterv(), instead of MPI_Scatter
   MPI_Scatterv(B+norm+1, rows_per_process, displ, MPI_FLOAT,local_matrix_B,local_no_of_rows, MPI_FLOAT, 
                                                              0, MPI_COMM_WORLD); 

   //lets broadcast the element against which matrix B will be normalized.
   local_norm_B = B[norm];
   MPI_Bcast(&local_norm_B, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   
   //each process performs the following elimination operation on their
   //share of the matrix A and B.
   eliminate(local_matrix_size,
		local_no_of_rows, 
		&local_norm_row[0], 
		&(local_matrix_A[0][0]),
		norm,
		&(local_matrix_B[0]),
		local_norm_B);   

   //we need to calculate the counts and displacement for the Gather operation
   //of the processed matrix A, after each iteration.
   int counts_for_gather[p];
     int displacements_for_gather[p];
    if(my_rank==0)
   {
     
     *(displacements_for_gather)=0;
     counts_for_gather[0] = rows_per_process[0]*local_matrix_size;
  	
     for(j=1;j<p;j++)
		{
		counts_for_gather[j] = rows_per_process[j]*local_matrix_size;
		 *(displacements_for_gather+j) = counts_for_gather[j-1]+ *(displacements_for_gather+j-1);
		}
   }


   //here we gather the processed matrix A from all processes and store it locally
   MPI_Gatherv(local_matrix_A,
		local_no_of_rows*local_matrix_size, 
		MPI_FLOAT,
                A+(N*(norm+1)),
		counts_for_gather, 
		displacements_for_gather,
                MPI_FLOAT, 
		0, 
		MPI_COMM_WORLD);

   //similarly we gather the processed matrix B.
   MPI_Gatherv(local_matrix_B,
		local_no_of_rows, 
		MPI_FLOAT,
                B+norm+1,
		rows_per_process, 
		displ,
                MPI_FLOAT, 
		0, 
		MPI_COMM_WORLD);



 }

  //We need to wait for al processes to complete before we go ahead with
  //back subsitution.
  MPI_Barrier(MPI_COMM_WORLD);

  //perform the back substitution operation only by process 0.
  int row,col;
  if(my_rank==0){
  /* Back substitution */
  for (row = N - 1; row >= 0; row--) {
    X[row] = B[row];
    for (col = N-1; col > row; col--) {
      X[row] -= *(A+(N*row)+col) * X[col];
    }
    X[row] /= *(A+(N*row)+col);
  }


  //Stop clock as operation is finished.
  end_time = MPI_Wtime();  
	
  //display X in matrix size is small.
  if (N < 100) {
    printf("\nX = [");
    for (row = 0; row < N; row++) {
      printf("%5.2f%s", X[row], (row < N-1) ? "; " : "]\n");
    }
  }

  //print the execution time for performance analysis purpose.
  printf("\n\nThe total execution time as recorded on process 0 = %f seconds!!\n!",end_time-start_time);
  
}
  MPI_Finalize();  

}


void scatter_data(

         int  norm_row    /* out */, 

         int     my_rank  /* in  */, 

         int     p        /* in  */,
	 float	*A,
	 int N,
	 int* local_no_of_rows,
	 int* local_matrix_size,
	 float* local_norm_row,
	 float* local_matrix_A,
	 int* rows_per_process) 
{			
    int source = 0;    /* All local variables used by */
    int dest;          /* MPI_Send and MPI_Recv       */
    int tag;
    MPI_Status status;
    
    //The data will be sent by the process 0 to all other processes.
    if (my_rank == 0){
        //as the datais being sent only to processes 1 and above, we need to
        //initialize the local variables for process zero explicitly, which
        //is done by the following code.
	*local_matrix_size=N;
	int i,j;
	if(p>1)
	{
		*local_no_of_rows = ((N-norm_row-1)/p)+1;
	}
	else
	{
		*local_no_of_rows = N-norm_row-1;
	}
		
	int curr_start_row=norm_row+1;	
		
	for(j=0;j<*local_matrix_size;j++)
		{
		*(local_norm_row+j) = *(A+(N*norm_row)+j);
		}
        //keep a record of all the number of rows being sent to each process,
        //as it will be needed while gathering the data back.
	for(i=0;i<*local_no_of_rows;i++){
		for(j=0;j<*local_matrix_size;j++)
		{
		*(local_matrix_A+N*i+j) = *(A+(N*curr_start_row)+N*i+j);
		}	
	}
	
        //This code segment sends the relevant data to all processes.
	curr_start_row=norm_row+(*local_no_of_rows)+1;
	int no_of_rows = *local_no_of_rows;	
	*(rows_per_process) = no_of_rows;
        for (dest = 1; dest < p; dest++){
		
	    //Sending number of rows
            tag = 0;
	    if((N-curr_start_row)<=*local_no_of_rows)
		{
			no_of_rows = N-curr_start_row;
		}
            MPI_Send(&no_of_rows,1, MPI_INT, dest, tag, 
		MPI_COMM_WORLD);

	    *(rows_per_process+dest) = no_of_rows;
	    
	    //sending Matrix size
	    tag = 1;
	    MPI_Send(&N, 1, MPI_INT, dest, tag, 
		MPI_COMM_WORLD);
		
	    //sending the normalization row	
            tag = 2;

            MPI_Send((A+(N*norm_row)), N, MPI_FLOAT, dest, tag, 
		MPI_COMM_WORLD);
	    
	    //sending the data for processing
	    tag = 3;
	    
	    MPI_Send(A+(N*curr_start_row), no_of_rows*N, MPI_FLOAT, dest, tag, 
		MPI_COMM_WORLD);
		
	    //move the start row
   	    curr_start_row+=no_of_rows;	

        }

    } else {
        //This code segment is executed by all other processes except 0, and they
        // receive the data sent by the process 0.
        tag = 0;

        MPI_Recv(local_no_of_rows, 1, MPI_INT, source, tag, 

            MPI_COMM_WORLD, &status);

        tag = 1;

        MPI_Recv(local_matrix_size, 1, MPI_INT, source, tag, 

            MPI_COMM_WORLD, &status);
	
        tag = 2;
	
        MPI_Recv(local_norm_row, *local_matrix_size, MPI_FLOAT, source, tag, 
	     MPI_COMM_WORLD, &status);

	tag = 3;

        MPI_Recv(local_matrix_A, (*local_matrix_size)*(*local_no_of_rows), MPI_FLOAT, source, tag, 

                MPI_COMM_WORLD, &status);
	
    }
}
/**
 * This method does the normalisation operation for a single row.
 * It is executed by each process, and iteratively normalise the rows
 * which belong to that particular process.
 *
 */
void eliminate(int local_matrix_size,
		int local_no_of_rows, 
		float* local_norm_row, 
		float* local_matrix_A, 
		int norm,
		float* local_matrix_B,
		float local_norm_B)
{
	int row,col;
	for (row = 0; row < local_no_of_rows; row++) {
		float multiplier = *(local_matrix_A+(row*local_matrix_size)+norm) 
					/ *(local_norm_row+norm);                	
	      for(col = norm; col < local_matrix_size; col++) {
				
		*(local_matrix_A+(row*local_matrix_size)+col) -= *(local_norm_row+col) * multiplier;	      
		
		}
	      *(local_matrix_B+row) -= (local_norm_B * multiplier);
	}

}
