
/*
 ------------------------------------------------------------------------
 FFT1D            c_fft1d(r,i,-1)
 Inverse FFT1D    c_fft1d(r,i,+1)
 ------------------------------------------------------------------------
*/
/* ---------- FFT 1D
   This computes an in-place complex-to-complex FFT
   r is the real and imaginary arrays of n=2^m points.
   isign = -1 gives forward transform
   isign =  1 gives inverse transform
*/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <mpi.h>

typedef struct {float r; float i;} complex;
static complex ctmp;
int N=512;

#define C_SWAP(a,b) {ctmp=(a);(a)=(b);(b)=ctmp;}

void c_fft1d(complex *r, int      n, int      isign)
{
   int     m,i,i1,j,k,i2,l,l1,l2;
   float   c1,c2,z;
   complex t, u;
   if (isign == 0) return;
	//printf("val=%f",r[0].r);
   /* Do the bit reversal */
   i2 = n >> 1;
   j = 0;
   for (i=0;i<n-1;i++) {
      if (i < j)
         C_SWAP(r[i], r[j]);
      k = i2;
      while (k <= j) {
         j -= k;
         k >>= 1;
      }
      j += k;
   }

   /* m = (int) log2((double)n); */
   for (i=n,m=0; i>1; m++,i/=2);

   /* Compute the FFT */
   c1 = -1.0;
   c2 =  0.0;
   l2 =  1;
   for (l=0;l<m;l++) {
      l1   = l2;
      l2 <<= 1;
      u.r = 1.0;
      u.i = 0.0;
      for (j=0;j<l1;j++) {
         for (i=j;i<n;i+=l2) {

            i1 = i + l1;

            /* t = u * r[i1] */
            t.r = u.r * r[i1].r - u.i * r[i1].i;
            t.i = u.r * r[i1].i + u.i * r[i1].r;

            /* r[i1] = r[i] - t */
            r[i1].r = r[i].r - t.r;
            r[i1].i = r[i].i - t.i;

            /* r[i] = r[i] + t */
            r[i].r += t.r;
            r[i].i += t.i;
         }
         z =  u.r * c1 - u.i * c2;

         u.i = u.r * c2 + u.i * c1;
         u.r = z;
      }

      c2 = sqrt((1.0 - c1) / 2.0);
      if (isign == -1) /* FWD FFT */
         c2 = -c2;
      c1 = sqrt((1.0 + c1) / 2.0);
   }

   /* Scaling for inverse transform */
   if (isign == 1) {       /* IFFT*/
      for (i=0;i<n;i++) {
         r[i].r /= n;
         r[i].i /= n;
      }
   }
}

void readComplexFile(FILE *fr, complex temp[N][N] )
{
	int i,j;
	float t;
	 for (i=0;i<N;i++)
		for (j=0;j<N;j++){
			fscanf(fr,"%f",&((temp[i][j]).r));
			(temp[i][j]).i= 0.0;
		}
}

void mmul_point(complex A[N][N], complex B[N][N])
{
    int i, j;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            A[i][j].r *= B[i][j].r;
            A[i][j].i *= B[i][j].i;
        }
    }
}

void readFromFile(const char *fname, float data[N][N])
{
    int i, j;
    FILE *fp;
    fp = fopen(fname, "r");
    for (i = 0; i < N; i++)
    {
        for(j = 0; j < N; j++)
            fscanf(fp, "%g", &data[i][j]);
    }
    fclose(fp);
}

void writeToFile(const char *fname, float data[N][N])
{
    int i, j;
    FILE *fp;
    fp = fopen(fname, "w");
    for (i = 0; i < N; i++)
    {
        for(j = 0; j < N; j++)
        {
            fprintf(fp, "%6.2g\t", data[i][j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);

}


void convertToComplex(complex C[N][N], float F[N][N])
{
    int i, j;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            C[i][j].r = F[i][j];
            C[i][j].i = 0;
        }
    }
}

void convertToReal(float F[N][N], complex C[N][N])
{
    int i, j;
    for (i= 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            F[i][j] = C[i][j].r;
        }
    }
}


void transpose(complex mat[N][N])
{
    int i, j;
    float tempr, tempi;
    for (i = 0; i < N; i++)
    {
        for (j = i + 1; j < N; j++)
        {
            tempr = mat[i][j].r;
            mat[i][j].r = mat[j][i].r;
            mat[j][i].r = tempr;

            tempi = mat[i][j].i;
            mat[i][j].i = mat[j][i].i;
            mat[j][i].i = tempi;
        }
    }
}


void c_rowwise_fft2d(complex *local_mat, int local_n)
{
    int i;
    for (i = 0; i < local_n; i++)
    {
        c_fft1d((local_mat+N*i), N, -1);
    }
}
void c_rowwise_inv_fft2d(complex *local_mat, int local_n)
{
    int i;
    for (i = 0; i < local_n; i++)
    {
        c_fft1d((local_mat+N*i), N, 1);
    }
}



void printMatRow(float mat[N][N],int row)
{
    int i, j;
        for(j = 0; j < N; j++)
        {
            printf("%f\t", mat[row][j]);
        }
        printf("\n");
}


void printNonZeroMat(float mat[N][N])
{
    int i, j;
    for (i = 0; i < N; i++)
    {
        for(j = 0; j < N; j++)
        {
            if(mat[i][j]!=0)
                printf("%f\t", mat[i][j]);
        }
        printf("\n");
    }
}

void printMat(float mat[N][N])
{
    int i, j;
    for (i = 0; i < N; i++)
    {
        for(j = 0; j < N; j++)
        {
            printf("%f\t", mat[i][j]);
        }
        printf("\n");
    }
}

void splitCommunicator(MPI_Comm *p1,MPI_Comm *p2,MPI_Comm *p3,MPI_Comm *p4)
{
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the group of processes in MPI_COMM_WORLD
  MPI_Group world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);

  int n = world_size/4;
  int ranks[2] = {0,1};

   // Construct a group containing the first 2 processes
   MPI_Group group;
   MPI_Group_incl(world_group, n, ranks, &group);
   // Create a new communicator based on the group
   MPI_Comm_create(MPI_COMM_WORLD, group,  p1);

    MPI_Group_free(&world_group);
    MPI_Group_free(&group);

}

int main(int argc, char **argv)
{

  int my_rank=0;   /* My process rank           */
  int p;         /* The number of processes   */
  //clock time recording variables
  double  start_time,end_time,comm_time_start,comm_time_end,total_comm_time=0.0;

  /* Let the system do what it needs to start up MPI */
   MPI_Init(&argc, &argv);

  /* Get my process rank */
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  /* Find out how many processes are being used */
   MPI_Comm_size(MPI_COMM_WORLD, &p);
   float dataA[N][N], dataB[N][N];

    complex A[N][N], B[N][N];


//create MPI type for complex datatype
   	MPI_Aint displ[2];
	displ[0] = 0;
  	displ[1] = sizeof(float);
  	MPI_Datatype mpi_complex,types[2];
  	types[0] = MPI_FLOAT;
  	types[1] = MPI_FLOAT;
  	int block_len[2];
  	block_len[0]= 1;
  	block_len[1]= 1;
 	MPI_Type_struct(2, block_len, displ, types,&mpi_complex);
  	MPI_Type_commit(&mpi_complex);

  	//create 4 communicators by splitting the default one
  	MPI_Comm new_comm;
    /* Find out how many processes are being used */
    int color = my_rank / 2;
    MPI_Comm_split(MPI_COMM_WORLD, color, my_rank, &new_comm);

    int row_rank, row_size;
    MPI_Comm_rank(new_comm, &row_rank);
    MPI_Comm_size(new_comm, &row_size);


//we will read the files by corresponding processes
  if(my_rank==0)
  {

	readFromFile("data/1_im1", dataA);


    convertToComplex(A, dataA);
  }
  else if(my_rank==2)
  {
    readFromFile("data/1_im2", dataB);
    convertToComplex(B, dataB);
  }

  if(my_rank==6)
  {
    //Start clock and record the start time.
    start_time = MPI_Wtime();
  }
  int local_n = N/row_size;
  int rows_per_process = N/row_size;
  complex local_A[N/row_size][N];
  complex local_B[N/row_size][N];
  //broadcast the number of ros each process will work on
  MPI_Bcast(&rows_per_process , 1, MPI_INT, 0,MPI_COMM_WORLD);

  //process 0 and 1 work on Matrix A
  if(my_rank<2)
  {
        if(row_rank==0)
        {
            comm_time_start=MPI_Wtime();
        }
        MPI_Scatter(&A[0][0], N*rows_per_process, mpi_complex,
               &local_A, N*rows_per_process, mpi_complex, 0,
               new_comm);
        if(row_rank==0)
        {
            comm_time_end=MPI_Wtime();
            total_comm_time+=comm_time_end-comm_time_start;
        }

        c_rowwise_fft2d(&local_A[0][0],rows_per_process);

        if(row_rank==0)
        {
            comm_time_start=MPI_Wtime();
        }
        MPI_Gather(&local_A, N*rows_per_process, mpi_complex,
               &A[0][0], N*rows_per_process, mpi_complex,
               0, new_comm);

        if(row_rank==0)
        {
            comm_time_end=MPI_Wtime();
            total_comm_time+=comm_time_end-comm_time_start;
        }


        if(row_rank==0)
            transpose(A);

         if(row_rank==0)
        {
            comm_time_start=MPI_Wtime();

        }

        MPI_Scatter(&A[0][0], N*rows_per_process, mpi_complex,
               &local_A, N*rows_per_process, mpi_complex, 0,
               new_comm);
        if(row_rank==0)
        {
            comm_time_end=MPI_Wtime();
            total_comm_time+=comm_time_end-comm_time_start;
        }


        c_rowwise_fft2d(&local_A[0][0],rows_per_process);

        if(row_rank==0)
        {
            comm_time_start=MPI_Wtime();
        }
        MPI_Gather(&local_A, N*rows_per_process, mpi_complex,
               &A[0][0], N*rows_per_process, mpi_complex,
               0, new_comm);
        if(row_rank==0)
        {
            comm_time_end=MPI_Wtime();
            total_comm_time+=comm_time_end-comm_time_start;
        }

    MPI_Barrier(new_comm);

    if(row_rank==0)
    {
             transpose(A);


            comm_time_start=MPI_Wtime();
            //send the intermediate matrix to process 4
            MPI_Send(A, N*N, mpi_complex, 4,11,MPI_COMM_WORLD);

            comm_time_end=MPI_Wtime();
            total_comm_time+=comm_time_end-comm_time_start;

    }

  }
  if(my_rank>=2 && my_rank<4)  //process 2 and 3 work on Matrix B
  {
      //printf(" %d ",rows_per_process);
      if(row_rank==0)
        {
            comm_time_start=MPI_Wtime();

        }

        MPI_Scatter(&B[0][0], N*rows_per_process, mpi_complex,
               &local_B, N*rows_per_process, mpi_complex, 0,
               new_comm);

        if(row_rank==0)
        {
            comm_time_end=MPI_Wtime();
            total_comm_time+=comm_time_end-comm_time_start;
        }


        c_rowwise_fft2d(&local_B[0][0],rows_per_process);

        if(row_rank==0)
        {
            comm_time_start=MPI_Wtime();
        }
        MPI_Gather(&local_B, N*local_n, mpi_complex,
               &B[0][0], N*rows_per_process, mpi_complex,
               0, new_comm);

        if(row_rank==0)
        {
            comm_time_end=MPI_Wtime();
            total_comm_time+=comm_time_end-comm_time_start;
        }

        if(row_rank==0)
            transpose(B);

        if(row_rank==0)
        {
            comm_time_start=MPI_Wtime();
        }
        MPI_Scatter(&B[0][0], N*rows_per_process, mpi_complex,
               &local_B, N*rows_per_process, mpi_complex, 0,
               new_comm);
         if(row_rank==0)
        {
            comm_time_end=MPI_Wtime();
            total_comm_time+=comm_time_end-comm_time_start;
        }

        c_rowwise_fft2d(&local_B[0][0],rows_per_process);

        if(row_rank==0)
        {
            comm_time_start=MPI_Wtime();
        }
        MPI_Gather(&local_B, N*local_n, mpi_complex,
               &B[0][0], N*rows_per_process, mpi_complex,
               0, new_comm);

        if(row_rank==0)
        {
            comm_time_end=MPI_Wtime();
            total_comm_time+=comm_time_end-comm_time_start;
        }

        MPI_Barrier(new_comm);

        if(row_rank==0)
        {
            transpose(B);
            //lets send intermediate matrix B to process 4
            MPI_Send(B, N*N, mpi_complex, 4, 12,MPI_COMM_WORLD);

        }

  }
  if(my_rank==4)  //process 4 multiplies the two matrices
  {
    MPI_Status status;

    comm_time_start=MPI_Wtime();

    MPI_Recv(A, N*N, mpi_complex, 0, 11,
                    MPI_COMM_WORLD, &status);
    MPI_Recv(B, N*N, mpi_complex, 2, 12,
                   MPI_COMM_WORLD, &status);

    comm_time_end=MPI_Wtime();
    total_comm_time+=comm_time_end-comm_time_start;


    mmul_point(A,B);
    if(row_rank==0)
    {
            comm_time_start=MPI_Wtime();

            MPI_Send(A, N*N, mpi_complex, 6, 13,MPI_COMM_WORLD);

            comm_time_end=MPI_Wtime();
            total_comm_time+=comm_time_end-comm_time_start;
    }

  }
  if(my_rank>=6 && my_rank<8) //process 6 and 7 does inverse transform on result
  {

    MPI_Status status;
    if(row_rank==0)
    {
            comm_time_start=MPI_Wtime();
            MPI_Recv(A, N*N, mpi_complex, 4, 13,
                    MPI_COMM_WORLD, &status);

            comm_time_end=MPI_Wtime();
            total_comm_time+=comm_time_end-comm_time_start;

    }

    if(row_rank==0)
        {
            comm_time_start=MPI_Wtime();
        }

    MPI_Scatter(&A[0][0], N*rows_per_process, mpi_complex,
               &local_A, N*rows_per_process, mpi_complex, 0,
               new_comm);
     if(row_rank==0)
        {
            comm_time_end=MPI_Wtime();
            total_comm_time+=comm_time_end-comm_time_start;
        }

        c_rowwise_inv_fft2d(&local_A[0][0],rows_per_process);

        if(row_rank==0)
        {
            comm_time_start=MPI_Wtime();
        }
        MPI_Gather(&local_A, N*rows_per_process, mpi_complex,
               &A[0][0], N*rows_per_process, mpi_complex,
               0, new_comm);
        if(row_rank==0)
        {
            comm_time_end=MPI_Wtime();
            total_comm_time+=comm_time_end-comm_time_start;
        }

    if(row_rank==0)
        transpose(A);

        if(row_rank==0)
        {
            comm_time_start=MPI_Wtime();
        }

        MPI_Scatter(&A[0][0], N*rows_per_process, mpi_complex,
               &local_A, N*rows_per_process, mpi_complex, 0,
               new_comm);
         if(row_rank==0)
        {
            comm_time_end=MPI_Wtime();
            total_comm_time+=comm_time_end-comm_time_start;
        }

        c_rowwise_inv_fft2d(&local_A[0][0],rows_per_process);

        if(row_rank==0)
        {
            comm_time_start=MPI_Wtime();
        }
        MPI_Gather(&local_A, N*rows_per_process, mpi_complex,
               &A[0][0], N*rows_per_process, mpi_complex,
               0, new_comm);
        if(row_rank==0)
        {
            comm_time_end=MPI_Wtime();
            total_comm_time+=comm_time_end-comm_time_start;
        }

    if(row_rank==0)
    {   transpose(A);


        //Stop clock as operation is finished.
        end_time = MPI_Wtime();
        //print the execution time for performance analysis purpose.
        printf("\n\nThe total execution time as recorded on process 0 = %f seconds!!\n!",end_time-start_time);

        convertToReal(dataA,A);
        writeToFile("final_output.txt", dataA);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
      //get a total of communication cost recorded by all processes
        MPI_Reduce(&total_comm_time, &total_comm_time, 1, MPI_FLOAT,
               MPI_SUM, 0, MPI_COMM_WORLD);
  if(my_rank ==0)
        {
                printf("\n\nThe total Communication time = %f seconds!!\n!",total_comm_time);
        }

  MPI_Comm_free(&new_comm);
  MPI_Finalize();

return 0;
}
