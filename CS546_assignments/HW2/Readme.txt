pthread-
The pthread code can be compiled using following command:
gcc - pthread gauss.c -o gauss

The pthread code can be executed using folloing command:
./gauss 2000

This will run the pthread code using 4 threads for a 2000X2000 matrix.


OpenMP-
The openMP code can be compiled using following command:
gcc -fopenmp gauss.c -o gauss

The openmp code can be executed using folloing command:
./gauss 2000

This will run the openmp code using 4 threads for a 2000X2000 matrix.
 