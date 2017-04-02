/**
* Author: Adithep Narula
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <string.h>

int main (int argc, char *argv[] )  {

    double tstart = 0.0;
    double ttaken;
    FILE *f;


	//check if user enters the correct number of arguments
    if(argc != 3)
    {
        printf("ERROR: RUN AGAIN AND ENTER A FILE NAME!\n");
        exit(0);
    }

    //get N
    int N = strtol(argv[1], NULL, 10);
    //get number of threads from command line
    int thread_count = strtol(argv[2], NULL, 10);

    tstart = omp_get_wtime();

    //step 1: generate all numbers from 2 to N
    //generate bit arrays for number 0 to N (easier for indexing)
    int *array = (int*) calloc(N+1, sizeof(int));

    //0 and 1 are not considered prime
    array[0] = 1;
    array[1] = 1;
    
    //step 2: determine the prime numbers
    long int start;
    int end = (int) floor((N+1)/2.0);
    
    #pragma omp parallel num_threads(thread_count) default(none) shared(end, N, array) private(start)
    for(start = 2; start <= end; start++){
        //if current number is not prime, go cross out all of its multiples
        if(array[start] == 0) {
            //int will cause seg fault
            long i;
            #pragma omp for private(i) 
            for(i = start+1; i <= N; i++){
                if(i % start == 0) {
                    array[i] = 1;
                }
            }
        }//end_if
    }//end_for
    
    ttaken = omp_get_wtime()-tstart;

    
    //step 3: print prime numbers
    if(omp_get_thread_num() == 0){
        //create file name
        
        char fileName[20];
        sprintf(fileName, "%d", N); 
        strcat(fileName,".txt"); 
        //open file
        f = fopen(fileName, "w");
        if(f == NULL){
            printf("Error opening file!\n");
            exit(1);
        }


        int j;
        int counter = 1;
        int previousPrime = 2;
        for(j = 2; j <= N; j++){
            if(array[j] == 0) {
                //printf("%d, %d, %d\n", counter, j, j-previousPrime);
                fprintf(f,"%d%s %d%s %d\n", counter, ",", j, ",", j-previousPrime);
                previousPrime = j;
                counter++;
            }
        }

        fclose(f);
    } 

    free(array);
    printf("Time taken for the main part: %f\n",ttaken);
   
 

   
    return 0;
}




