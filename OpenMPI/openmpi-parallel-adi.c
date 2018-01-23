#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

/***** Globals ******/
float *a; /* The coefficients - scatter (done) */
float *x;  /* initial values - broadcast*/
float *b;  /* The constants - scatter (done)*/
float err; /* The absolute relative error -broadcast */
int num = 0;  /* number of unknowns - broadcast (done)*/ 

/*** MPI variables ****/
int world_rank;
int comm_sz;
MPI_Comm comm;


/****** Function declarations */
void check_matrix(); /* Check whether the matrix will converge */
void get_input();  /* Read input from file */
void scatter_vector(float *matrix, float local_row[], int local_n, int my_rank,MPI_Comm comm);

void calculateXNewVector(
                           float *local_xNewArray, //out
                           float *local_row, //in
                           float *local_constants_array, //in
                           float *global_xOldArray, //in
                           int local_row_per_process, // in
                           int global_numUnknowns, //in
                           int local_current_row_index, //in
                           int world_rank
                        );

void calculateRelativeError(
                            int *finishPointer, //out
                            float *local_xNewArray,   //in
                            float *global_xOldArray,  //in
                            int local_current_row_index,  //in
                            float global_erorrLimit, //in
                            int local_row_per_process, //in
                            int world_rank
                            );

/********************************/



/* Function definitions: functions are ordered alphabetically ****/
/*****************************************************************/

/* 
   Conditions for convergence (diagonal dominance):
   1. diagonal element >= sum of all other elements of the row
   2. At least one diagonal element > sum of all other elements of the row
 */
void check_matrix()
{
  int bigger = 0; /* Set to 1 if at least one diag element > sum  */
  int i, j;
  float sum = 0;
  float aii = 0;
  
  for(i = 0; i < num; i++)
  {
    sum = 0;
    aii = fabs(a[num*i+i]);
    
    for(j = 0; j < num; j++)
       if( j != i)
	 sum += fabs(a[num*i+j]);
       
    if( aii < sum)
    {
      printf("The matrix will not converge\n");
      exit(1);
    }
    
    if(aii > sum)
      bigger++;
    
  }
  
  if( !bigger )
  {
     printf("The matrix will not converge\n");
     exit(1);
  }
}


/******************************************************/
/* Read input from file */
void get_input(char filename[])
{
  FILE * fp;
  int i,j;  
 
  fp = fopen(filename, "r");
  if(!fp)
  {
    printf("Cannot open file %s\n", filename);
    exit(1);
  }

 fscanf(fp,"%d ",&num);
 fscanf(fp,"%f ",&err);

 /* Now, time to allocate the matrices and vectors */
 a = (float*)malloc(num * num * sizeof(float));
 if( !a)
  {
	printf("Cannot allocate a!\n");
	exit(1);
  }
 
 x = (float *) malloc(num * sizeof(float));
 if( !x)
  {
	printf("Cannot allocate x!\n");
	exit(1);
  }

 b = (float *) malloc(num * sizeof(float));
 if( !b)
  {
	printf("Cannot allocate b!\n");
	exit(1);
  }

 /* Now .. Filling the blanks */ 

 /* The initial values of Xs */
 for(i = 0; i < num; i++)
	fscanf(fp,"%f ", &x[i]);
 
 for(i = 0; i < num; i++)
 {
   for(j = 0; j < num; j++)
     fscanf(fp,"%f ",&a[num*i+j]);
   
   /* reading the b element */
   fscanf(fp,"%f ",&b[i]);
 }
 
 fclose(fp); 

}


/************************************************************/


int main(int argc, char *argv[])
{
  
  int global_numUnknowns = 0;
  float *global_xOldArray = NULL;
  float *process0_matrix = NULL;
  float *process0_constants_array = NULL;
  float *local_constants_array; //constants are the b's
  float *local_row; //receive buffer - row of x's for the process to solve
  float *local_xNewArray;
  float *local_error_array_boolean;
  float global_err = 1000000000;
  float *global_xNewArray = NULL;
  int local_finish;
  int local_finish_result; 
  int *global_displs_constants = NULL; //will rename below
  int process0_sum = 0; // Sum of counts, used to calculate displacements
  int nit = 0;




  // Initialize the MPI environment
  MPI_Init(&argc, &argv);

  // Get the number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

  // Get the rank of the process
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  //initializations
  int *process0_sendcounts_matrix = (int*)malloc(sizeof(int)*comm_sz);
  int *process0_sendcounts_constants = (int*)malloc(sizeof(int)*comm_sz);
  int *process0_displs_matrix = (int*)malloc(sizeof(int)*comm_sz);
  int *process0_displs_constants = (int*)malloc(sizeof(int)*comm_sz);



  //process 0 reads the file and performs initializations
  if(world_rank == 0) {
     int i;
     int nit = 0; /* number of iterations */

     
     if( argc != 2)
     {
       printf("Usage: gsref filename\n");
       exit(1);
     }
     /* Read the input file and fill the global data structure above */ 
       get_input(argv[1]); 
      
     /* Check for convergence condition */
      check_matrix();

  
     //rename
     global_numUnknowns = num;
     process0_matrix = a;
     process0_constants_array = b;
     global_err = err;  


  } 


  //need to initialize global_numuknowns
  //wait for process 0 to finish reading the file before begin
  MPI_Barrier(MPI_COMM_WORLD);

  //broadcast the number of unknowns so every process has it
  MPI_Bcast(&global_numUnknowns, 1, MPI_INT, 0, MPI_COMM_WORLD);

  //initialize 
  global_xOldArray = (float *)malloc(global_numUnknowns * sizeof(float));
  global_xNewArray = (float *)malloc(global_numUnknowns * sizeof(float));

  //copy values of unknowns into global array and broadcast that 
  int i;
  if(world_rank == 0){
    for(i = 0; i < global_numUnknowns; i++){
      global_xOldArray[i] = x[i];
    }
  }

  //broadcast error limit
  MPI_Bcast(&global_err, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
 
  //broadcast the initial values (xOldArray) to each process
  MPI_Bcast(global_xOldArray, global_numUnknowns, MPI_INT, 0, MPI_COMM_WORLD);

  if(world_rank == 0) {
      //how many rows each process takes
      int number_of_rows;
      //divide equally, add more rows if necessary
      int rows_before_add = global_numUnknowns/comm_sz;
      //how many rows left that we need to add
      int num_rows_remain = global_numUnknowns%comm_sz;

     for(i = 0; i < comm_sz; i++){
      //if rank is less than remain, add one otherwise don't add
      if(i < num_rows_remain) {
        number_of_rows = rows_before_add + 1;
      }else{
        number_of_rows = rows_before_add;
      }
      //number of elements to distribute for coefficients matrix per process
      process0_sendcounts_matrix[i] = number_of_rows * global_numUnknowns;
      //displacement for matrix
      process0_displs_matrix[i] = process0_sum * global_numUnknowns;

      //for constants, corresponds to number of rows
      process0_sendcounts_constants[i] = number_of_rows;
      process0_displs_constants[i] = process0_sum;

      //add displacement
      process0_sum += number_of_rows;

    }//end for

  }//end if


  
  //<<<calculations to determine size of local recieve buffer for coefficients and constants>>>>
  //how many rows each process takes
  int number_of_rows;
  //divide equally, add more rows if necessary
  int rows_before_add = global_numUnknowns/comm_sz;
  //how many rows left that we need to add
  int num_rows_remain = global_numUnknowns%comm_sz;
  //if rank is less than remain, add one otherwise don't add
  if(world_rank < num_rows_remain) {
     number_of_rows = rows_before_add + 1;
  }else{
    number_of_rows = rows_before_add;
  }

  int local_num_data = number_of_rows * global_numUnknowns;
  int local_num_constants = number_of_rows;

  //create local_row receive buffer to store coefficients data for scatter
  local_row = (float*) malloc (local_num_data* sizeof(float));
  //for constants
  local_constants_array = (float *) malloc(local_num_constants * sizeof(float));

  //divide the data among processes as described by sendcounts and displs
  MPI_Scatterv(process0_matrix, process0_sendcounts_matrix, process0_displs_matrix, MPI_FLOAT, local_row, local_num_data, MPI_FLOAT, 0, MPI_COMM_WORLD);

  //scatter constants to each process
  MPI_Scatterv(process0_constants_array, process0_sendcounts_constants, process0_displs_constants, MPI_FLOAT, local_constants_array, local_num_constants, MPI_FLOAT, 0, MPI_COMM_WORLD);
  
  //broadcast error limit
  MPI_Bcast(&global_err, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);


  //broadcast global displacement constants
  MPI_Bcast(process0_displs_constants, comm_sz, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(process0_sendcounts_constants, comm_sz, MPI_INT, 0, MPI_COMM_WORLD);

  //array of new x values (number of elements in xNewArray is same as number of constants that the process gets)
  local_xNewArray = (float *) malloc(local_num_constants* sizeof(float));
    
  //how to find the beginning of global row index for each process? - ie, corresponding global row number
  //currentRow = world_rank * global_numUknowns/comm_sz
  //this corresponds to the number of rows
  int local_current_row_index = process0_displs_constants[world_rank];


  //how many rows of coefficients the current process takes 
  int local_row_per_process;
  if(world_rank < comm_sz-1){
     local_row_per_process = process0_displs_constants[world_rank+1] - process0_displs_constants[world_rank];
  }else{
    local_row_per_process = global_numUnknowns - process0_displs_constants[world_rank];
  }



  int continueLoop = 1;

  //take timing, syncronize processes first
  MPI_Barrier(MPI_COMM_WORLD);
  double local_start_time = MPI_Wtime();
  
  while(continueLoop) {
    //count the number of iterations
    nit++;

    //calculate x new for each process
    calculateXNewVector(local_xNewArray, local_row, local_constants_array, global_xOldArray, local_row_per_process, global_numUnknowns, local_current_row_index, world_rank);
 
    //set flag to finish, if local_xarray contains error more than threshold, finish changes to 0
    local_finish = 1;
    calculateRelativeError(&local_finish, local_xNewArray, global_xOldArray, local_current_row_index, global_err,local_row_per_process, world_rank);

    //receive buffer
    local_finish_result;

    //reduce the bits-boolean array
    MPI_Allreduce(&local_finish, &local_finish_result, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

    //local_finish will be 1 if all process is done
    //so have each process break out of the loop 
    if(local_finish_result == 1) {
      break;
    }
    //else make xOld=xNew
    //all gather the x new array
    MPI_Allgatherv(local_xNewArray, local_row_per_process, MPI_FLOAT, global_xNewArray, process0_sendcounts_constants, process0_displs_constants, MPI_FLOAT, MPI_COMM_WORLD);
  

    //each process copies x_new into x_old
    for(i = 0; i < global_numUnknowns; i++){
       global_xOldArray[i] = global_xNewArray[i];
    }//end for loop


  }//end while loop

  double local_finish_time = MPI_Wtime();
  double local_elapsed_time = local_finish_time - local_start_time;
  double elapsed;
  MPI_Reduce(&local_elapsed_time, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if(world_rank == 0){
    printf("----------------------------\n");
    printf("Elapsed time in milliseconds is: %f\n", elapsed * 1000);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  
  if(world_rank == 0){
    printf("----------------------------\n");
    printf("Finishing X vector: \n");     
        for(i = 0; i < global_numUnknowns; i++){
          printf("%f ",global_xNewArray[i]);
        }
        printf("\n");
        printf("----------------------------\n");
        printf("number of iterations: %d\n ", nit);
  }
 
  

  //free a bunch of things
  free(global_xOldArray);
  free(global_xNewArray);
  free(process0_matrix);
  free(process0_constants_array);
  free(local_row);
  free(local_constants_array);
  
  MPI_Finalize();


}

/**
* This function creates a bit-boolean error array
* if element is 0 then the corresponding xNew passes threshold 
**/
void calculateRelativeError(
                            int *finishPointer, //out
                            float *local_xNewArray,   //in
                            float *global_xOldArray,  //in
                            int local_current_row_index,  //in
                            float global_erorrLimit, //in
                            int local_row_per_process, //in
                            int world_rank
                            )
{
  int row;
  for(row = 0; row < local_row_per_process; row++){
    float xOld = global_xOldArray[local_current_row_index+row];
    float xNew = local_xNewArray[row];
    float error = (xNew - xOld)/xNew;
    float errorAbs = fabsf(error);
    if(errorAbs > global_erorrLimit){
      *finishPointer = 0;
      break;
    }
  }//end for 

}

void calculateXNewVector(
                           float *local_xNewArray, //out
                           float *local_row, //in - row of coefficients 
                           float *local_constants_array, //in 
                           float *global_xOldArray, //in
                           int local_row_per_process, // in
                           int global_numUnknowns, //in
                           int local_current_row_index, //in
                           int world_rank
                        )
{


  int row, col;
  int numberOfIt = 0;
  float tempArray[global_numUnknowns+1];

  //calculate xNew for each row
  for(row = local_current_row_index; row < local_current_row_index+local_row_per_process; row++){
    
    //global_numknowns is same as number of columns
    //add one because we will also store the constant 
    //tempArray will store one row 
    
    if(tempArray == NULL){
      perror("tempArray malloc error");
      exit(1);
    }
    
    //step 1: multiply each element except Xrow by negative 1
    for (col = 0; col < global_numUnknowns; col++){
      if(col != row){
        //row = numberOfIt
        tempArray[col] = -1 * local_row[global_numUnknowns*numberOfIt+col];
      } else{
        tempArray[col] = local_row[global_numUnknowns*numberOfIt+col];
      }
    }

  
    //store the last element of tempArray as the constant
    tempArray[global_numUnknowns] = local_constants_array[numberOfIt];

    //step 2: divide every number by Xrow (including constant)
    float xRow =  local_row[global_numUnknowns*numberOfIt+row];
    for(col = 0; col < global_numUnknowns + 1; col++){
      tempArray[col] = tempArray[col]/xRow;
    }

    //step 3: multiply (ie substitution) every element except Xrow and constant by Xold 
    for(col = 0; col < global_numUnknowns; col++){
      if(col != row){
        tempArray[col] = tempArray[col] * global_xOldArray[col];
      }
    }
    //step 4: add all elements (including constant) together except Xrow
    float xNew = 0;
    for(col = 0; col < global_numUnknowns+1; col++){
      if(col != row){
        xNew = xNew + tempArray[col];
      }
    }
    //step 5: add xNew to xNewArray 
    local_xNewArray[numberOfIt] = xNew; 
 
    numberOfIt++;

  }//end outer for



}
  

