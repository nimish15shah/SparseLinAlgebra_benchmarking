
#include<stdio.h>
#include<stdlib.h>
#include <assert.h>
#include<math.h>
#include <iostream>
#include <mkl.h>
#include "mkl_dss.h"
#include "mkl_types.h"
#include <string.h>

// headers from common directory
#define MIC_TARGET
#include "mmio.h"
#include "util.h"
#include "paths.h"

using namespace std;

class CPUBenchmark {
  /** read matrix market file for A matrix */
  MKL_INT nRows, nCols, nNNZ;
  MKL_INT *Ax, *Ay;
  _DOUBLE_PRECISION_t *Anz;

  /** construct csr storage for A matrix */
  //MKL_INT* A_col_idx;
  //MKL_INT* A_row_ptr;
  _INTEGER_t* A_col_idx;
  _INTEGER_t* A_row_ptr;
  _DOUBLE_PRECISION_t* Aval;


  /** rhs detail */
  MKL_INT nRhs= 1;
  _DOUBLE_PRECISION_t *rhs;

  public:
    CPUBenchmark(char* strpathA);
    void print_matrix_info();
    MKL_INT dss_routine(int n_solve_iter=1);
  private:
    MKL_INT construct_matrix(char* strpathA);

};

void CPUBenchmark::print_matrix_info() {
  cout << "Size of the matrix: "<< nRows << " x " << nCols << endl;
  cout << "Number of nnz elements: "<< nNNZ << endl;
  cout << A_row_ptr[0] << endl;
  for(int i = 0; i < 5; ++i)
      cout << A_row_ptr[i] << ",";
  cout << endl;
  cout << A_col_idx[0] << endl;
  cout << A_col_idx[nNNZ - 1] << endl;
  cout << sizeof(nRows) << endl;
}

CPUBenchmark::CPUBenchmark (char* strpathA) {
  construct_matrix(strpathA);
  print_matrix_info();
}

MKL_INT CPUBenchmark::construct_matrix(char* strpathA) {
  read_mm(strpathA, &nRows, &nCols, &nNNZ, &Ax, &Ay, &Anz);

  A_col_idx = (_INTEGER_t*) mkl_malloc( nNNZ * sizeof( _INTEGER_t ), 64 );
  A_row_ptr = (_INTEGER_t*) mkl_malloc( (nRows+1) * sizeof(_INTEGER_t), 64 );
  //A_col_idx = (MKL_INT*) mkl_malloc( nNNZ * sizeof( MKL_INT ), 64 );
  //A_row_ptr = (MKL_INT*) mkl_malloc( (nRows+1) * sizeof( MKL_INT ), 64 );
  Aval = (double*) mkl_malloc( nNNZ * sizeof( double ),  64 );
  coo_to_csr(nRows, nNNZ, Ax, Ay, Anz, A_row_ptr, A_col_idx, Aval);

  rhs = (_DOUBLE_PRECISION_t*) mkl_malloc( nCols * sizeof( _DOUBLE_PRECISION_t ),  64 );
  for(int i = 0; i < nCols + 1; ++i) {
    rhs[i] = i;
  }
}

MKL_INT CPUBenchmark::dss_routine (int n_solve_iter)
{
  double time, time_st, time_end;
  /* Allocate storage for the solver handle and the right-hand side. */
  _DOUBLE_PRECISION_t *solValues;
  solValues= (_DOUBLE_PRECISION_t*) mkl_malloc( nRows * sizeof( _DOUBLE_PRECISION_t ),  64 ); 

  _MKL_DSS_HANDLE_t handle;
  _INTEGER_t error;
  _DOUBLE_PRECISION_t statOut[8];

  MKL_INT opt = MKL_DSS_DEFAULTS;
  //MKL_INT opt = MKL_DSS_ZERO_BASED_INDEXING;
  //MKL_INT sym = MKL_DSS_SYMMETRIC;
  MKL_INT sym = MKL_DSS_NON_SYMMETRIC;
  MKL_INT type = MKL_DSS_POSITIVE_DEFINITE;
  //MKL_INT type = MKL_DSS_INDEFINITE;

/* --------------------- */
/* Initialize the solver */
/* --------------------- */
  error = dss_create (handle, opt);
  if (error != MKL_DSS_SUCCESS)
    goto printError;
/* ------------------------------------------- */
/* Define the non-zero structure of the matrix */
/* ------------------------------------------- */
  error = dss_define_structure (handle, sym, A_row_ptr, nRows, nCols, A_col_idx, nNNZ);
  if (error != MKL_DSS_SUCCESS)
    goto printError;
/* ------------------ */
/* Reorder the matrix */
/* ------------------ */
  cout << "dss_reorder" << endl;
  time_st = dsecnd();
  error = dss_reorder (handle, opt, 0);
  time_end = dsecnd();
  time = (time_end - time_st);
  cout << "Measured Reorder time: " << time << endl;
  if (error != MKL_DSS_SUCCESS)
    goto printError;
/* ------------------ */
/* Factor the matrix  */
/* ------------------ */
  cout << "dss_factor" << endl;
  time_st = dsecnd();
  error = dss_factor_real (handle, type, Aval);
  time_end = dsecnd();
  time = (time_end - time_st);
  cout << "Measured Factor time: " << time << endl;
  if (error != MKL_DSS_SUCCESS)
    goto printError;
/* ------------------------ */
/* Get the solution vector  */
/* ------------------------ */
  cout << "dss_solve" << endl;
  time_st = dsecnd();
  for(int i = 0; i < n_solve_iter ; ++i) {
    error = dss_solve_real (handle, opt, rhs, nRhs, solValues);
  }
  time_end = dsecnd();
  time = (time_end - time_st)/n_solve_iter;
  cout << "Measured Solve time: " << time << endl;
  if (error != MKL_DSS_SUCCESS)
    goto printError;
/* ------------------------ */
/* Get the determinant (not for a diagonal matrix) */
/*--------------------------*/
  if (nRows < nNNZ)
    {
      _CHARACTER_t statIn[] = "determinant,ReorderTime,FactorTime,SolveTime";
      error = dss_statistics (handle, opt, statIn, statOut);
      if (error != MKL_DSS_SUCCESS)
        goto printError;
/*-------------------------*/
/* print determinant       */
/*-------------------------*/
      printf (" determinant power is %g \n", statOut[0]);
      printf (" determinant base is %g \n", statOut[1]);
      printf (" Determinant is %g \n", (pow (10.0, statOut[0])) * statOut[1]);
      _CHARACTER_t statIn_2[] = "ReorderTime,FactorTime,SolveTime,Flops";
      error = dss_statistics (handle, opt, statIn_2, statOut);
      if (error != MKL_DSS_SUCCESS)
        goto printError;
      cout << "statistics reorder time: " << statOut[0] << endl;
      cout << "statistics factor time: " << statOut[1] << endl;
      cout << "statistics solve time: " << statOut[2] << endl;
      cout << "factor flops: " << statOut[3] << endl;
      cout << "Mflops/sec in factor step: " << statOut[3]/(1e6*statOut[1]) << endl;
    }
/* -------------------------- */
/* Deallocate solver storage  */
/* -------------------------- */
  error = dss_delete (handle, opt);
  if (error != MKL_DSS_SUCCESS)
    goto printError;
/* ---------------------- */
/* Print solution vector  */
/* ---------------------- */
  printf (" Solution array: ");
  //for (int i = 0; i < nCols; i++)
  for (int i = 0; i < 10; i++)
    printf (" %g", solValues[i]);
  printf ("\n");
  return 0;
printError:
  printf ("Solver returned error code %d\n", error);
  exit (1);
}

int main(int argc, char* argv[]) {
  /** usage */
  int nrequired_args = 3;
  if (argc != nrequired_args){
      fprintf(stderr, "NAME:\n\tbenchmark_dss : script to benchmark mkl DSS performance\n");
      fprintf(stderr, "\nSYNOPSIS:\n");
      fprintf(stderr, "\tmkl_spmv MATRIX_A NUMBER_OF_THREADS\n");
      fprintf(stderr, "\nDESCRIPTION:\n");
      fprintf(stderr, "\tNUMBER_OF_THREADS: {0,1,2,...}\n");
      fprintf(stderr, "\t\t0: Use number of threads determined by MKL\n");
      fprintf(stderr, "\nSAMPLE EXECUTION:\n");
      fprintf(stderr, "\t%s test.mtx 2\n", argv[0]);
      exit(1);
  }
  /** parse arguments */
  int iarg = 1;
  char* strpathA = argv[iarg];    iarg++;
  int nthreads = atoi(argv[iarg]);    iarg++;
  assert(nrequired_args == iarg);

  /** MKL general setting */
  if (nthreads == 0) {
    nthreads = mkl_get_max_threads();
  }
  cout << "Parallel threads: " << nthreads << endl;
  mkl_set_num_threads(nthreads);

  /** instantiate a benchmarking object **/
  //char *strpathA= &path_sample1[0]; 
  CPUBenchmark bench_obj(strpathA);
  
  int n_solve_iter= 10;
  for (int i = 0; i < 1; i++) {
    bench_obj.dss_routine(n_solve_iter);
  }
}

