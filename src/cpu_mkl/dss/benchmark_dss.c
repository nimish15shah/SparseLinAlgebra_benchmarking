#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <mkl.h>
#include "mkl_dss.h"
#include "mkl_types.h"
#include <string.h>

// headers from common directory
#define MIC_TARGET
#include "mmio.h"
#include "util.h"
#include "paths.h"


int main() {
  char *strpathA= &path_sample1[0]; 

  /** read matrix market file for A matrix */
  MKL_INT nRows, nCols, nNNZ;
  MKL_INT *Ax, *Ay;
  _DOUBLE_PRECISION_t *Anz;
  read_mm(strpathA, &nRows, &nCols, &nNNZ, &Ax, &Ay, &Anz);

  /** construct csr storage for A matrix */
  //MKL_INT* A_col_idx;
  //MKL_INT* A_row_ptr;
  _INTEGER_t* A_col_idx;
  _INTEGER_t* A_row_ptr;
  _DOUBLE_PRECISION_t* Aval;


  /** rhs detail */
  MKL_INT nRhs= 1;
  _DOUBLE_PRECISION_t *rhs;


  A_col_idx = (_INTEGER_t*) mkl_malloc( nNNZ * sizeof( _INTEGER_t ), 64 );
  A_row_ptr = (_INTEGER_t*) mkl_malloc( (nRows+1) * sizeof(_INTEGER_t), 64 );
  //A_col_idx = (MKL_INT*) mkl_malloc( nNNZ * sizeof( MKL_INT ), 64 );
  //A_row_ptr = (MKL_INT*) mkl_malloc( (nRows+1) * sizeof( MKL_INT ), 64 );
  Aval = (double*) mkl_malloc( nNNZ * sizeof( double ),  64 );
  coo_to_csr(nRows, nNNZ, Ax, Ay, Anz, A_row_ptr, A_col_idx, Aval);

  rhs = (_DOUBLE_PRECISION_t*) mkl_malloc( nCols * sizeof( _DOUBLE_PRECISION_t ),  64 );
  _DOUBLE_PRECISION_t *solValues;
  solValues= (_DOUBLE_PRECISION_t*) mkl_malloc( nRows * sizeof( _DOUBLE_PRECISION_t ),  64 ); 

  _MKL_DSS_HANDLE_t handle;
  _INTEGER_t error;
  _DOUBLE_PRECISION_t statOut[8];

  MKL_INT opt = MKL_DSS_DEFAULTS;
  //MKL_INT opt = MKL_DSS_ZERO_BASED_INDEXING;
  MKL_INT sym = MKL_DSS_SYMMETRIC;
  //MKL_INT type = MKL_DSS_POSITIVE_DEFINITE;
  MKL_INT type = MKL_DSS_INDEFINITE;

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
  opt= MKL_DSS_AUTO_ORDER;
  return 1;
  error = dss_reorder (handle, opt, 0);
  if (error != MKL_DSS_SUCCESS)
    goto printError;
/* ------------------ */
/* Factor the matrix  */
/* ------------------ */
  error = dss_factor_real (handle, type, Aval);
  if (error != MKL_DSS_SUCCESS)
    goto printError;
/* ------------------------ */
/* Get the solution vector  */
/* ------------------------ */
  error = dss_solve_real (handle, opt, rhs, nRhs, solValues);
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
  for (int i = 0; i < nCols; i++)
    printf (" %g", solValues[i]);
  printf ("\n");
  exit (0);
printError:
  printf ("Solver returned error code %d\n", error);
  exit (1);

}

