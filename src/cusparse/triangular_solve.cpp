
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <assert.h>

// CUDA Runtime
#include <cuda_runtime.h>

// Using updated (v2) interfaces for CUBLAS and CUSPARSE
#include <cusparse.h>
#include <cublas_v2.h>
#include "cusolverSp.h"

// Utilities and system includes
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper for CUDA error checking
#include "helper_cusolver.h"

#define FLOAT_T float

template <typename T_ELEM>
int loadMMSparseMatrix(
    char *filename,
    char elem_type,
    bool csrFormat,
    int *m,
    int *n,
    int *nnz,
    T_ELEM **aVal,
    int **aRowInd,
    int **aColInd,
    int extendSymMatrix);

struct trsolve_testOpts {
  char *sparse_mat_filename;   // by switch -F<filename>
  const char *testFunc; // by switch -R<name>
  const char *reorder; // by switch -P<name>
  int lda; // by switch -lda<int>
  bool do_lu; // by switch -lu
};

void UsageSP(void)
{
    printf( "<options>\n");
    printf( "-h          : display this help\n");
    printf( "-R=<name>   : choose a linear solver\n");
    printf( "              chol (cholesky factorization), this is default\n");
    printf( "              qr   (QR factorization)\n");
    printf( "              lu   (LU factorization)\n");
    printf( "-P=<name>    : choose a reordering\n");
    printf( "              symrcm (Reverse Cuthill-McKee)\n");
    printf( "              symamd (Approximate Minimum Degree)\n");
    printf( "              metis  (nested dissection)\n");
    printf( "-file=<filename> : filename containing a matrix in MM format\n");
    printf( "-lu : whether to do lu decomposition before triangular solver, depending on whether the original marix is triangular or not\n");
    printf( "-device=<device_id> : <device_id> if want to run on specific GPU\n");

    exit( 0 );
}

void parseCommandLineArguments(int argc, char *argv[], struct trsolve_testOpts &opts)
{
    memset(&opts, 0, sizeof(opts));

    if (checkCmdLineFlag(argc, (const char **)argv, "-h"))
    {
        UsageSP();
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "R"))
    {
        char *solverType = NULL;
        getCmdLineArgumentString(argc, (const char **)argv, "R", &solverType);

        if (solverType)
        {
            if ((STRCASECMP(solverType, "chol") != 0) && (STRCASECMP(solverType, "lu") != 0) && (STRCASECMP(solverType, "qr") != 0))
            {
                printf("\nIncorrect argument passed to -R option\n");
                UsageSP();
            }
            else
            {
                opts.testFunc = solverType;
            }
        }
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "P"))
    {
        char *reorderType = NULL;
        getCmdLineArgumentString(argc, (const char **)argv, "P", &reorderType);

        if (reorderType)
        {
            if ((STRCASECMP(reorderType, "symrcm") != 0) &&
				(STRCASECMP(reorderType, "symamd") != 0) &&
				(STRCASECMP(reorderType, "metis" ) != 0)    )
            {
                printf("\nIncorrect argument passed to -P option\n");
                UsageSP();
            }
            else
            {
                opts.reorder = reorderType;
            }
        }
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "file"))
    {
        char *fileName = 0;
        getCmdLineArgumentString(argc, (const char **)argv, "file", &fileName);

        if (fileName)
        {
            opts.sparse_mat_filename = fileName;
        }
        else
        {
            printf("\nIncorrect filename passed to -file \n ");
            UsageSP();
        }
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "lu"))
    {
      opts.do_lu= 1;
    }
    else 
    {
      opts.do_lu= 0;
    }
}

int main (int argc, char *argv[])
{
    struct trsolve_testOpts opts;

    cudaStream_t stream = NULL;
    checkCudaErrors(cudaStreamCreate(&stream));

    cusparseMatDescr_t descrA = NULL;
    
    int rowsA = 0; /* number of rows of A */
    int colsA = 0; /* number of columns of A */
    int nnzA  = 0; /* number of nonzeros of A */
    int baseA = 0; /* base index in CSR format */

    /* CSR(A) from I/O */
    int *h_csrRowPtrA = NULL;
    int *h_csrColIndA = NULL;
    FLOAT_T *h_csrValA = NULL;

    FLOAT_T *h_x  = NULL; /* x = A \ b */
    FLOAT_T *h_b  = NULL; /* b = ones(n,1) */
    FLOAT_T *h_r  = NULL; /* r = b - A*x */

/* device copy of A*/
    int *d_csrRowPtrA = NULL;
    int *d_csrColIndA = NULL;
    FLOAT_T *d_csrValA = NULL;

    parseCommandLineArguments(argc, argv, opts);

    printf("step 1: read matrix market format\n");
    if (opts.sparse_mat_filename == NULL)
    {
        fprintf(stderr, "Error: input matrix is not provided\n");
        return EXIT_FAILURE;
    }
    if (loadMMSparseMatrix<FLOAT_T>(opts.sparse_mat_filename, 'd', true , &rowsA, &colsA,
           &nnzA, &h_csrValA, &h_csrRowPtrA, &h_csrColIndA, true))
    {
        exit(EXIT_FAILURE);
    }
    baseA = h_csrRowPtrA[0]; // baseA = {0,1}
    printf("sparse matrix A is %d x %d with %d nonzeros, base=%d\n", rowsA, colsA, nnzA, baseA);

    if ( rowsA != colsA ){
        fprintf(stderr, "Error: only support square matrix\n");
        return 1;
    }


    checkCudaErrors(cusparseCreateMatDescr(&descrA));
    checkCudaErrors(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    if (baseA)
    {
        checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE));
    }
    else
    {
        checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
    }
    printf("Warning: Assumed an upper triangular matrix. The code will fail but may not throw errors for other matrix types\n");
    cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_UPPER);
    cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT);

    //display_matrix(rowsA, colsA, nnzA,descrA, h_csrValA, h_csrRowPtrA, h_csrColIndA);
    display_matrix(2, colsA, nnzA,descrA, (double *)h_csrValA, h_csrRowPtrA, h_csrColIndA);

    checkCudaErrors(cudaMalloc((void **)&d_csrRowPtrA, sizeof(int)*(rowsA+1)));
    checkCudaErrors(cudaMalloc((void **)&d_csrColIndA, sizeof(int)*nnzA));
    checkCudaErrors(cudaMalloc((void **)&d_csrValA   , sizeof(FLOAT_T)*nnzA));
    printf("step 4: prepare data on device\n");
    checkCudaErrors(cudaMemcpyAsync(d_csrRowPtrA, h_csrRowPtrA, sizeof(int)*(rowsA+1), cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_csrColIndA, h_csrColIndA, sizeof(int)*nnzA     , cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_csrValA   , h_csrValA   , sizeof(FLOAT_T)*nnzA  , cudaMemcpyHostToDevice, stream));

    h_x = (FLOAT_T*)malloc(sizeof(FLOAT_T)*colsA);
    h_b = (FLOAT_T*)malloc(sizeof(FLOAT_T)*rowsA);
    h_r = (FLOAT_T*)malloc(sizeof(FLOAT_T)*rowsA);
    assert(NULL != h_x);
    assert(NULL != h_b);
    assert(NULL != h_r);

    printf("step 3: b(j) = 1 + j/n \n");
    for(int row = 0 ; row < rowsA ; row++)
    {
        h_b[row] = 1.0 + ((FLOAT_T)row)/((FLOAT_T)rowsA);
    }

    FLOAT_T *d_x = NULL; /* x = A \ b */
    FLOAT_T *d_b = NULL; /* a copy of h_b */
    checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(FLOAT_T)*colsA));
    checkCudaErrors(cudaMalloc((void **)&d_b, sizeof(FLOAT_T)*rowsA));
    checkCudaErrors(cudaMemcpyAsync(d_b         , h_b         , sizeof(FLOAT_T)*rowsA , cudaMemcpyHostToDevice, stream));

    /* Create CUSPARSE context */
    cusparseHandle_t cusparseHandle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle);

    checkCudaErrors(cusparseStatus);

    cusolverSpHandle_t cusolverHandle = NULL;
    checkCudaErrors(cusolverSpCreate(&cusolverHandle));

/* bind stream to cusparse and cusolver*/
    checkCudaErrors(cusolverSpSetStream(cusolverHandle, stream));
    checkCudaErrors(cusparseSetStream(cusparseHandle, stream));

    // LU decomposition
    if (opts.do_lu) 
    {
      printf("LU factorization is not implemented. The original matrix is supposed to be upper triangular\n");
      exit(1);
      //printf("step 5: solve A*x = b on CPU \n");
    }

    /* This will pick the best possible CUDA capable device */
    cudaDeviceProp deviceProp;
    int devID = findCudaDevice(argc, (const char **)argv);
    printf("GPU selected Device ID = %d \n", devID);

    if (devID < 0)
    {
        printf("Invalid GPU device %d selected,  exiting...\n", devID);
        exit(EXIT_SUCCESS);
    }

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    /* Statistics about the GPU device */
    printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
           deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    /* create the analysis info object for the A matrix */
    cusparseSolveAnalysisInfo_t infoA = 0;
    cusparseStatus = cusparseCreateSolveAnalysisInfo(&infoA);
    double start, stop;
    start = second();
    cusparseStatus = cusparseScsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             rowsA, nnzA, descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA , infoA);
    stop = second();
    double time_analysis= stop - start;

    // Back Substitution
    const float floatone = 1.0;
    start = second();
    cusparseStatus = cusparseScsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, rowsA, &floatone, descrA,
                                          d_csrValA, d_csrRowPtrA, d_csrColIndA, infoA, d_b, d_x);
    stop = second();
    double time_solve= stop - start;

    checkCudaErrors(cusparseStatus);
    
    printf("analysis time: %10.6f sec, triangular solve time: %10.6f sec", time_analysis, time_solve);
}
