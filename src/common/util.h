#ifndef UTIL_H
#define UTIL_H

#define OPTION_NOPRINT_MATRICES 0
#define OPTION_PRINT_MATRICES 1
MIC_TARGET int option_print_matrices = 0;

/** Converts COO matrix to CSR matrix */
void coo_to_csr(int m, int nnz, int* I, int* J, _DOUBLE_PRECISION_t* val, MKL_INT* AI, MKL_INT* AJ, _DOUBLE_PRECISION_t* Aval) { /*{{{*/
    MKL_INT info = 0;
    MKL_INT job[8];
    job[1] = 1; // one based indexing in csr

    job[2] = 0; // zero based indexing in coo
    job[3] = 2; // I don't know
    job[4] = nnz; // nnz

    job[0] = 1; // coo to csr
    job[5] = 0; // Acsr and AJR allocated by user
    mkl_dcsrcoo(job, &m, Aval, AJ, AI, &nnz, val, I, J, &info);
} /* ENDOF coo_to_csr }}}*/

/** Prints matrix in CSR format */
void MIC_TARGET printmm_one(int m, double* Aval, int* AJ, int* AI) { //{{{

    if (option_print_matrices == OPTION_NOPRINT_MATRICES)
        return;
    int i;
    for (i = 0; i < m; i++) {
        printf("%d: ", i + 1);
        int j;
        for (j = AI[i] - 1; j < AI[i + 1] - 1; j++) {
            printf("%d:%g  ", AJ[j], Aval[j]);
        }
        printf("\n");
    }
    printf("\n");
}//}}}

/** Writes matrix in CSR format in to a file using Matrix Market format */
void MIC_TARGET printfilemm_one(char* file, int m, int n, double* Aval, int* AJ, int* AI) {//{{{

    FILE* f = fopen(file, "w");
    if (f == NULL) {
        printf("%s %s %d: %s cannot be opened to write matrix\n", __FILE__, __PRETTY_FUNCTION__, __LINE__, file);
        exit(1);
    }
    int i;
    fprintf(f, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(f, "%d %d %d\n", m, n, AI[m] - 1);
    for (i = 0; i < m; i++) {
        int j;
        for (j = AI[i] - 1; j < AI[i + 1] - 1; j++) {
            fprintf(f, "%d %d %g\n", i + 1, AJ[j], Aval[j]);
        }
    }
    fclose(f);
}//}}}
#endif
