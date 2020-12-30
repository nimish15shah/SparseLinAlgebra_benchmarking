#ifndef CSR_H
#define CSR_H

typedef int csi;
typedef double csv;
csv zero = 0.0;
#define CS_MAX(a,b) (((a) > (b)) ? (a) : (b))

typedef struct csr_t {
    csi m;
    csi n;
    csi nzmax;
    csi nr;
    csi* r;
    csi* p;
    csi* j;
    csv* x;
} csr;
/* method declarations {{{*/
csr *csr_spfree(csr *A);
/*}}}*/

/*csr util{{{*/

/* free workspace and return a sparse matrix result */
csr *csr_done(csr *C, void *w, void *x, csi ok) {
    return (ok ? C : csr_spfree(C)); /* return result if OK, else free it */
}

/* wrapper for free */
void *cs_free(void *p) {
    if (p)
        free(p); /* free p if it is not already NULL */
    return (NULL); /* return NULL to simplify the use of cs_free */
}

/* wrapper for realloc */
void *csr_realloc(void *p, csi n, size_t size, csi *ok) {
    void *pnew = NULL;
    pnew = realloc(p, CS_MAX(n, 1) * size); /* realloc the block */
    *ok = (pnew != NULL); /* realloc fails if pnew is NULL */
    if (pnew == NULL) {
        printf("%d:reallocation failed, pnew is NULL\n", __LINE__);
    }
    return ((*ok) ? pnew : p); /* return original p if failure */
}

/* wrapper for realloc */
void *cs_realloc(void *p, csi n, size_t size, csi *ok) {
    void *pnew = NULL;
    pnew = realloc(p, CS_MAX(n, 1) * size); /* realloc the block */
    *ok = (pnew != NULL); /* realloc fails if pnew is NULL */
    if (pnew == NULL) {
        printf("reallocation failed\n");
    }
    return ((*ok) ? pnew : p); /* return original p if failure */
}

/* change the max # of entries sparse matrix */
csi csr_sprealloc(csr *A, csi nzmax) {
    csi ok, oki = 0, okj = 1, okx = 1;
    if (!A)
        return (0);
    if (nzmax <= 0)
        nzmax = A->p[A->m];
    A->j = (int*) csr_realloc(A->j, nzmax, sizeof (csi), &oki);
    if (A->x)
        A->x = (csv*) csr_realloc(A->x, nzmax, sizeof (csv), &okx);
    ok = (oki && okj && okx);
    if (ok)
        A->nzmax = nzmax;
    return (ok);
}

/* free a sparse matrix */
csr *csr_spfree(csr *A) {
    if (!A)
        return (NULL); /* do nothing if A already NULL */
    cs_free(A->p);
    A->p = NULL;
    cs_free(A->j);
    A->j = NULL;
    cs_free(A->x);
    A->x = NULL;
    cs_free(A->r);
    A->r = NULL;
    cs_free(A); /* free the cs struct and return NULL */
    return NULL;
}

/* allocate a sparse matrix (triplet form or compressed-ROW form) */
csr *csr_spalloc(csi m, csi n, csi nzmax, int values, int triplet, csv f) {
    csr* A = (csr*) calloc(1, sizeof (csr)); /* allocate the cs struct */
    if (!A) {
        perror("sparse allocation failed");
        return (NULL); /* out of memory */
    }
    A->m = m; /* define dimensions and nzmax */
    A->n = n;
    A->nzmax = nzmax = CS_MAX(nzmax, 0);
    A->nr = 0; // number of nonzero rows
    A->p = (csi*) calloc(m + 2, sizeof (csi));
    A->j = (csi*) calloc(CS_MAX(nzmax, 1), sizeof (csi));
    A->x = (csv*) calloc(CS_MAX(nzmax, 1), sizeof (csv));
    return ((!A->p || !A->j || !A->x) ? csr_spfree(A) : A);
}/*}}}*/

/** Multiply two sparse matrices which are stored in CSR format. MKL is used */
csr *csr_multiply(csi Am, csi An, csi Anzmax, const csi* Ap, const csi* Aj, const csv* Ax, csi Bm, csi Bn, csi Bnzmax, const csi* Bp, const csi* Bj, const csv* Bx, long* nummult, csi* xb, csv* x) { /*{{{*/
    csv tf = 0;
    csi p, jp, j, kp, k, i, nz = 0, anz, *Cp, *Cj, m, n,
            bnz, values = 1;
    csv *Cx;
    csr *C;
    if (An != Bm)
        return (NULL);
    if (Anzmax == 0 || Bnzmax == 0) {
        C = csr_spalloc(Am, Bn, 0, values, 0, tf);
        return C;
    }
    m = Am;
    anz = Ap[Am];
    n = Bn;
    bnz = Bp[Bm];
    for (i = 0; i < n; i++) xb[i] = 0;
    for (i = 0; i < n; i++)
        xb[i] = 0;
    values = (Ax != NULL) && (Bx != NULL);
    csi tnz = (anz + bnz) * 2;
    C = csr_spalloc(m, n, tnz, values, 0, tf); /* allocate result */
    if (!C || !xb || (values && !x))
        return (csr_done(C, xb, x, 0));
    Cp = C->p;
    for (i = 0; i < m; i++) {
        if (((nz + n) > C->nzmax)) {
            if (!csr_sprealloc(C, (2 * (C->nzmax) + n))) {
                return (csr_done(C, xb, x, 0)); // out of memory
            } else {
            }
        }
        Cj = C->j;
        Cx = C->x; /* C->j and C->x may be reallocated */
        Cp[i] = nz; /* row i of C starts here */
        for (jp = Ap[i]; jp < Ap[i + 1]; jp++) {
            j = Aj[jp];
            for (kp = Bp[j]; kp < Bp[j + 1]; kp++) {
                k = Bj[kp]; /* B(i,j) is nonzero */
                if (xb[k] != i + 1) {
                    xb[k] = i + 1; /* i is new entry in column j */
                    Cj[nz++] = k; /* add i to pattern of C(:,j) */
                    if (x) {
                        x[k] = Ax[jp] * Bx[kp]; /* x(i) = beta*A(i,j) */
                        (*nummult)++;
                    }
                } else if (x) {
                    x[k] += (Ax[jp] * Bx[kp]); /* i exists in C(:,j) already */
                    (*nummult)++;
                }
            }
        }
        if (values)
            for (p = Cp[i]; p < nz; p++)
                Cx[p] = x[Cj[p]];
    }
    Cp[m] = nz; /* finalize the last row of C */
    csr_sprealloc(C, 0); /* remove extra space from C */
    xb = NULL;
    x = NULL;
    return C;
}/*}}}*/

#endif

