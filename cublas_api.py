# coding:utf-8: © Arno Pähler, 2007-08
# NP: remove absolute path in CB

from cublas_defs import *

## Loading the library below causes a Segmentation fault
## upon exit - functionality itself is not affected
## This is a known bug in cleanup - see NVIDIA CUDA forum
##
CB = "libcublas.so"
cb = CDLL(CB)

##/* CUBLAS helper functions */
##
##cublasStatus cublasInit (void);
##cublasStatus cublasShutdown (void);
##cublasStatus cublasGetError (void);
##cublasStatus cublasAlloc (int n, int elemSize, void **devicePtr);
##cublasStatus cublasFree (const void *devicePtr);
cublasInit = cb.cublasInit
cublasInit.restype = cublasStatus
cublasInit.argtypes = None

cublasShutdown = cb.cublasShutdown
cublasShutdown.restype = cublasStatus
cublasShutdown.argtypes = None

cublasGetError = cb.cublasGetError
cublasGetError.restype = cublasStatus
cublasGetError.argtypes = None

cublasAlloc = cb.cublasAlloc
cublasAlloc.restype = cublasStatus
cublasAlloc.argtypes = [ c_int, c_int, POINTER(c_void_p) ]

cublasFree = cb.cublasFree
cublasFree.restype = cublasStatus
cublasFree.argtypes = [ c_void_p ]

##cublasStatus cublasSetVector (int n, int elemSize, const void *x, 
##                                        int incx, void *devicePtr, int incy);
##cublasStatus cublasGetVector (int n, int elemSize, const void *x, 
##                                        int incx, void *y, int incy);
##cublasStatus cublasSetMatrix (int rows, int cols, int elemSize, 
##                                        const void *A, int lda, void *B, 
##                                        int ldb);
##cublasStatus cublasGetMatrix (int rows, int cols, int elemSize, 
##                                        const void *A, int lda, void *B,
##                                        int ldb);
cublasSetVector = cb.cublasSetVector
cublasSetVector.restype = cublasStatus
cublasSetVector.argtypes = [ c_int, c_int, c_void_p, c_int, c_void_p, c_int ]

cublasGetVector = cb.cublasGetVector
cublasGetVector.restype = cublasStatus
cublasGetVector.argtypes = [ c_int, c_int, c_void_p, c_int, c_void_p, c_int ]

cublasSetMatrix = cb.cublasSetMatrix
cublasSetMatrix.restype = cublasStatus
cublasSetMatrix.argtypes = [ c_int, c_int, c_int,
                             c_void_p, c_int, c_void_p, c_int ]

cublasGetMatrix = cb.cublasGetMatrix
cublasGetMatrix.restype = cublasStatus
cublasGetMatrix.argtypes = [ c_int, c_int, c_int,
                             c_void_p, c_int, c_void_p, c_int ]

##/* ---------------- CUBLAS single-precision BLAS1 functions ---------------- */
##
##int cublasIsamax (int n, const float *x, int incx);
##int cublasIsamin (int n, const float *x, int incx);
##float cublasSasum (int n, const float *x, int incx);
cublasIsamax = cb.cublasIsamax
cublasIsamax.restype = c_int
cublasIsamax.argtypes = [ c_int, c_void_p, c_int ]

cublasIsamin = cb.cublasIsamin
cublasIsamin.restype = c_int
cublasIsamin.argtypes = [ c_int, c_void_p, c_int ]

cublasSasum = cb.cublasSasum
cublasSasum.restype = c_float
cublasSasum.argtypes = [ c_int, c_void_p, c_int ]

##void cublasSaxpy (int n, float alpha, const float *x, int incx, 
##                            float *y, int incy);
##void cublasScopy (int n, const float *x, int incx, float *y, int incy);
##float cublasSdot (int n, const float *x, int incx, const float *y, int incy);
cublasSaxpy = cb.cublasSaxpy
cublasSaxpy.restype = None
cublasSaxpy.argtypes = [ c_int, c_float, c_void_p, c_int,c_void_p, c_int ]

cublasScopy = cb.cublasScopy
cublasScopy.restype = None
cublasScopy.argtypes = [ c_int, c_void_p, c_int, c_void_p, c_int ]

cublasSdot = cb.cublasSdot
cublasSdot.restype = c_float
cublasSdot.argtypes = [ c_int, c_void_p, c_int, c_void_p, c_int ]

##float cublasSnrm2 (int n, const float *x, int incx);
##void cublasSrot (int n, float *x, int incx, float *y, int incy, 
##                           float sc, float ss);
##void cublasSrotg (float *sa, float *sb, float *sc, float *ss);
##void cublasSrotm(int n, float *x, int incx, float *y, int incy, 
##                           const float* sparam);
##void cublasSrotmg (float *sd1, float *sd2, float *sx1, 
##                             const float *sy1, float* sparam);
##void cublasSscal (int n, float alpha, float *x, int incx);
##void cublasSswap (int n, float *x, int incx, float *y, int incy);
cublasSnrm2 = cb.cublasSnrm2
cublasSnrm2.restype = c_float
cublasSnrm2.argtypes = [ c_int, c_void_p, c_int ]

cublasSrot = cb.cublasSrot
cublasSrot.restype = None
cublasSrot.argtypes = [ c_int, c_void_p, c_int, c_void_p, c_int,
                        c_float, c_float ]

cublasSrotg = cb.cublasSrotg
cublasSrotg.restype = None
cublasSrotg.argtypes = [ c_float_p, c_float_p, c_float_p, c_float_p ]

cublasSrotm = cb.cublasSrotm
cublasSrotm.restype = None
cublasSrotm.argtypes = [ c_int, c_void_p, c_int, c_void_p, c_int, c_void_p ]

cublasSrotmg = cb.cublasSrotmg
cublasSrotmg.restype = None
cublasSrotmg.argtypes = [ c_float_p, c_float_p, c_float_p,
                          c_float_p, c_float_p ]

cublasSscal = cb.cublasSscal
cublasSscal.restype = None
cublasSscal.argtypes = [ c_int, c_float, c_void_p, c_int ]

cublasSswap = cb.cublasSswap
cublasSswap.restype = None
cublasSswap.argtypes = [ c_int, c_void_p, c_int, c_void_p, c_int ]

##
##      NOT YET IMPLEMENTED in Python / complex BLAS1
##

##/* ----------------- CUBLAS single-complex BLAS1 functions ----------------- */
##
##void cublasCaxpy (int n, cuComplex alpha, const cuComplex *x, 
##                            int incx, cuComplex *y, int incy);
##void cublasCcopy (int n, const cuComplex *x, int incx, cuComplex *y,
##                            int incy);
##void cublasCscal (int n, cuComplex alpha, cuComplex *x, int incx);
##void cublasCsscal (int n, float alpha, cuComplex *x, int incx);
##void cublasCswap (int n, cuComplex *x, int incx, cuComplex *y,
##                            int incy);
##cuComplex cublasCdotu (int n, const cuComplex *x, int incx, 
##                                 const cuComplex *y, int incy);
##cuComplex cublasCdotc (int n, const cuComplex *x, int incx, 
##                                 const cuComplex *y, int incy);
##int cublasIcamax (int n, const cuComplex *x, int incx);
##int cublasIcamin (int n, const cuComplex *x, int incx);
##float cublasScasum (int n, const cuComplex *x, int incx);
##float cublasScnrm2 (int n, const cuComplex *x, int incx);

##/* --------------- CUBLAS single precision BLAS2 functions  ---------------- */
##
##void cublasSgbmv (char trans, int m, int n, int kl, int ku, 
##                            float alpha, const float *A, int lda, 
##                            const float *x, int incx, float beta, float *y, 
##                            int incy);
cublasSgbmv = cb.cublasSgbmv
cublasSgbmv.restype = None
cublasSgbmv.argtypes = [ c_char, c_int, c_int, c_int, c_int,
                         c_float, c_void_p, c_int,
                         c_void_p, c_int, c_float, c_void_p, c_int ]

##void cublasSgemv (char trans, int m, int n, float alpha,
##                            const float *A, int lda, const float *x, int incx,
##                            float beta, float *y, int incy);
cublasSgemv = cb.cublasSgemv
cublasSgemv.restype = None
cublasSgemv.argtypes = [ c_char, c_int, c_int, c_float, c_void_p, c_int,
                         c_void_p, c_int, c_float, c_void_p, c_int ]

##void cublasSger (int m, int n, float alpha, const float *x, int incx,
##                           const float *y, int incy, float *A, int lda);
##
cublasSger = cb.cublasSger
cublasSger.restype = None
cublasSger.argtypes = [ c_int, c_int, c_float, c_void_p, c_int,
                        c_void_p, c_int, c_float, c_void_p, c_int ]

##void cublasSsbmv (char uplo, int n, int k, float alpha, 
##                            const float *A, int lda, const float *x, int incx, 
##                            float beta, float *y, int incy);
cublasSsbmv = cb.cublasSsbmv
cublasSsbmv.restype = None
cublasSsbmv.argtypes = [ c_char, c_int, c_int, c_float, c_void_p, c_int,
                         c_void_p, c_int, c_float, c_void_p, c_int ]

##void cublasSspmv (char uplo, int n, float alpha, const float *AP, 
##                            const float *x, int incx, float beta, float *y,
##                            int incy);
cublasSspmv = cb.cublasSspmv
cublasSspmv.restype = None
cublasSspmv.argtypes = [ c_char, c_int, c_float, c_void_p,
                         c_void_p, c_int, c_float, c_void_p, c_int ]

##void cublasSspr (char uplo, int n, float alpha, const float *x,
##                           int incx, float *AP);
cublasSspr = cb.cublasSspr
cublasSspr.restype = None
cublasSspr.argtypes = [ c_char, c_int, c_float, c_void_p, c_int, c_void_p ]

##void cublasSspr2 (char uplo, int n, float alpha, const float *x, 
##                            int incx, const float *y, int incy, float *AP);
cublasSspr2 = cb.cublasSspr2
cublasSspr2.restype = None
cublasSspr2.argtypes = [ c_char, c_int, c_float, c_void_p, c_int,
                         c_void_p, c_int, c_void_p ]

##void cublasSsymv (char uplo, int n, float alpha, const float *A,
##                            int lda, const float *x, int incx, float beta, 
##                            float *y, int incy);
cublasSsymv = cb.cublasSsymv
cublasSsymv.restype = None
cublasSsymv.argtypes = [ c_char, c_int, c_float, c_void_p, c_int,
                         c_void_p, c_int, c_float, c_void_p, c_int ]

##void cublasSsyr (char uplo, int n, float alpha, const float *x,
##                           int incx, float *A, int lda);
cublasSsyr = cb.cublasSsyr
cublasSsyr.restype = None
cublasSsyr.argtypes = [ c_char, c_int, c_float, c_void_p, c_int,
                        c_void_p, c_int ]

##void cublasSsyr2 (char uplo, int n, float alpha, const float *x, 
##                            int incx, const float *y, int incy, float *A, 
cublasSsyr2 = cb.cublasSsyr2
cublasSsyr2.restype = None
cublasSsyr2.argtypes = [ c_char, c_int, c_float, c_void_p, c_int,
                        c_void_p, c_int, c_void_p ]

##void cublasStbmv (char uplo, char trans, char diag, int n, int k, 
##                            const float *A, int lda, float *x, int incx);
cublasStbmv = cb.cublasStbmv
cublasStbmv.restype = None
cublasStbmv.argtypes = [ c_char, c_char, c_char, c_int, c_int,
                         c_void_p, c_int, c_void_p, c_int ]

##void cublasStbsv (char uplo, char trans, char diag, int n, int k, 
##                            const float *A, int lda, float *x, int incx);
cublasStbsv = cb.cublasStbsv
cublasStbsv.restype = None
cublasStbsv.argtypes = [ c_char, c_char, c_char, c_int, c_int,
                         c_void_p, c_int, c_void_p, c_int ]

##void cublasStpmv (char uplo, char trans, char diag, int n,
##                            const float *AP, float *x, int incx);
## not in the library !!!
##cublasStmpv = cb.cublasStmpv
##cublasStmpv.restype = None
##cublasStmpv.argtypes = [ c_char, c_char, c_char, c_int,
##                         c_void_p, c_void_p, c_int ]

##void cublasStpsv (char uplo, char trans, char diag, int n,
##                            const float *AP, float *x, int incx);
cublasStpsv = cb.cublasStpsv
cublasStpsv.restype = None
cublasStpsv.argtypes = [ c_char, c_char, c_char, c_int,
                         c_void_p, c_void_p, c_int ]

##void cublasStrmv (char uplo, char trans, char diag, int n, 
##                            const float *A, int lda, float *x, int incx);
cublasStrmv = cb.cublasStrmv
cublasStrmv.restype = None
cublasStrmv.argtypes = [ c_char, c_char, c_char, c_int,
                         c_void_p, c_int, c_void_p, c_int ]

##void cublasStrsv (char uplo, char trans, char diag, int n, 
##                            const float *A, int lda, float *x, int incx);
cublasStrsv = cb.cublasStrsv
cublasStrsv.restype = None
cublasStrsv.argtypes = [ c_char, c_char, c_char, c_int,
                         c_void_p, c_int, c_void_p, c_int ]

##
##      NOT YET IMPLEMENTED in Python / complex BLAS2
##

##/* ----------------- CUBLAS single complex BLAS2 functions ----------------- */
##void cublasCgemv (char trans, int m, int n, cuComplex alpha,
##                            const cuComplex *A, int lda, const cuComplex *x, 
##                            int incx, cuComplex beta, cuComplex *y, int incy);
##void cublasCgbmv (char trans, int m, int n, int kl, int ku,
##                            cuComplex alpha, const cuComplex *A, int lda,
##                            const cuComplex *x, int incx, cuComplex beta,
##                            cuComplex *y, int incy);
##void cublasChemv (char uplo, int n, cuComplex alpha, 
##                            const cuComplex *A, int lda, const cuComplex *x,
##                            int incx, cuComplex beta, cuComplex *y, int incy);
##void cublasChbmv (char uplo, int n, int k, cuComplex alpha,
##                            const cuComplex *A, int lda, const cuComplex *x,
##                            int incx, cuComplex beta, cuComplex *y, int incy);
##void cublasChpmv (char uplo, int n, cuComplex alpha,
##                            const cuComplex *AP, const cuComplex *x, int incx,
##                            cuComplex beta, cuComplex *y, int incy);
##void  cublasCtrmv (char uplo, char trans, char diag, int n,
##                             const cuComplex *A, int lda, cuComplex *x,
##                             int incx);
##void cublasCtbmv (char uplo, char trans, char diag, int n, int k, 
##                            const cuComplex *A, int lda, cuComplex *x,
##                            int incx);
##void cublasCtpmv (char uplo, char trans, char diag, int n,
##                            const cuComplex *AP, cuComplex *x, int incx);
##void cublasCtrsv (char uplo, char trans, char diag, int n,
##                            const cuComplex *A, int lda, cuComplex *x,
##                            int incx);
##void cublasCtbsv (char uplo, char trans, char diag, int n, int k, 
##                            const cuComplex *A, int lda, cuComplex *x,
##                            int incx);
##void cublasCtpsv (char uplo, char trans, char diag, int n,
##                            const cuComplex *AP, cuComplex *x, int incx);
##void cublasCgeru (int m, int n, cuComplex alpha, const cuComplex *x,
##                            int incx, const cuComplex *y, int incy,
##                            cuComplex *A, int lda);
##void cublasCgerc (int m, int n, cuComplex alpha, const cuComplex *x,
##                            int incx, const cuComplex *y, int incy,
##                            cuComplex *A, int lda);
##void cublasCher (char uplo, int n, cuComplex alpha, 
##                           const cuComplex *x, int incx, cuComplex *A, 
##                           int lda);
##void cublasChpr (char uplo, int n, cuComplex alpha,
##                           const cuComplex *x, int incx, cuComplex *AP);
##void cublasCher2 (char uplo, int n, cuComplex alpha,
##                            const cuComplex *x, int incx, const cuComplex *y,
##                            int incy, cuComplex *A, int lda);
##void cublasChpr2 (char uplo, int n, cuComplex alpha,
##                            const cuComplex *x, int incx, const cuComplex *y,
##                            int incy, cuComplex *AP);

##/* ---------------- CUBLAS single precision BLAS3 functions ---------------- */
##
##void cublasSgemm (char transa, char transb, int m, int n, int k, 
##                            float alpha, const float *A, int lda, 
##                            const float *B, int ldb, float beta, float *C, 
##                            int ldc);
cublasSgemm = cb.cublasSgemm
cublasSgemm.restype = None
cublasSgemm.argtypes = [ c_char, c_char, c_int, c_int, c_int,
                         c_float, c_void_p, c_int, c_void_p, c_int,
                         c_float, c_void_p, c_int ]

##void cublasSsymm (char side, char uplo, int m, int n, float alpha, 
##                            const float *A, int lda, const float *B, int ldb,
##                            float beta, float *C, int ldc);
cublasSsymm = cb.cublasSsymm
cublasSsymm.restype = None
cublasSsymm.argtypes = [ c_char, c_char, c_int, c_int,
                         c_float, c_void_p, c_int, c_void_p, c_int,
                         c_float, c_void_p, c_int ]

##void cublasSsyrk (char uplo, char trans, int n, int k, float alpha, 
##                            const float *A, int lda, float beta, float *C, 
##                            int ldc);
cublasSsyrk = cb.cublasSsyrk
cublasSsyrk.restype = None
cublasSsyrk.argtypes = [ c_char, c_char, c_int, c_int, c_float,
                         c_void_p, c_int, c_float, c_void_p, c_int ]

##void cublasSsyr2k (char uplo, char trans, int n, int k, float alpha, 
##                             const float *A, int lda, const float *B, int ldb, 
##                             float beta, float *C, int ldc);
cublasSsyr2k = cb.cublasSsyr2k
cublasSsyr2k.restype = None
cublasSsyr2k.argtypes = [ c_char, c_char, c_int, c_int, c_float,
                          c_void_p, c_int, c_void_p, c_int,
                          c_float, c_void_p, c_int ]

##void cublasStrmm (char side, char uplo, char transa, char diag,
##                            int m, int n, float alpha, const float *A, int lda,
##                            float *B, int ldb);
cublasStrmm = cb.cublasStrmm
cublasStrmm.restype = None
cublasStrmm.argtypes = [ c_char, c_char, c_char, c_char,
                          c_int, c_int, c_float,
                          c_void_p, c_int, c_void_p, c_int]
##void cublasStrsm (char side, char uplo, char transa, char diag,
##                            int m, int n, float alpha, const float *A, int lda,
##                            float *B, int ldb);
cublasStrsm = cb.cublasStrsm
cublasStrsm.restype = None
cublasStrsm.argtypes = [ c_char, c_char, c_char, c_char,
                          c_int, c_int, c_float,
                          c_void_p, c_int, c_void_p, c_int]

##
##      NOT YET IMPLEMENTED in Python / complex BLAS3
##

##/* ----------------- CUBLAS single complex BLAS3 functions ----------------- */
##
##void cublasCgemm (char transa, char transb, int m, int n, int k, 
##                            cuComplex alpha, const cuComplex *A, int lda,
##                            const cuComplex *B, int ldb, cuComplex beta,
##                            cuComplex *C, int ldc);
##
##void cublasCsymm (char side, char uplo, int m, int n,
##                            cuComplex alpha, const cuComplex *A, int lda,
##                            const cuComplex *B, int ldb, cuComplex beta,
##                            cuComplex *C, int ldc);
##void cublasChemm (char side, char uplo, int m, int n,
##                            cuComplex alpha, const cuComplex *A, int lda,
##                            const cuComplex *B, int ldb, cuComplex beta,
##                            cuComplex *C, int ldc);
##void cublasCsyrk (char uplo, char trans, int n, int k,
##                            cuComplex alpha, const cuComplex *A, int lda,
##                            cuComplex beta, cuComplex *C, int ldc);
##void cublasCherk (char uplo, char trans, int n, int k,
##                            cuComplex alpha, const cuComplex *A, int lda,
##                            cuComplex beta, cuComplex *C, int ldc);
##void cublasCsyr2k (char uplo, char trans, int n, int k,
##                             cuComplex alpha, const cuComplex *A, int lda,
##                             const cuComplex *B, int ldb, cuComplex beta,
##                             cuComplex *C, int ldc);
##void cublasCher2k (char uplo, char trans, int n, int k,
##                             cuComplex alpha, const cuComplex *A, int lda,
##                             const cuComplex *B, int ldb, cuComplex beta,
##                             cuComplex *C, int ldc);
##void cublasCtrmm (char side, char uplo, char transa, char diag,
##                            int m, int n, cuComplex alpha, const cuComplex *A,
##                            int lda, cuComplex *B, int ldb);
##void cublasCtrsm (char side, char uplo, char transa, char diag,
##                            int m, int n, cuComplex alpha, const cuComplex *A,
##                            int lda, cuComplex *B, int ldb);
##void cublasXerbla (const char *srName, int info);
