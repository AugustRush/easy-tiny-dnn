//
//  accelerate.h
//  Easy
//
//  Created by pingwei liu on 2019/3/13.
//  Copyright © 2019 pingwei liu. All rights reserved.
//

#ifndef accelerate_h
#define accelerate_h

#include <stdio.h>

#if defined __cplusplus
extern "C" {
#endif
#ifdef __APPLE__
#if defined(CNN_USE_DOUBLE)
#define element_type double
#else
#define element_type float
#endif
    //vector
    void apple_tanh(element_type *dst, const element_type *src, int count);
    void apple_vsq(element_type *dst, const element_type *src, int count);
    void apple_pow(float *dst, const float *src, int count, float scalar);
    void apple_caxpy(float *y, const float *x, float alpha, const int count);// y[i] = x[i] * alpha + y[i]
    void apple_copy(float *y, const float *x, const int strideX, const int strideY, const int count);// y[i] = x[i]
    void apple_clear(float *y, const int stride, const int count);// y[i*stride] = 0.0
    void apple_fill(element_type *y, element_type scalar, const int stride, const int count); // y[i * stride] = scalar
    void apple_max(const float *x, float *scalar, const int stride, const int count);
    void apple_maxi(const float *x, float *scalar, unsigned long *index, const int stride, const int count);
    void apple_min(const float *x, float *scalar, const int stride, const int count);
    void apple_mini(const float *x, float *scalar, unsigned long *index, const int stride, const int count);
    void apple_saxpby(float *dst, const float alpha, const float *src, const float beta, const int count); //dst[i] = dst[i] * alpha + src[i] * beta
    void apple_vadd(float *z, const float *x, const float *y, const int count);// z[i] = x[i] + y[i]
    void apple_vmul(float *z, const float *x, const float *y, const int count);// z[i] = x[i] * y[i]
    void apple_vsub(float *z, const float *x, const float *y, const int count);// z[i] = x[i] - y[i]
    void apple_vdiv(float *z, const float *x, const float *y, const int count);// z[i] = x[i] / y[i]
    void apple_vam(float *d, const float *a, const float *b, const float *c, const int count);// d[i] = (a[i] + b[i]) * c[i]
    void apple_vma(float *d, const float *a, const float *b, const float *c, const int count);// d[i] = a[i] * b[i] + c[i]
    void apple_vmsb(float *d, const float *a, const float *b, const float *c, const int count);// d[i] = a[i] * b[i] - c[i]
    void apple_vsbm(float *d, const float *a, const float *b, const float *c, const int count);// d[i] = (a[i] - b[i]) * c[i]
    void apple_vasm(float *d, const float *a, const float *b, const float c, const int count);// d[i] = (a[i] + b[i]) * c
    void apple_vsbsm(float *d, const float *a, const float *b, const float c, const int count);// d[i] = (a[i] - b[i]) * c
    void apple_vmsa(float *d, const float *a, const float *b, const float c, const int count);// d[i] = a[i] * b[i] + c
    void apple_vmmsb(float *e, const float *a, const float *b, const float *c, const float *d, const int count);// e[i] = a[i] * b[i] - c[i] * d[i]
    void apple_vaam(float *e, const float *a, const float *b, const float *c, const float *d, const int count);// e[i] = (a[i] + b[i]) * (c[i] + d[i])
    void apple_vsbsbm(float *e, const float *a, const float *b, const float *c, const float *d, const int count);// e[i] = (a[i] - b[i]) * (c[i] - d[i])
    void apple_vasbm(float *e, const float *a, const float *b, const float *c, const float *d, const int count);// e[i] = (a[i] + b[i]) * (c[i] - d[i])
    void apple_vsadd(float *c, const float *a, const float b, const int count);// c[i] = a[i] + b
    void apple_vsmul(float *c, const float *a, const float b, const int count);// c[i] = a[i] * b
    void apple_vsdiv(float *c, const float *a, const float b, const int count);// c[i] = a[i] / b
    void apple_svdiv(float *c, const float a, const float *b, const int count);// c[i] = a / b[i]
    void apple_vsma(float *d, const float *a, const float b, const float *c, const int count);// d[i] = a[i] * b + c[i]
    void apple_vsmsa(float *d, const float *a, const float b, const float c, const int count);// d[i] = a[i] * b + c
    void apple_vsmsma(float *e, const float *a, const float b, const float *c, const float d, const int count);// d[i] = a[i] * b + c[i] * d
    void apple_vabs(float *c, const float *a, const int count);// c[i] = |a[i]|
    void apple_vneg(float *c, const float *a, const int count);// c[i] = -a[i]
    void apple_sve(float *c, const float *a, const int count);// *c = sum(a[i])
    void apple_thres(float *c, const float *a, const float threshold , const int count);// > threshold stay origin, < threshold change to threshold
    /*
     for (n = 0; n < N; ++n)
        if (*B <= A[n*IA])
            D[n*ID] = *C;
        else
            D[n*ID] = -(*C);
     */
    void apple_limit(float *c, const float *a, const float limit, const float scalar, const int count);//
    void apple_exp(float *c, const float *a, const int count);
    
    /**
     for (n = 0; n < N; ++n) {
        if (A[n*IA] < *B)
            D[n*ID] = *B;
        else if (A[n*IA] > *C)
            D[n*ID] = *C;
        else
            D[n*ID] = A[n*IA];
     }

     @param c output
     @param a input
     @param min low clipping threshold
     @param max high clipping threshold
     @param count count of elements
     */
    void apple_clip(float *c, const float *a, const float min, const float max, const int count);// if < min
    void apple_reverse(float *c, const float *a, const int count);
    
    /**
     Vector index in-place sort; single precision.

     @param c output sorted indexes
     @param a input vector
     @param order 1 ascending  -1 descending
     @param count process numbers count
     */
    void apple_vsorti(unsigned long *c, const float *a, const int order, const int count);
    
    void apple_meanv(float *scalar, const float *a, const int stride, const int count);
    void apple_measqv(float *scalar, const float *a, const int stride, const int count);
#pragma mark --------
    
    //matrix
    void apple_mmul(float *c, const float *a, const float *b, const int m, const int n, const int p);
    void apple_mtrans(float *c, const float *a, const int m, const int n);//
    void apple_sgemm(float *c, const float *a, int trans_a, const float *b, int trans_b, const int m, const int n, const int k, const int lda, const int ldb, const int ldc, const float alpha, const int beta);//C=alpha*A*B+beta*C  {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113,AtlasConj=114}
    
    /**
     convolution a use b store in c

     @param c output
     @param a image matrix
     @param a_r rows of a
     @param a_c columns of b
     @param f filter / kernel matrix
     @param p rows of f, must be odd
     @param q coloumns of f, must be odd
     */
    void apple_imgfir(float *c, const float *a, const int a_r, const int a_c, const float *f, const int p, const int q);
    /*
     Single-precision real input submatrix.
     
     __C
     Single-precision real output submatrix.
     
     __M
     Number of columns in A and C
     
     __N
     Number of rows in A and C
     
     __TA
     Number of columns in the matrix of which A is a submatrix.
     
     __TC
     Number of columns in the matrix of which C is a submatrix.
     */
    void apple_mmov(float *c, const float *a, const int m, const int n, const int ta, const int tc);
    
#pragma mark --- threads
    
    void apple_dispatch_apply(size_t iterations,void(^dispatch_block)(size_t idx));
    
#endif
#if defined __cplusplus
};
#endif

#endif /* accelerate_h */
