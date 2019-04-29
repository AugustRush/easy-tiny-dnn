//
//  accelerate.c
//  Easy
//
//  Created by pingwei liu on 2019/3/13.
//  Copyright © 2019 pingwei liu. All rights reserved.
//

#include "apple_math.h"

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#if defined(__i386__) || defined(__x86_64__)
#include <x86intrin.h>
#endif
#if defined(_M_ARM64)
#include <arm64intr.h>
#endif

void apple_tanh(element_type *dst, const element_type *src, int count) {
#if defined(CNN_USE_DOUBLE)
    vvtanh(dst, src, &count);
#else
    vvtanhf(dst, src, &count);
#endif
}

void apple_vsq(element_type *dst, const element_type *src, int count) {
#if defined(CNN_USE_DOUBLE)
    vDSP_vsqD(src, 1, dst, 1, count);
#else
    vDSP_vsq(src, 1, dst, 1, count);
#endif
}

void apple_pow(float *dst, const float *src, int count, float scalar) {
    vvpowsf(dst, &scalar, src, &count);
}

void apple_caxpy(float *y, const float *x, float alpha, const int count) {
    cblas_caxpy(count, &alpha, x, 1, y, 1);
}

void apple_copy(float *y, const float *x, const int strideX, const int strideY, const int count) {
    cblas_ccopy(count, x, strideX, y, strideY);
}

void apple_clear(float *y, const int stride, const int count) {
    vDSP_vclr(y, stride, count);
}

void apple_fill(element_type *y, element_type scalar, const int stride, const int count) {
#if defined(CNN_USE_DOUBLE)
    vDSP_vfillD(&scalar, y, stride, count);
#else
    vDSP_vfill(&scalar, y, stride, count);
#endif
}

void apple_max(const float *x, float *scalar, const int stride, const int count) {
    vDSP_maxv(x, stride, scalar, count);
}

void apple_maxi(const float *x, float *scalar, unsigned long *index, const int stride, const int count) {
    vDSP_maxvi(x, stride, scalar, index, count);
}

void apple_min(const float *x, float *scalar, const int stride, const int count) {
    vDSP_minv(x, stride, scalar, count);
}

void apple_mini(const float *x, float *scalar, unsigned long *index, const int stride, const int count) {
    vDSP_minvi(x, stride, scalar, index, count);
}

void apple_saxpby(float *dst, const float alpha, const float *src, const float beta, const int count) {
    catlas_saxpby(count, beta, src, 1, alpha, dst, 1);
}

void apple_vadd(float *z, const float *x, const float *y, const int count) {
    vDSP_vadd(x, 1, y, 1, z, 1, count);
}

void apple_vmul(float *z, const float *x, const float *y, const int count) {
    vDSP_vmul(x, 1, y, 1, z, 1, count);
}

void apple_vsub(float *z, const float *x, const float *y, const int count) {
    vDSP_vsub(x, 1, y, 1, z, 1, count);
}

void apple_vdiv(float *z, const float *x, const float *y, const int count) {
    vDSP_vdiv(x, 1, y, 1, z, 1, count);
}

void apple_vam(float *d, const float *a, const float *b, const float *c, const int count) {
    vDSP_vam(a, 1, b, 1, c, 1, d, 1, count);
}

void apple_vma(float *d, const float *a, const float *b, const float *c, const int count) {
    vDSP_vma(a, 1, b, 1, c, 1, d, 1, count);
}

void apple_vmsb(float *d, const float *a, const float *b, const float *c, const int count) {
    vDSP_vmsb(a, 1, b, 1, c, 1, d, 1, count);
}

void apple_vsbm(float *d, const float *a, const float *b, const float *c, const int count) {
    vDSP_vsbm(a, 1, b, 1, c, 1, d, 1, count);
}

void apple_vasm(float *d, const float *a, const float *b, const float c, const int count) {
    vDSP_vasm(a, 1, b, 1, &c, d, 1, count);
}

void apple_vsbsm(float *d, const float *a, const float *b, const float c, const int count) {
    vDSP_vsbsm(a, 1, b, 1, &c, d, 1, count);
}

void apple_vmsa(float *d, const float *a, const float *b, const float c, const int count) {
    vDSP_vmsa(a, 1, b, 1, &c, d, 1, count);
}

void apple_vmmsb(float *e, const float *a, const float *b, const float *c, const float *d, const int count) {
    vDSP_vmmsb(a, 1, b, 1, c, 1, d, 1, e, 1, count);
}

void apple_vaam(float *e, const float *a, const float *b, const float *c, const float *d, const int count) {
    vDSP_vaam(a, 1, b, 1, c, 1, d, 1, e, 1, count);
}

void apple_vsbsbm(float *e, const float *a, const float *b, const float *c, const float *d, const int count) {
    vDSP_vsbsbm(a, 1, b, 1, c, 1, d, 1, e, 1, count);
}

void apple_vasbm(float *e, const float *a, const float *b, const float *c, const float *d, const int count) {
    vDSP_vasbm(a, 1, b, 1, c, 1, d, 1, e, 1, count);
}

void apple_vsadd(float *c, const float *a, const float b, const int count) {
    vDSP_vsadd(a, 1, &b, c, 1, count);
}

void apple_vsmul(float *c, const float *a, const float b, const int count) {
    vDSP_vsmul(a, 1, &b, c, 1, count);
}

void apple_vsdiv(float *c, const float *a, const float b, const int count) {
    vDSP_vsdiv(a, 1, &b, c, 1, count);
}

void apple_svdiv(float *c, const float a, const float *b, const int count) {
    vDSP_svdiv(&a, b, 1, c, 1, count);
}

void apple_vsma(float *d, const float *a, const float b, const float *c, const int count) {
    vDSP_vsma(a, 1, &b, c, 1, d, 1, count);
}

void apple_vsmsa(float *d, const float *a, const float b, const float c, const int count) {
    vDSP_vsmsa(a, 1, &b, &c, d, 1, count);
}

void apple_vsmsma(float *e, const float *a, const float b, const float *c, const float d, const int count) {
    vDSP_vsmsma(a, 1, &b, c, 1, &d, e, 1, count);
}

void apple_vabs(float *c, const float *a, const int count) {
    vDSP_vabs(a, 1, c, 1, count);
}

void apple_vneg(float *c, const float *a, const int count) {
    vDSP_vneg(a, 1, c, 1, count);
}

void apple_sve(float *c, const float *a, const int count) {
    vDSP_sve(a, 1, c, count);
}

void apple_thres(float *c, const float *a, const float threshold , const int count) {
    vDSP_vthr(a, 1, &threshold, c, 1, count);
}

void apple_limit(float *c, const float *a, const float limit, const float scalar, const int count) {
    vDSP_vlim(a, 1, &limit, &scalar, c, 1, count);
}

void apple_exp(float *c, const float *a, const int count) {
    vvexpf(c, a, &count);
}

void apple_clip(float *c, const float *a, const float min, const float max, const int count) {
    vDSP_vclip(a, 1, &min, &max, c, 1, count);
}

void apple_reverse(float *c, const float *a, const int count) {
    cblas_ccopy(count, a, 1, c, 1);
    vDSP_vrvrs(c, 1, count);
}

void apple_mmov(float *c, const float *a, const int m, const int n, const int ta, const int tc) {
    vDSP_mmov(a, c, m, n, ta, tc);
}

void apple_vsorti(unsigned long *c, const float *a, const int order, const int count) {
    vDSP_vsorti(a, c, NULL, count, order);
}

void apple_meanv(float *scalar, const float *a, const int stride, const int count) {
    vDSP_meanv(a, stride, scalar, count);
}

void apple_measqv(float *scalar, const float *a, const int stride, const int count) {
    vDSP_measqv(a, stride, scalar, count);
}

#pragma mark ----------------

void apple_mmul(float *c, const float *a, const float *b, const int m, const int n, const int p) {
    vDSP_mmul(a, 1, b, 1, c, 1, m, n, p);
//    cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, m, n, p, 1, a, n, b, n, 0, c, n);
}

void apple_mtrans(float *c, const float *a, const int m, const int n) {
    vDSP_mtrans(a, 1, c, 1, m, n);
}

void apple_sgemm(float *c, const float *a, int trans_a, const float *b, int trans_b, const int m, const int n, const int k, const int lda, const int ldb, const int ldc, const float alpha, const int beta) {
    cblas_sgemm(CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void apple_imgfir(float *c, const float *a, const int a_r, const int a_c, const float *f, const int p, const int q) {
    vDSP_imgfir(a, a_r, a_c, f, c, p, q);
}

#pragma mark --- threads

static dispatch_queue_t __queue() {
    static dispatch_queue_t __q = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        dispatch_queue_attr_t attr = dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_CONCURRENT, QOS_CLASS_USER_INTERACTIVE, -1);
        __q = dispatch_queue_create("com.easynet.august", attr);
    });
    
    return __q;
}

void apple_dispatch_apply(size_t iterations,void(^dispatch_block)(size_t idx)) {
    dispatch_apply(iterations, __queue(), dispatch_block);
}

#endif
