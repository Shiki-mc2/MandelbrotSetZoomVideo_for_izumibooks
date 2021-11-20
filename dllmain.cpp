#include "CalMandelbrotSet.h"
#include <string>
#include <stdio.h>
#include <mpirxx.h>
#include <omp.h> 

void CalMandelbrotSet(int row, int col, int prec, char* c_xmin, char* c_ymax,
    char* c_dpp, int N, float R, int n_threads, int nmg[]) {

    mpf_class xmin(c_xmin, prec);
    mpf_class ymax(c_ymax, prec);
    mpf_class dpp(c_dpp, prec);
    float R2 = R * R;

    printf("num thread = %d ", n_threads);
    printf("(max = %d)\n", omp_get_max_threads());
    
    #pragma omp parallel for  num_threads(n_threads)
    for (int i = 0; i < row; i++) {
        mpf_class c_imag(0.0, prec);
        mpf_class c_real(0.0, prec);
        mpf_class z_real(0.0, prec);
        mpf_class z_imag(0.0, prec);
        mpf_class z_real_tmp(0.0, prec);

        c_imag = ymax - dpp * i;
        for (int j = 0; j < col; j++) {
            z_real = 0.0;
            z_imag = 0.0;
            z_real_tmp = 0.0;
            c_real = xmin + dpp * j;
            int k = 0;
            for (k = 0; k < N; k++) {
                z_real_tmp = z_real * z_real - z_imag * z_imag + c_real;
                z_imag = 2 * z_real * z_imag + c_imag;
                z_real = z_real_tmp;
                if (z_real * z_real + z_imag * z_imag > R2) {
                    break;
                }
            }
            nmg[col * i + j] = k;
        }
    }
}