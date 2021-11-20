# -*- coding: utf-8 -*-
import numpy as np
from   matplotlib import cm
import cv2
from   mpmath import mp
from numba import jit
from numba import uint8, uint16, uint32, int32
from ctypes import cdll, c_int, c_char_p, c_float, c_void_p, byref

def CalMandelbrotSet_mpf(row, col, x, y, r, N, R, colorlist, n_threads):
    dll = cdll.LoadLibrary("./CalMandelbrotSet.dll")
    CalMandelbrotSet = dll.CalMandelbrotSet
    CalMandelbrotSet.argtypes = [c_int,    c_int,    c_int,
                                 c_char_p, c_char_p, c_char_p,
                                 c_int,    c_float,  c_int,
                                 c_void_p]

    nmg  = np.zeros((row * col),dtype=np.int32)
    xmin = x - r * (col-1)/(row-1)
    ymax = y + r
    dpp  = 2*r/(row-1)
    
    prec = int(max(mp.log(1/dpp)/mp.log(2),49) + 3)
    print("prec=", prec)
    
    nmg = np.ctypeslib.as_ctypes(nmg)
    CalMandelbrotSet(row, col, prec,
                  c_char_p(("%s" % xmin).encode('utf-8')),
                  c_char_p(("%s" % ymax).encode('utf-8')),
                  c_char_p(("%s" % dpp ).encode('utf-8')),
                  N, R, n_threads, byref(nmg))
    nmg = np.ctypeslib.as_array(nmg)
    nmg = nmg.reshape(row, col)
    img = PlotMandelbrotSet(row, col, N, nmg, colorlist)
    return img, nmg

@jit((uint16, uint16, uint32, int32[:,:], uint8[:,:]))
def PlotMandelbrotSet(row, col, N, nmg, colorlist):
    img = np.zeros((row,col,3),dtype=np.uint8)
    
    for i in range(row):
        for j in range(col):
            if nmg[i,j] < N:
                img[i,j,:] = colorlist[nmg[i,j]]
    return img

def main():
    row    = 540
    col    = 720
    
    mp.dps = 30
    x      = mp.mpf("-1.26222162762384535")
    y      = mp.mpf("-0.04591700163513884")
    r      = mp.mpf("0.000000000000005")
    
    N      = 1000
    R      = 2.0
    
    n_threads = 2
    
    colorlist = np.zeros((N,3),dtype=np.uint8)
    for i in range(N):
        color = cm.jet(i/64 % 1)
        for j in range(3):
            colorlist[i,j] = int(color[j]*255)

    img, nmg = CalMandelbrotSet_mpf(row, col, x, y, r, N, R, colorlist,
                                   n_threads)

    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    cv2.imwrite("MandelbrotSet_MultiplePrecision2.png",img)
    print("r=%.5e n min=%5d n max=%5d" % (r, np.min(nmg), np.max(nmg)))

if __name__ == "__main__":
    main()
