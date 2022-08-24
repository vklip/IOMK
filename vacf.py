#!/usr/bin/env python3
import sys
import numpy as np
from scipy.fftpack import fft, ifft,ifftshift,fftshift
import h5py as h5


def readtrj(trjfile):
    hf = h5.File(trjfile,"r")
    vel  = hf["particles"]["all"]["velocity"]["value"][:,:,:]
    time = hf["particles"]["all"]["velocity"]["time"][:]
    
    return   vel, time

    
def correlate(x,y):
    xp = ifftshift((x))
    yp = ifftshift((y))
    n, = xp.shape
    xp = np.r_[xp[:n//2], xp[n//2:]]
    yp = np.r_[yp[:n//2],  yp[n//2:]]
    fx = fft(xp)
    fy = fft(yp)
    p = fx*np.conj(fy)
    pi = ifft(p)
    return np.real(pi)[:n//2]/(np.arange(n//2)[::-1]+n//2)

def autocorrelation(x):
    xp = ifftshift((x))
    n, = xp.shape
    xp = np.r_[xp[:n//2], xp[n//2:]]
    f = fft(xp)
    p = np.absolute(f)**2
    pi = ifft(p)
    return np.real(pi)[:n//2]/(np.arange(n//2)[::-1]+n//2)

    
def main(*args):
    trjfile = sys.argv[1]
    out_file = sys.argv[2]
        
    vel,time = readtrj(trjfile)
    natoms = len(vel[0,:,0])
    vacf = np.zeros(len(autocorrelation(vel[:,0,0])))   
    dt = time[1]-time[0]
    for i in range(len(time)):
        time[i] = i*dt
    for i in range(natoms):
        vacf += autocorrelation(vel[:,i,0])
        vacf += autocorrelation(vel[:,i,1])
        vacf += autocorrelation(vel[:,i,2])
            

    vacf = vacf/natoms/3
    vacf = np.array([time[:len(vacf)],vacf])
    vacf = vacf.T
    
    np.savetxt(out_file+"vacf",vacf,fmt ="%6.6e")
    
main()        
      
        
      
      
        
