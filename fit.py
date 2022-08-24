import numpy as np
import os
from scipy.optimize import curve_fit
import sys

def wrapper(x,n_osc, *args):
    params = list(args)
    return fitfun(x,params,n_osc)   



def fitfun(x,params,n_osc):
    out = np.zeros(len(x))
    for i in range(n_osc):
        a = params[i*4]
        e = params[i*4+1]
        f = params[i*4+2]
        d = params[i*4+3]
            
        c = a*f/2./d -a*e/d/2.
        b = 2.*e + 2*c*d/a
     
        out += -2.*np.exp(-0.5*a*x)*(np.sin(d*x)*(a*c-2.*b*d)+np.cos(d*x)*(a*b+2.*c*d))/(a**2. + 4.* d**2.) +2.*((a*b+2.*c*d))/(a**2. + 4.* d**2.)
   
    return out

def dfun(x,params,n_osc):
    out = np.zeros(len(x))
    for i in range(n_osc):
        a = params[i*4]
        e = params[i*4+1]
        f = params[i*4+2]
        d = params[i*4+3]
            
        c = a*f/2./d - a*e/d/2.
        b = 2.*e + 2*c*d/a
        
        out += np.exp(-0.5*a*x)*(b*np.cos(d*x)+c*np.sin(d*x))
    return out


def fit(infile,  folder, n_osc,final, name, col):
    memory = np.loadtxt(infile)
    tableout = np.zeros([len(memory[:,0]),2])
    params = []
    lbounds = []
    ubounds = []
    try:
        os.mkdir(folder)
    except:
        pass
    
    #these are the initial guesses and bounds for the fitting procedure which work for the given example. They might have to be changed from case to case 
    for i in range(n_osc):
        params.append(0.1/(np.sqrt(float(i))*6.0+1.))
        params.append(0.00003/n_osc)
        params.append(0.00003/n_osc)
        params.append(0.01/(4*i+1))
        lbounds +=[0.005,0.000001,0.000001,0.00000001]
        ubounds +=[1.0,50000.0,50000.0,1.]
    
    bounds = (lbounds,ubounds)
    x = memory[:,0]
    params0 = params
    np.savetxt(folder+"/starting_K_params", params)
    x = memory[:,0]
    params, error = curve_fit(lambda x, *params:wrapper(x,n_osc,*params),x[:final],memory[:final,col],p0=params,maxfev = 80000,bounds = bounds)# one cane increase the fiting accuracy by setting ftol and xtol
    
    #we assume memory[0,0] = 0.0
    x = memory[:,0]
    x = np.linspace(0,x[1]*len(x)*5,num=len(x)*5+1)
    tableout = np.zeros((len(x),2))
    tableout[:,0] = x
    tableout[:,1] = fitfun(x,params,n_osc)
    np.savetxt(folder+"/"+name,tableout,fmt ="%6.6e")
    tableout[:,1] = dfun(x,params,n_osc)
    np.savetxt(folder+"/d"+name,tableout,fmt ="%6.6e")
    for i in range(n_osc):
        tableout[:,1] = fitfun(x,params[i*4:i*4+4],1)
        np.savetxt(folder+"/"+name+"_"+str(i),tableout,fmt ="%6.6e")
    
    np.savetxt(folder+"/"+name+"_K_params", params)

#derive parameters for GLE thermostat matrix from fitting parameters for a single dampened oscillator
def gle_param(a, e, f,d):
    c = a*f/2./d -a*e/d/2.
    b = 2.*e + 2*c*d/a
    ao = np.sqrt(0.5*b-c*d/a)
    bo =  np.sqrt(0.5*b+c*d/a)
    co = a
    do = 0.5*np.sqrt(4*d**2.+a**2.)
    
    #representation of one dampened oscillator the the GLE thermostat matrix
    #A = np.array([ [0.0, ao, bo],
    #               [-ao, co, do ],
    #               [-bo, -do, 0.0]])
    return (ao,bo,co,do)    

def write_Amatrix(folder,n_osc):
    params = np.loadtxt(folder+"/fit_K_params")
    n_exp = int((len(params)-4*n_osc)/2)
    n = int(len(params)/4)
    fullA = np.zeros((n_osc*2+n_exp+1,n_osc*2+n_exp+1))
    fsum = np.sum(params[2::4])    
    for i in range(n_osc):
        a,b,c,d = gle_param(params[i*4],params[i*4+1],params[i*4+2],params[i*4+3])
        fullA[0,1+i*2] = a
        fullA[0,2+i*2] = b
        fullA[1+i*2,0] = -a
        fullA[2+i*2,0] = -b
        fullA[1+i*2,1+i*2] = c
        fullA[1+i*2,2+i*2] = d
        fullA[2+i*2,1+i*2] = -d
    
    for i in range(n_exp):
        a = params[n_osc*4+i*2]
        b = params[n_osc*4+i*2+1]
        fullA[0,1+n_osc*2+i] = -np.sqrt(a/b)
        fullA[1+n_osc*2+i,0] = np.sqrt(a/b)
        fullA[1+n_osc*2+i,1+n_osc*2+i] = 1./b
        
        fullA
    np.savetxt(folder+"/Amatrix",fullA)


def main():
    name = sys.argv[1]
    n_osc = int(sys.argv[2])
    nsteps = int(sys.argv[3])
    
    fit(name , name+"_fit", n_osc,nsteps, "fit", 1)
    write_Amatrix(name+"_fit", n_osc)

main()   

