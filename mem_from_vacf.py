import numpy as np
from scipy import integrate
import sys 

def mem_from_vacf(vacf):
    
    vacf = np.loadtxt(vacf)[:2000]
    memory = np.zeros(vacf.shape)
    memory[:,0] = vacf[:,0]
    dt = vacf[1,0]-vacf[0,0]
    
    for i in range(1,len(vacf)-1):
        memory[i,1]  = (1.-vacf[i,1]/vacf[0,1])/(0.5*dt)
        
        for j in range(1,i):
            memory[i,1] += -2*memory[j,1]*vacf[i-j,1]/vacf[0,1]
        
        temp  = (1.-vacf[i+1,1]/vacf[0,1])/(0.5*dt)
        
        for j in range(1,i+1):
            temp += -2*memory[j,1]*vacf[i+1-j,1]/vacf[0,1]
        
        memory[i,1] = (memory[i-1,1]+3.*memory[i,1]+temp)/5.
    
    memory[len(memory)-1,1] = temp
    dmem = np.zeros_like(memory)
    dmem[:,0] = memory[:,0]
    dmem[:len(dmem)-1,1] = np.diff(memory[:,1])/dt
    return memory, dmem

def main():
    vacf_file  = sys.argv[1]
    mem,dmem = mem_from_vacf(vacf_file)
    np.savetxt(vacf_file+"_imem",mem)
    np.savetxt(vacf_file+"_mem",dmem)

main()
