import numpy as np
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
    #VACF from the last iteration
    it_vacf_file  = sys.argv[1]
    #integrated memory kernel from standard CG-MD simulation
    a_file = sys.argv[2]
    #target integrated memory kernel
    fg_file = sys.argv[3]
    #tilde{K}_i (last iteration)
    tilde_file =sys.argv[4]
    #filename for integrated memory kernel for the next iteration
    out_file = sys.argv[5]
   
    mem,dmem = mem_from_vacf(it_vacf_file)
    
    np.savetxt(it_vacf_file+"_imem",mem)
    np.savetxt(it_vacf_file+"_mem",dmem)

    a = np.loadtxt(a_file)[1:,1]
    K_GLE = mem[1:,1]
    K_FG = np.loadtxt(fg_file)[1:,1]
    K_tilde = np.loadtxt(tilde_file)[:len(a)+1:,:]
    K_tilde[:,0]  = np.loadtxt(a_file)[:,0] 
    K_tilde[1:,1] = (K_FG-a)/(K_GLE-a)*K_tilde[1:,1]
    np.savetxt(out_file,K_tilde)

main()
