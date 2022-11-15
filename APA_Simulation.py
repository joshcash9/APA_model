# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 11:41:06 2021

@author: D Tanis
    
"""

import numpy as np
from casadi import *
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, lfilter
from timeit import default_timer


def GenerateForces(amp, dur, lat, time, slope = 0, type='impulse', fc=0):
    #This function generates force timeseries data for the F1 force on the mover
    #amp indicates the amplitude of the force
    #dur indicates the duration in timesteps (applies only to "impulse" types)
    #lat indicates the latency when the force steps on
    #time is the time vector of the simulation
    #slope indicates the ramp-up slope of the force. If zero, the force will be a step-function
    #type indicates whether the force trace is an impulse (ie ON for dur then OFF) or if it a step (ie ON for the remainder of the time)
    #fc indicates the filtering coefficient on the force, if any
    
    force = np.zeros([len(time),1])
    tstep = time[1]-time[0]
    if (type=='impulse'):
        force[int(lat/tstep):int((lat+dur)/tstep)] = amp
    elif (type == 'step'):
        if (slope == 0):
            force[int(lat/tstep):len(force)] = amp
        else:
            endsamp = int(round(amp/(slope*tstep)))
            if (endsamp + lat/tstep > len(force)):
                raise ValueError('slope too low')
            s_temp = np.arange(1,endsamp,1)
            f_temp = s_temp*slope*tstep
            force[int(lat/tstep):(int(lat/tstep)+endsamp-1),0] = f_temp
            force[int(lat/tstep)+endsamp-1:] = amp
            
    if (fc!=0):
        #if fc is not zero, filter the force signal at the given fc:
        fs = 1/tstep
        fc_norm = 2*fc/fs
        b, a = butter(2, fc_norm, btype='low')
        force = np.transpose(lfilter(b, a, np.transpose(force)))
        
    return force


save_loc = 'C:\\Users\\cominnadm\\Desktop\\Raw Data.txt'  #save location for output data


do_optimization = 1

tic = default_timer()

peaks = []
peak_locs = []
starts = []
AUCs = []
points = []

pred_red = "#ff4346"
prey_blue = '#1f77b4'

#data points holds values of parameters to be run
data_points = 10**np.arange(0, 5.2, 0.2)       #Q/R ratio

#if only one value is found in data_points, this flag will tell the code to plot differently
do_loop = 0
if len(data_points)>1:
    do_loop = 1
    
loop_count = -1;
    
for thisinput in data_points:

    loop_count+=1;
    
    #setup parameters:
    tau = 0.04      #filtering coefficient on the F2 control signal
    # tau = thisinput
    tau_f1 = 0.04   #filtering coefficient on the F1 (preset) control signal
    
    #masses:
    m1 = 1      #mover
    m2 = 20     #body
    # m1 = thisinput*m2   #change m1 relative to constant m2
    # m2 = m1/thisinput

    #Spring constants:
    k1 = 10000
    k2 = 200000

    #damping coefficients:
    d1= 0.9*2*np.sqrt(k1*m1)
    d2 = 0.9*2*np.sqrt(k2*m2)
    
    #setup times:
    tstart = -0.15
    tstep = 0.001
    tend = 0.5
    t = np.arange(tstart, tend, tstep)
    
    #plotting limits:
    xlims = [-0.25, 0.25]
    xticks = [-0.25, 0, 0.25]
    
    #setup F1:
    A1 = k1*1;  #amplitude
    D1 = .2;    #duration
    L1 = -tstart
    
    #setup F2: (if explicitly defined)
    A2 = 0;
    D2 = 2;
    L2 = 4;
       
    #generate the F1 control signal:
    F1_control = GenerateForces(A1, D1, L1, t, slope=0, type='step', fc=0)

    #generate the F2 control signal (used only if do_optimization if false):
    F2 = GenerateForces(A1*0, D1, L1, t, 'step') + GenerateForces(A2, D2, L2, t,'impulse') #This is the APA force if optimization is not performed
    
    
    f2 = SX.sym('f2')   #set F2 as a symbolic value, which will get solved by the optimizer
    params = f2 #params is a list of things whose values gets set by the optimizer
    
    #setup states:
    u = SX.sym('u',6)
    u_prime = SX.sym('u_prime',6)  
    
    #states are as follows:
    #   u[0]: F2 force (APA)
    #   u[1]: position of M1 mass
    #   u[2]: position of M2 mass
    #   u[3]: velocity of M1 mass
    #   u[4]: velocity of M2 mass
    #   u[5]: placeholder for F1 force (preset above)
    
    #below are the first derivatives of the above states:
    u_prime[0] = (f2 - u[0])/tau    #F2'
    u_prime[1] = u[3]       #velocity of M1 mass
    u_prime[2] = u[4]       #velocity of M2 mass
    u_prime[3] = -(u[3]*d1-u[4]*d1+u[1]*k1-u[2]*k1-u[5]) / m1     #acceleration of M1 mass
    u_prime[4] = -(u[4]*(d1+d2)-u[3]*d1+u[2]*(k1+k2)-u[1]*k1+u[0]) / m2     #acceleration of M2 mass
    u_prime[5] = 0 #d/dt of f1 (not used)

    
    #starting conditions:
    u_0 = [0, 0, 0, 0, 0, 0]
    
    qr_ratio = 1000     #ratio of Q to R cost coefficients
    # qr_ratio = thisinput
    
    R = 1  #effort cost coefficient
    Q = 1*qr_ratio*k2  #body position cost coefficent

    
    #indicate where the M2 mass should be at all timepoints:
    target = 0
    m2_target = np.zeros(t.shape)
    m2_target[:] = np.nan
    m2_target[:] = 0
    #m2_target[int(1/tstep):] = 0
    
    #cost function given to the optimizer.
    #Note that only the effort cost term is included here, but the accuracy cost term is added later
    costfunc= R*((f2)/(m2/20))**2       #The m2/20 term is used for when the m2 mass is changed. By default, m2 = 20, therefore m2/20 = 1
    
    #setup optimization parameters
    dae = {'x':u, 'p':params, 'ode':u_prime, 'quad':costfunc}
    opts = {'tf':tstep}
    f = integrator('F', 'cvodes', dae, opts)
    
    #setup idx to run:
    idx = np.arange(0, len(t)-1, 1)
    
    #Run a preliminary Euler integration to generate F1 from the F1 control signal 
    f1 = np.zeros([len(idx)+1,1])
    for n in idx:
        f1_prime = (F1_control[n] - f1[n])/tau_f1   #1st order filter on F1 control
        f1[n+1] = f1[n] + tstep*f1_prime
    
    
    if do_optimization:
        opti = Opti()
        
        xf = opti.variable(len(idx)+1,6)
        
        APA = opti.variable(len(F1_control),1)
        
        f1_p = opti.parameter(1,1)  

        
    else:
        xf = np.zeros([len(idx)+1,7])
        APA = F2
    
    #initialize starting state:
    xf[0,:] = u_0
    APA[0] = 0
    
    #recursively run the integration through each index:
    J = 0
    cost_delay = 0
    prevAPAout = 0
    prevAPA = 0
    
    
    for n in idx:
        # U(n+1) = U(n) + tstep * U'(n)
        
        #set F1:
        xf[n,5] = f1[n]
        
        ps = APA[n]
         
        sol = f(x0 = xf[n,:], p = ps)
        
        if do_optimization:
            xf[n+1,:] = sol['xf']
        else:
            xf[n+1,:] = np.transpose(sol['xf'])

        #get cost of effort:
        J += (sol['qf'])
        
        #add in the cost on body position:
        a = 1
        if not np.isnan(m2_target[n]) and n>cost_delay:
            temp = Q*(xf[n+1-cost_delay,2]-m2_target[n])**2
            
            #remove cost if body position is below the target (asymmetric cost)
            # temp = if_else((xf[n+1,2] - m2_target[n]) > 0, temp, 0)
                
            J += a*temp 
    
    if do_optimization:
        opti.minimize(J)
        
        opti.solver('ipopt',{},dict(tol=1e-5,print_frequency_iter=50,max_iter=3000))
        sol2 = opti.solve() # Actually Solving
    
        APA_in = sol2.value(APA)
        xf_out = sol2.value(xf)
        APA_out = xf_out[:,0]
        J_out = sol2.value(J)
        
    else:
        APA_out = APA
        xf_out = xf
        J_out = J
        
    #extract stats (latency and intensity of APA)
    t1 = int(0)
    t2 = int(L1/tstep)
    
    rng = max(APA_in[10:-10]) - 0
    start = list(np.where(APA_in > 0.01*rng))
    startF1 = np.where(f1 > 2)
    
    if start[0].size == 0:
        start[0] = [np.nan]
        
    start2 = (start[0][0] - startF1[0][0])*tstep

    starts.append(start2)
    
    thesenums = APA_out[np.where(APA_out[0:t2] > 0.01*rng)]
    if thesenums.size == 0:
        thisAUC = 0
    else:
        thisAUC = np.cumsum(thesenums)[-1]*tstep
        
    AUCs.append(thisAUC)
    
    points.append(thisinput)
    
    #extract max acc of APA
    if loop_count == 0:
        maxacc = max(np.diff(np.diff(APA_in[0:-100]))/(tstep*tstep));
    
    #save results:
    with open(save_loc,'w') as f:
        f.write('Data\n\nAUCs: ')
        f.write(str(AUCs))
        f.write('\n\nStarts: ')
        f.write(str(starts))
        f.write('\n\nData Points: ')
        f.write(str(data_points))
    
    #Print out results:
    toc = default_timer() - tic
    print('')
    print('Total time elapsed: ',toc/60,' minutes. Average time per loop: ', toc/(60*len(AUCs)) ,' minutes')
    print('Current loop:', thisinput)
    print('AUCs:',  AUCs)
    print('Starts:', starts)
    print('Data points:', points)
    print('')
    print('--------------------------------------------------------------')
    print('')
    
    #Plot individual loop results:
    plot_during = True;
    if plot_during and do_loop:
        plot_to = int((tend-0.25-tstart)/tstep); 
        fig1 = plt.figure();
        ax1 = plt.gca();
        ax1.plot(t[0:plot_to],xf_out[0:plot_to,1]/1.05,prey_blue, linewidth=2)
        ax1.set_ylim([0, 5]);
        
        ax1.tick_params(axis='y',color = prey_blue, labelcolor = prey_blue, labelsize = 12)
        ax3 = ax1.twinx();
        ax3.plot(t[0:plot_to],xf_out[0:plot_to,2]/0.05,pred_red, linewidth=2)
        ax3.set_ylim([0,2.5]);
        
        fig2 = plt.figure()
        ax2 = plt.gca();
        ax2.plot(t[0:plot_to],F1_control[0:plot_to], prey_blue, linewidth=2)
        ax2.plot(t[0:plot_to],APA_in[0:plot_to],pred_red, linewidth=2)
        ax2.set_ylim([0, 50000]);
        
    
pred_red = "#ff4346"
prey_blue = '#1f77b4'


if not do_loop:
    #if only one data_point was run, plot the result of that simulation
    
    thisfigsize = [4,1.5];
    thislabelsize = 16;
    
    fig1 = plt.figure(dpi = 100, figsize=thisfigsize);
    
    ax1 = plt.gca();
    
    #new fig size should be 2 by 0.75
    
    fig1.patch.set_facecolor('none')
    ax1.set_facecolor('none')
    
    
    
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)
    
            
    plot_to = int((tend-0.25-tstart)/tstep);   
    
    plt.subplots_adjust(left=0.1,right=0.85, top = 0.9, bottom = 0.2);
    ax1.plot(t[0:plot_to],xf_out[0:plot_to,1]/1.05,prey_blue, linewidth=2)
    
    ax1.tick_params(axis='y',color = prey_blue, labelcolor = prey_blue, labelsize = thislabelsize)
    ax3 = ax1.twinx();
    ax3.set_facecolor('none')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_linewidth(2)
    ax3.tick_params(axis='y',color = pred_red, labelcolor = pred_red, labelsize = thislabelsize)
    ax1.tick_params(axis='x',labelsize = thislabelsize)
    ax3.tick_params(axis='x',labelsize = thislabelsize)

    ax3.plot(t[0:plot_to],xf_out[0:plot_to,2]/0.05,pred_red, linewidth=2)
    ax3.set_yticks(np.round([0, 1],2))
    # ax3.set_ylim([0, 0.1])
    ylim = ax1.get_ylim()
    # ax3.set_ylim(np.array(ylim)/200)
    ax1.plot([0,0],ax1.get_ylim(),'black',linestyle='dotted')
    # axs[0].plot(t[0:plot_to],m2_target[0:plot_to],"red",linestyle="dotted")
    # axs[0].legend(["Mover (m1) position","Body (m2) position"])
    ax1.set_ylim(ylim)
    ax1.set_yticks(np.round([0, 1],1))
    ax1.set_xlim(xlims)
    ax1.set_xticks(np.round(xticks,2))
    
    # fig1.tight_layout();

    # axs[0].grid(True)
    # axs[0].set_title(f"Cost = {J_out}")
    
    fig2 = plt.figure(dpi = 100, figsize=thisfigsize);
    fig2.patch.set_facecolor('none')
    
    ax2 = plt.gca();
    ax2.set_facecolor('none')
    
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_linewidth(2)
    ax2.spines['left'].set_linewidth(2)
    
    ax2.ticklabel_format(axis='y',style='sci', scilimits=(0,0));
    plt.subplots_adjust(left=0.1,right=0.85, top = 0.9, bottom = 0.2);
    
    ax2.plot(t[0:plot_to],F1_control[0:plot_to], prey_blue, linewidth=2)
    # axs[1].plot(t[0:plot_to]-L1,-APA_out[0:plot_to],"red")  #force signal
    ax2.plot(t[0:plot_to],APA_in[0:plot_to],pred_red, linewidth=2) #control signal

    # ax2.set_ylim([-100, 2000])
    ylim = ax2.get_ylim()
    ax2.plot([0,0],ax2.get_ylim(),'black',linestyle='dotted')
    ax2.set_ylim(ylim)
    ax2.set_yticks(np.round([0, 1e4],1))
    ax2.set_xticks(np.round([-.5, 0, 0.5, 1],1))
    ax2.tick_params(axis='y',labelsize = thislabelsize)
    ax2.tick_params(axis='x',labelsize = thislabelsize)
    
    ax2.set_xlim(xlims)
    ax2.set_xticks(np.round(xticks,2))
    
    # fig2.tight_layout();

if do_loop:
    #if multiple data points were run, plot both latency and intensity of APA across loops:
    fig = plt.figure(dpi = 100, figsize = [4,3])
    # fig.patch.set_facecolor('none')
    axs = plt.gca();
    axs.set_facecolor('none')
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.spines['bottom'].set_linewidth(2)
    axs.spines['left'].set_linewidth(2)
    axs.plot(data_points, np.multiply(starts,1000), linewidth = 2, color='black')
    axs.tick_params(axis='both',labelsize = 16)
    # axs.plot(data_points, peak_locs,'ro-')
    # axs.legend(["APA onset latency","APA peak latency"])
    # plt.title('F1 Onset latency')
    # plt.ylabel('Latency (sec)')
    # plt.xlabel('Q (m2 position) to R (effort) ratio')
    # plt.xlabel('M1/M2 ratio')
    #plt.xlabel('Q/R ratio')
    # plt.xlabel('k1 value')
    # plt.xlabel('Maximum permitted F1')
    # axs.set_xscale('log')
    plt.ylim([-200, 100])
    axs.set_yticks(np.round([-200, -100, 0],0))
    fig.tight_layout();
    
    # plt.figure()
    # plt.plot(data_points, peaks)
    # axs = plt.gca()
    # axs.set_facecolor('none')
    # axs.spines['right'].set_visible(False)
    # axs.spines['top'].set_visible(False)
    # axs.spines['bottom'].set_linewidth(2)
    # axs.spines['left'].set_linewidth(2)
    # plt.title('APA Peak Force')
    # plt.ylabel('Force (N)')
    # #plt.xlabel('Q (m2 position) to R (effort) ratio')
    # #plt.xlabel('Q/R ratio')
    # # plt.ylim([-3, 120])
    # #plt.xlabel('k2 value')
    # #axs.set_xscale('log')
    # plt.xlabel('Maximum permitted F1')
    # #axs.legend()
    
    fig2 = plt.figure(dpi=100, figsize = [4,3])
    axs2 = plt.gca();
    axs2.set_facecolor('none')
    axs2.spines['right'].set_visible(False)
    axs2.spines['top'].set_visible(False)
    axs2.spines['bottom'].set_linewidth(2)
    axs2.spines['left'].set_linewidth(2)
    axs2.plot(data_points, AUCs, linewidth = 2, color='black')
    axs2.tick_params(axis='both',labelsize = 16)
    # plt.title('APA Area Under Curve')
    # plt.ylabel('Area')
    # plt.xlabel('Q (m2 position) to R (effort) ratio')
    # plt.xlabel('M1/M2 ratio')
    # plt.xlabel('k1 value')
    # plt.xlabel('Maximum permitted F1')
    # axs2.set_xscale('log')
    fig.tight_layout();
    
    
    
plt.show()
    
