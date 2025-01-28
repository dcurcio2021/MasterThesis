#%%import stuff
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from matplotlib import rcParams
from scipy.optimize import root_scalar
from scipy.optimize import minimize_scalar

rcParams['font.family'] = 'sans serif'
rcParams['font.serif'] = ['Helvetica']
#%%load data an prep it
directory='C:\\Users\\domec\\OneDrive - ETH Zurich\\MA\\Simulations&Evaluation'
#directory='C:\\Users\\curcio_d\\OneDrive - ETH Zurich\\MA\\Simulations&Evaluation'
os.chdir(directory)
#load data (column 7 and 8 have °CA and Load force [kN] inside)
Loadcurve_Simon=pd.read_excel('Lastkurve_Masse-_Gas-_Ventilabhebung_Rod Reversal LP250V L-0208.xlsx')
x_values=Loadcurve_Simon['Unnamed: 7'][8:188]*np.pi/180
y_valve_unloading=Loadcurve_Simon['Unnamed: 8'][8:188]
y_valve_unloading_inertia=Loadcurve_Simon['Unnamed: 14'][8:188]
y_valve_unloading_gas=Loadcurve_Simon['Unnamed: 17'][8:188]

y_inertia=np.array(y_valve_unloading_inertia,dtype='float')
y_gas=np.array(y_valve_unloading_gas,dtype='float')


y_values_steppiston=Loadcurve_Simon['Unnamed: 2'][8:188]
y_steppiston=np.array(y_values_steppiston,dtype='float')

y_values_uniform=Loadcurve_Simon['Unnamed: 5'][8:188]
y_uniform=np.array(y_values_uniform,dtype='float')

x = np.array(x_values, dtype='float')
x_deg=x*180/np.pi
#fuer Experimente #2
y = np.array(y_valve_unloading, dtype='float')
#fuer Experiment#1
#y = np.array(y_uniform, dtype='float')

#y = np.array(y_values_steppiston, dtype='float')
#scaling factors for normalization
y_min = np.min(y)
y_max = np.max(y)
y_normalized = (y-y_min/2) / (y_max-y_min)
y_factor_orig=y_max-y_min
y_add_orig=y_min/2
#%% change offset and scaling factor of fit curve
offset=0
y_add=y_add_orig+offset #for a y-axis offset
scale=1
y_factor=y_factor_orig*scale
#%% fitting sinsoidal curve (here up to 4th order)
def sinus_function(x, a0, a1,c1, a2,c2, a3,c3, a4,c4):
    return a0 + a1*np.sin(1*x+c1) + a2*np.sin(2*x+c2) + a3*np.sin(3*x+c3) + a4*np.sin(4*x+c4)
# First derivative
def sinus_function_prime(x, a1, c1, a2, c2, a3, c3, a4, c4):
    return (
        a1 * np.cos(1 * x + c1)
        + 2 * a2 * np.cos(2 * x + c2)
        + 3 * a3 * np.cos(3 * x + c3)
        + 4 * a4 * np.cos(4 * x + c4)
    )

# Second derivative
def sinus_function_double_prime(x, a1, c1, a2, c2, a3, c3, a4, c4):
    return (
        -a1 * np.sin(1 * x + c1)
        - 4 * a2 * np.sin(2 * x + c2)
        - 9 * a3 * np.sin(3 * x + c3)
        - 16 * a4 * np.sin(4 * x + c4)
    )
#initial parameters
p0=[0, 1,0, 0.5,0, 0.25,0, 0.125,0]
#fit
params, params_covariance = curve_fit(sinus_function, x, y_normalized, p0)
params2=np.round(params, decimals=2)
#most closely reconstructed curve
def reconstructed_sinfunc(x):
    return y_factor_orig*sinus_function(x, *params)+y_add_orig
#new curve
def wrapped_sinus_function(x): 
    return y_factor*sinus_function(x, *params)+ y_add
# Rekonstruktion der Kurve
y_reconstructed_normalized=sinus_function(x,*params)
y_reconstructed=y_reconstructed_normalized*y_factor_orig+y_add_orig
y_new=(y_reconstructed_normalized) * y_factor + y_add

y_new_off62_s0_25 = (y_reconstructed_normalized) * (y_factor_orig*0.25) + y_add_orig +62
y_new_off15_s0_5 = (y_reconstructed_normalized) * (y_factor_orig*0.5) + y_add_orig +15
y_new_off_neg35_s0_75=(y_reconstructed_normalized) * (y_factor_orig*0.75) + y_add_orig-35

#Experiment 1
y_new_uni_off60_s0_28=(y_reconstructed_normalized) * (y_factor_orig*0.28) + y_add_orig+60
y_new_uni_off0_s0_55=(y_reconstructed_normalized) * (y_factor_orig*0.55) + y_add_orig
y_new_uni_off_neg60_s0_8=(y_reconstructed_normalized) * (y_factor_orig*0.8) + y_add_orig-60

y_recon_norm_rounded2=sinus_function(x,*params2)
y_recon_rounded2=y_recon_norm_rounded2*y_factor_orig+y_add_orig
y_new_rounded2=y_recon_norm_rounded2*y_factor + y_add
#%% fitting sinsoidal curve (no phase shifts)
def sinus_function_noshift(x, a0, a1):#, a2, a3, a4):
    return a0 + a1*np.sin(1*x)# + a2*np.sin(2*x) + a3*np.sin(3*x) + a4*np.sin(4*x)
#initial parameters
p0_noshift=[0, 1]#, 0.5, 0.25, 0.125]
#fit
params_noshift, params_noshift_covariance = curve_fit(sinus_function_noshift, x, y_normalized, p0_noshift)
#most closely reconstructed curve
def reconstructed_sinfunc_noshift(x):
    return y_factor_orig*sinus_function_noshift(x, *params_noshift)+y_add_orig
#new curve
def wrapped_sinus_function_noshift(x): 
    return y_factor*sinus_function_noshift(x, *params_noshift)+ y_add
# Rekonstruktion der Kurve
y_recon_norm_noshift=sinus_function_noshift(x,*params_noshift)
y_recon_noshift=y_recon_norm_noshift*y_factor_orig+y_add_orig # not close enough
#y_new_noshift = (y_recon_norm_noshift) * y_factor + y_add
#%% for multiple revolutions
x2=np.append(x,x+2*np.pi)
x1_2=x+2*np.pi
x4=np.append(x2,x2+4*np.pi)
x_deg2=np.append(x_deg,x_deg+360)
x_deg4=np.append(x_deg2,x_deg2+720)
y2=np.append(y,y)
y4=np.append(y2,y2)
y_norm2=np.append(y_normalized,y_normalized)
y_reconstructed2=np.append(y_reconstructed,y_reconstructed)
y_reconstructed4=np.append(y_reconstructed2,y_reconstructed2)
#%% Plotten der Original- und rekonstruierten Daten
plt.plot(x_deg, y, label='Original')
#plt.plot(x_deg, y_gas, label='Gas Force')
#plt.plot(x_deg, y_inertia, label='Mass Force')

#plt.plot(x_deg, y, label='Ventilabhebung')
#plt.plot(x_deg, y_uniform, label='Normal Operation', color='r')
#plt.plot(x_deg, y_valve_unloading, label='50% Valve Unloading')
#plt.plot(x_deg, y_steppiston, label='Step-Piston')
#plt.plot(x_deg, y_new, label='New (offset:'+str(offset)+'|scale:'+str(scale)+')', linestyle='--',color='k')
plt.plot(x_deg, y_reconstructed, label='Reconstructed', linestyle='-.')
#plt.plot(x_deg, y_reconstructed_normalized, label='Reconstructed', linestyle='-.')

#Lastkurven fuer Experimente #2 (Exzenterbuchse)
plt.plot(x_deg, y_new_off62_s0_25, label='LC #2.1')
plt.plot(x_deg, y_new_off15_s0_5, label='LC #2.2')
plt.plot(x_deg, y_new_off_neg35_s0_75, label='LC #2.3')

#plt.plot(x_deg, y_recon_noshift,label='no phase shift')

#Lastkurven fuer Experiment mit Buchse Zweistoff mit Langlochkontur und erhoehtem Oeldruck
# plt.plot(x_deg, y_new_uni_off60_s0_28, label='Lastkurve #1.1')
# plt.plot(x_deg, y_new_uni_off0_s0_55, label='Lastkurve #1.2')
# plt.plot(x_deg, y_new_uni_off_neg60_s0_8, label='Lastkurve #1.3')

# fig, ax1 = plt.subplots() 
# ax1.plot(x_deg, 52.36*np.cos(x)*125/720*0.09, label='$v_{rel}$'+' max:'+str(np.round(max(52.36*np.cos(x)*125/720*0.09),2))+'m/s') #for relative sliding velocity
# ax1.set_xlabel('Crank Angle [°]')
# ax1.set_ylabel('$v_{rel}$ [m/s]', color='b')
# ax1.legend()
# ax2=ax1.twinx()
# ax2.plot(x_deg, 1000/2*125/720*np.sin(x), label='$x^{\prime}_{disp}$'+'max:'+str(np.round(max(1000/2*125/720*np.sin(x)),1))+'\u03BCm', color='r')
# ax2.set_ylabel('$x^{\prime}_{disp}$ [\u03BCm]', color='r')
# ax2.legend()
# ax1.grid(1)

plt.xlim(0,360)
plt.xticks([0,90,180,270,360])
plt.xlabel('Crank Angle [°]')
plt.ylabel('Load on Connecting Rod [kN]')
#plt.ylabel('Relative Sliding Velocity [m/s]')

plt.grid(1)
plt.title('Eccentric Displacement vs °CA')
plt.title('Relative Sliding Velocity & Eccentric Displacement vs °CA')
plt.axhline(y=0, linestyle='dashed', color='black')

plt.legend()

#plt.show()
#%% reversal calculation (all from "wrapped_sinus_function")
if np.any(y_new>0):
    #reversal duration [°deg]
    init_guess_deg=[200,280] #guess from plot where zeroes are
    initial_guesses =[init_guess_deg[0]*np.pi/180, init_guess_deg[1]*np.pi/180]
    r = fsolve(wrapped_sinus_function, initial_guesses)
    r_unique = np.unique(np.round(r, decimals=5))
    nullstellen = r_unique[(r_unique >= 0) & (r_unique <= 2 * np.pi)]
    nullstellen=nullstellen*180/np.pi
    r_duration=float(round(max(nullstellen)-min(nullstellen),0))
    #reversal amplitude [%]
    maxmin=np.array([min(wrapped_sinus_function(x)),max(wrapped_sinus_function(x))])
    r_amplitude=float(round((min(np.abs(maxmin))/max(np.abs(maxmin))*100),0))
else:
    r_duration=0
    r_amplitude=0
#%% Final Output (negative force=tension on conrod, 0°=TDC)
n=2 #round on how many digits (2 seems good enough)
print('max Force:'+str(round(max(maxmin),0))+'| min Force:'+str(round(min(maxmin),0)))
print('RR: Amplitude='+str(r_amplitude)+ '| Duration='+str(r_duration))
#Print whole fit function (with 4 orders)
print('y='+str(round(y_factor*params[0]+y_add,n))+'+'+str(round(y_factor*params[1],n))+'*sin(1*x+'+str(round(params[2],n))+') + '+str(round(y_factor*params[3],n))+'*sin(2*x+'+str(round(params[4],n))+')'+'+'+str(round(y_factor*params[5],n))+'*sin(3*x+'+str(round(params[6],n))+') + '+str(round(y_factor*params[7],n))+'*sin(4*x+'+str(round(params[8],n))+')')

#parameter print
# nparams=np.array(params)
# rparams=np.round(nparams,decimals=3)
# print(rparams)
#%% plot comparison curve over multiple revolutions
plt.plot(x_deg, wrapped_sinus_function(x1_2),label='360-720°',linestyle='-')
plt.plot(x_deg, wrapped_sinus_function(x),label='0-360°',linestyle='-.')
plt.legend()
plt.show()

#%%
# Coefficients from your array
a0, a1, c1, a2, c2, a3, c3, a4, c4 = params2

# Function wrapper for second derivative
def second_derivative(x):
    return sinus_function_double_prime(x, a1, c1, a2, c2, a3, c3, a4, c4)

# Find x where second derivative is 0 in a given range
root_result = minimize_scalar(lambda x: abs(second_derivative(x)), bounds=(0, 2 * np.pi), method='bounded')

if root_result.success:
    x_zero = root_result.x
    print(f"x where second derivative is 0: {x_zero}")

    # Evaluate the first derivative at this x
    max_rate_of_change = sinus_function_prime(x_zero, a1, c1, a2, c2, a3, c3, a4, c4)*y_factor_orig
    print(f"Maximum rate of change: {max_rate_of_change}")
else:
    print("Root finding failed.")
#%% for exporting plots as svg to folder ...\plot_svg, has to be inserted before plt.show
os.chdir(directory+'\\plot_svg')
pltname='sliding_velocity_vs_xdisp.svg'
plt.savefig(pltname, format="svg")
#fig.savefig(pltname, format="svg") for subplots

