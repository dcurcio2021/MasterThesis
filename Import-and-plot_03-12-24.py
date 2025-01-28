#%% Initialization
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib import rcParams
from cycler import cycler
directory='C:\\Users\\domec\\OneDrive - ETH Zurich\\MA\\Simulations&Evaluation'  #from private pc
#directory='C:\\Users\\curcio_d\\OneDrive - ETH Zurich\\MA\\Simulations&Evaluation' #from BCAG pc
#%%Figure & Plot Parameters
# rcParams dict
rcParams['axes.labelsize'] = 35
rcParams['axes.titlesize'] = 35
rcParams['xtick.labelsize'] = 24
rcParams['ytick.labelsize'] = 24
rcParams['legend.fontsize'] = 24
rcParams['font.family'] = 'sans serif'
rcParams['font.serif'] = ['Helvetica']
rcParams['figure.titlesize'] = 35
rcParams['figure.figsize'] = 20, 20
rcParams['axes.prop_cycle']=cycler(color='bkycrmg')
xticks=180 #distance between x-axis ticks [°]
legendloc=(0.5,1.075)
#%% Insert folder names here
Folders=[]
# Definition Function Folders extension from jobx until joby, and extend folders and create list jobnrs
def folderext(x,y):
    for i in range(y-x+1):
        Folders.append('job'+str(i+x))
folderext(2,9)
folderext(22,26)
folderext(28,45)
folderext(47,70)
folderext(72,88)
folderext(63,70)
alljobs=[]
for i in range(len(Folders)):
    alljobs.append(int(Folders[i].split('job')[1]))
#%% Get Table which job is which
os.chdir(directory)
DOE_total=pd.read_excel('DOE Table.xlsx')
loadcurve2=np.loadtxt('Load curve 2-RRd15A6.txt')
loadcurve3=np.loadtxt('Load curve 3-RRd15A6.txt')
DOE=DOE_total[DOE_total['Job Nr'].isin(alljobs)]
relevantcols=DOE.columns[:10]
DOE=DOE.loc[:,relevantcols]

#%%Bearings/parts definition
name_bearing_outer='LAGER_AUSSEN' #name of the file
name_bearing_inner='LAGER_MITTE' #name of the file
comp_names=['LagerMitte','LagerAussen','BuchseStange','zapfen','buchse'] #name of the parts in first
#%% Import data
home=directory+'\\Daten First'

Bearings = dict()   #Bearing surfaces only
#Datastructure Bearings: nested dicts: 1st: 'job number', 2nd: 'component name', 3rd: 'variable name' (force, hmin, pmax...)

Components = [] #which bearing surface (Buchse/LagerAussen/LagerMitte/BuchseStange...has to have the same elements as comp_names)

for idx, folder in enumerate(Folders): #loop over jobs
    # Updata Datastructure
    Bearings[folder]=dict()
    Components.append([])
    
    # Switch to datafolder
    os.chdir(os.path.join(home,folder))
    
    # Get all datafiles
    Files = os.listdir()

    # Read the Files
    for filenam in Files:
        if os.path.splitext(filenam)[1]!='.gnu'and os.path.isfile(filenam) and filenam!='.xpost-index' and os.path.splitext(filenam)[1]!='.dat':
            if filenam[:2]=='hy':
                comp = filenam.split('~')[0].split('_')[1]+filenam.split('~')[0].split('_')[2]
                if len(filenam.split('~'))==3 and filenam.split('~')[1]=='kw_flw-sum': #for flow sum distinction pressure side and exit side
                    param=filenam.split('~')[1].split('_')[1]+'-'+filenam.split('~')[2].split('_')[2]
                else:
                    param = filenam.split('~')[1].split('_')[1]
            elif filenam.split('~')[0]=='buchse' and len(filenam.split('~'))==3 and filenam.split('~')[1]=='kw_d1-max':  #for deformation: put values from buchse to where inner and outer bearings are
                if filenam.split('~')[2]==name_bearing_outer:
                    comp=comp_names[1]
                    param=filenam.split('~')[1].split('_')[1]
                elif filenam.split('~')[2]==name_bearing_inner:
                    comp=comp_names[0]
                    param=filenam.split('~')[1].split('_')[1]
                else:       # still deformation: surface between bearing and conrod stays with bearing
                    comp=filenam.split('~')[0]
                    param=filenam.split('~')[1].split('_')[1]+'-'+filenam.split('~')[2]
            else:
                comp=filenam.split('~')[0]
                param = filenam.split('~')[1].split('_')[1]
            if comp not in Components[idx]:
                Components[idx].append(comp)
                Bearings[folder][comp]=dict()
            
            Data = np.loadtxt(filenam)
            Bearings[folder][comp][param]=Data[:,1]        #write whatever parameter values into datastructure "Bearings"
            Bearings[folder][comp]['alpha'] = Data[:,0]    #write °CAs into datastructure "Bearings"
#%% Definition functions misc
#simulations list (which job number correspond to a certain parameter) everything from excel file DOE
def findjobs(df,joblist,parametername,parametervalue):
    selectedrows=df[df['Job Nr'].isin(joblist)]
    jobs=selectedrows[selectedrows[parametername]==parametervalue]['Job Nr'].tolist()
    return jobs
#return similarities and differences of jobs (for title and legend)
def find_differences(df, jobs):
    if len(jobs)==1:
        title='Loadcurve:'+str(df[df['Job Nr']==jobs[0]]['Loadcurve'].iloc[0])+'| Eccentricity:'+str(df[df['Job Nr']==jobs[0]]['Eccentricity'].iloc[0])+'mm | Contour:'+str(df[df['Job Nr']==jobs[0]]['Contour'].iloc[0])+'° | Clearance:'+str(df[df['Job Nr']==jobs[0]]['Clearance A'].iloc[0])+'\u2030'
        labels=['']
    else:
        selected_rows = df[df['Job Nr'].isin(jobs)]
        sames = {}
        differences = {}
        for i in range(len(jobs)): #compare all jobs with each other
            for j in range(i+1,len(jobs)):
                row1 = selected_rows.iloc[i]
                row2 = selected_rows.iloc[j]
                diff = row1 != row2
                same = row1 == row2
                identical_cells = same[same].index.tolist()
                non_identical_cells = diff[diff].index.tolist()
                sames[(jobs[i],jobs[j])] = identical_cells
                differences[(jobs[i], jobs[j])] = non_identical_cells
        sames_common_elements = set.intersection(*map(set, sames.values()))
        elements_to_remove=['Rigid','Duration','Amplitude']
        if 'Loadcurve' in sames_common_elements and df[df['Job Nr']==jobs[0]]['Loadcurve'].iloc[0]!=0:
            elements_to_remove.append('Max Force')
        elif 'Loadcurve' in sames_common_elements and df[df['Job Nr']==jobs[0]]['Loadcurve'].iloc[0]==0:
            elements_to_remove.append('Loadcurve')
        for element in elements_to_remove:
            sames_common_elements.discard(element)
        tit=list(sames_common_elements)
        diff_longest_key = max(differences, key=lambda k: len(differences[k]))
        diff_longest_list = differences[diff_longest_key]
        elements_to_remove_label=['Rigid','Duration','Amplitude']
        diff_longest_list= [element for element in diff_longest_list if element not in elements_to_remove_label]
        leg=diff_longest_list[1:]
        labels=[]
        title=''
        for i in range(len(tit)):
            title=title+tit[i]+':'+str(df[df['Job Nr']==jobs[0]][tit[i]].iloc[0])+'|'
        for k in range(len(jobs)):
             label=''
             if 'Clearance M' in leg and df[df['Job Nr']==jobs[k]]['Clearance A'].iloc[0]==df[df['Job Nr']==jobs[k]]['Clearance M'].iloc[0]:
                 leg.remove('Clearance M')
             for j in range(len(leg)):
                 label=label+leg[j]+':'+str(df[df['Job Nr']==jobs[k]][leg[j]].iloc[0])+'|'
             labels.append(label)
    return title,labels
#create current joblist
def fsim(simdef):
    global simt
    simt=[]
    for i in range(len(simdef)):
        simt.append('job'+str(simdef[i]))
    return simt
#create current variable list
def fvart(var,vdef):
    global vart
    vart=[]
    for i in range(len(vdef)):
        vart.append(var[vdef[i]])
    return vart
#expansion load curve to have more revolutions and matches the 2 deg simulation steps
def lcexpansion(loadcurve):
    loadcurvemod=loadcurve[:,0]
    for i in range(2):
        loadcurvemod=np.append(loadcurvemod,loadcurvemod)
    loadcurvemod=np.delete(loadcurvemod, np.arange(1,loadcurvemod.size,2))
    return loadcurvemod
#sum of forces through aussen and mitte
def fsumfor(s,namefor):
    nametotal=namefor+'-total'
    a=comp_names[1]
    b=comp_names[0]
    Bearings[s][a][nametotal]=-1*(Bearings[s][a][namefor]+Bearings[s][b][namefor])
    Bearings[s][b][nametotal]=-1*(Bearings[s][a][namefor]+Bearings[s][b][namefor])
#calculating the °CA when the bearing "switches" from moving within contour to moving within the clearance (con=contour angle)
def alphastartcontour(con):
    temp=np.arcsin(con/10)*180/np.pi
    global xcuts
    xcuts=[temp,180-temp,180+temp,360-temp]
    return xcuts
#change axs settings for 1 revolution or all revolutions
def all(ax,sim,var,everything):
    global startx
    global xstart
    global xend
    global endx
    if everything==0:
        startx=-180
        xend=Bearings[sim[0]][var[0][3]]['alpha'][-1]
        endx=np.where(Bearings[sim[0]][var[0][3]]['alpha']==xend)[0][0]
        nfirst=int(Bearings[sim[0]][var[0][3]]['alpha'][-1]/360-1)
        xend=xend-360*nfirst
        xstart=Bearings[sim[0]][var[0][3]]['alpha'][startx]
        startx=endx-180
        xstart=xstart-360*nfirst
        ax.set_xticks(np.arange(0, xend + 1, 90))     #x-axis setting ticks
        ax.set_xlim(xstart,xend)      #x-axis set lower and upper limit
    elif everything==1:
        xend=Bearings[sim[0]][var[0][3]]['alpha'][-1]
        endx=-1
        xstart=Bearings[sim[0]][var[0][3]]['alpha'][startx]
        ax.set_xlim(0,xend)      #x-axis set lower and upper limit
        ax.set_xticks(np.arange(0, xend+1, xticks))     #x-axis setting ticks
        startx=0
#%% Definition Plot Functions        
def plotnx1(s,v,b,axs,j,var,label):
    axs[j].set_ylabel(v+' ['+var[j][2]+']')     #y-axis label
    x=Bearings[s][b]['alpha'][startx:endx]-Bearings[s][b]['alpha'][startx]+xstart #x-axis definition
    y=Bearings[s][b][v][startx:endx]/var[j][1]   #y-axis definition and unit conversion
    if maxmin==1 and v=='hmin':
        label=label+' min:'+str(Bearings[s][b]['minhmin'])+'\u03BC'+'m'
    elif maxmin==1 and v=='pmax':
        label=label+' max:'+str(Bearings[s][b]['maxpmax'])+'bar'
    elif maxmin==1 and v=='flw-sum-druck':
        label=label+' mean:'+str(Bearings[s][b]['mean_flw'])+'l/min'
    elif maxmin==1 and v=='flw-sum-rand':
        label=label+' mean:'+str(Bearings[s][b]['mean_flw'])+'l/min'
    elif maxmin==1 and v=='flw-net':
        label=label+' mean:'+str(Bearings[s][b]['mean_flw_net'])+'l/min'
    axs[j].plot(x, y, label=label)
    axs[j].grid(True)
    
def plotnx2(s,v,b,axs,j,i,var,label):
    axs[j,0].set_ylabel(v+' ['+var[j][2]+']')     #y-axis label
    x=Bearings[s][b]['alpha'][startx:endx]-Bearings[s][b]['alpha'][startx]+xstart #x-axis definition
    y=Bearings[s][b][v][startx:endx]/var[j][1]   #y-axis definition and unit conversion
    if maxmin==1 and v=='hmin':
        label=label+' min:'+str(Bearings[s][b]['minhmin'])+'\u03BC'+'m'
    elif maxmin==1 and v=='pmax':
        label=label+' max:'+str(Bearings[s][b]['maxpmax'])+'bar'
    elif maxmin==1 and v=='flw-sum-druck':
        label=label+' mean:'+str(Bearings[s][b]['mean_flw'])+'l/min'
    elif maxmin==1 and v=='flw-sum-rand':
        label=label+' mean:'+str(Bearings[s][b]['mean_flw'])+'l/min'
    elif maxmin==1 and v=='flw-net':
        label=label+' mean:'+str(Bearings[s][b]['mean_flw_net'])+'l/min'
    axs[j,i].plot(x, y, label=label)
    axs[j,i].sharey(axs[j,i-1])
    axs[j,i].grid(True)
    
def plot1x1(s,v,b,axs,j,i,var,label):
    axs.set_ylabel(v+' ['+var[j][2]+']')     #y-axis label
    x=Bearings[s][b]['alpha'][startx:endx]-Bearings[s][b]['alpha'][startx]+xstart #x-axis definition
    y=Bearings[s][b][v][startx:endx]/var[j][1]   #y-axis definition and unit conversion
    if maxmin==1 and v=='hmin':
        label=label+' min:'+str(Bearings[s][b]['minhmin'])+'\u03BC'+'m'
    elif maxmin==1 and v=='pmax':
        label=label+' max:'+str(Bearings[s][b]['maxpmax'])+'bar'
    elif maxmin==1 and v=='flw-sum-druck':
        label=label+' mean:'+str(Bearings[s][b]['mean_flw'])+'l/min'
    elif maxmin==1 and v=='flw-sum-rand':
        label=label+' mean:'+str(Bearings[s][b]['mean_flw'])+'l/min'
    elif maxmin==1 and v=='flw-net':
        label=label+' mean:'+str(Bearings[s][b]['mean_flw_net'])+'l/min'
    axs.plot(x, y, label=label)
    #axs.set_title(b)
    axs.grid(True)

def plot1xn(s,v,b,axs,j,i,var,label):
    axs[i].set_xlabel('°CA')    #x-axis setting label
    axs[0].set_ylabel(v+' ['+var[j][2]+']')     #y-axis label
    x=Bearings[s][b]['alpha'][startx:endx]-Bearings[s][b]['alpha'][startx]+xstart #x-axis definition
    y=Bearings[s][b][v][startx:endx]/var[j][1]   #y-axis definition and unit conversion
    if maxmin==1 and v=='hmin':
        label=label+' min:'+str(Bearings[s][b]['minhmin'])+'\u03BC'+'m'
    elif maxmin==1 and v=='pmax':
        label=label+' max:'+str(Bearings[s][b]['maxpmax'])+'bar'
    elif maxmin==1 and v=='flw-sum-druck':
        label=label+' mean:'+str(Bearings[s][b]['mean_flw'])+'l/min'
    elif maxmin==1 and v=='flw-sum-rand':
        label=label+' mean:'+str(Bearings[s][b]['mean_flw'])+'l/min'
    elif maxmin==1 and v=='flw-net':
        label=label+' mean:'+str(Bearings[s][b]['mean_flw_net'])+'l/min'
    axs[i].plot(x, y, label=label)
    axs[i].sharey(axs[i-1])
    axs[i].set_title(b)
    axs[i].grid(True)
#%% Calculations
for i in range(len(Bearings)):
    #Force total through Aussen and Mitte
    s=Folders[i]
    fsumfor(s,'for1')
    #Force resulting from x- and y-direction
    Bearings[s][comp_names[0]]['for-res']=(Bearings[s][comp_names[0]]['for1']**2+Bearings[s][comp_names[0]]['for2']**2)**(1/2)
    Bearings[s][comp_names[1]]['for-res']=(Bearings[s][comp_names[1]]['for1']**2+Bearings[s][comp_names[1]]['for2']**2)**(1/2)  
    #oil created
    if 'flw-sum-druck' in Bearings[s][comp_names[1]] and 'flw-sum-rand' in Bearings[s][comp_names[1]]:
        Bearings[s][comp_names[1]]['flw-net']=Bearings[s][comp_names[1]]['flw-sum-rand']+Bearings[s][comp_names[1]]['flw-sum-druck']
        Bearings[s][comp_names[0]]['flw-net']=Bearings[s][comp_names[0]]['flw-sum-rand']+Bearings[s][comp_names[0]]['flw-sum-druck']   
    #minimal hmin in micron (only from last revolution)
    Bearings[s][comp_names[0]]['minhmin']=round(np.min(Bearings[s][comp_names[0]]['hmin'][-180:]*1e6),2)
    Bearings[s][comp_names[1]]['minhmin']=round(np.min(Bearings[s][comp_names[1]]['hmin'][-180:]*1e6),2)
    #maximum pmax in bar (only from last revolution)
    Bearings[s][comp_names[0]]['maxpmax']=round(np.max(Bearings[s][comp_names[0]]['pmax'][-180:]*1e-5),0)
    Bearings[s][comp_names[1]]['maxpmax']=round(np.max(Bearings[s][comp_names[1]]['pmax'][-180:]*1e-5),0)
    #mean oil flow over 'rand' in l/min and mean net flow (should be ~0)
    if 'flw-sum-rand' in Bearings[s][comp_names[1]]:
        Bearings[s][comp_names[0]]['mean_flw']=round(np.mean(Bearings[s][comp_names[0]]['flw-sum-rand'][-180:]*60*1e3),1)
        Bearings[s][comp_names[1]]['mean_flw']=round(np.mean(Bearings[s][comp_names[1]]['flw-sum-rand'][-180:]*60*1e3),1)
        Bearings[s][comp_names[1]]['mean_flw_net']=round(np.mean(Bearings[s][comp_names[1]]['flw-net'][-180:]*60*1e3),2)
        Bearings[s][comp_names[0]]['mean_flw_net']=round(np.mean(Bearings[s][comp_names[0]]['flw-net'][-180:]*60*1e3),2)
    #Moment around z-axis from hydrodynamic forces -->check if same as buchse_stange mom3
    Bearings[s][comp_names[0]]['hy_mom']=Bearings[s][comp_names[0]]['for1']*DOE[DOE['Job Nr']==int(s.split('b')[1])]['Eccentricity'].iloc[0]/1000
    #Bearings[s][comp_names[1]]['hy_mom']=Bearings[s][comp_names[1]]['for1'][-180:]*DOE[DOE['Job Nr']==int(s.split('b')[1])]['Eccentricity'].iloc[0]/1000/2 # should be 0 because eccentricity to outer bearing surface=0
    #Bearings[s][comp_names[2]]['mom_res']=Bearings[s][comp_names[0]]['hy_mom']#+Bearings[s][comp_names[1]]['hy_mom']
    #alpha max (°CA max)
    Bearings[s][comp_names[0]]['alphamax']=Bearings[s][comp_names[0]]['alpha'][-1]


#%% induced forces
#forappl=applied force constant        
def forinduc(simdef,forappl):
    forappl=forappl*1e3
    fsim(simdef)
    bearn=[comp_names[1],comp_names[0]]
    namefor='for1-applied'
    for i in range(len(simt)): #loop over folders
        s=simt[i]
        for j in range(len(bearn)):
            b=bearn[j]
            Bearings[s][b]['for1-induced']=Bearings[s][b]['for1'].copy()
            for k in range(len(Bearings[s][b]['for1'])):
                if np.sign(forappl) ==1:
                    if Bearings[s][b]['for1'][k]>forappl: #if actual force bigger than applied force
                        Bearings[s][b]['for1-induced'][k]=Bearings[s][b]['for1'][k].copy()-forappl
                    elif Bearings[s][b]['for1'][k]<forappl and Bearings[s][b]['for1'][k]>0: #if actual force smaller than applied force -> induced force =0
                        Bearings[s][b]['for1-induced'][k]=0
                    elif Bearings[s][b]['for1'][k]<0:
                        Bearings[s][b]['for1-induced'][k]=Bearings[s][b]['for1'][k].copy()
                elif np.sign(forappl)==-1:
                    if Bearings[s][b]['for1'][k]<forappl: #if actual force bigger than applied force
                        Bearings[s][b]['for1-induced'][k]=Bearings[s][b]['for1'][k].copy()-forappl
                    elif Bearings[s][b]['for1'][k]>forappl and Bearings[s][b]['for1'][k]<0: #if actual force smaller than applied force -> induced force =0
                        Bearings[s][b]['for1-induced'][k]=0
                    elif Bearings[s][b]['for1'][k]>0:
                        Bearings[s][b]['for1-induced'][k]=Bearings[s][b]['for1'][k].copy()
            Bearings[s][b][namefor]=Bearings[s][b]['for1']-Bearings[s][b]['for1-induced']
        fsumfor(s,namefor)

#include and subtract induced forces, const load
simdefconst=findjobs(DOE,alljobs,'Loadcurve',0)
#100kN (Compression)
forappltemp=100 #applied force in kN
simdeftemp=findjobs(DOE,simdefconst,'Max Force',-forappltemp)
forinduc(simdeftemp,forappltemp)
#-100kN (Tension)
forappltemp=-100 #applied force in kN
simdeftemp=findjobs(DOE,simdefconst,'Max Force',-forappltemp)
forinduc(simdeftemp,forappltemp)
#-50kN (Tension)
forappltemp=-50 #applied force in kN
simdeftemp=findjobs(DOE,simdefconst,'Max Force',-forappltemp)
forinduc(simdeftemp,forappltemp)
#240kN (Compression)
forappltemp=240 #applied force in kN
simdeftemp=findjobs(DOE,simdefconst,'Max Force',-forappltemp)
forinduc(simdeftemp,forappltemp)
#0kN
forappltemp=0 #applied force in kN
simdeftemp=findjobs(DOE,simdefconst,'Max Force',-forappltemp)
forinduc(simdeftemp,forappltemp)

#induced forces with load curve
def forindulc(simdef,lc):
    lc=lc*-1e3
    fsim(simdef)
    bearn=[comp_names[1],comp_names[0]]
    namefor='for1-applied'
    for i in range(len(simt)): #loop over folders
        s=simt[i]
        for j in range(len(bearn)):
            b=bearn[j]
            Bearings[s][b]['for1-induced']=Bearings[s][b]['for1'].copy()
            for k in range(len(Bearings[s][b]['for1'])):
                if np.sign(lc[k])==1:
                    if Bearings[s][b]['for1'][k]>lc[k]: #if actual force bigger than applied force
                        Bearings[s][b]['for1-induced'][k]=Bearings[s][b]['for1'][k].copy()-lc[k]
                    elif Bearings[s][b]['for1'][k]<lc[k] and Bearings[s][b]['for1'][k]>0: #if actual force smaller than applied force -> induced force =0
                        Bearings[s][b]['for1-induced'][k]=0
                    elif Bearings[s][b]['for1'][k]<0:
                        Bearings[s][b]['for1-induced'][k]=Bearings[s][b]['for1'][k].copy()
                elif np.sign(lc[k])==-1:
                    if Bearings[s][b]['for1'][k]<lc[k]: #if actual force bigger than applied force
                        Bearings[s][b]['for1-induced'][k]=Bearings[s][b]['for1'][k].copy()-lc[k]
                    elif Bearings[s][b]['for1'][k]>lc[k] and Bearings[s][b]['for1'][k]<0: #if actual force smaller than applied force -> induced force =0
                        Bearings[s][b]['for1-induced'][k]=0
                    elif Bearings[s][b]['for1'][k]>0:
                        Bearings[s][b]['for1-induced'][k]=Bearings[s][b]['for1'][k].copy()
            Bearings[s][b][namefor]=Bearings[s][b]['for1']-Bearings[s][b]['for1-induced']
        fsumfor(s,namefor)
    
#induced and applied forces for jobs with load curve 2
simdef2=findjobs(DOE,alljobs,'Loadcurve',2)
simdef2neg=findjobs(DOE,simdef2,'Max Force',-240)
lc=lcexpansion(loadcurve2)
forindulc(simdef2neg,lc)
simdef2pos=findjobs(DOE,simdef2,'Max Force',240)
lc=lcexpansion(loadcurve2)
forindulc(simdef2pos,-lc)

#induced and applied forces for jobs with load curve 3
simdef=findjobs(DOE,alljobs,'Loadcurve',3)
loadcurve3[:,0]=loadcurve3[:,0]*0.5 #lc3's magnitude is half the imported file
lc=lcexpansion(loadcurve3)
forindulc(simdef,lc)

def lc4():
    x = Bearings['job65'][comp_names[1]]['alpha']
    lc4=3e2*(-0.43+0.4*np.sin(x*np.pi/180)+0.1*np.sin(2*x*np.pi/180))
    return lc4

def lc6():
    x = Bearings['job80'][comp_names[1]]['alpha']
    lc6=-173.99-89.038*np.sin(x*np.pi/180+0.45)+84.991*np.sin(2*x*np.pi/180-1.076)-3.305*np.sin(3*x*np.pi/180-0.732)-10.717*np.sin(4*x*np.pi/180-0.48)
    return lc6
def lc7():
    x = Bearings['job82'][comp_names[1]]['alpha']
    lc7=-140.164-70.293*np.sin(x*np.pi/180+0.45)+67.098*np.sin(2*x*np.pi/180-1.076)-2.609*np.sin(3*x*np.pi/180-0.732)-8.461*np.sin(4*x*np.pi/180-0.48)
    return lc7

if findjobs(DOE,alljobs,'Loadcurve',4)!=[]:
    simdef4=findjobs(DOE,alljobs,'Loadcurve',4)
    lc4=lc4()
    forindulc(simdef4,lc4)
if findjobs(DOE,alljobs,'Loadcurve',5)!=[]:
    simdef5=findjobs(DOE,alljobs,'Loadcurve',5)
    lc5=lc4*0.5
    forindulc(simdef5,lc5)
if findjobs(DOE,alljobs,'Loadcurve',6)!=[]:
    simdef6=findjobs(DOE,alljobs,'Loadcurve',6)
    lc6=lc6()
    forindulc(simdef6,-lc6)
if findjobs(DOE,alljobs,'Loadcurve',7)!=[]:
    simdef7=findjobs(DOE,alljobs,'Loadcurve',7)
    lc7=lc7()
    forindulc(simdef7,-lc7)


#%% Plot a x b grid Function definition from dome
def plotaxb(var,bdef,sim, legend, title):
    if legend==[] and title=='':
        labels=[]
        title=''
        title,labels=find_differences(DOE,simdef)
    else:
        labels=legend
        title=title
    if bdef==0: #number of columns
        ncol=2
    else:
        ncol=1     
    fig, axs = plt.subplots(len(var),ncol, sharex=True)    #Plots plotcol x plotrow grid
    if ncol*len(var)>1:
        for ax in axs.flat:         #plot settings for all plots on grid (x-axis limits and ticks)
            all(ax,sim,var,everything)
    elif ncol*len(var)==1:
        all(axs,sim,var,everything)
    fig.suptitle(title)
    #plot
    for e in range(len(sim)):
        s=sim[e]
        for i in range(ncol):          #loop over columns (bearings)
            if len(var)>1:
                if bdef==0: #if a x 2 grid
                    axs[len(var)-1,i].set_xlabel('°CA')    #x-axis setting label
                    for j in range(len(var)):      #loop over rows
                        if len(var[j])>4:    
                            b=var[j][3+i] #bearing component
                            if i==0:
                                b1='Middle Bearing'
                            else:
                                b1='Outer Bearing'                                
                        else:
                            b=var[j][3]
                            b1='Middle Bearing'
                        v=var[j][0]
                        plotnx2(s,v,b,axs,j,i,var,labels[e])
                    axs[0,0].legend(loc='upper center', bbox_to_anchor=(1.1,1.2), ncol=round(np.sqrt(len(sim)*ncol*len(var))))
                    axs[0,i].set_title(b1)
                else:   #if a x 1 grid
                    axs[len(var)-1].set_xlabel('°CA')    #x-axis setting label on bottom plot
                    title='|'
                    for j in range(len(var)):      #loop over rows  
                        v=var[j][0]    
                        if bdef==2:
                            if len(var[j])>4:
                                b=var[j][4]
                                b1='Outer Bearing'
                            else:
                                b=var[j][3]
                                b1='Middle Bearing'
                            plotnx1(s,v,b,axs,j,var,labels[e])
                            title=b1
                        elif bdef==1:
                            b=var[j][3]
                            plotnx1(s,v,b,axs,j,var,labels[e])
                            title=b
                        elif bdef==3:
                            if len(var[j])>4:
                                for k in range(2):
                                    b=var[j][3+k]
                                    if k==0:
                                        b1='Middle Bearing'
                                    else:
                                        b1='Outer Bearing'
                                    plotnx1(s,v,b,axs,j,var,labels[e]+b1)
                                title=''
                            else:
                                b=var[j][3]
                                plotnx1(s,v,b,axs,j,var,labels[e])
                                title=title+b+'|'
                        axs[j].legend(loc='upper right', bbox_to_anchor=(1, 1.2), ncol=round(np.sqrt(len(sim)*len(var))))                
                    axs[0].set_title(title)
                    
            elif ncol*len(var)==1:   #if only one plot    
                axs.set_xlabel('°CA')    #x-axis setting label
                j=0
                v=var[j][0]
                if bdef==3:
                    for a in range(2):
                        b=var[j][3+a]
                        if a==0:
                            b1='Middle Bearing'
                        else:
                            b1='Outer Bearing'
                        plot1x1(s,v,b,axs,j,i,var,labels[e]+b1)
                elif bdef==2 and len(var[j])>4:
                    b=var[j][4]
                    plot1x1(s,v,b,axs,j,i,var,labels[e])
                elif bdef==2 and len(var[j])<4:
                    b=var[j][3]
                    plot1x1(s,v,b,axs,j,i,var,labels[e])
                elif bdef==1:
                    b=var[j][3]
                    plot1x1(s,v,b,axs,j,i,var,labels[e])
                axs.legend(loc='upper center', bbox_to_anchor=legendloc, ncol=round(np.sqrt(len(sim))))
            elif len(var)==1 and ncol>1: #if multiple columns but only 1 row
                j=0
                b=var[j][3+i]                
                v=var[j][0]
                plot1xn(s,v,b,axs,j,i,var,labels[e])
                axs[0].legend(loc='upper center', bbox_to_anchor=(1,1.2), ncol=round(np.sqrt(len(sim))))
    return fig
#%%Define everything
title='Loadcurve:2|Contour:0|Clearance:1.5\u2030|Eccentricity:1|'
simdef=[75,70]  #define which folder (which simulation job which number after 'job')
#(shortest running job always first, exept if everything=1, then longest running first)
#simdef=findjobs(DOE,alljobs,'Loadcurve',2) #find all jobs from DOE table with same entry in column ''
#simdef=findjobs(DOE,findjobs(DOE,alljobs,'Loadcurve',0),'Eccentricity',1) #nested listmaking
#legend=['Zug ', 'Druck ','A','M']
legend=[ 'Tension ', 'Compression ']
#'\u2030' promille 
#'\u00B1' plus minus symbol

#"bearing"/Which part of the model to analyse
bdef=3 #definition of bearing surface(s) to be plotted (columns)
#0 = variable with according bearings outer and inner next to each other
#1 = variable with according bearings but only inner bearing
#2 = variable with according bearings but only outer bearing
#3 = variable with according bearings outer and inner on top of each other
vdef=[0,4] #definition of rows
#Variables              conversion      unit
var=[['hmin'            ,1e-6   ,'\u03BC'+'m', comp_names[0],comp_names[1]],  #0  #comp_names[0]=LagerMitte, [1]=LagerAussen 
     ['pmax'            ,1e5           ,'bar', comp_names[0],comp_names[1]],  #1
     ['for1'            ,1e3            ,'kN', comp_names[0],comp_names[1]],  #2  #only applicable with bears from hydrodynamic calculations
     ['for2'            ,1e3            ,'kN', comp_names[0],comp_names[1]],  #3
     ['for1-total'      ,1e3            ,'kN', comp_names[0],comp_names[1]],  #4  #total force (Aussen+Mitte) in x-direction
     ['for-res'         ,1e3            ,'kN', comp_names[0],comp_names[1]],  #5  #resulting force (sqrt(for1^2+for2^2))
     ['flw-sum-druck'   ,1/6e4       ,'l/min', comp_names[0],comp_names[1]],  #6  #ONLY 'LagerAussen'/'LagerMitte'
     ['flw-sum-rand'    ,1/6e4       ,'l/min', comp_names[0],comp_names[1]],  #7
     ['flw-net'         ,1/6e4       ,'l/min', comp_names[0],comp_names[1]],  #8  #difference flw-rand and flw-druck -> mean_flw_net should be 0 (net flow over 1 revolution=0, only true with cavitation)
     ['phmi1'           ,1               ,'?', comp_names[0],comp_names[1]],  #9  #position hmin in x-direction (conrod direction)  
     ['phmi2'           ,1               ,'?', comp_names[0],comp_names[1]],  #10 #'' in y-direction (in eccenter direction) 
     ['phmi3'           ,1               ,'?', comp_names[0],comp_names[1]],  #11 #'' in z-direction
     ['for1-induced'    ,1e3            ,'kN', comp_names[0],comp_names[1]],  #12 #self induced forces
     ['for1-applied'    ,1e3            ,'kN', comp_names[0],comp_names[1]],  #13
     ['for1-applied-total',1e3          ,'kN', comp_names[0],comp_names[1]],  #14
     ['prei'            ,1e3            ,'kW', comp_names[0],comp_names[1]],  #15
     ['d1-max'          ,1e-6  ,'\u03BC'+'m?', comp_names[0],comp_names[1]],  #16 #max dislocation in x-direction (conrod direction) only time dependance ("rigid dislocation")
     ['hy_mom'          ,1              ,'Nm', comp_names[0]],                #17 Moment resulting from hydrodynamic forces from LagerMitte (for1*ecc)
     ['mom_res'         ,1              ,'Nm', comp_names[2]],                #18 #ONLY 'BuchseStange': Moment resulting from hydrodynamic forces from both Bearing Surfaces added
     ['mom3'            ,1              ,'Nm', comp_names[2]],                #19 #ONLY BuchseStange moment of bearing
     ['rot'            ,np.pi/180,'° Con-rod', comp_names[3]],                #20 #ONLY Zapfen
     ['d1-max-AUGE'     ,1e-6  ,'\u03BC'+'m?', comp_names[4]]]                #21 #max dislocation in x-direction (conrod direction) relative to position at time=0 (deformation+dislocation) (AUGE = name conrod small end model part)
     
everything=0 #plot everything on x-axis or only the last revolution (1/0)
maxmin=1 #put max or min into legend (1/0)
fsim(simdef)
fvart(var,vdef)
#%% Plot
fig=plotaxb(vart,bdef,simt,[],'')   
#%% Plot custom legend
fig=plotaxb(vart,bdef,simt,legend,title)  

#%%exporting plots as svg to folder ...\plot_svg
os.chdir(directory+'\\plot_svg')
pltname='pltLC2_h_tensionvscompression.svg'
fig.savefig(pltname, format="svg")
#%% Plot3d
sim=findjobs(DOE,findjobs(DOE,alljobs,'Contour',0),'Loadcurve',2) #Loadcurve 2 sims w/o contour
fsim(sim)
zaxis='minhmin'
#zaxis='maxpmax'
x=[]
y=[]
z=[]
for i in range(len(simt)):
    x.append(float(DOE[DOE['Job Nr']==sim[i]]['Eccentricity'].iloc[0]))
    y.append(float(DOE[DOE['Job Nr']==sim[i]]['Clearance A'].iloc[0]))
    z.append(np.min([Bearings[simt[i]][comp_names[1]][zaxis],Bearings[simt[i]][comp_names[0]][zaxis]]))
    #z.append(np.max([Bearings[simt[i]][comp_names[1]][zaxis],Bearings[simt[i]][comp_names[0]][zaxis]]))
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
#ax.bar3d(x,y,0,0.05,0.05,z, shade=True)
ax.scatter(x,y,z)
ax.set_xlabel('Eccentricity [mm]')
ax.set_ylabel('Clearance [\u2030]')
ax.set_zlabel(zaxis)
#%% Plot minhmin/maxpmax
for i in range(len(Folders)):
    plt.plot(Folders[i],Bearings[Folders[i]][comp_names[1]]['minhmin'],'*')
#%%plot maxpmax    
for i in range(len(Folders)):
    plt.plot(Folders[i],Bearings[Folders[i]][comp_names[1]]['maxpmax'],'*')

#%% calc
x=Bearings[0][0]['alpha'][-180:]
y=np.sin(x*np.pi/180)*10
y2=y.copy()
y6=y.copy()
alphastartcontour(2)
x2cuts=xcuts
alphastartcontour(6)
x6cuts=xcuts

for i in range(len(Bearings[0][0]['alpha'])):
    if abs(y[i])<2:
        y2[i]=0
    if abs(y[i])<6:
        y6[i]=0

plt.plot(x[-180:],y2[-180:])


