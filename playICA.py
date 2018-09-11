import os,sys
root=os.getcwd()+'/'+os.pardir+'/'
azpath=root+'Aztica/'
sys.path.append(azpath)
from aztica import *
import warnings
#warnings.simplefilter("error",UserWarning)
np.set_printoptions(3,suppress=True)

length_per_pix=3                       # the arcsec extent of a pixel
beam_FWHM=int( (30./length_per_pix) )  # the beam extent in pixels
fwhm_eff=np.sqrt(2)*beam_FWHM          # the effect of double smoothing with 30 arcsec
beams_excluded = 2.                    # number of beams excluded around a pixel-source
apix=(length_per_pix*(1./60))**2       # area of a pixel in arcmin^2
#
datadir=root+'data/'
plottodir='outputs/'
saveflag=False
#
m0name=datadir+'M0/pca25sigma.nc'
m=Azmap(m0name)
m0s=1e3*m.fSignal
m0w=1e-3*np.sqrt(m.fWeight)
imshape=m0s.shape
imsize=m0s.size
#hitm is the hit-map, or the map of coverture
hitm=m0w**2/np.max(m0w**2)
hitm += 1e-14 #this is just to avoid division by zero


mask380,idx380,l380=makeWmask(hitm,area=380,apix=apix,l_init=0.19)
mask270,idx270,l270=makeWmask(hitm,area=270,apix=apix,l_init=0.42)
m0s -= np.median(m0s[mask270])
m0_counts=countBrightSources(m0s*m0w,beams_excluded,beam_FWHM,mask270,printflag=True,SNtol=4)

cm=ContourMap(hitmap=hitm,xytitle=(0.5,0.96),plottodir=plottodir,filetype='png',
    pixsize=3,xlim=(-585,585),ylim=(-695,695),rotangle=20,xyoffset=(30,-25),
    levels=[l380,l270],level_labels=["380","270"])
cm.plot(m0s*mask380,m0_counts['locations'],
        figname="real_M0_S",figtitle=r"GOODS-S $M_0$",
        clearcanv=True,saveflag=saveflag,vmin=0,vmax=6.5)


redundant_dir=datadir+'/tch120/'
map_names=get_filepaths(redundant_dir)
M_s=[];M_w=[]
for mname in map_names:
    m=Azmap(mname)
    ms=1e3*m.fSignal
    mw=1e-3*np.sqrt(m.fWeight)
    M_s.append(ms)
    M_w.append(mw+1e-13)
M_s=[m-np.mean(m[mask270]) for m in M_s]
Nb=len(M_s)
nm=range(1,Nb+1)


areica=380
masika,idxica,lk=makeWmask(hitm,areica,apix,l_init=1.2)
n=4; maxit=1000; tol=1e-14
w_init=np.identity(n)
try:
    S,A,U=aztica(M_s,n,masika,max_iter=maxit,tol=tol,w_init=w_init)
except UserWarning:
    raise UserWarning("fallo fastica")
S=S[[0,1,3,2]]; A=A.T[[0,1,3,2]].T; U=U[[0,1,3,2]]
print J(S,mask270)


# FIT-wise CALIBRATION steps
# residual2data() is the target function to be minimized, it makes
# a simultaneous fit of n IC to the subset rangeM of redundant maps M_s
# Choose wisely rangeM for each S_j. I choose the largest correlated maps
# but I also check the hessian error size

# From a fit to redundant maps:
# find the scales of S[0]:=S1, the point source component
# and S[1]:=S2, the confusion background
rangeM_S1=[4,5,6,7,8]
x0=[0,1,0,1,0,1,0,1] #init guess: (mean,std) for each IC
calib4S1=op.minimize(residual2data,x0,args=(S,M_s,rangeM_S1,mask380),tol=5e-5)
if not calib4S1.success:
    raise AssertionError("S2 could not be calibrated")
a0_fit,a0std_fit=calib4S1.x[1],np.sqrt(calib4S1.hess_inv[1,1])
a1,a1std=calib4S1.x[3],np.sqrt(calib4S1.hess_inv[3,3])

rangeM_S2=[1,2]
x0=[0,1,0,1,0,1,0,1]
calib4S2=op.minimize(residual2data,x0,args=(S,M_s,rangeM_S2,mask380),tol=1e-5)
a2,a2std=calib4S2.x[5],np.sqrt(calib4S2.hess_inv[5,5])
S[2] *= a2; A.T[2] /= a2; U[2] *= a2

rangeM_S3=[0]
calib4S3=op.minimize(residual2data,x0,args=(S,M_s,rangeM_S3,mask380),tol=0.1)
a3,a3std=calib4S3.x[7],np.sqrt(calib4S3.hess_inv[7,7])
S[3] *= a3; A.T[3] /= a3; U[3] *= a3



# Calibrate S[0]:=S1, the point-like map.
# a0 and a0std are the result of a previous witness (brutal force) step,
# which yielded the scale factor and uncertainty
a0=1.221;a0std=0.0536
S[0] -= np.mean(S[0][masika])
S[0] *= a0; A.T[0] /= a0; U[0] *= a0
S[0] -= np.median(S[0][mask270])
Sstd=np.std(S[0][mask270])
Sstd_err=Sstd*a0std
# Create the W0 weight map and count sources on point source component
W=makeWeightMap(Sstd,M_s[5:],M_w[5:],mask270,Sstd_err,nPars=4)
sn_ica=S[0]*W
ica_counts=countBrightSources(sn_ica,beams_excluded,beam_FWHM,mask270,printflag=True,SNtol=4)

S[1] *= a1; A.T[1] /= a1; U[1] *= a1
s1std=np.std(S[1][mask270])
s1std_err=s1std*a1std
w1=makeWeightMap(s1std,M_s[5:],M_w[5:],mask270,s1std_err,nPars=4)
s1_counts=countBrightSources(S[1]*w1,beams_excluded,beam_FWHM,mask270,printflag=True,SNtol=3.5)

# Now plot'em out
cm.plot(S[0]*mask380,ica_counts['locations'],
        figname="real_S1",figtitle=r"GOODS-S  $S_1$",
        plot_circles=True,plot_numbers=True,
        clearcanv=True,saveflag=saveflag,vmin=0,vmax=6.5)

cm.plot(S[1]*mask380,s1_counts['locations'],
        figname="real_S2",figtitle=r"GOODS-S  $S_2$",
        plot_circles=True,plot_numbers=True,
        clearcanv=True,saveflag=saveflag,vmin=-3,vmax=3)

v=15
cm.plot(S[2]*mask380,
        figname="real_S3",figtitle=r"GOODS-S  $S_3$",
        plot_circles=True,plot_numbers=True,
        clearcanv=True,saveflag=saveflag,vmin=-v,vmax=v)
v=30
cm.plot(S[3]*mask380,
        figname="real_S4",figtitle=r"GOODS-S  $S_4$",
        plot_circles=True,plot_numbers=True,
        clearcanv=True,saveflag=saveflag,vmin=-v,vmax=v)


legends={'loc':'best','markerfirst':False,'frameon':True,'fancybox':True,
    'ncol':2,'columnspacing':1,'shadow':True,'framealpha':0.85,'fontsize':12}
plotSeq(nm,A.T[0],
        figname="real_mix",clearcanv=True,label=r"$A_1$",figsize=(6.5,4),
        c='royalblue',ls='-',lw=4,alpha=0.7)
plotSeq(nm,A.T[1],
        figname="real_mix",clearcanv=False,label=r"$A_2$",
        c='darkorange',ls='-',lw=2)
plotSeq(nm,A.T[2],
        figname="real_mix",clearcanv=False,label=r"$A_3$",
        c='m',ls='--',lw=2)
plotSeq(nm,A.T[3],
        figname="real_mix",figtitle="GOODS-S mixing matrix",label=r"$A_4$",
        c='darkgreen',ls='-.',lw=2,legends=legends,
        xstart=0,xend=110,xmin=0,xmax=40,xstep=4,
        ystart=-2,yend=2,ystep=0.5,ymin=-0.9,ymax=1.9,
        xscale='linear',yscale='linear',xlabel="Map number",ylabel=None,
        clearcanv=False,plottodir=plottodir,saveflag=saveflag,filetype='png')

