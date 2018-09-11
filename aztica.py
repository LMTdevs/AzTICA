#!/opt/local/bin/python2.7
import os,sys
import numpy as np
import random
import scipy.optimize as op
from scipy import ndimage as nd
import matplotlib.pyplot as plt
from matplotlib import mlab
from matplotlib import transforms as tr
from netCDF4 import Dataset as NetCDFFile
from sklearn.decomposition import FastICA
from scipy import interpolate


class Azmap(object):
    """This is a short version of AztecMap."""
    def __init__(self,filename):
        self.filename=filename
        try:
            self.ncf=NetCDFFile(filename)
            self.readmaps(self.ncf)
        except IOError:
            raise NameError("Failed reading map {}".format(self.filename))
            self.ncf.close()
    
    def readmaps(self,ncf):
        try:
            self.signal=(np.array(ncf.variables['signal'][:]).T)[::-1,::-1]
            self.weight=(np.array(ncf.variables['weight'][:]).T)[::-1,::-1]
            self.fSignal=(np.array(ncf.variables['filteredSignal'][:]).T)[::-1,::-1]
            self.fWeight=(np.array(ncf.variables['filteredWeight'][:]).T)[::-1,::-1]
        except Exception:
            raise Exception("File {} is damaged".format(self.filename))
        finally:
            # Si todo va bien, que cierre el ncf,
            # si ocurre una excepcion, que cierre el ncf
            ncf.close()

def J(maps,mask=None,fun='logcosh'):
    """ Negentropy approximation of maps,
        it uses only the useful pixels if a mask is provided.
        two non-quadratic functions are available:
            'logcosh' and 'exp' """

    if np.array(maps).ndim <= 2:
        maps=[maps]
    
    if isinstance(mask,np.ndarray):
        idxgood=np.where(mask > 0)
        maps=np.array([m[idxgood] for m in maps])

    negs=[]
    for m in maps:
        m=m-np.mean(m)
        m=m/np.std(m)
        m=m.flatten()
        # you may convince yourself that the numerical value of v
        # used here is the value that you get when G is a really
        # really big gaussian distribution.
        if fun=='logcosh':
            #G=np.array([random.gauss(0,1) for i in range(imsize*100)])
            #v=np.mean(np.log(np.cosh(G)))
            v=0.37456749067925998
            y=np.mean(np.log(np.cosh(m)))
            negs.append( 2*(v-y)/v*10 )
        elif fun=='exp':
            #G=[random.gauss(0,1) for i in range(imsize*100)]
            v=-0.7072548132591665  #-np.mean(np.exp(-G**2/2))
            y=-np.mean(np.exp(-m**2/2))
            negs.append( 2*(v-y)/np.abs(v)*10 )
    if len(negs) == 1:
        return negs[0]
    elif len(negs) > 1:
        return np.array(negs)
    else:
        print "Impossible to compute negentropies"
        return -1


import re
def tryint(s):
    try:
        return int(s)
    except:
        return s
def alphanum_key(s):
    """ Turns a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"] """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]
def get_filepaths(directory):
    """Docstring:
        This function generates the file names in a directory tree
        by walking the tree either top-down or bottom-up
        for each directory inside the tree.
        """
    file_paths = []  # List which will store all of the full filepaths.
    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.
    if len(file_paths)==0:
        raise NameError("File names not found in disk")
    file_paths.sort(key=alphanum_key) # re-order the names alpha-numerically
    return file_paths


def aztica(maps,n,mask=None,seed=None,
           return_low_mem=False,*args,**kwargs):
    """ Takes a list of (PCA)maps and returns a list of ICA-separated maps.
        If a mask is provided, it takes only the useful pixels.
        By default, the independent maps are normalized to unit-std and
        ordered from more compact to more extended sources. """
    Nmaps = len(maps)
    imsize = maps[0].size
    imshape = maps[0].shape
    flatMaps=np.array(maps).reshape([Nmaps,imsize])
    if isinstance(seed,int): np.random.seed(seed)
    #
    maxit = kwargs['max_iter'] if kwargs.has_key('max_iter') else 20000
    tol = kwargs['tol'] if kwargs.has_key('tol') else 0.01
    w_init=kwargs['w_init'] if kwargs.has_key('w_init') else np.identity(n)
    ica=FastICA(n_components=n,w_init=w_init,max_iter=maxit,tol=tol,*args)
    #
    if isinstance(mask,np.ndarray):
        idxgood=np.where(mask)
        X=np.array([m[idxgood] for m in maps]).T
    else:
        X=flatMaps.T
    #
    s=ica.fit_transform(X).T
    A=(ica.mixing_).T   # (transpose of) mixing matrix
    U=ica.components_   # unmixing matrix
    # Ad hoc right-skewness and unit-scale calibration
    for i in range(n):
        if np.sum(A[i]/np.max(np.abs(A[i])))<0:
           s[i] *= -1
           A[i] *= -1
           U[i] *= -1
        a=1./np.std(s[i])
        s[i] *= a
        A[i] /= a
        U[i] *= a
    #
    # This is an ordering convention.
    # Let Z_i=sum(A_i)/max(A_i).
    # It turns out that the more compact are the sources, the larger is Z_i.
    # Then, we can always identify the map of compact sources
    # and use it as a criterion to order the rest of the components.
    Z=np.array([np.abs(np.sum(A[i]))/np.max(np.abs(A[i])) for i in range(n)])
    inds=Z.argsort()[::-1]
    s=s[inds]
    A=A[inds]
    U=U[inds]
    #
    S=np.dot(U,flatMaps)
    S=S.reshape(n,imshape[0],imshape[1])
    if return_low_mem:
        return S[0]
    else:
        return S,A.T,U



def makeWmask(wmap,area,apix,l_init=1.,ntrials=20,fsolvetol=1e-5,verbose=0,
              fullreturn=True):
    """ Builds a mask from a weight-map or (its normalized version) hit-map.
        
        The user request the unmasked area size and it is computed according
        to the scanning coverage level.
        
        Input parameters:
            wmap: map of sqrt-weights
            area: desired unmasked area [arcmin] (filled with ones)
            apix: area per pixel [arcmin]
            l_init: initial guess for the wmap-level at area
            ntrials:
            fsolvetol: is an accuracy tolerance parameter.
        
        Returns:
        If fullreturn==True (default), then gives back:
            mask,indices of non-masked pixels,fitted wmap-level
        otherwise, only the mask
            
        """
    area=np.int(area)
    if area > np.where(wmap > 0.05)[0].size*apix:
        raise ValueError("You are asking a mask size greater than provided weight area")
    # func_level is the target function that will be used by fsolve
    # to find the weight-level that correspons to the requested area.
    func_level = lambda lev: area - len(np.where(wmap > lev)[0])*apix
    for tryi in range(ntrials):
        try:
            level=op.fsolve(func_level,l_init,xtol=fsolvetol)[0]
            mask = wmap > level
            idx=np.where(mask)
            # check that the unmasked area fits the requested area.
            # if you don't like rtol to be hard-coded you may add it to
            # the input parameters.
            if not np.any(mask) or not np.allclose(area,idx[0].size*apix,rtol=0.1):
                raise AssertionError("could not fit the requested area")
            break
        except (TypeError,RuntimeError,AssertionError):
            l_init=np.random.uniform(0,2)
            if verbose > 0:
                print 'trying another init: ',l_init
    if tryi == ntrials-1:
        raise RuntimeError("l_init cant be found, increase ntrials")
    idx=np.where(mask)
    if not np.allclose(area,idx[0].size*apix,rtol=0.1):
        raise AssertionError("failed making required mask")
    if fullreturn:
        return mask.astype(bool),idx,level
    else:
        return mask.astype(bool)



def makeWitness(NumWit,referMap,Fluxrange):
    # makes desired number NumWit of artificial sources
    # with a random flux in Fluxrange, and whose map shape=referMap.shape
    # returns the map and parameters of each artificial source in a dictionary
    # The makeotherbump_flag prevents to make bumps closer than 3-sgimas
    imshape = referMap.shape
    bumps = []
    for wit_i in range(NumWit):
        makeotherbump_flag = True
        while makeotherbump_flag:
            rndbump = makeRandBump(referMap,Amprange=Fluxrange,max_prop_radius=0.5)
            if len(bumps) == 0:
                makeotherbump_flag = False
            else:
                for bump in bumps:
                    rndmaxwidth = 3*np.max(  [rndbump['widths'][0] , rndbump['widths'][1]] )
                    bumpmaxwidth = 3*np.max( [bump['widths'][0] , bump['widths'][1]]  )
                    maxwidth = 3*(rndmaxwidth + bumpmaxwidth)
                    radiusbumps = np.array(rndbump['bump_cord']) - np.array(bump['bump_cord'])
                    magn_radiusbumps = np.sqrt( radiusbumps[0]**2 + radiusbumps[1]**2 )
                    if magn_radiusbumps > maxwidth:
                        makeotherbump_flag = False
                    else:
                        makeotherbump_flag = True
        # after we are sure that rndbump is far from previous bumps we accept it...
        bumps.append( rndbump )
    # Now, make a map with the NumWit bumps and return parameters
    bmap = np.zeros(imshape)
    for bump in [x['bump'] for x in bumps]:
        bmap += bump
    bcord = tuple([x['bump_cord'] for x in bumps])
    fluxes = [x['Amplitude'] for x in bumps]
    widths = np.array([x['widths'] for x in bumps])
    witness = {'bmap':bmap,'imshape':imshape,'bcord':bcord,'fluxes':fluxes,'widths':widths}
    return witness





def FitGauss2D(map, x,y, guess):
    # x,y must come in a meshgrid
    sigma = np.sqrt(x**2 + y**2)+1.0
    try:
        popt,pcov=op.curve_fit(f=Gaussian2D, xdata=(x.ravel(),y.ravel()),
                    ydata=map.ravel(), p0=guess, sigma=sigma.ravel())
        print "Im being used"
    except RuntimeError or TypeError:
            raise
    return popt,pcov



def SetGaussFit_toWit(witness,map,bn=2,beam_size=10,tol=5,verbose=False):
    # wsize controls the desired area to fit,
    # i.e. a map as big as (wsize*2*wx) * (wsize*2*wy) pixels
    numWitness = len(witness['fluxes'])
    wmap = witness['map']
    fluxes = witness['fluxes']
    cords = witness['locations']
    widths = witness['widths']
    
    outPars={'wit_idx':[],'background':[],'fit_fluxes':[],'fit_widths':[],
            'local_positions':[],'width_errors':[],'flux_errors':[]}
    
    for wit_i in range(numWitness):
        # set the values of each bump as an initial guess for the gaussian fit
        # c0-background, c1-amplitude, x0,y0-position,
        # wx,wy,theta-widths and elliptical rotation
        flux = fluxes[wit_i]
        wx = widths[wit_i][0]
        wy = widths[wit_i][1]
        wxy = widths[wit_i][2]
        pars_per_ws=[]
        for ws in np.arange(0, np.int(bn*beam_size)): # ws := window size
            # let us make the local map ws*sigma from the bump center
            wxs=wx+ws; wys=wy+ws
            local_witness_map = map[ cords[wit_i][0]-wxs : cords[wit_i][0]+wxs,
                                     cords[wit_i][1]-wys : cords[wit_i][1]+wys ]
            c0=0.5 ; x0=0. ; y0=0.
            c1 = np.median(local_witness_map)
            guess = [c0, c1, x0, y0, wx, wy, wxy]
            local_shape = local_witness_map.shape
            y = np.arange(-local_shape[0]/2,local_shape[0]/2)
            x = np.arange(-local_shape[1]/2,local_shape[1]/2)
            x,y = np.meshgrid(x,y)
            try:
                p , pcov = FitGauss2D(map=local_witness_map, x=x,y=y, guess=guess)
                pars = {"background":p[0],"fit_flux":p[1],"local_postition":[p[2],p[3]],
                        "fit_widths":[np.abs(p[4]),np.abs(p[5]),p[6]]}
            except RuntimeError or TypeError:
                # If fitting is taking too long (RuntimeError) we don't want to stop
                # the pipeline, we just discard this witness and move on.
                # Sometimes the witness lied too close map edges, you should try to
                # avoid this situation setting makeGaussianWitness:bn larger than
                # SetGaussFit_toWit:bn. In (rare) case where this would happen anyways,
                # FitGauss2D raises a TypeError and SetGaussFit_toWit will simply ignore
                # current witness.
                continue
            fit_widths = pars['fit_widths']
            width_error = [ (wx-fit_widths[0])/wx*100 , (wy-fit_widths[1])/wy*100 ]
            fit_flux = pars['fit_flux']
            flux_error = (flux - fit_flux)/flux * 100
            werr = np.max(  [ np.abs(width_error[0]) , np.abs(width_error[1]) ]  )
            pars['width_error'] = werr
            pars['flux_error'] = flux_error
            pars_per_ws.append( pars )
            #
        w_errs = np.array([ x['width_error'] for x in pars_per_ws ])
        if w_errs.size==0:
            print "impossible to fit this witness "
            tol_str = repr(tol)
            #raise HighPercentErr(tol_str+' %')
            return 0
        lowest_error_index = np.argmin(w_errs)
        pars_best_ws = pars_per_ws[lowest_error_index]
        if pars_best_ws['width_error'] < tol:
            #print "werr_{} = {}".format(wit_i, pars_best_ws['width_error'] )
            if verbose:
                print "werr_{} = {}".format(wit_i, pars_best_ws['width_error'] )
            outPars['wit_idx'].append(          wit_i                           )
            #continue
        #
        outPars['background'].append(       pars_best_ws['background']      )
        outPars['fit_fluxes'].append(       pars_best_ws['fit_flux']        )
        outPars['fit_widths'].append(       pars_best_ws['fit_widths']      )
        outPars['local_positions'].append(  pars_best_ws['local_postition'] )
        outPars['width_errors'].append(     pars_best_ws['width_error']     )
        outPars['flux_errors'].append(      pars_best_ws['flux_error']      )
    return outPars


def Gaussian2D( (x,y), c0, c1, x0,y0, sx, sy, theta ):
    """sx,sy are sigmax and sigmay
        """
    xp = (x-x0)*np.cos(theta) - (y-y0)*np.sin(theta)
    yp = (x-x0)*np.sin(theta) + (y-y0)*np.cos(theta)
    expn = (xp/sx)**2 + (yp/sy)**2
    ff = c0 + c1*np.exp(-0.5*expn)
    return ff


def Gaussian2D( (x,y), dc,a,b, A, x0,y0, sx, sy, theta ):
    """ Docstring:
        This is a (rotated) bivariate gaussian function and a plane.
        It is mostly used to fit a point source.
        
        Parameters:
        x,y: are the x,y coordinates coming in a meshgrid
        dc: is the background (noise?) level
        a,b: are the dc plane slopes in the non-rotated reference frame.
        A: is the amplitude of the gaussian
        x0,y0: the position of the gaussian center
        sx,sy: are sigmax and sigmay in the non-rotated ref frame.
        theta: is the angle of rotation.
        
        Returns:
        2D array: the gaussian function evaluated on the parameters.
        """
    xp = (x-x0)*np.cos(theta) - (y-y0)*np.sin(theta)
    yp = (x-x0)*np.sin(theta) + (y-y0)*np.cos(theta)
    argument = (xp/sx)**2 + (yp/sy)**2
    ff = dc + a*xp + b*yp + A*np.exp(-0.5*argument)
    return ff


def fitGauss2D(local_map):
    # I should rewrite this function with a fit other than op.curve_fit
    # because it doesnt have an option for constrained search (no tol parameter)
    local_shape=local_map.shape
    y=np.arange(-local_shape[0]/2,local_shape[0]/2)
    x=np.arange(-local_shape[1]/2,local_shape[1]/2)
    x,y=np.meshgrid(x,y)
    # weigh more the map center pixels
    # in order to force less importance at edges
    sigma=np.sqrt(x**2+y**2)+1.0
    #sx=np.float(local_shape[1])/n
    #sy=np.float(local_shape[0])/n
    for n in range(0,16):
        #print "n = " , n
        dc=0.                     # background
        a=0.; b=0.                # plane slopes
        A=np.max(local_map)    # amplitude
        x0=0. ; y0=0.
        sx=(8.+n)/2.35
        sy=(8.+n)/2.35
        guess=[dc,a,b,A,x0,y0,sx,sy,0.]
        #       0,1,2,3, 4, 5, 6, 7,8
        bounding=([-np.inf,-np.inf,-np.inf,0.,-np.inf,-np.inf,0.,0.,0.],
                  [np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.pi])
        try:
            fitpars,fitcov=op.curve_fit(f=Gaussian2D,
                            xdata=(x.ravel(),y.ravel()),ydata=local_map.ravel(),
                            p0=guess, sigma=sigma.ravel(),bounds=bounding)
            pars = {"background":fitpars[0:3],
                    "amplitude":fitpars[3],
                    "local_postition":[fitpars[4],fitpars[5]],
                    "sigmas":[np.abs(fitpars[6]),np.abs(fitpars[7])],
                    "angle":fitpars[8]}
            break
        except RuntimeError or TypeError or ValueError or OptimizeWarning or (np.isinf(fitcov)).any():
            #print "  Error en fit Gaussiano de mapa %s,  n = %d" % (repr(local_map),n)
            pars = {"background":np.nan,"amplitude":np.nan,
                    "local_postition":[np.nan,np.nan],
                    "sigmas":[np.nan,np.nan],"angle":np.nan}
            fitcov=np.nan
            continue
    return pars,fitcov


def FitPerBeam( signal_map,source_cord,source_range=15,verbose=0):
    """ Fits a specific source point according to source cordinates
        in a set of signal maps. User must provide the range of 
        cordinates of the specific source point, e.g. 10 pixels.
        It returns the fitted fluxes of both the source and background """
    source_map = signal_map[source_cord[0]-source_range:source_cord[0]+source_range, \
                            source_cord[1]-source_range:source_cord[1]+source_range]
    fit = fitGauss2D(source_map)
    if np.isnan(fit['amplitude']) and verbose > 0:
        print "Bad fit at cooridnates (%s,%s)" % source_cord
    return fit



def FitSourceFlux( a,signal_maps,source_cord,random_a=True,printn=False):
    """ Fits a specific source point in a set of signal maps,
        you must provide the range of cordinates of the specific source point.
        It returns the fitted fluxes of both the source and background """
    if random_a:
        random.seed(1)
        scale = array(random.sample(a,len(a)))
    else:
        scale = a.copy()
    #
    source_fit = []
    for n in range(len(scale)):
        if printn:
            print n
        source_map = scale[n] * signal_maps[n][source_cord[0]:source_cord[1],source_cord[2]:source_cord[3]]
        fit = SetGaussFit(source_map)
        if np.isnan(fit['flux']):
            continue
        else:
            source_fit.append( fit )
    source_fluxes = [ x['flux'] for x in source_fit ]
    source_backgrounds = [ x['background'] for x in source_fit  ]
    return source_fluxes,source_backgrounds



def cleanoutliers(variable,sigs_tol=1):
    """ Cleans variable off outliers and returns it
        sigs_tol is how many std deviations should clean off
        """
    var = np.array(variable).ravel()
    reject_outliers_flag = True
    while reject_outliers_flag:
        len1 = len(var)
        varmean = var.mean() ; varstd = var.std()
        vardist = np.array([ np.abs(x-varmean)/varstd    for x in var ])
        bad = np.where(vardist > sigs_tol)
        var = np.delete(var,bad)
        len2 = len(var)
        if len1 == len2:
            reject_outliers_flag = False
    return var


def candidatePixels(loc_idx,loc0,bn=2,beam=10):
    """bn is the number of beams to be excluded"""
    y,x=loc_idx
    y0,x0=loc0
    a=[];b=[]
    for yi,xi in zip(y,x):
        if np.sqrt( (x0-xi)**2 + (y0-yi)**2 ) > bn*beam:
            a.append(yi)
            b.append(xi)
    return (np.array(a),np.array(b))

def discardEdges(imshape,cords,edge_size=30):
    edsz=np.int(np.round(edge_size,decimals=0))
    edge_rows=[i for i in cords[0] if i<edsz or i>imshape[0]-edsz]
    edge_cols=[i for i in cords[1] if i<edsz or i>imshape[1]-edsz]
    y,x=cords
    a=[];b=[]
    for yi,xi in zip(y,x):
        if (yi not in edge_rows) and (xi not in edge_cols):
            a.append(yi)
            b.append(xi)
    return (np.array(a),np.array(b))


def countBrightSources(snmap,bn,beam=10,mask=None,SNtol=3.5,edge=15,
        printflag=False,toomany=200):
    """ Given a signal-to-noise map, this function counts how many
        (groups of) pixels are above SNtol. A mask could be useful to
        discard in advance unwanted pixels. By default it also discards
        pixels at the edges of the map, the edge parameter should be
        larger than beam.
        
        The steps:
        It first locates all unmasked pixels above the SNtol,
        picks the brightest and discards locations of pixels within
        a radius of bn*beam. From the remaining pixels, it picks the
        new brightest and discards its underlying vicinity again.
        The process stops either when no more pixels are above SNtol
        or the the function has found toomany bright-sources.
        
        Returns: the number of sources detected, their pixel locations,
        and their SN-value.
        
        """
    imshape=snmap.shape
    try:
        if isinstance(mask,type(None)):
            non_discarded_idx=np.where(snmap > SNtol)
        else:
            non_discarded_idx=np.where( (snmap > SNtol)*mask )
        non_discarded_idx=discardEdges(imshape,non_discarded_idx,edge)
    except TypeError:
        print TypeError("Seems that the input snmap or mask is ill")
        raise
    if len(non_discarded_idx[0])==0:
        if printflag:
            print "**** Warning: No sources above SN tolerance in this map ***"
        return {'N':0,'locations':(np.array(0),np.array(0)),'SN':[0]}
    max_sn=np.max(snmap[non_discarded_idx])
    locs=[]
    sn=[]
    count=0
    while np.size(non_discarded_idx) > 2 :
        max_sn=np.max(snmap[non_discarded_idx])
        max_idx=np.where( snmap == max_sn)
        max_idx=(max_idx[0][0],max_idx[1][0])
        if np.size(max_idx) > 2:
            print "error fatal! the same sn-pixel was found twice"
            break
        non_discarded_idx=candidatePixels(non_discarded_idx,max_idx,bn,beam)
        locs.append((max_idx[0],max_idx[1]))
        sn.append(max_sn)
        count += 1
        if count > toomany and printflag:
            print "**** Warning: too many incredible sources"
    outdict={}
    outdict['N']=count
    outdict['locations']=locs
    outdict['SN']=np.array(sn)
    if printflag==True:
        print "      {} bright sources found with SNR>{}".format(count,SNtol)
    return outdict





def isprime(n):
    """Simply tells whether an integer is prime or not, returns bool.
        If you provide a non-valid number, returns none."""
    if not isinstance(n,int):
        print "Only integers may be prime numbers"
        return
    if n in [2,3]: return True
    if n < 2 or n%2 == 0: return False
    answer=True
    for i in range(3,n,2):
        if n%i == 0:
            answer=False
            break
    return answer


def integerticks(vmin,vmax,nticks=None,centredflag=False):
    """Given a minimum (vmin) and maximum (vmax) tick-value,
       estimates the optimal stepsize, taking into account
       that the user wants between 3 and 10 ticks (may be changed).
       "Optimal" step is meant as the integer (vmax-vmin)/nsteps,
       i.e. with zero residual.
       Worst case scenario: if it can't find a zero residual,
       it takes a step=(vmax-vmin)/9.
       
       Returns an integer array, arange(vmin,vmax,step)."""
    vmin=np.int(vmin); vmax=np.int(vmax)
    vm=vmax-vmin
    step=1
    if isinstance(nticks,int):
        ticks=list(np.linspace(vmin,vmax,nticks+2))
        ticks.pop(0);ticks.pop(-1)
        ticks=np.array(ticks)
    else:
        if vm > 10 and isprime(vm):
                vmax -= 1
                vm=vmax-vmin
        if vm > 10:
            for nsteps in range(3,10)[::-1]:
                if vm%nsteps==0:
                    step=vm/nsteps
                    break
            if step == 1:
                step = vm/9
        #
        ticks=np.arange(vmin,vmax+1,step,dtype=int)
    if centredflag:
        ticks=ticks-np.mean(ticks)
        ticks=ticks.astype(int)
    return ticks







class ContourMap(object):

    def __init__(self,figsize=(5.5,5),xytitle=(0.45,0.96),
                 arclabel='arcmin',plottodir=None,filetype='png',
                 pixsize=1,dpi='figure',nxticks=5,nyticks=5,
                 rotangle=0,xyoffset=0,xlim=None,ylim=None,
                 hitmap=None,levels=None,level_labels=None):

        self.figsize=figsize
        self.xytitle=xytitle
        self.arclabel=arclabel
        self.plottodir=plottodir
        self.filetype=filetype
        self.nxticks=nxticks
        self.nyticks=nyticks
        self.pixsize=pixsize
        self.dpi=dpi
        self.rotangle=rotangle
        self.xyoffset=xyoffset
        self.xlim=xlim
        self.ylim=ylim
        self.hitmap=hitmap
        self.levels=levels
        self.level_labels=level_labels
    

    def plot(self,M,source_locs=None,source0_locs=None,
             figname=None,figtitle=None,units=r"mJy beam$^{-1}$",
             clearcanv=True,saveflag=False,
             plot_circles=True,plot_circles0=False,
             plot_numbers=False,plot_numbers0=False,
             m_circ_radius=10,m0_circ_radius=10,
             sourcec='white',source0c='yellow',
             paletteflag=True,ori='vertical',shr=1.0,*args,**kwargs):
        """Docstring:
            """
        if isinstance(figname,str):
            fig=plt.figure(figname,figsize=self.figsize)
            if clearcanv:
                fig.clf()
        else:
            fig=plt.figure(figsize=self.figsize)
        if isinstance(figtitle,str):
            suptitle=fig.suptitle(figtitle,x=self.xytitle[0],y=self.xytitle[1],fontsize=20)
        else:
            suptitle=''
        #
        ax=fig.gca()
        map=np.array(M).astype(float).copy()
        imshape=map.shape
        ra_m=imshape[1]*self.pixsize/2
        de_m=imshape[0]*self.pixsize/2
        ra =np.linspace(-ra_m,ra_m,imshape[1],dtype=int)
        dec=np.linspace(-de_m,de_m,imshape[0],dtype=int)
        ext=( ra.min(), ra.max(), dec.min(), dec.max() ) #left,right,bottom,top
        m_circ_radius=m_circ_radius*self.pixsize
        m0_circ_radius=m0_circ_radius*self.pixsize
        if not kwargs.has_key('cmap'): kwargs['cmap']='CMRmap'
        #
        transformacion=ax.transData
        if self.rotangle != 0:
            cy,cx=np.array(imshape)/2  # the center of the map
            rotation=tr.Affine2D().rotate_deg_around(cx,cy,self.rotangle)
            transformacion += rotation
        if self.xyoffset != 0:
            xoff = -7.5 if self.xyoffset==1 else self.xyoffset[0]
            yoff =  4   if self.xyoffset==1 else self.xyoffset[1]
            xoff *= self.pixsize
            yoff *= self.pixsize
            translation=tr.Affine2D().translate(xoff,yoff)
            transformacion += translation
        an=np.linspace(0,2*np.pi,50)
        #
        if not isinstance(source_locs,type(None)) and plot_circles:
            # the minus sign is because the default numpy origin is left top,
            # then, every coordinate below should have negative y-axis.
            # Instead we multiply by -1 to keep positive y axis.
            y=np.array([loc[0] for loc in source_locs])
            x=np.array([loc[1] for loc in source_locs])
            y= -(y-imshape[0]/2)*self.pixsize
            x= (x-imshape[1]/2)*self.pixsize
            counter=1
            for yi,xi in zip(y,x):
                ax.plot( xi+(m_circ_radius)*np.cos(an), yi+(m_circ_radius)*np.sin(an),
                        color=sourcec,ls='solid',lw=2,transform=transformacion)
                if plot_numbers:
                    # you may change the offsets where the numbers will be plotted
                    # double digit numbers need more offset
                    xo=-9
                    yo=-13
                    ax.annotate(str(counter),xy=(xi,yi),xycoords=transformacion,
                                xytext=(xo,yo),textcoords='offset points',
                                color=sourcec,size='xx-small')
                counter+=1
        #
        if not isinstance(source0_locs,type(None)) and plot_circles0:
            y0=np.array([loc[0] for loc in source0_locs])
            x0=np.array([loc[1] for loc in source0_locs])
            y0=-(y0-imshape[0]/2)*self.pixsize
            x0=(x0-imshape[1]/2)*self.pixsize
            counter=1
            for yi,xi in zip(y0,x0):
                ax.plot( xi+(m0_circ_radius)*np.cos(an), yi+(m0_circ_radius)*np.sin(an),
                        color=source0c,ls='dotted',lw=2,transform=transformacion)
                if plot_numbers0:
                    xo=6
                    yo=1
                    ax.annotate(str(counter),xy=(xi,yi),xycoords=transformacion,
                                xytext=(xo,yo),textcoords='offset points',
                                color=source0c,size='xx-small')
                counter+=1
        im=ax.imshow(map,extent=ext, *args,**kwargs)
        im.set_transform(transformacion)
        #
        if paletteflag:
            cbar=fig.colorbar(im,ax=ax,orientation=ori,shrink=shr)
            cbar.set_label(units)
            cbar.ax.tick_params(labelsize=15)
            vmin=np.int(kwargs['vmin']) if kwargs.has_key('vmin') else np.int(map.min())
            vmax=np.int(kwargs['vmax']) if kwargs.has_key('vmax') else np.int(map.max())
            if vmax-vmin > 2:
                bticks=integerticks(vmin,vmax)
                cbar.set_ticks(bticks)
        #
        if isinstance(self.hitmap,np.ndarray) and isinstance(self.levels,list):
            if self.rotangle != 0:
                hitmap=nd.rotate(self.hitmap,self.rotangle,reshape=False)
            else:
                hitmap=self.hitmap.copy()
            yr,xr=np.meshgrid(ra,dec)
            #yr=yr[::-1,::-1] # reverse the origin
            cn=ax.contour(yr,xr,hitmap,levels=self.levels,colors='g',linewidths=1)
            fmt={}
            if isinstance(self.level_labels,type(None)):
                # if labels are not provided, compute the labels according to
                # the level-information provided
                apix = ( self.pixsize**2 * (1./60)**2 )
                lab0 = int(len(np.where(hitmap > self.levels[0])[0])*apix)
                lab0 = '\t'+str(lab0)+' arcmin'+r'$^2$'+'\t'
                lab1 = int(len(np.where(hitmap > self.levels[1])[0])*apix)
                lab1 = '\t'+str(lab1)+' arcmin'+r'$^2$'+'\t'
                self.level_labels=[lab0,lab1]
            else:
                # if labels are provided, set in to string and add units
                lab0 = '\t'+str(self.level_labels[0])+' arcmin'+r'$^2$'+'\t'
                lab1 = '\t'+str(self.level_labels[1])+' arcmin'+r'$^2$'+'\t'
                level_labels=[lab0,lab1]
            for l,s in zip(cn.levels,level_labels):
                fmt[l]=s
            #
            ax.clabel(cn,cn.levels,inline=True,fmt=fmt,fontsize=12)


        plt.xlabel(self.arclabel,fontsize=15)
        plt.ylabel(self.arclabel,fontsize=15)
        #
        xticks=integerticks(ra.min(),ra.max(),self.nxticks,True)
        if self.arclabel == 'arcsec':
            xticklabs=integerticks(ra.min(),ra.max(),self.nxticks,True)
        else:
            xticklabs=integerticks(ra.min()/60,ra.max()/60,self.nxticks,True)
        plt.xticks(xticks,xticklabs,fontsize=15)
        #
        yticks=integerticks(dec.min(),dec.max(),self.nyticks,True)
        if self.arclabel == 'arcsec':
            yticklabs=integerticks(dec.min(),dec.max(),self.nyticks,True)
        else:
            yticklabs=integerticks(dec.min()/60,dec.max()/60,self.nyticks,True)
        plt.yticks(yticks,yticklabs,fontsize=15)
        #
        xlimits = (-585,585) if self.xlim==1 else self.xlim
        ylimits = (-695,695) if self.ylim==1 else self.ylim
        ax.set_xlim(xlimits)
        ax.set_ylim(ylimits)
        #controls the space (0) between subplots and give space (1-0.95) for suptitle
        fig.tight_layout(rect=[0,0,1,0.95])
        if saveflag:
            plottodir = self.plottodir if isinstance(self.plottodir,str) else os.getcwd()
            fname=plottodir+figname+'.'+self.filetype
            fig.savefig(fname,format=self.filetype,
                        transparent=True,bbox_extra_artists=(suptitle,),dpi=self.dpi)
            # savefig cant control the title, we need to use artists
        fig.show()
        return




def plotSeq(xseq,yseq=None,zseq=None,
            clearcanv=True,figname=None,figtitle=None,figsize=(6.5,5),
            xstart=None,xend=None,xstep=5,xmin=None,xmax=None,
            ystart=None,yend=None,ystep=5,ymin=None,ymax=None,
            xlabel=None,ylabel=None,xscale=None,yscale=None,
            one2one=False,plottodir=None,saveflag=False,
            transparent=True,legends={}, *args, **kwargs):
    """Docstring:
        Plots a sequence of numbers
        """
    if isinstance(figname,str):
        fig=plt.figure(figname,figsize=figsize)
        if clearcanv:
            fig.clf()
    else:
        fig=plt.figure(figsize=figsize)
    if isinstance(figtitle,str):
        suptitle=fig.suptitle(figtitle,x=0.5,y=0.96,fontsize=20)
    if kwargs.has_key('filetype'):
        filetype=kwargs['filetype']
        del kwargs['filetype']
    else:
        filetype='png'
    ax=fig.gca()
    #
    if type(xseq)!=type(None) and type(yseq)==type(None) and type(zseq)==type(None):
        if type(xstart)==type(None): xstart=0
        if type(xend)==type(None): xend=xstart+len(xseq)
        if type(ystart)==type(None): ystart=np.min(xseq)
        if type(yend)==type(None): yend=np.max(xseq)
        ax.plot(np.arange(xstart,xend),xseq, *args, **kwargs)
    elif type(xseq)!=type(None) and type(yseq)!=type(None) and type(zseq)==type(None):
        if type(xstart)==type(None): xstart=np.min(xseq)
        if type(xend)==type(None): xend=np.max(xseq)
        if type(ystart)==type(None): ystart=np.min(yseq)
        if type(yend)==type(None): yend=np.max(yseq)
        ax.plot(xseq,yseq, *args, **kwargs)
    elif type(xseq)!=type(None) and type(yseq)!=type(None) and type(zseq)!=type(None):
        if type(xstart)==type(None): xstart=np.min(xseq)
        if type(xend)==type(None): xend=np.max(xseq)
        if type(ystart)==type(None): ystart=np.min(yseq)
        if type(yend)==type(None): yend=np.max(yseq)
        ax.errorbar(xseq,yseq,zseq, *args, **kwargs)
    if one2one:
        start=np.min([xstart,ystart])
        end=np.max([xend,yend])
        seq=np.linspace(start,end,10)
        ax.plot(seq,seq,'k',lw=2)
    if isinstance(xscale,str): plt.xscale(xscale)
    if isinstance(yscale,str): plt.yscale(yscale)
    plt.xticks( np.arange(xstart,xend,xstep),fontsize=15)
    plt.yticks( np.arange(ystart,yend,ystep),fontsize=15)
    if isinstance(xlabel,str): plt.xlabel(xlabel,fontsize=15)
    if isinstance(ylabel,str): plt.ylabel(ylabel,fontsize=15)
    ax.tick_params('both',right=True,top=True,direction='in')
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    if kwargs.has_key('label'):
        lg=ax.legend(**legends)
    fig.tight_layout(rect=[0,0,1,0.95])
    if saveflag:
        if not isinstance(plottodir,str):
            plottodir=os.getcwd()
        fig.savefig(plottodir+figname+'.'+filetype,format=filetype,
                        transparent=transparent,bbox_extra_artists=(suptitle,))
    fig.show()
    return


def makeWeightMap(signal_std,signal_maps,weight_maps,mask=None,
        signal_std_err=0,nPars=5,check_fit=False,return_std_only=False):
    """ Docstring:
        The standard deviations of signal and weight maps are related
        by a linear equation. To convince yourself, take the log of both
        signal and weight standard deviations and plot them together,
        you will see that one is a linear function of the other.
        Here this function performs a fit to them. Thus, given a
        singal-map standard deviation (sigma_s), this infers the weight-map
        standard deviation weight_sig. Then, given a hit-map h,
        the weight map is estimated as weight_sig*h.
        
        Parameters:
        signal_std: float or array. the std of the signal map
                    (inside some uniformly covered area)
        signal_maps: a set of signal redundant maps.
        weight_maps: the sqrt-weight partners of signa_maps
        mask: mask untrusted pixels
        signal_std_err: if signal_std has an uncertainty,
                    then it adds up in quadratures.
        
        Returns:
        The weight-map(s). If return_std_only==True, returns only the weight_sig.
        """
    # 20 is hard-coded but we checked that in practice the choice is irrelevant
    hitmap=weight_maps[20]/np.max(weight_maps[20])
    hitmap += 1e-14 # this avoids 1/0
    if isinstance(mask,np.ndarray):
        ns=np.array([np.std(ss[mask]) for ss in signal_maps])
        #effective sensitivities
        sens=[np.mean(hitmap[mask]/(w[mask]+1e-13)) for w in weight_maps]
    else:
        ns=np.array([np.std(ss) for ss in signal_maps])
        #effective sensitivities
        sens=np.array([np.mean(hitmap/(w+1e-13)) for w in weight_maps])
    #coeffs=np.polyfit(ns,sens,nPars)
    #Poln=np.poly1d(coeffs)
    #sig_eff=Poln(signal_std)
    func=interpolate.interp1d(ns,sens,kind='cubic')
    sig_eff=func(signal_std)
    #
    if check_fit:
        down=np.min(ns)
        upp=np.max(ns)
        x=np.linspace(down,upp,100)
        #y=Poln(x)
        y=func(x)
        fig=plt.figure("checking fit"); fig.clf(); ax=fig.gca()
        ax.plot(ns,sens,lw=3,label="Actual maps")
        ax.plot(x,y,lw=3,label="fit")
        ax.legend(loc='best')
        plt.xlabel("Signal maps std")
        plt.ylabel("Sensitivities")
        print "sig_eff={}".format(sig_eff)
        plt.show()
    if np.asarray(sig_eff).size > 1:
        W=[]
        for sig in sig_eff:
            w=hitmap/sig
            W.append( (w**(-2.) + signal_std_err**2)**(-0.5) )
        W=np.array(W)
    else:
        W=hitmap/sig_eff
    if return_std_only:
        return sig_eff
    else:
        return W



def residual2data(p,s,ms,rangeM,mask):
    """Docstring:
        This function is used to calibrate n ICA maps, 
        using a pixel-fit to a subset of redundant maps, and a mask.
        
        Parameters:
        All parameters are mandatory because calibration must be thoughtful!
        p: array containing 2*n parameters of calibration
            p0,p1 are the media and scale factor of s0
            p2,p3 are the media and scale factor of s1, and so on
        s: the n IC maps
        ms: the Nb-1 redundant maps
        rangeM: the indexes of redundant maps to be used for the fit
        mask: a 2D array. Shapes of mask, ms, and s must be the same.
        
        Returns:
        the residual of the pixel-fit (kind-of the chi-2) that will be minimized.
        """
    s=np.array(s).copy()
    if len(s.shape) == 2:
        n=1
        s=np.array([s])
    elif len(s.shape) == 3:
        n=s.shape[0]
    else:
        raise TypeError("map s is neither a map nor sequence of maps")
    idxgood=np.where(mask > 0)
    numdata=idxgood[0].size
    model=np.zeros(numdata)
    for j in range(n):
        model += p[2*j+1]*(p[2*j]+s[j][idxgood])
    Xi=0.
    for i in rangeM:
        Xi += np.sum(np.abs(model - ms[i][idxgood]))
    return Xi/numdata
