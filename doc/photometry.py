# -*- coding: utf-8 -*-
"""Adds photometric tools to package"""

# Python Imports
from os.path import join

# 3rd Party Imports
import numpy as np
import xarray as xr
import pandas as pd
import pyregion as pyreg
import pysynphot as psp
from sklearn.metrics import (pairwise_distances as pdist,
                             pairwise_distances_argmin as pdistargmin)

# Astro and ESO
from astroptical.photometry import magnitude, magerr


# --- Calc Mags ---------------------------------------------------------------
def calcesomags(cats, zpts, filt=False):
    """Calculates the Magnitudes of each source since SExtractor cannot be
    trusted to do so"""

    # Calculate Mags
    mags = {}
    for key in [275, 475, 814, 160]:
        mags[key], _ = magnitude(
            cats.sel(measurement='FLUX_APER', filt=key).values,
            zpts['abZpt'][key], zpts['GalExt'][key], zpts['ApCor'][key],
            zpts['kcor'][key]
        )

    # Calculate Errors
    errs = {}
    for key in mags:
        errs[key] = magerr(
            cats.sel(measurement='FLUX_APER',    filt=key).values,
            cats.sel(measurement='FLUXERR_APER', filt=key).values
        )/zpts['DrizCor'][key]

    # Remove by Color
    if filt:
        x = mags[275] - mags[475]
        y = mags[475] - mags[814]
        gdInd1 = np.logical_and(x > -2, x < 4)
        gdInd2 = np.logical_and(y > -2, y < 2)
        msk = np.logical_and(gdInd1,gdInd2)
        cats = cats[:,msk,:]
        for key in mags:
            mags[key] = mags[key][msk]
            errs[key] = errs[key][msk]

    return mags, errs, cats

# --- Combine Sources if Low Error --------------------------------------------
def combinesrc(cats,zpts,roi,pixDistTsh=153,colorTsh=0.3,maxIter=1000):
    '''Comb the list and add sources with high error
    '''

    # Loop Until no more changes made or counter runs out
    cnt = 0
    while cnt < maxIter:

        # Get Params to Check
        mags   = calcesomags(cats,zpts)[0]
        colorY = mags[475] - mags[814]
        dists  = pdist(cats.sel(measurement=['X_IMAGE','Y_IMAGE'],filt=475).values)
        errsY  = cats.sel(measurement='MAGERR_APER',filt=[475,814]).values.transpose()
        colorErr = np.linalg.norm(errsY,axis=1)

        # Pare Down Checks
        np.fill_diagonal(dists,np.Inf)

        # Get Index Pairs that lie in pixel tolerence
        minPairs  = np.vstack(np.nonzero(dists <= pixDistTsh)).transpose()
        minPairs  = np.unique(np.sort(minPairs),axis=0)

        # Only keep pairs that have color difference within error
        colorDiff = np.diff(colorY[minPairs]).flatten()
        combErr   = np.linalg.norm(colorErr[minPairs],axis=1)
        minPairs  = minPairs[np.abs(colorDiff) <= combErr,:]

        # Start Making Combos
        comboMade = False
        pairsMade = []
        for i, (obs1, obs2) in enumerate(minPairs):

            # Only proceed if one of the errors is flagged as greater than 1
            if (np.any(np.linalg.norm(errsY[[obs1,obs2],:],axis=1) > colorTsh)
                    and obs1 != obs2 and [obs1,obs2] not in pairsMade):

                # Identify that a combo has been made.
                comboMade = True
                pairsMade.append([obs1,obs2]); pairsMade.append([obs2,obs1])

                # Replacea all remaining instances of obs2
                inds1 = np.flatnonzero(minPairs[:,0] == obs2)
                inds2 = np.flatnonzero(minPairs[:,1] == obs2)
                minPairs[inds1[inds1>i],0] = obs1
                minPairs[inds2[inds2>i],1] = obs1

                # Adjust Catalog
                ent1 = cats[:,obs1,:]
                ent2 = cats[:,obs2,:]
                newRad  = np.maximum(ent1.loc[:,'AperRad'],ent2.loc[:,'AperRad'])
                newRad += 25
                ent1.loc[:,'AperRad']      = newRad
                ent1.loc[:,'X_IMAGE']      = (ent1.loc[:,'X_IMAGE'] +
                                              ent2.loc[:,'X_IMAGE'])/2
                ent1.loc[:,'Y_IMAGE']      = (ent1.loc[:,'Y_IMAGE'] +
                                              ent2.loc[:,'Y_IMAGE'])/2
                ent1.loc[:,'FLUX_APER']    = (ent1.loc[:,'FLUX_APER'] +
                                              ent2.loc[:,'FLUX_APER'])
                ent1.loc[:,'FLUXERR_APER'] = (ent1.loc[:,'FLUXERR_APER'] +
                                              ent2.loc[:,'FLUXERR_APER'])
                ent1.loc[:,'MAGERR_APER']  = magerr(ent1.loc[:,'FLUX_APER'],
                                                    ent1.loc[:,'FLUXERR_APER'])
                ent1.loc[:,'Merged']      += 1
                cats[:,obs1,:] = ent1
                cats[:,obs2,:] = None

        # Drop Bad Cat Entries or break no entries combined
        cats = cats.dropna('observation','all')
        if not comboMade:
            break

        cnt += 1

    # Remove High Error Sources
    if roi.lower() != 'control':
        goodInd     = np.logical_and(cats.sel(measurement='MAGERR_APER',filt=475) < 1,
                                     cats.sel(measurement='MAGERR_APER',filt=814) < 1)
        if roi.lower() in ['halpha','h-alpha']:
            goodInd = np.logical_and(
                np.logical_and(
                    goodInd, cats.sel(measurement='MAGERR_APER',filt=160) < 1),
                cats.sel(measurement='MAGERR_APER',filt=275) < 1)

        # Remove
        cats = cats[:,goodInd.values,:]

    # Return
    return cats


# --- Sources in Region -------------------------------------------------------
def srcinreg(fileName,cats,dropIn=True):
    """Checks to see if a source is given region and drops it if dropIn=True
    or keeps it if dropIn=False"""

    # Read File First
    dRegs = pyreg.open(fileName)

    if dropIn:
        for dReg in dRegs:
            xI = cats.sel(measurement='X_IMAGE', filt=475).values
            yI = cats.sel(measurement='Y_IMAGE', filt=475).values
            if dReg.name == 'circle':
                x,y,r = dReg.coord_list
                r     = r**2
                kpInd = (x - xI)**2 + (y - yI)**2 > r
            elif dReg.name == 'ellipse':
                x,y,a,b,th = dReg.coord_list
                th = np.deg2rad(th)
                X = (x - xI)*np.cos(th) + (y - yI)*np.sin(th)
                Y = (x - xI)*np.sin(th) - (y - yI)*np.cos(th)
                kpInd = (X/a)**2 + (Y/b)**2 > 1
            cats = cats[:, kpInd, :]
    else:
        xI   = cats.sel(measurement='X_IMAGE',filt=475).values
        yI   = cats.sel(measurement='Y_IMAGE',filt=475).values
        good = np.zeros(len(xI),dtype=np.bool)
        for dReg in dRegs:
            if dReg.name == 'circle':
                x,y,r = dReg.coord_list
                r     = r**2
                good  = np.logical_or(good, (x - xI)**2 + (y - yI)**2 <= r)
            elif dReg.name == 'ellipse':
                x,y,a,b,th = dReg.coord_list
                th = np.deg2rad(th)
                X = (x - xI)*np.cos(th) + (y - yI)*np.sin(th)
                Y = (x - xI)*np.sin(th) - (y - yI)*np.cos(th)
                good = np.logical_or(good, (X/a)**2 + (Y/b)**2 <= 1)
        cats = cats[:,good,:]

    return cats


# --- Remove Sources ----------------------------------------------------------
def rmsrcs(regPath,roi,detImg,cats,zpts,maxFlux=60):
    """Systematically removes sources from the catalogs by location and error"""

    # Check ROI
    roi = roi.lower()

    # Remove Bad Flux & Error
    mags, errs  = calcesomags(cats, zpts)[:2]
    badInd = np.zeros_like(mags[475], dtype='bool')
    for key in mags:
        badCur = np.logical_not(np.isfinite(mags[key])) | (errs[key] > 1)
        badInd = badInd | badCur
    cats  = cats[:, np.logical_not(badInd), :]

    # Remove MW Stars first
    cats = srcinreg(join(regPath, 'starsInH2Reg-img.reg') ,cats)
    cats = srcinreg(join(regPath, 'ESO_Guide_Star_Coords-img.reg') ,cats)

    # Remove Other High Flux Sources in F814W
    if roi != 'control':
        potMWStar = (cats.sel(measurement='FLUX_APER',filt=[475,814]) > maxFlux).values
        potMWStar = np.any(potMWStar,0)
        cats = cats[:,np.logical_not(potMWStar),:]

    # Remove Spurrious F275W Sources
    detImg = detImg.upper()
    if detImg.upper() == 'F275W':
        cats = srcinreg(join(regPath, 'spurriousF275W-img.reg'), cats)

    # Remove Edges
    cats = srcinreg(join(regPath, detImg + '_Edge-img.reg'), cats)

    # Changes based on ROI
    if roi in ['tail','tailreg','tail-reg']:
        cats = srcinreg(join(regPath, 'tail-img.reg'), cats, False)
    elif roi in ['halpha','h-alpha']:
        cats = srcinreg(join(regPath, 'sun-imgCoord.reg'), cats, False)
    elif roi == 'control':
        cats = srcinreg(join(regPath, 'tail-img.reg'), cats, True)
    elif roi == 'galaxy':
        cats = srcinreg(join(regPath, 'sun-imgCoord.reg'), cats, True)
        cats = srcinreg(join(regPath, 'galaxy-img.reg'), cats, False)
    elif roi in ['foss','fossati']:
        cats = srcinreg(join(regPath, 'Fossati-img.reg'), cats, False)

    return cats

# --- Calculate the Magnitudes ---------------------------------------------- #
def calcsspmag(srcs, filts):
    """Calculates the Magnitude of SSP models"""

    # Loop through the years
    nSrcs = len(srcs)
    mags = {}
    mags[275] = np.zeros(nSrcs)
    mags[475] = np.zeros(nSrcs)
    mags[814] = np.zeros(nSrcs)
    mags[160] = np.zeros(nSrcs)
    for i in np.arange(nSrcs):
        mags[275][i] = psp.Observation(srcs[i],filts[275]).effstim('abmag')
        mags[475][i] = psp.Observation(srcs[i],filts[475]).effstim('abmag')
        mags[814][i] = psp.Observation(srcs[i],filts[814]).effstim('abmag')
        mags[160][i] = psp.Observation(srcs[i],filts[160]).effstim('abmag')
    return pd.DataFrame(mags)

# --- Combine Sources ---------------------------------------------------------
def combinebyreg(cats, fsEWs, roiPath, zpts):
    '''Combines all sources falling in an ROI circlular regions'''

    # Get Regions
    regs = pyreg.open(roiPath + 'Fossati-FK5.reg')
    regs = np.array([reg.coord_list for reg in regs])

    # Sort
    fsReg  = np.array(fsEWs[['RA','DEC']])
    srtInd = pdistargmin(fsReg,regs[:, :2], 0)
    regs = pyreg.open(roiPath + 'Fossati-img.reg')
    regs = np.array([reg.coord_list for reg in regs])
    regs = regs[srtInd, :]

    # Source X, Y
    srcX = cats.loc[475, :, 'X_IMAGE'].values
    srcY = cats.loc[475, :, 'Y_IMAGE'].values
    errs = cats.loc[[275, 475, 814], :, 'MAGERR_APER'].values
    errs = np.all(errs < 1, 0)

    # Loop Through Regions
    regCats = xr.full_like(cats, np.NaN)
    regCats = regCats.loc[:, :, ['FLUX_APER','MAGERR_APER']]
    regCats = regCats[:, :len(regs), :]
    regCats = regCats.assign_coords(observation=pd.Int64Index(range(len(regs))))
    for i, (regX, regY, regR) in enumerate(regs):

        # Keep Ind
        kpInd = ((regX - srcX)**2 + (regY - srcY)**2 <= regR**2)
        kpInd = np.logical_and(kpInd, errs)

        # Combine
        if np.any(kpInd):
            regCats.loc[:, i, 'FLUX_APER'] = cats.loc[:, kpInd, 'FLUX_APER'].sum(
                dim='observation', skipna=True)
            regFluxErr = cats.loc[:,kpInd,'FLUXERR_APER'].sum(
                dim='observation', skipna=True)
            regCats.loc[:, i, 'MAGERR_APER'] = magerr(regCats.loc[:, i, 'FLUX_APER'],
                                                      regFluxErr)

    return calcesomags(regCats,zpts,False)[:2]
