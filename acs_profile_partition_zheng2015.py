#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Partition a_nw into a_ph, a_d and a_g.

Reference: Zheng et al. 2015, JGR.

@author: Yangyang Liu (yangyang.liu@awi.de), November 2019.
"""
import glob, os, shutil, math, itertools, scipy.optimize
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import pandas as pd
import scipy.interpolate as inp
from scipy.signal import savgol_filter
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
def readConfig(config):
    paramDict = {}
    try:
        with open(config) as f:
            for line in f:
                if line.startswith('#')==False and len(line)>1:
                    key = line.strip().split('=')[0]
                    try:
                        value = line.strip().split('=')[1].strip()
                    except:
                        value = None
                    paramDict[key] = value
    except Exception as e:
        print(e)
    finally:
        return paramDict
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
def mergeFilesinSubDir(filename, dirs):
    '''
    merge files with completely same structure (i.e. header) and filenames 
    in the directories "dirs".
    '''
    merged_data = pd.DataFrame()
    for i, dir_data in enumerate(dirs):
        try:
            data = pd.read_csv(os.path.join(dir_data,filename), comment='%', sep='\t')
            merged_data = merged_data.append(data)
            del data     
        except:
            print(f'Warning: {filename} does not exist in {dir_data}!!!')   
    return merged_data

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
def createDir(dirname, overwrite=False):
    
    if overwrite:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        else:
            shutil.rmtree(dirname) #removes all the existing directories!
            os.makedirs(dirname)
    else:
        if not os.path.exists(dirname):
            os.makedirs(dirname)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
def spectra_normalize_cluster(dfspectra, wavelength, wl1, wl2, n_clusters=5,
                               keyword='NAP'):
    '''
    normalize a spectrum by sum and HCA.
    '''    
    dfspectra = dfspectra.dropna()
    pos1 = np.where(np.array(wavelength) >= wl1)[0][0]
    pos2 = np.where(np.array(wavelength) >= wl2)[0][0]
    dfspectra = dfspectra.iloc[:,pos1:pos2+1]
    dfspectra.index = range(len(dfspectra))
    wl = wavelength[pos1:pos2+1]
    
    spectra_norm = dfspectra.divide(np.sum(dfspectra,1), axis='index')
    cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', 
                                      linkage='ward')
    cluster_labels = pd.DataFrame(cluster.fit_predict(spectra_norm), 
                                  index=spectra_norm.index)
    spectra_basis = pd.concat([cluster_labels, spectra_norm], axis=1)
    spectra_basis_median = spectra_basis.groupby(pd.Grouper(key=0)).median() 
    spectra_basis_sd = spectra_basis.groupby(pd.Grouper(key=0)).agg(np.std, ddof=1)
    
    spectra_basis_median.to_csv('basis_vector_'+keyword+'.txt', index=True, header=True, 
                                encoding='utf-8', sep='\t') 
    spectra_basis_sd.to_csv('basis_vector_sd_'+keyword+'.txt', index=True, header=True, 
                                encoding='utf-8', sep='\t') 
    
    #plot basis vectors
    font = {'family': 'Arial', 'weight': 'bold', 'size': 15}
    plt.rcParams['mathtext.default'] = 'regular'
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    ax.plot(wl,spectra_basis.iloc[:,1:].transpose(), color='0.75', 
            linestyle='dashed', linewidth=1)
    ax.plot(wl,spectra_basis_median.transpose(),
            linewidth=2.5)
    plt.xlim(wl1, wl2)
    plt.tick_params(labelsize=12)
    plt.xlabel('Wavelength [nm]', fontdict=font)
    if keyword=='NAP':
        plt.ylabel(r'$\hat a_{NAP}(\lambda)\ [m^{-1}]$', fontdict=font)
    elif keyword == 'CDOM':
        plt.ylabel(r'$\hat a_{CDOM} [m^{-1}]$', fontdict=font)    
    plt.tight_layout()
    fig.savefig('basis_vector_'+keyword+'.jpg', dpi=300)
    plt.close(fig)
    
    #dendrogram
    plt.figure()
    shc.dendrogram(shc.linkage(spectra_norm , method='ward'))
    plt.ylabel('Euclidean Distance', fontdict=font)
    plt.tight_layout()
    plt.savefig('dendrogram_'+keyword+'.jpg', dpi=300)
    plt.close()
    
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------    
class PartitionModel2015():
    
    def __init__(self, config='config_PartitionModel2015.txt'):
        paraDict = readConfig(config)
        #self.path_acs = paraDict['path_acs']
        self.path_qft = paraDict['path_qft']
        self.path_lwcc = paraDict['path_lwcc']
        self.ap = paraDict['ap']
        self.aph = paraDict['aph']
        self.anap = paraDict['anap']
        self.cdom = paraDict['cdom']
        self.cruises = paraDict['cruises_involved'].split(',')
        qft_wl_select = paraDict['qft_wl_select'].split(',')
        self.qft_wl_select = [float(wl) for wl in qft_wl_select]
        self.quantiles = [0.01, 0.5, 0.99]
        #self.quantiles = [0, 0.5, 1]
        self.percentiles = ['1st_percentile', '50th_percentile(median)', 
                        '99th_percentile']
        #self.percentiles = ['0th_percentile', '50th_percentile(median)', 
        #                '100th_percentile']
        self.wl_cluster = [400, 730]
        self.merged_qft_aph = 'merged_qft_aph.txt'
        self.merged_qft_anap = 'merged_qft_anap.txt'
        self.merged_lwcc_cdom = 'merged_lwcc_cdom.txt'
        self.qft_bandratio_constraints = 'qft_bandratio_constraints.txt'
        self.matched_qft_anw = 'matched_qft_lwcc_anw.txt'
        self.matched_qft_adg = 'matched_qft_lwcc_adg.txt'
        self.matched_qft_aph = 'matched_qft_aph.txt'
        self.matched_qft_anap = 'matched_qft_anap.txt'
        self.matched_lwcc_cdom = 'matched_lwcc_cdom.txt'
        self.matched_decomp_aph = 'decomposed_aph_median.txt'
        self.matched_decomp_anap = 'decomposed_anap_median.txt'
        self.matched_decomp_acdom = 'decomposed_acdom_median.txt'
        
    #--------------------------------------------------------------------------
    def getWL(self, df):
        
        wl = [float(col.replace('wl','')) for i, col in enumerate(df.columns) if 'wl' in col]
        
        pos0= [i for i, col in enumerate(df.columns) if 'wl' in col]
        
        pos = [np.where(np.array(wl) >= wvl)[0][0] for wvl in self.qft_wl_select] 
        return wl, pos0, pos
    
    #--------------------------------------------------------------------------
    def interpSpectra(self, wl, dfSpec, wl_new, columns):
        #interpolate data on wavelengths
        func = inp.interp1d(wl, dfSpec, 'linear', fill_value='extrapolate')
        interpSpec = pd.DataFrame(func(wl_new), index=dfSpec.index, columns=columns)
        return interpSpec
    
    #-------------------------------------------------------------------------- 
    def plotSpec_decomMeas(self, dfSpec, dfSpec_10per, dfSpec_90per, dfMeas, dirfig):
        
        wl_decomp, pos0, pos = self.getWL(dfSpec)
        wl_meas, pos0_meas, pos = self.getWL(dfMeas)
        createDir(dirfig, overwrite=True)
        for i in range(len(dfSpec)):
            fig = plt.figure(figsize=(8.5, 6.5))
            ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
            ax.tick_params(axis="both", labelsize=10)
            ax.plot(wl_decomp, dfSpec.iloc[i,pos0[0]:].transpose(), 
                    'k-', linewidth=1.5)
            ax.plot(wl_decomp, dfSpec_10per.iloc[i,pos0[0]:].transpose(), 
                    'b--', linewidth=1.5)
            ax.plot(wl_decomp, dfSpec_90per.iloc[i,pos0[0]:].transpose(), 
                    'b--', linewidth=1.5)
            ax.plot(wl_meas, dfMeas.iloc[i,pos0_meas[0]:].transpose(), 
                    'r-', linewidth=1.5)
            ax.legend(['median','10 percentile','90 percentile', 'measured'], loc='upper right',
                       bbox_to_anchor=(1, 1), fontsize='small')
            ax.set_xlabel('Wavelength [$nm$]', fontsize=12)
            ax.set_ylabel('$Absorption(\lambda)\ [m^{-1}]$', fontsize=12)
            ax.set_xlim([400,720])
            fig.savefig(os.path.join(dirfig, dfSpec.iloc[i,2]+'_'+
                                     str(int(dfSpec.iloc[i,5]))+'m.jpg'), dpi=200)
            plt.close(fig)
    
    #--------------------------------------------------------------------------
    def mergeQFT_BandRatios(self, smoothNAP=True, histogram=True):
        
        #merge files
        dirs = []
        for i, cruise in enumerate(self.cruises):
            dir0 = os.path.join(self.path_qft, cruise)
            dirs.append(dir0)
    
        merged_ap = mergeFilesinSubDir(self.ap, dirs)
        merged_anap = mergeFilesinSubDir(self.anap, dirs)
        
        #data quality control
        wl, pos0, pos = self.getWL(merged_ap)
        #updated self.qft_wl_select
        qft_wl_select = [wl[loc] for i, loc in enumerate(pos)]
        pos = [pos0[0] + loc for i, loc in enumerate(pos)]

        #smooth NAP data
        if smoothNAP:
            data = savgol_filter(merged_anap.iloc[:,pos0[0]:], 5, 1)
            merged_anap[merged_anap.columns[pos0[0]:]] = data
            merged_aph = merged_ap.copy()
            data = merged_ap[merged_anap.columns[pos0[0]:]] - \
            merged_anap[merged_anap.columns[pos0[0]:]]
            merged_aph[merged_anap.columns[pos0[0]:]] = data
        else:
            merged_aph = mergeFilesinSubDir(self.aph, dirs)

        #data points with aph(675)>0.7*aph(442), anap(730)>0.7*anap(442) and 
        #anap(490)>anap(469) are set as NaN
        conditions = np.where( (merged_aph.iloc[:,pos[1]].divide(merged_aph.iloc[
                :,pos0[0]+np.where(np.array(wl)>=675)[0][0]])<=1) | 
                (merged_anap.iloc[:,pos[-1]].divide(merged_anap.iloc[:,pos[1]])>=0.7)
                | (merged_anap.iloc[:,pos[3]].divide(merged_anap.iloc[:,pos[2]])>1) )[0]
        merged_aph.iloc[conditions,pos0[0]:] = np.nan
        merged_anap.iloc[conditions,pos0[0]:] = np.nan
        
        #ranges of ap(442), aph(442), anap(442)
        range_ap = merged_ap.iloc[:,pos[1]]
        range_aph = merged_aph.iloc[:,pos[1]]
        range_anap = merged_anap.iloc[:,pos[1]]
        ranges = pd.concat([pd.DataFrame([range_ap.quantile(self.quantiles[0]), 
                            range_ap.quantile(self.quantiles[1]),
                            range_ap.quantile(self.quantiles[2])]),
                            pd.DataFrame([range_aph.quantile(self.quantiles[0]), 
                            range_aph.quantile(self.quantiles[1]),
                            range_aph.quantile(self.quantiles[2])]),
                            pd.DataFrame([range_anap.quantile(self.quantiles[0]), 
                            range_anap.quantile(self.quantiles[1]),
                            range_anap.quantile(self.quantiles[2])])], axis=1)
        ranges.index = self.percentiles
        ranges.columns = ['ap(' + str(int(qft_wl_select[1])) + ')', 
                          'aph(' + str(int(qft_wl_select[1])) + ')', 
                          'anap(' + str(int(qft_wl_select[1])) + ')']
        ranges.to_csv('merged_qft_ranges.txt', index=True, header=True, 
                         encoding='utf-8', sep='\t') 
        merged_ap.to_csv(self.merged_qft_aph.replace('aph','ap'), index=False, 
                         header=True, encoding='utf-8', sep='\t') 
        merged_aph.to_csv(self.merged_qft_aph, index=False, header=True, 
                          encoding='utf-8', sep='\t') 
        merged_anap.to_csv(self.merged_qft_anap, index=False, header=True, 
                           encoding='utf-8', sep='\t')   
        
        #calculate aph band ratios and determine constraints
        aph_ratio = merged_aph.iloc[:,:pos0[0]]
        ix = [(0,1), (3,1), (2,0), (4,3), (5,1)]
        wl_pairs = []
        for i in range(len(ix)-1):
            pairs = (str(int(qft_wl_select[ix[i][0]])), 
                         str(int(qft_wl_select[ix[i][1]])))
            wl_pairs.append(pairs)
            aph_ratio[wl_pairs[i][0]+'/'+wl_pairs[i][1]] = merged_aph.iloc[:,
        pos[ix[i][0]]]/merged_aph.iloc[:,pos[ix[i][1]]]  
        
        aph_ratio = aph_ratio.dropna()
        ratios = aph_ratio.iloc[:,pos0[0]:]
        bounds = pd.concat([ratios.quantile(self.quantiles[0]), 
                            ratios.quantile(self.quantiles[1]),
                            ratios.quantile(self.quantiles[2])], axis=1)
        
        #calculate anap band ratio and determine constraints
        anap_ratio = merged_anap.iloc[:,:pos0[0]]
        anap_ratio[str(int(qft_wl_select[ix[-1][0]]))+'/'+
                   str(int(qft_wl_select[ix[-1][1]]))] = merged_anap.iloc[:,
        pos[ix[-1][0]]]/merged_anap.iloc[:,pos[ix[-1][1]]] 
        anap_ratio = anap_ratio.dropna()
        ratios_nap = anap_ratio.iloc[:,pos0[0]:]
        bounds = bounds.append(pd.concat([ratios_nap.quantile(self.quantiles[0]), 
                            ratios_nap.quantile(self.quantiles[1]),
                            ratios_nap.quantile(self.quantiles[2])], axis=1))
            
        bounds.columns = self.percentiles
        bounds.index = ['aph'+idx for idx in bounds.index]
        bounds = bounds.rename(index={str(bounds.index[-1]) : 
            str(bounds.index[-1]).replace('aph','anap')})
        
        bounds.to_csv(self.qft_bandratio_constraints, index=True, header=True, 
                         encoding='utf-8', sep='\t') 

        #histograms
        if histogram:
            font = {'family': 'Arial', 'weight': 'bold', 'size': 16}
            plt.rcParams['mathtext.default'] = 'regular'
            plt.style.use('ggplot')
            
            fig, axes = plt.subplots(2, 2, sharey=True, tight_layout=True, 
                                     figsize=(8.5, 6.5))
            k = 0
            for i in range(2):
                for j in range(2):
                    axes[i,j].hist(ratios.iloc[:,k], bins=20, alpha=0.5, 
                        facecolor='k', histtype='bar', ec='black', linewidth=1.2) 
                    axes[i,j].tick_params(labelsize=12)
                    axes[i,j].set_xlabel('$a_{ph}($' + str(wl_pairs[k][0]) + 
                        '$)/a_{ph}($' + str(wl_pairs[k][1]) +'$)$', fontdict=font)
                    k += 1
            plt.ylim(0, 120)
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top='off', bottom='off', 
                            left='off', right='off')
            plt.grid(False)
            plt.ylabel('Number of observations',fontdict=font, labelpad=10)
            plt.tight_layout()
            plt.savefig('aph_bandratio_hist.jpg', dpi=300)
            plt.close()
        
    #--------------------------------------------------------------------------
    def mergeLWCC(self):
        dirs = []
        for i, cruise in enumerate(self.cruises):
            dir0 = os.path.join(self.path_lwcc, cruise)
            dirs.append(dir0)
    
        filenames = sorted(glob.glob(os.path.join(dirs[0], self.cdom)))
        merged_cdom = pd.DataFrame(columns=pd.read_csv(filenames[0], sep='\t').columns)
        for fn in filenames:
            file = os.path.basename(fn)
            tmp = mergeFilesinSubDir(file, dirs) 
            merged_cdom = merged_cdom.append(tmp, ignore_index=True)
        
        wl, pos0, pos = self.getWL(merged_cdom)
        pos = np.where(np.array(wl) >= self.qft_wl_select[1])[0][0] 
        
        #ranges of acdom(442)
        range_cdom = merged_cdom.iloc[:,pos+pos0[0]]
        range_cdom = pd.DataFrame([range_cdom.quantile(self.quantiles[0]), 
                            range_cdom.quantile(self.quantiles[1]),
                            range_cdom.quantile(self.quantiles[2])])
        range_cdom.index = self.percentiles
        range_cdom.columns = ['a_CDOM(' + str(int(wl[pos])) + ')']
        
        range_cdom.to_csv('merged_lwcc_range.txt', index=True, header=True, 
                           encoding='utf-8', sep='\t') 
        merged_cdom.to_csv(self.merged_lwcc_cdom, index=False, header=True, 
                           encoding='utf-8', sep='\t') 
        
    #--------------------------------------------------------------------------    
    def spectraHCA(self, n_clusters=5, keyword='NAP'):
        
        if keyword == 'NAP':
            merged_data = pd.read_csv(self.merged_qft_anap,comment='%', sep='\t')
        elif keyword == 'CDOM':
            merged_data = pd.read_csv(self.merged_lwcc_cdom,comment='%', sep='\t')
        
        wavelength, pos0, pos = self.getWL(merged_data)
        dfspectra = merged_data.iloc[:, pos0[0]:]
        spectra_normalize_cluster(dfspectra, wavelength, self.wl_cluster[0], 
                                  self.wl_cluster[1], n_clusters=n_clusters,
                               keyword=keyword)
        
    #--------------------------------------------------------------------------    
    def matchup_QFT_LWCC(self):
        
        aph = pd.read_csv(self.merged_qft_aph, sep='\t') 
        anap = pd.read_csv(self.merged_qft_anap, sep='\t') 
        acdom = pd.read_csv(self.merged_lwcc_cdom, sep='\t') 
        
        acdom.index = acdom.iloc[:,0]+'_'+acdom.iloc[:,2]+'_'+acdom.iloc[:,5
                                ].astype(int).astype(str)
        aph.index = aph.iloc[:,0]+'_'+aph.iloc[:,2]+'_'+aph.iloc[:,5
                                ].astype(int).astype(str)
        anap.index = anap.iloc[:,0]+'_'+anap.iloc[:,2]+'_'+anap.iloc[:,5
                                ].astype(int).astype(str)
        
        wl_qft, pos0, pos = self.getWL(aph)
        pos_qft = pos0[0]
        wl_lwcc, pos0, pos = self.getWL(acdom)
        pos_lwcc = pos0[0]

        #interpolate QFT data on LWCC wavelengths
        aph_interp = self.interpSpectra(wl_qft, aph.iloc[:,pos_qft:], wl_lwcc,
                                        range(len(wl_lwcc)))
        anap_interp = self.interpSpectra(wl_qft, anap.iloc[:,pos_qft:], wl_lwcc,
                                         range(len(wl_lwcc)))
        
        #match QFT and LWCC data, and calculate anw
        data_match = pd.concat([aph_interp, anap_interp, acdom], axis=1, 
                               ).dropna()
        acdom_match = data_match.iloc[:,len(wl_lwcc)*2:]
        aph_match = data_match.iloc[:,:len(wl_lwcc)]
        aph_match.columns = acdom_match.iloc[:,pos_lwcc:].columns
        anap_match = data_match.iloc[:,len(wl_lwcc):len(wl_lwcc)*2]
        anap_match.columns = aph_match.columns
        anw_match = aph_match + anap_match + acdom_match.iloc[:,pos_lwcc:]
        anw_match = pd.concat([acdom_match.iloc[:,:pos_lwcc], anw_match], axis=1)
        adg_match = anap_match + acdom_match.iloc[:,pos_lwcc:]
        adg_match = pd.concat([acdom_match.iloc[:,:pos_lwcc], adg_match], axis=1)
        aph_match = pd.concat([acdom_match.iloc[:,:pos_lwcc], aph_match], axis=1)
        anap_match = pd.concat([acdom_match.iloc[:,:pos_lwcc], anap_match], axis=1)
        anw_match.to_csv(self.matched_qft_anw, index=False, header=True, 
                         encoding='utf-8', sep='\t')
        adg_match.to_csv(self.matched_qft_adg, index=False, header=True, 
                         encoding='utf-8', sep='\t')
        aph_match.to_csv(self.matched_qft_aph, index=False, header=True, 
                         encoding='utf-8', sep='\t')
        anap_match.to_csv(self.matched_qft_anap, index=False, header=True, 
                         encoding='utf-8', sep='\t')
        acdom_match.to_csv(self.matched_lwcc_cdom, index=False, header=True, 
                         encoding='utf-8', sep='\t')
        
    #--------------------------------------------------------------------------    
    def anw_partition(self, plot=True):
            
        constraints = pd.read_csv(self.qft_bandratio_constraints, index_col=0,
                              sep='\t').round(2)
        x = np.arange(constraints.iloc[0,0],constraints.iloc[0,2]+0.01,0.01)
        y = np.arange(constraints.iloc[1,0],constraints.iloc[1,2]+0.01,0.01)
     
        anap_basis = pd.read_csv('basis_vector_NAP.txt', index_col=0,
                              sep='\t')
        acdom_basis = pd.read_csv('basis_vector_CDOM.txt', index_col=0,
                              sep='\t') 
        anw = pd.read_csv(self.matched_qft_anw, sep='\t')
     
        wl_basis, pos0, pos_basis = self.getWL(anap_basis)
        pos0_basis = pos0[0]
        wl_anw, pos0, pos_anw = self.getWL(anw)
        pos0_anw = pos0[0]
        pos_700 = np.where(np.array(wl_anw)>700)[0][0] + pos0_anw

        #interpolate anap_basis and acdom_basis on wl_anw 
        anap_basis_interp = self.interpSpectra(wl_basis, anap_basis.iloc[:,pos0_basis:], 
                                            wl_anw, anw.columns[pos0_anw:])
        acdom_basis_interp = self.interpSpectra(wl_basis, acdom_basis.iloc[:,pos0_basis:], 
                                            wl_anw, anw.columns[pos0_anw:])
        
        #combinations
        combo = [ list(range(len(x))), list(range(len(y))), 
              list(range(len(anap_basis))), list(range(len(acdom_basis))) ]
        combo = list(itertools.product(*combo))
     
        aph_optimal = pd.DataFrame(columns=range(len(anw)))
        aph_range_10th = aph_optimal.copy()
        aph_range_90th = aph_optimal.copy()
        anap_optimal = aph_optimal.copy()
        anap_range_10th = aph_optimal.copy()
        anap_range_90th = aph_optimal.copy()
        acdom_optimal = aph_optimal.copy()
        acdom_range_10th = aph_optimal.copy()
        acdom_range_90th = aph_optimal.copy()
        for j in range(len(anw)):
            #print('j=',str(j))
            aph_feasible = pd.DataFrame(columns=anw.iloc[:,pos0_anw:].columns)
            anap_feasible = pd.DataFrame()
            acdom_feasible = pd.DataFrame()
            for i in range(len(combo)):
                Quant1 = anap_basis.iloc[combo[i][2], pos0_basis+pos_basis[0]] - \
                x[combo[i][0]] * anap_basis.iloc[combo[i][2], pos0_basis+pos_basis[1]]
                Quant2 = acdom_basis.iloc[combo[i][-1], pos0_basis+pos_basis[0]] - \
                x[combo[i][0]] * acdom_basis.iloc[combo[i][-1], pos0_basis+pos_basis[1]]
                Quant3 = anap_basis.iloc[combo[i][2], pos0_basis+pos_basis[3]] - \
                y[combo[i][1]] * anap_basis.iloc[combo[i][2], pos0_basis+pos_basis[1]]
                Quant4 = acdom_basis.iloc[combo[i][-1], pos0_basis+pos_basis[3]] - \
                y[combo[i][1]] * acdom_basis.iloc[combo[i][-1], pos0_basis+pos_basis[1]]
                Diff1 = anw.iloc[j,pos_anw[0]]-x[combo[i][0]]*anw.iloc[j,pos_anw[1]]
                Diff2 = anw.iloc[j,pos_anw[3]]-y[combo[i][1]]*anw.iloc[j,pos_anw[1]]
             
                factor_nap_tmp = (Quant4*Diff1-Quant2*Diff2)/(Quant1*Quant4-Quant2*Quant3)
                factor_cdom_tmp = (Quant1*Diff2-Quant3*Diff1)/(Quant1*Quant4-Quant2*Quant3) 
                anap = factor_nap_tmp * anap_basis_interp.iloc[combo[i][2], :]
                acdom = factor_cdom_tmp * acdom_basis_interp.iloc[combo[i][-1], :]
                aph_tmp = anw.values[j,pos0_anw:] - anap.values - acdom.values
                #constraints
                aph_tmp_ratio3 = aph_tmp[pos_anw[2]]/aph_tmp[pos_anw[0]]
                aph_tmp_ratio4 = aph_tmp[pos_anw[4]]/aph_tmp[pos_anw[3]]
                anap_ratio = anap[pos_anw[-1]]/anap[pos_anw[1]]
                constaints345 = (aph_tmp_ratio3 >= constraints.iloc[2,0]) & \
                (aph_tmp_ratio3 <= constraints.iloc[2,2]) & \
                (aph_tmp_ratio4 >= constraints.iloc[3,0]) & \
                (aph_tmp_ratio4 <= constraints.iloc[3,2]) & \
                (anap_ratio >= constraints.iloc[4,0]) & \
                (anap_ratio <= constraints.iloc[4,2]) & \
                (anap[pos_anw[1]]> 0) & (acdom[pos_anw[1]]> 0) & \
                (aph_tmp[pos_anw[4]]>0) & (aph_tmp[pos_700]>=0) & \
                (abs(aph_tmp[pos_anw[-1]])<=1e-3)
                if constaints345:
                    aph_feasible = aph_feasible.append(pd.Series(
                         aph_tmp, index=anw.iloc[:,pos0_anw:].columns, 
                         name=str(i))) 
                    anap_feasible = anap_feasible.append(pd.Series(
                         factor_nap_tmp*anap_basis.iloc[combo[i][2],:], 
                         name=str(i)), ignore_index=True) 
                    acdom_feasible = acdom_feasible.append(pd.Series(
                         factor_cdom_tmp*acdom_basis.iloc[combo[i][-1],:], 
                         name=str(i)), ignore_index=True)                           
            #median, optimal  
            if len(aph_feasible)>0:
                aph_optimal[j] = aph_feasible.median()
                aph_range_10th[j] = aph_feasible.quantile(0.1)
                aph_range_90th[j] = aph_feasible.quantile(0.9)
            if len(anap_feasible)>0:
                anap_optimal[j] = anap_feasible.median()
                anap_range_10th[j] = anap_feasible.quantile(0.1)
                anap_range_90th[j] = anap_feasible.quantile(0.9)
            if len(acdom_feasible)>0:
                acdom_optimal[j] = acdom_feasible.median()
                acdom_range_10th[j] = acdom_feasible.quantile(0.1)
                acdom_range_90th[j] = acdom_feasible.quantile(0.9)
         
        
        decomposed_aph_median = pd.concat([anw.iloc[:,:pos0_anw], 
                                           aph_optimal.transpose()], axis=1)
        decomposed_aph_10per = pd.concat([anw.iloc[:,:pos0_anw], 
                                           aph_range_10th.transpose()], axis=1)
        decomposed_aph_90per = pd.concat([anw.iloc[:,:pos0_anw], 
                                           aph_range_90th.transpose()], axis=1)
        decomposed_aph_median.to_csv(self.matched_decomp_aph, index=False, 
                                     header=True, encoding='utf-8', sep='\t')
        decomposed_aph_10per.to_csv('decomposed_aph_10per.txt', index=False, 
                                    header=True, encoding='utf-8', sep='\t')
        decomposed_aph_90per.to_csv('decomposed_aph_90per.txt', index=False, 
                                    header=True, encoding='utf-8', sep='\t')
        
        decomposed_anap_median = pd.concat([anw.iloc[:,:pos0_anw], 
                                           anap_optimal.transpose()], axis=1)
        decomposed_anap_10per = pd.concat([anw.iloc[:,:pos0_anw], 
                                           anap_range_10th.transpose()], axis=1)
        decomposed_anap_90per = pd.concat([anw.iloc[:,:pos0_anw], 
                                           anap_range_90th.transpose()], axis=1)
        decomposed_anap_median.to_csv(self.matched_decomp_anap, index=False, 
                                      header=True, encoding='utf-8', sep='\t')
        decomposed_anap_10per.to_csv('decomposed_anap_10per.txt', index=False, 
                                     header=True, encoding='utf-8', sep='\t')
        decomposed_anap_90per.to_csv('decomposed_anap_90per.txt', index=False, 
                                     header=True, encoding='utf-8', sep='\t')
        
        decomposed_acdom_median = pd.concat([anw.iloc[:,:pos0_anw], 
                                           acdom_optimal.transpose()], axis=1)
        decomposed_acdom_10per = pd.concat([anw.iloc[:,:pos0_anw], 
                                           acdom_range_10th.transpose()], axis=1)
        decomposed_acdom_90per = pd.concat([anw.iloc[:,:pos0_anw], 
                                           acdom_range_90th.transpose()], axis=1)
        decomposed_acdom_median.to_csv(self.matched_decomp_acdom, index=False, 
                                       header=True, encoding='utf-8', sep='\t')
        decomposed_acdom_10per.to_csv('decomposed_acdom_10per.txt', index=False,
                                      header=True, encoding='utf-8', sep='\t')
        decomposed_acdom_90per.to_csv('decomposed_acdom_90per.txt', index=False, 
                                      header=True, encoding='utf-8', sep='\t')
        
        if plot:
            aph_meas = pd.read_csv(self.matched_qft_aph, sep='\t')
            anap_meas = pd.read_csv(self.matched_qft_anap, sep='\t')
            acdom_meas = pd.read_csv(self.matched_lwcc_cdom, sep='\t')
            self.plotSpec_decomMeas(decomposed_aph_median, decomposed_aph_10per, 
                                 decomposed_aph_90per, aph_meas, 'decomp_aph')
            self.plotSpec_decomMeas(decomposed_anap_median, decomposed_anap_10per, 
                                 decomposed_anap_90per, anap_meas, 'decomp_anap')
            self.plotSpec_decomMeas(decomposed_acdom_median, decomposed_acdom_10per, 
                                 decomposed_acdom_90per, acdom_meas, 'decomp_acdom')

    


if __name__ == '__main__':
    
    wd = '/Users/yliu/Data/cruises/adg_anw'
    os.chdir(wd)
    
    model = PartitionModel2015(config='config_PartitionModel2015.txt')
    model.mergeQFT_BandRatios()
    model.mergeLWCC()
    model.spectraHCA(n_clusters=6, keyword='NAP')
    model.spectraHCA(n_clusters=5, keyword='CDOM')
    model.matchup_QFT_LWCC()
    model.anw_partition()
    











    
    
