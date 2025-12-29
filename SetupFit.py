import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec#
import math
from astropy.modeling.models import BlackBody
from astropy import units as u
import scipy.stats
#import corner

try:
    plt.style.use(['science','ieee', 'no-latex'])
except:
    plt.style.use(['default'])
from astropy.modeling import models, fitting

import pandas as pd
from astropy.table import Table
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline as IUSScipy
from astropy.modeling import Fittable1DModel
from astropy.modeling import Parameter
from astropy.modeling.functional_models import Gaussian1D
from astropy.modeling.physical_models import Drude1D
import os
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.special import gamma, factorial, legendre

from jax import random
import jax as jaxx
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from scipy.interpolate import interp1d
import pickle

import jax
jax.config.update("jax_enable_x64", True)

#import pymc3 as pm
numpyro.set_platform("cpu")
#import pymc.sampling_jax
dir_path = os.path.dirname(os.path.realpath(__file__))


#ls_i, ice = np.loadtxt("IceExt_v2.txt", unpack = True, usecols=[0,1])
#ls_i, ice = np.loadtxt(dir_path+"/IceExt.txt", unpack = True, usecols=[0,1])


#ls_C, CH = np.loadtxt("CHExt_v2.txt", unpack = True, usecols=[0,1])





# Stellar_x1, Stellar_y1 = np.loadtxt(dir_path+'/SSP_10.0_highres.txt', unpack=True)
# Stellar_y1 /= np.interp(1.6, Stellar_x1, Stellar_y1)
# Stellar_y1 /= 100.0
# #Stellar_x_org1 = Stellar_x1
# #Stellar_x1 = np.linspace(0.9, 28, 10000)
# #Stellar_y1 = np.interp(Stellar_x1, Stellar_x_org1, Stellar_y1)/100.0
# Stellar_y_smooth1 = scipy.ndimage.gaussian_filter1d(Stellar_y1, 999)#/100.0


# Stellar_x2, Stellar_y2 = np.loadtxt(dir_path+'/SSP_8.0_highres.txt', unpack=True)
# Stellar_y2 /= np.interp(1.6, Stellar_x2, Stellar_y2)
# Stellar_y2 /= 100.0

# #Stellar_x_org2 = Stellar_x2
# #Stellar_x2 = np.linspace(0.9, 28, 10000)
# #Stellar_y2 = np.interp(Stellar_x2, Stellar_x_org2, Stellar_y2)/100.0
# Stellar_y_smooth2 = scipy.ndimage.gaussian_filter1d(Stellar_y2, 999)#/100.0


# plt.figure()
# plt.plot(Stellar_x1, Stellar_y1)
# plt.plot(Stellar_x1, Stellar_y_smooth1)
# plt.plot(Stellar_x1, Stellar_y2)
# plt.plot(Stellar_x1, Stellar_y_smooth2)

# plt.show()




class Fit():
    def __init__(self, objName, specdata, z, lam_range,   ExtType='Differential', Ices_6micron = False, ExtCurve = 'D23ExtCurve', EmCurve='D24Emissivity', MIR_CH = 'CHExt_v3', NIR_CH = 'CH_NIR', Fit_NIR_CH = False, NIR_Ice_ = 'NIR_Ice', NIR_CO2_ = 'NIR_CO2', Cont_Only = False, St_Cont = True, Extend = False, Fit_CO = False, spec_res = 'h'):
        lam, flux, flux_err = specdata
        lam = lam/(1.0+z)

        
        
        scale = np.mean(flux)
        self.scale = scale
        maxlam = lam_range[1]
       # print(maxlam)
        minlam = lam_range[0]
        flux = flux[(lam>=minlam)&(lam<=maxlam)]
        flux_err = flux_err[(lam>=minlam)&(lam<=maxlam)]#/10.0
        lam = lam[(lam>=minlam)&(lam<=maxlam)]

        global ls, S, ls_Em, Em, ls_C, CH, ls_C_NIR, CH_NIR, ls_NIR_Ice, NIR_Ice, ls_NIR_CO2, NIR_CO2
        ls, S = np.loadtxt(dir_path+'/Ext.Curves/'+ExtCurve+'.txt', unpack=True)
        ls_Em, Em = np.loadtxt(dir_path+'/Emissivity/'+EmCurve+'.txt', unpack = True, usecols=[0,1])


        # ls_H20_Template, H20_Template = np.loadtxt(dir_path+"/IceTemplates/NGC5728_Ice_Template.txt", unpack = True, usecols=[0,1])

        ls_C, CH = np.loadtxt(dir_path+"/IceTemplates/MIR_CH/"+MIR_CH+".txt", unpack = True, usecols=[0,1])
        ls_C_NIR, CH_NIR = np.loadtxt(dir_path+"/IceTemplates/NIR_CH/"+NIR_CH+".txt", unpack = True, usecols=[0,1])

        ls_NIR_Ice, NIR_Ice= np.loadtxt(dir_path+"/IceTemplates/NIR_H2O/"+NIR_Ice_+".txt", unpack = True, usecols=[0,1])
        ls_NIR_CO2, NIR_CO2= np.loadtxt(dir_path+"/IceTemplates/NIR_CO2/"+NIR_CO2_+".txt", unpack = True, usecols=[0,1])

        extended = Extend



        global Stellar_y1, Stellar_y2, Stellar_y_smooth1, Stellar_y_smooth2, Stellar_x1, Stellar_x2

        # if (spec_res == 'h'):
        #     Stellar_x1, Stellar_y1 = np.loadtxt(dir_path+'/SSP_10.0_highres.txt', unpack=True)
        # else:
        Stellar_x1, Stellar_y1 = np.loadtxt(dir_path+'/SSP_10.0_medres.txt', unpack=True)
        #Stellar_x1, Stellar_y1 = np.loadtxt(dir_path+'/SSP_10.0_medres.txt', unpack=True)
       # Stellar_x1, Stellar_y1 = np.loadtxt(dir_path+'/SSP_6.0_lowZ_nebcont.txt', unpack=True)


        Stellar_y1 /= np.interp(1.6, Stellar_x1, Stellar_y1)
        Stellar_y1 /= 100.0

        #Stellar_x_org1 = Stellar_x1
        #Stellar_x1 = np.linspace(0.9, 28, 10000)
        #Stellar_y1 = np.interp(Stellar_x1, Stellar_x_org1, Stellar_y1)/100.0
        Stellar_y_smooth1 = scipy.ndimage.gaussian_filter1d(Stellar_y1, 999)#/100.0
        Stellar_y1 = np.interp(lam, Stellar_x1, Stellar_y1)
        Stellar_y_smooth1 = np.interp(lam, Stellar_x1, Stellar_y_smooth1)
        Stellar_x1 = np.copy(lam)

        # if (spec_res == 'h'):
        #     Stellar_x2, Stellar_y2 = np.loadtxt(dir_path+'/SSP_8.0_highres.txt', unpack=True)
        # else:
        Stellar_x2, Stellar_y2 = np.loadtxt(dir_path+'/SSP_8.0_medres.txt', unpack=True)

        #print('NIRSpec Spectral Res:', spec_res)
        Stellar_y2 /= np.interp(1.6, Stellar_x2, Stellar_y2)
        Stellar_y2 /= 100.0

        #Stellar_x_org2 = Stellar_x2
        #Stellar_x2 = np.linspace(0.9, 28, 10000)
        #Stellar_y2 = np.interp(Stellar_x2, Stellar_x_org2, Stellar_y2)/100.0
        Stellar_y_smooth2 = scipy.ndimage.gaussian_filter1d(Stellar_y2, 999)#/100.0
        Stellar_y2 = np.interp(lam, Stellar_x2, Stellar_y2)
        Stellar_y_smooth2 = np.interp(lam, Stellar_x2, Stellar_y_smooth2)
        Stellar_x2 = np.copy(lam)


        global grid_size, tau_lim

        if (extended == True):
            tau_lim = 67.29
            grid_size = [20, 25]
        else:
            grid_size = [20, 20]
            tau_lim = 15.0
        self.grid_size = grid_size

        flux /=scale
        flux_err /=scale
        

        self.ObjName = objName
        
        flux_orig = flux*flux/flux
        
        filt = scipy.signal.medfilt(flux,  41)
        flux_sub = flux-filt
        flux_sub_orig = flux_sub
        stellar_est = np.interp(1.6, lam, filt)*1000.0

        if (min(lam)>4.0):
            stellar_est = np.mean(filt[lam<6.0])*1000.0

        if (stellar_est <=0.0):
            stellar_est = np.mean(filt[lam>6.0])*1000.0

########################################




        # Set up parameters dataframe
        self.parameters = pd.DataFrame(columns=['Section', 'Component','Name', 'Description','Value', '+Error', '-Error','Prior','Prior Type', 'Fixed'])
        
        # Initialise Model Components
        
        #Emission lines
        lines =["BrGamma","BrBeta","Pf11", "H2 O(4)", "PfEpsilon", "H2 O(5)","PfDelta", "H2 O(6)", "H2 S(15)", "H2 S(14)", "PfGamma", 
                "H2 O(7)",  "H2 S(13)", "H2 S(12)", "BrAlpha","Hu13","HeI","Hu12","H2 S(10)", "PfBeta",
            "H2 S(8)", "[Fe II]", "[Fe VIII]",
            
            "H2 S(7)", "[Mg V]","HuGamma",
            "H2 S(6)","H2 S(5)","[Ar II]", "[Na III]","PfAlpha", "HuBeta","[NeVI]",
            "[FeVII]","[Ar V]",
             "H2 S(4)","[Ar III]",
             "[FeVIII]",
             "H2 S(3)", "[S IV]",  "H2 S(2)",
             "HuAlpha",
             "[Ne II]", "[ArV]",
             "[Ne V] ", "[Ne III]", "H2 S(1)",
             "[FeII]",
             "[S III]","HI 8-7","[Fe III]",
             "[NeV_]", "[O IV]", "[Fe II_]", "[SIII_]", "[Si II]"]#, "C2H2", "HCN"]
        cents =  [2.1655 , 2.63, 2.87, 3.00387, 3.04, 3.2350, 3.2961, 3.50081, 3.62636, 3.72558, 3.74, 3.80740, 3.84723, 3.99692, 4.05, 4.18, 4.30, 4.38, 4.40997, 4.6525,
            5.053, 5.340, 5.447,
            5.511, 5.609,5.9066,
            6.109, 6.910, 6.985, 7.318, 7.460, 7.5005, 7.6524,
            7.8145, 7.901,
            8.025, 8.991,
            9.527,
            9.665, 10.511, 12.279,
            12.370,
            12.8135, 13.102,
            14.3217, 15.555, 17.035,
            17.936,
            18.713, 19.052, 22.925,
            24.3175, 25.910, 25.989, 33.480, 34.815]

        widths = [0.034, 0.034,0.034,0.034,0.034,0.034,0.034,0.034,0.034, 0.053, 0.053,0.053,0.053,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.14,0.14,0.14,0.14,0.34,0.34,0.34,0.34, 0.34, 0.34, 0.1, 0.1] # Spitzer
        vals = [0.000757799755090427, 0.00029842545884589856, 0.00409849610889481, 0.045355842828490714, 0.030825202361984858, 0.013389637498470245, 0.013389637498470245, 0.023938848033417545, 0.01829149007506314, 0.04773620924836646,0.000757799755090427, 0.00029842545884589856, 0.00409849610889481, 0.045355842828490714, 0.030825202361984858, 0.013389637498470245, 0.023938848033417545, 0.01829149007506314, 0.04773620924836646, 0.01969443135828826, 0.39347772642591167, 0.01816892897538409, 0.2640011203300603, 0.035319393780289365, 0.5476647269591665, 0.003334490580185104, 0.0011990672414200704, 0.004236638465562518, 0.024856938574108832, 0.7463738764070731, 0.9867613111587851, -0.0031418002198900824, -0.0031418002198900824]
        widths = np.full(len(cents), 0.01)
        vals =np.full(len(cents), 1.0)
        
        self.Nlines = 0
        self.linecents=[]
        self.linenames=[]
        self.H2linecents=[]
        self.H2width_limits=[]
        self.H2_indices=[]
        mfs=[]
        #mfs_prev = 0.0
        for i in range(len(lines)):
            if (np.min(lam) <= cents[i] <= np.max(lam)):
                self.Nlines += 1
                

                
                

                amp_lower = 0.0001
                amp_upper = 100.0
                amp_l_br = 0.0
                amp_u_br = 1.0
                cent_range =0.03


                if (cents[i]*(1.0+z) < 2.5):
                    width = 0.001
                    cent_range =0.1

                if (2.5<=cents[i]*(1.0+z) < 4.9):
                    width = 0.003
                    cent_range =0.1

                
                if (4.9 <= cents[i]*(1.0+z) <= 7.65):
                    width = 0.005
                    cent_range =0.1

                    
                if (7.51 <= cents[i]*(1.0+z) <= 11.71):
                    width = 0.012
                    cent_range =0.1

                if (10.5 <= cents[i]*(1.0+z) <= 18.2):
                    cent_range =0.1
                    width = 0.015
                if (17.71 <= cents[i]*(1.0+z) <= 28.1):
                    cent_range =0.1
                    width = 0.02
                
                # Find line centre
               #  lam_range = lam[(lam>(cents[i]-cent_range)) & (lam<(cents[i]+cent_range))]
               # #print(lam_range)
               #  indx = np.where(np.in1d(lam, lam_range))[0]
               #  flux_range = flux_sub[indx]
               #  flux_err_range = flux_err[indx]
               #  #mfs_prev =max(flux_range)
               #  cents[i] = lam_range[flux_range == max(flux_range)][0]
                #mfs.append(max(flux_range))

                cent_range = 5.0*width#0.01
                mfs.append(cent_range)
                #flux_sub -= Gauss(lam,  lam_range[flux_range == max(flux_range)][0], 0.001, max(flux_range), jax=False)

                # Only mask line if it is actually detected
                if (np.any(flux_sub[(lam>(cents[i]-cent_range)) & (lam<(cents[i]+cent_range))])): # Check for NIRSpec detector gap
                    if (np.max(flux_sub[(lam>(cents[i]-cent_range)) & (lam<(cents[i]+cent_range))])>2.0*np.mean(flux_err[(lam>(cents[i]-cent_range)) & (lam<(cents[i]+cent_range))]) and cents[i]!=3.2961):
                        flux_cut = np.concatenate((filt[(lam<(cents[i]-cent_range))], filt[(lam>(cents[i]+cent_range))])) 
                        lam_cut =  np.concatenate((lam[(lam<(cents[i]-cent_range))], lam[(lam>(cents[i]+cent_range))])) 
                        flux[(lam>(cents[i]-cent_range)) & (lam<(cents[i]+cent_range))] = interp1d(lam_cut, flux_cut, kind='linear', fill_value='extrapolate')(lam[(lam>(cents[i]-cent_range)) & (lam<(cents[i]+cent_range))])
                        print('Masking', lines[i], 'S/N=', np.max(flux_sub[(lam>(cents[i]-cent_range)) & (lam<(cents[i]+cent_range))])/np.mean(flux_err[(lam>(cents[i]-cent_range)) & (lam<(cents[i]+cent_range))]))
                    else:
                        print('Not Masking', lines[i], 'S/N=', np.max(flux_sub[(lam>(cents[i]-cent_range)) & (lam<(cents[i]+cent_range))])/np.mean(flux_err[(lam>(cents[i]-cent_range)) & (lam<(cents[i]+cent_range))]))
               # flux[(lam>(cents[i]-cent_range)) & (lam<(cents[i]+cent_range))] = filt[(lam>(cents[i]-cent_range)) & (lam<(cents[i]+cent_range))]
                plt.axvline(cents[i]-cent_range, ls='dashed', color='grey', lw=.5)
                plt.axvline(cents[i]+cent_range, ls='dashed', color='grey', lw=.5)


               # if (lines[i] == "C2H2" or lines[i]=="HCN"):
                  #  amp_lower = 0.0
                 #   amp_upper = -0.0
                  #  amp_l_br = 0.0
                   # amp_u_br = 0.0

                self.linecents.append(cents[i])
                self.linenames.append(lines[i])

                if (lines[i].startswith("H2")):
                    self.H2linecents.append(cents[i])
                    self.H2width_limits.append(cent_range)
                    self.H2_indices.append(np.where((lam>(cents[i]-cent_range)) & (lam<(cents[i]+cent_range))))                
       # print(self.H2linecents)
        #plt.plot(lam, flux_sub_orig, lw=0.2)
        #plt.errorbar(cents, 0.25*np.ones(len(cents)), ls='none', marker='.', color='green', ms=0.3)
        #plt.errorbar(self.linecents, mfs, ls='none', marker='.', color='tab:red', ms=0.7)
        plt.plot(lam, flux, lw=0.2)
        plt.savefig('LineCentTest.pdf')
        np.savetxt('LineCentTest.txt', np.transpose([lam, flux_sub_orig, flux_err]))
       # plt.show()








        # # Calcuate Correct Noise for Spectrum

        y_filt = scipy.signal.medfilt(np.copy(flux),  21)
        binsize =100
        N_bins = int(len(y_filt)/binsize)
        noise = np.empty(len(y_filt))
        for i in range(N_bins):
            noise[int(i*binsize):int(i*binsize + binsize)] = np.ones(binsize)*np.std(flux[int(i*binsize):int(i*binsize + binsize)] - y_filt[int(i*binsize):int(i*binsize + binsize)])
        noise[-binsize:] = noise[-binsize-1]
        p_fit = scipy.signal.medfilt(noise, int(binsize*10+1))
        p_fit = scipy.ndimage.gaussian_filter1d(p_fit, 250)#np.poly1d(np.polyfit(lam, p_fit/y_filt, 3))(lam) #scipy.ndimage.gaussian_filter1d(p_fit, 250)
        flux_err = p_fit#*y_filt


        if '_Template' in objName:
            flux_err[(lam>5.0) & (lam<7.5)] *= 1000

        # plt.figure(figsize=(15,5))
        # plt.errorbar(lam, flux, flux_err, ls='none', marker= '.', markeredgewidth=0.0, color='black', ms= 0.1)
        # plt.savefig('NoiseTest.pdf')
        # plt.show()
        # plt.close()


        #print(flux_err)


        # #Mask CO abs
        # if (min(lam)<4.7):
        # #     # flux_cut = np.concatenate((filt[(lam<(4.725-0.285))], filt[(lam>(4.725+0.285))])) 
        # #     # lam_cut =  np.concatenate((lam[(lam<(4.725-0.285))], lam[(lam>(4.725+0.285))])) 
        # #     # flux[(lam>(4.725-0.285)) & (lam<(4.725+0.285))] = interp1d(lam_cut, flux_cut, kind='linear', fill_value='extrapolate')(lam[(lam>(4.725-0.285)) & (lam<(4.725+0.285))])

        # flux_cut = np.concatenate((filt[(lam<(4.45))], filt[(lam>(5.5))])) 
        # lam_cut =  np.concatenate((lam[(lam<(4.45))], lam[(lam>(5.5))])) 
        # flux[(lam>(4.45)) & (lam<(5.5))] =  interp1d(lam_cut, flux_cut, kind='linear', fill_value='extrapolate')(lam[(lam>(4.45)) & (lam<(5.5))])#scipy.ndimage.gaussian_filter1d(flux[(lam>(4.41)) & (lam<(5.1))], 51)


        # flux_cut = np.concatenate((filt[(lam<(4.15))], filt[(lam>(4.33))])) 
        # lam_cut =  np.concatenate((lam[(lam<(4.15))], lam[(lam>(4.33))])) 
        # flux[(lam>(4.15)) & (lam<(4.33))] =  interp1d(lam_cut, flux_cut, kind='linear', fill_value='extrapolate')(lam[(lam>(4.15)) & (lam<(4.33))])#scipy.ndimage.gaussian_filter1d(flux[(lam>(4.41)) & (lam<(5.1))], 51)



     #   plt.close()
        # plt.figure(figsize=(15,5))
        # plt.plot(lam ,flux_orig, lw=0.2, color='black')
        # plt.plot(lam ,flux, lw=0.2, color='tab:red')
        # plt.plot(lam ,filt, lw=0.2, color='tab:purple', alpha=0.5)
        # for i in range(len(self.linecents)):
        #     plt.axvline(self.linecents[i] - np.array(mfs)[i] , lw=0.1)
        #     plt.axvline(self.linecents[i] + np.array(mfs)[i] , lw=0.1)
        # plt.savefig('LineCentTest.pdf')
        # plt.show()
        # plt.close()


        #plt.plot(lam ,filt, lw=0.2)

        #plt.show()
                
                
                
                
                

        # PAH features
        cents = [3.29, 3.395, 3.405, #3.40, 

        3.47, 5.175, 5.24, 
        5.45, 5.53,  
        5.64, 5.70, 5.76,
        5.87,
        6.0, 6.2, #6.30, 
        6.69,  7.1, 
        7.42, 7.55, 7.61, 7.82, 8.33, 8.5, 8.61, 10.60, 10.74,
         11.0,#11.04,
         11.20, #11.22,
         11.26, 11.25,
         11.99,
        # 12.69, 12.75, 
        12.6, 12.77, 13.15,
         13.55, 14.04, 
         14.19,  15.90, 16.45, 17.04, 17.375]#, 18.92]




        widths = [0.043, 0.00995, 0.02691, #0.031, 

        0.1, 0.05, 0.1, 
        0.15, 0.1, 
        0.1, 0.1, 0.1,
        0.15,
        0.2, 0.15, 
        0.4,  0.4, 
        0.935, 0.3, 0.1, 0.4, 0.2, 0.2, 0.336, 0.1, 0.1,
        0.1,
        0.1, 
        0.3, 0.2,
        0.54,
       #0.53, 0.12, 
        0.5, 0.15, 0.5,
        0.2, 0.2, 
        0.2,  0.318, 0.230, 1.108, 0.209]#, 0.5]

        asymm_pr = [0.52, 0.01, 0.01, #-10.0, 

        -0.8, 0.01, -3.0, 
        0.01, 0.01, 
        0.01, 0.01, 0.01,
        0.01, 
        0.01, -6, 
        0.01, 0.01, 
        0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, #(0.16)
        -1.1, 
        0.01, 
        -10.0, 0.01,
        0.01, 
        #1.8, 0.01,
        0.01, 0.01, 0.01, 
        -5.0, 0.01,
        -5.0, 0.01, 0.01, 0.01, 0.01]#, 10.0]#, 

        vals=[1.02049784e+00,1.61445514e-01,  1.61445514e-01, 1.11919928e-01, 1.00000000e-04,
       3.91156556e-01, 2.85140435e-02, 9.34855689e-03, 3.42602724e-01,
       4.03655576e-01, 2.89336300e-01, 1.83215171e-01, 7.72098763e-01,
       5.71966818e+00, 5.26999745e-01, 7.24688723e-01, 4.78873962e-01,
       4.24412068e+00, 1.84347745e+00, 7.25828768e+00, 1.29273645e+00,
       2.73093136e-01, 4.33143305e+00, 4.62446547e-01, 3.44995609e-01,
       1.14277385e+00, 4.87549187e+00, 9.73253531e+00, 1e-2,1.36019370e+00, 
       4.09349980e+00, 2.04890018e+00, 5.70992428e-01, 8.30259350e-01,
       3.06186582e-01, 7.44791244e-01, 1.88968165e-01, 1.97294262e+00,
       1.89570594e+00, 1.06383830e+00]
        cent_vals = [ 3.28943501,  3.395, 3.405, #3.40273685, 

         3.47694295,  5.18533981,  5.24326482,
        5.44536806,  5.53428643,  5.63543196,  5.69302699,  5.74934365,
        5.87123814,  6.01489636,  6.21472461,  6.70591766,  7.08235452,
        7.43655478,  7.560782  ,  7.61374321,  7.8267868 ,  8.30953977,
        8.49605243,  8.60238948, 10.58960401, 10.73957233, 11.01618187,
       11.19664681, 11.25751649, 11.25, 11.98139372, 12.57885785, 12.74709422,
       13.18283048, 13.52444972, 14.01275159, 14.21893892, 15.86544906,
       16.44343821, 17.03789296, 17.39272733]
        cent_3sig = [0.00365353, 0.00526731, 0.00700842, 0.00589258, 0.01149442,
       0.02949958, 0.02362638, 0.01731085, 0.00849338, 0.00909641,
       0.010298  , 0.00021863, 0.00312717, 0.00671888, 0.00030997,
       0.01020114, 0.03262956, 0.02403464, 0.0300232 , 0.00218542,
       0.05092131, 0.01095223, 0.01562484, 0.02232456, 0.01593271,
       0.02139681, 0.02639707, 0.025, 0.06538198, 0.04039679, 0.02518112,
       0.00010234, 0.03206979, 0.05672565, 0.01569095, 0.02567448,
       0.01613086, 0.09252194, 0.04761093]

        width_vals = [0.03961628, 0.00995, 0.02691,# 0.03359109, 
        0.10955406, 0.02246635, 0.0508955 ,
       0.07878673, 0.046191  , 0.06124202, 0.06942872, 0.07276268,
       0.06434912, 0.10106664, 0.14431082, 0.33466758, 0.40443953,
       0.37594713, 0.28948679, 0.09909238, 0.43759236, 0.21965121,
       0.1563172 , 0.33445923, 0.09904063, 0.08338632, 0.09803608,
       0.05605736, 0.17821722, 0.2, 0.41194933, 0.5268334 , 0.15004281,
       0.41596002, 0.21753518, 0.20454655, 0.19773274, 0.22136101,
       0.13604213, 0.95261695, 0.15425702]
        width_3sig = [0.00510483, 0.00093305, 0.00126951, 0.00056374, 0.03206734,
       0.07132892, 0.00983252, 0.03722295, 0.03274502, 0.04383078,
       0.00980545, 0.04209973, 0.02380731, 0.17376503, 0.23646113,
       0.01200266, 0.12988249, 0.04574837, 0.02296572, 0.0020436 ,
       0.15491928, 0.05749563, 0.06136432, 0.0864642 , 0.05029047,
       0.02894166, 0.05999254, 0.06, 0.42607294, 0.08629872, 0.09322867,
       0.27220214, 0.02308651, 0.07628964, 0.12043646, 0.27229986,
       0.07222395, 0.49086803, 0.12431081]

        asymm_vals = [ 9.22497776e-01,  5.79763989e-03,  5.79763989e-03, #-1.36078393e+01, 

        -1.17173057e+00,  6.17167586e-03,
       -4.39343547e+00,  5.82367209e-03,  5.79763989e-03,  5.79694266e-03,
        6.07091214e-03,  6.49173155e-03,  6.49916483e-03,  5.79223502e-03,
       -6.03254558e+00,  5.72099351e-03,  7.13708823e-03,  5.72091838e-03,
        5.59207931e-03,  6.47812331e-03,  5.31345850e-03,  1.28720985e-02,
        1.00024396e-02,  1.00447280e-02,  7.57695577e-03,  9.29273266e-03,
       -1.27234603e+00,  7.88760752e-03, -1.14621847e+01,  6.47376827e-03, 6.47376827e-03,
        1.46058146e-02,  1.43709527e-02,  5.71394712e-03, -2.57189717e+00,
        1.21230234e-02, -6.17433237e+00,  1.42784586e-02,  5.46642276e-03,
        5.29244729e-03,  7.13779216e-03]

        asymm_3sig = [2.62356513e+00, 3.41791328e+00, 2.74609693e-02, 3.72321637e-03,
       1.25635224e-01, 2.80655802e-04, 5.99465580e-05, 3.01373006e-05,
       2.74110564e-03, 6.94832652e-03, 6.99935092e-03, 6.72612301e-05,
       2.87884585e+00, 1.38858897e-04, 9.46335133e-03, 1.56938133e-04,
       3.89833021e-04, 6.98572759e-03, 3.77571032e-04, 9.53600332e-03,
       1.26180047e-02, 1.32290745e-02, 9.64632530e-03, 1.24229594e-02,
       1.45838756e+00, 1.09247476e-02, 9.81160089e+00,  7.45683805e-03, 7.45683805e-03,
       5.69927133e-04, 4.58303371e-04, 2.43019663e-04, 4.98112242e-01,
       1.09581045e-02, 3.83519948e+00, 2.94939181e-04, 3.73312011e-04,
       5.53321277e-04, 9.55146167e-03]

        #print(len(cents), len(vals))
        feats = ['{:.3f}'.format(x) for x in cents]
        self.Npah = 0
        self.pahcents=[]
        self.pahnames=[]
        for i in range(len(cents)):
            if (np.min(lam) <= cents[i] <= np.max(lam)):
                self.Npah +=1
                self.pahcents.append(cents[i])
               # if (cents[i]==12.77):
                #    self.parameters = self.parameters.append({ 'Section': 'PAH', 'Component': feats[i],'Name': "AMP("+"PAH"+feats[i]+")",'Description': 'PAH flux', 'Value': 1.0, '+Error': 0.0, '-Error': 0.0,'Prior': [0.8, 1.5],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
               # else:
                # if (cents[i]==6.2 or cents[i]==11.2 or cents[i] == 11.26 or cents[i] == 3.29 or cents[i] == 3.40):
                #     self.parameters = self.parameters.append({ 'Section': 'PAH', 'Component': feats[i],'Name': "AMP("+"PAH"+feats[i]+")",'Description': 'PAH flux', 'Value': vals[i]/10.0, '+Error': 0.0, '-Error': 0.0,'Prior': [0.0001, 100.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)#[0.001, 50.0]
                # else:
                #     self.parameters = self.parameters.append({ 'Section': 'PAH', 'Component': feats[i],'Name': "AMP("+"PAH"+feats[i]+")",'Description': 'PAH flux', 'Value': 0.000101, '+Error': 0.0, '-Error': 0.0,'Prior': [0.0001, 0.00011],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)#[0.001, 50.0]

                if (Cont_Only == True):# or cents[i] == 10.6 or cents[i]==10.74 or cents[i]==11.0):
                    self.parameters = self.parameters.append({ 'Section': 'PAH', 'Component': feats[i],'Name': "AMP("+"PAH"+feats[i]+")",'Description': 'PAH flux', 'Value': 0.000101, '+Error': 0.0, '-Error': 0.0,'Prior': [0.0001, 0.00011],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)#[0.001, 50.0]

                else:
                    self.parameters = self.parameters.append({ 'Section': 'PAH', 'Component': feats[i],'Name': "AMP("+"PAH"+feats[i]+")",'Description': 'PAH flux', 'Value': vals[i]/10.0, '+Error': 0.0, '-Error': 0.0,'Prior': [0.0001, 100.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)#[0.001, 50.0]
                

                self.parameters = self.parameters.append({ 'Section': 'PAH', 'Component': feats[i],'Name': "CENT("+"PAH"+feats[i]+")",'Description': 'PAH centre', 'Value': cent_vals[i], '+Error': 0.0, '-Error': 0.0,'Prior': [cents[i] - 0.0025*cents[i], cents[i] + 0.0025*cents[i]],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
                self.parameters = self.parameters.append({ 'Section': 'PAH', 'Component': feats[i],'Name': "FWHM("+"PAH"+feats[i]+")",'Description': 'PAH width', 'Value': width_vals[i], '+Error': 0.0, '-Error': 0.0,'Prior': [widths[i] - 0.1*widths[i], widths[i] + 0.1*widths[i]],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
                

               # print(asymm_vals[i])
                if (cents[i] == 3.29):
                    self.parameters = self.parameters.append({ 'Section': 'PAH', 'Component': feats[i],'Name': "ASYM("+"PAH"+feats[i]+")",'Description': 'PAH asymm', 'Value': asymm_vals[i], '+Error': 0.0, '-Error': 0.0,'Prior': [asymm_pr[i]-0.5*abs(asymm_pr[i]), asymm_pr[i]+10.0*abs(asymm_pr[i])],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)

                else:
                    self.parameters = self.parameters.append({ 'Section': 'PAH', 'Component': feats[i],'Name': "ASYM("+"PAH"+feats[i]+")",'Description': 'PAH asymm', 'Value': asymm_vals[i], '+Error': 0.0, '-Error': 0.0,'Prior': [asymm_pr[i]-0.5*abs(asymm_pr[i]), asymm_pr[i]+0.5*abs(asymm_pr[i])],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)

        print('Npah', self.Npah)
        # Continuum
        A_vals = [-6.67755505e-01, -9.84765631e-01, -1.47283802e+00, -2.33090007e+00, -4.18732033e+00, -5.53325122e+00, -5.22575974e+00, -4.78881924e+00, -5.28970031e+00, -7.41599353e+00, -9.46513113e+00, -9.80150445e+00, -1.00000000e+01, -1.00000000e+01, -1.00000000e+01, -1.00000000e+01, -1.00000000e+01, -9.95632173e+00, -9.90722865e+00, -1.00000000e+01, -6.93719063e-01, -1.00835119e+00, -1.49353218e+00, -2.34793418e+00, -4.18536559e+00, -5.51505951e+00, -5.23398336e+00, -4.80670103e+00, -5.29936839e+00, -6.76160898e+00, -7.68619100e+00, -1.00000000e+01, -1.00000000e+01, -1.00000000e+01, -1.00000000e+01, -1.00000000e+01, -9.32693948e+00, -1.00000000e+01, -9.90103961e+00, -1.00000000e+01, -7.20536118e-01, -1.03248011e+00, -1.51430543e+00, -2.36415423e+00, -4.18026507e+00, -5.49015992e+00, -5.22393888e+00, -4.78585200e+00, -5.19632194e+00, -6.64179942e+00, -8.48803709e+00, -8.43199476e+00, -1.00000000e+01, -1.00000000e+01, -1.00000000e+01, -1.00000000e+01, -1.00000000e+01, -1.00000000e+01, -1.00000000e+01, -1.00000000e+01, -7.48133586e-01, -1.05711530e+00, -1.53497951e+00, -2.37906903e+00, -4.17068347e+00, -5.45542138e+00, -5.19541577e+00, -4.74456999e+00, -5.10503093e+00, -6.60287741e+00, -8.65084076e+00, -1.00000000e+01, -9.38221077e+00, -1.00000000e+01, -1.00000000e+01, -1.00000000e+01, -1.00000000e+01, -9.81726104e+00, -9.84075543e+00, -1.00000000e+01, -7.75580692e-01, -1.08193426e+00, -1.55595327e+00, -2.39385772e+00, -4.15957069e+00, -5.41415527e+00, -5.16629831e+00, -4.70656703e+00, -5.01859344e+00, -7.01005246e+00, -8.11134389e+00, -9.33410775e+00, -9.30634710e+00, -1.00000000e+01, -1.00000000e+01, -1.00000000e+01, -1.00000000e+01, -1.00000000e+01, -9.99793481e+00, -1.00000000e+01, -8.00737895e-01, -1.10636632e+00, -1.57868500e+00, -2.41226589e+00, -4.15566232e+00, -5.36841520e+00, -5.14146206e+00, -4.68010558e+00, -4.86795326e+00, -6.37727244e+00, -8.70856926e+00, -8.49175821e+00, -8.94145318e+00, -9.03337740e+00, -1.00000000e+01, -1.00000000e+01, -1.00000000e+01, -9.59295107e+00, -9.89155641e+00, -1.00000000e+01, -8.20341893e-01, -1.12933556e+00, -1.60553585e+00, -2.44186372e+00, -4.17546460e+00, -5.31377368e+00, -5.08933092e+00, -4.64891815e+00, -4.81398372e+00, -6.22786594e+00, -8.45675112e+00, -8.15552413e+00, -1.00000000e+01, -1.00000000e+01, -1.00000000e+01, -1.00000000e+01, -9.32859960e+00, -1.00000000e+01, -9.88351298e+00, -1.00000000e+01, -8.31478734e-01, -1.14785020e+00, -1.63588523e+00, -2.49060334e+00, -4.25418543e+00, -5.27291116e+00, -5.00268680e+00, -4.48918766e+00, -4.43577644e+00, -5.99186494e+00, -7.15805364e+00, -8.35922540e+00, -9.99231201e+00, -8.98663520e+00, -1.00000000e+01, -1.00000000e+01, -1.00000000e+01, -9.46347092e+00, -9.64179969e+00, -9.99212006e+00, -8.35079489e-01, -1.15556203e+00, -1.65639788e+00, -2.55326637e+00, -4.50027156e+00, -5.34955077e+00, -4.89886986e+00, -4.15341236e+00, -3.88541577e+00, -5.58622099e+00, -7.04086299e+00, -8.52197987e+00, -9.48090522e+00, -1.00000000e+01, -1.00000000e+01, -9.86473158e+00, -9.42870994e+00, -1.00000000e+01, -9.96351724e+00, -9.87337922e+00, -8.40124165e-01, -1.14405330e+00, -1.62951973e+00, -2.55823175e+00, -5.35539571e+00, -5.67055499e+00, -4.76956020e+00, -3.98223723e+00, -3.63773472e+00, -4.61318676e+00, -6.72441625e+00, -8.27603022e+00, -8.64967875e+00, -1.00000000e+01, -1.00000000e+01, -1.00000000e+01, -1.00000000e+01, -1.00000000e+01, -9.53288461e+00, -9.99936417e+00, -8.64902413e-01, -1.11167392e+00, -1.50897438e+00, -2.30435200e+00, -6.34338965e+00, -5.39140895e+00, -4.79319358e+00, -4.18662880e+00, -3.71978379e+00, -4.07821034e+00, -6.41781861e+00, -7.10998745e+00, -7.58024461e+00, -8.38575555e+00, -8.59886435e+00, -1.00000000e+01, -9.14867570e+00, -1.00000000e+01, -9.33977576e+00, -9.54419237e+00, -9.32801905e-01, -1.07039327e+00, -1.30364781e+00, -1.74450527e+00, -2.51470697e+00, -3.47500376e+00, -4.32171358e+00, -4.64235546e+00, -4.68002241e+00, -3.94556402e+00, -5.24284398e+00, -6.79481914e+00, -7.63825405e+00, -7.72998200e+00, -9.95026639e+00, -8.16251086e+00, -8.57547183e+00, -1.00000000e+01, -9.09003941e+00, -9.37734268e+00, -1.06822326e+00, -1.03340582e+00, -1.07008555e+00, -1.23466304e+00, -1.54795911e+00, -1.99124136e+00, -2.50054871e+00, -3.22285956e+00, -9.46134635e+00, -2.76006686e+00, -2.57907307e+00, -3.20859406e+00, -6.55940719e+00, -6.71276020e+00, -6.50288612e+00, -9.93554378e+00, -9.50565100e+00, -8.35027872e+00, -1.00000000e+01, -1.00000000e+01, -1.29959244e+00, -1.01435254e+00, -8.59134393e-01, -8.40742874e-01, -9.49283686e-01, -1.15793808e+00, -1.41288666e+00, -1.64720827e+00, -1.75814248e+00, -1.65494002e+00, -1.70110502e+00, -2.07365928e+00, -2.83117487e+00, -3.70161997e+00, -4.37948943e+00, -5.81405921e+00, -6.88877396e+00, -7.82419129e+00, -8.84152638e+00, -9.48378675e+00, -1.66851780e+00, -1.02151666e+00, -6.93021405e-01, -5.49537220e-01, -5.40947298e-01, -6.29963645e-01, -7.70731550e-01, -9.07177116e-01, -1.00246182e+00, -1.08708791e+00, -1.23228105e+00, -1.52679355e+00, -2.03326645e+00, -2.76821361e+00, -3.79232381e+00, -8.56739202e+00, -6.36303961e+00, -8.78774128e+00, -1.00000000e+01, -8.04168676e+00, -2.24566155e+00, -1.05485690e+00, -5.78929599e-01, -3.46817646e-01, -2.65419782e-01, -2.94827025e-01, -4.02587646e-01, -5.48144901e-01, -6.97456567e-01, -8.56554773e-01, -1.03536646e+00, -1.29522789e+00, -1.73920458e+00, -2.58472906e+00, -9.92026528e+00, -2.51275962e+00, -2.29051297e+00, -3.34174470e+00, -9.96465359e+00, -6.52728627e+00, -3.15994512e+00, -1.10848619e+00, -5.19315093e-01, -2.23779524e-01, -9.56489153e-02, -9.97340226e-02, -2.24469395e-01, -4.50260819e-01, -7.13128050e-01, -9.31708765e-01, -1.08757395e+00, -1.30590395e+00, -1.73979774e+00, -2.52386643e+00, -3.19357320e+00, -2.39093184e+00, -2.19658373e+00, -7.35339048e+00, -9.28716472e+00, -7.65894047e+00, -4.52755077e+00, -1.17617572e+00, -5.13324043e-01, -1.72509832e-01, -1.38091771e-02, -1.05348497e-02, -1.79993240e-01, -5.80625871e-01, -1.20750684e+00, -1.43947154e+00, -1.42007310e+00, -1.55841090e+00, -2.03187755e+00, -3.04507984e+00, -5.32294012e+00, -6.68939310e+00, -3.04418491e+00, -3.00032477e+00, -2.59373185e+00, -2.10758712e+00, -5.21584205e+00, -1.25820345e+00, -5.52912170e-01, -1.78804613e-01, -6.87934904e-04, 0.00000000e+00, -2.12021638e-01, -8.44050338e-01, -8.99478482e+00, -2.94409944e+00, -2.15099099e+00, -2.19765127e+00, -2.83966572e+00, -5.36339537e+00, -6.94290970e+00, -7.21681953e+00, -7.05744133e+00, -6.68964924e+00, -5.89341955e+00, -3.00798860e+00, -5.43868786e+00, -1.35853674e+00, -6.19458124e-01, -2.16585398e-01, -2.41606055e-02, -2.98680149e-02, -2.76583260e-01, -9.02151847e-01, -2.35363789e+00, -5.10134026e+00, -5.97090836e+00, -6.25715380e+00, -6.79170484e+00, -7.13744948e+00, -7.36124568e+00, -7.58358547e+00, -7.78627235e+00, -7.89074456e+00, -7.94604086e+00, -7.94580598e+00]
        if (ExtType == 'Differential'):
            #Temps=[35.0, 40.0, 50.0, 65.0, 90.0, 135.0, 200.0, 300.0,  500.0, 5000.0]

            #self.grid_size = [20, 20]
            #for i in range(int(self.grid_size[0]*self.grid_size[1] +self.grid_size[1])):
            for i in range(int(self.grid_size[0]*self.grid_size[1])):

                 #self.parameters = self.parameters.append({ 'Section': 'Continuum', 'Component': 'Continuum','Name': "A"+str(i+1),'Description': 'Psi Value', 'Value': np.random.uniform(0.1   , 1.0), '+Error': 0.0, '-Error': 0.0,'Prior': [0.0001   , 1.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
                # if (i==19 or i==39 or i==59 or i ==79 or i ==99 or i == 119 or i ==139 or i ==159 or i ==179 or i ==199 or i ==219 or i== 239 or i== 259):
                #     self.parameters = self.parameters.append({ 'Section': 'Continuum', 'Component': 'Continuum','Name': "A"+str(i+1),'Description': 'Psi Value', 'Value': np.random.uniform(-9.999), '+Error': 0.0, '-Error': 0.0,'Prior': [-10.0   , -9.99],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
                # else:
                try:
                    self.parameters = self.parameters.append({ 'Section': 'Continuum', 'Component': 'Continuum','Name': "A"+str(i+1),'Description': 'Psi Value', 'Value': A_vals[i], '+Error': 0.0, '-Error': 0.0,'Prior': [-10.0   , 0.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
                except:
                    self.parameters = self.parameters.append({ 'Section': 'Continuum', 'Component': 'Continuum','Name': "A"+str(i+1),'Description': 'Psi Value', 'Value': -9.0, '+Error': 0.0, '-Error': 0.0,'Prior': [-10.0   , 0.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)


            self.parameters = self.parameters.append({ 'Section': 'Continuum', 'Component': 'Continuum','Name': "Scale",'Description': 'Scale Factor', 'Value': 99.24, '+Error': 0.0, '-Error': 0.0,'Prior': [1.0   , 1000.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)

            if (np.min(lam)>7.0 or St_Cont == True):
                stellar_est = 0.00011
            self.parameters = self.parameters.append({ 'Section': 'Stellar', 'Component': 'Stellar','Name': "Star Scale1",'Description': 'Scale Factor', 'Value': stellar_est/2.0, '+Error': 0.0, '-Error': 0.0,'Prior': [0.0001   , 50.0*stellar_est],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
            self.parameters = self.parameters.append({ 'Section': 'Stellar', 'Component': 'Stellar','Name': "Star Scale2",'Description': 'Scale Factor', 'Value': stellar_est/2.0, '+Error': 0.0, '-Error': 0.0,'Prior': [0.0001   ,  50.0*stellar_est],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
   
            if (min(lam)<4.5):
                self.parameters = self.parameters.append({ 'Section': 'Stellar', 'Component': 'Ices','Name': "Ice Frac",'Description': 'NIR Ices Frac', 'Value': 0.9, '+Error': 0.0, '-Error': 0.0,'Prior': [0.001, 1.5],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)

            #self.parameters = self.parameters.append({ 'Section': 'Stellar', 'Component': 'Stellar','Name': "VelocityDisp",'Description': 'Dispersion', 'Value': 100.0, '+Error': 0.0, '-Error': 0.0,'Prior': [10.0   , 300.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
            self.parameters = self.parameters.append({ 'Section': 'Stellar', 'Component': 'Stellar','Name': "Star Ext.",'Description': 'Stellar Extinction', 'Value': 0.5, '+Error': 0.0, '-Error': 0.0,'Prior': [0.0001   , 5.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)

        



        elif (ExtType == 'Differential_Parametric'):
            #Temps=[35.0, 40.0, 50.0, 65.0, 90.0, 135.0, 200.0, 300.0,  500.0, 5000.0]

            self.grid_size = [20, 20]
            for i in range(int(3*self.grid_size[1] +self.grid_size[1])):
                 self.parameters = self.parameters.append({ 'Section': 'Continuum', 'Component': 'Continuum','Name': "A"+str(i+1),'Description': 'Psi Value', 'Value': 0.5, '+Error': 0.0, '-Error': 0.0,'Prior': [0.00001   , 10.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
            
            self.parameters = self.parameters.append({ 'Section': 'Continuum', 'Component': 'Continuum','Name': "Scale",'Description': 'Scale Factor', 'Value': 0.5, '+Error': 0.0, '-Error': 0.0,'Prior': [0.001   , 1000.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)



        elif (ExtType == 'Screen' or ExtType == 'Mixed'):
            Temps =  np.concatenate((np.logspace(np.log10(20), np.log10(1500), 20), [5000.0]))
            NBBs = 20#len(Temps)
            for i in range(NBBs):
                self.parameters = self.parameters.append({ 'Section': 'Continuum', 'Component': 'Continuum','Name': "A"+str(i+1),'Description': 'BB Amp', 'Value': 1.0, '+Error': 0.0, '-Error': 0.0,'Prior': [0.0	, 1000.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
            self.parameters = self.parameters.append({ 'Section': 'Continuum', 'Component': 'Continuum','Name': "tau_9.8",'Description': 'Optical Depth', 'Value': .1, '+Error': 0.0, '-Error': 0.0,'Prior': [0.001	, 10.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
            self.parameters = self.parameters.append({ 'Section': 'Stellar', 'Component': 'Stellar','Name': "Star Scale1",'Description': 'Scale Factor', 'Value': 9.5, '+Error': 0.0, '-Error': 0.0,'Prior': [0.001   , 100.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
            self.parameters = self.parameters.append({ 'Section': 'Stellar', 'Component': 'Stellar','Name': "Star Scale2",'Description': 'Scale Factor', 'Value': 0.00101, '+Error': 0.0, '-Error': 0.0,'Prior': [0.001   , 0.0011],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
            #self.parameters = self.parameters.append({ 'Section': 'Stellar', 'Component': 'Stellar','Name': "VelocityDisp",'Description': 'Dispersion', 'Value': 100.0, '+Error': 0.0, '-Error': 0.0,'Prior': [10.0   , 300.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
            self.parameters = self.parameters.append({ 'Section': 'Stellar', 'Component': 'Stellar','Name': "Star Ext.",'Description': 'Stellar Extinction', 'Value': 0.5, '+Error': 0.0, '-Error': 0.0,'Prior': [0.0001   , 5.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)






        # Ices and Silicate Emission Parameters
        if (Ices_6micron==True):
            upper= 3.0
        else:
            upper = 0.0011   

        print(upper)
        self.parameters = self.parameters.append({ 'Section': 'Extinction', 'Component': 'Ice','Name': "\u03C4_Ice",'Description': 'Ice Opt Depth', 'Value': 1.0, '+Error': 0.0, '-Error': 0.0,'Prior': [0.001,upper],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
        self.parameters = self.parameters.append({ 'Section': 'Extinction', 'Component': 'Ices','Name': "\u03C4_CH",'Description': 'CH Opt Depth', 'Value': 0.1, '+Error': 0.0, '-Error': 0.0,'Prior': [0.001, 1.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)


        if (min(lam)<4.5):
            self.parameters = self.parameters.append({ 'Section': 'Extinction', 'Component': 'Ices','Name': "H2O",'Description': '3 MicronIce Opt Depth', 'Value': 1.0, '+Error': 0.0, '-Error': 0.0,'Prior': [0.01, 10.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
            self.parameters = self.parameters.append({ 'Section': 'Extinction', 'Component': 'Ices','Name': "CO2",'Description': 'CO2 Opt Depth', 'Value': 1.0, '+Error': 0.0, '-Error': 0.0,'Prior': [0.01, 10.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
            if (Fit_NIR_CH == True):
                CH_up_lim = 3.0
            else:
                CH_up_lim = 0.0101
            self.parameters = self.parameters.append({ 'Section': 'Extinction', 'Component': 'Ices','Name': "CH_NIR",'Description': 'CH Op Depth', 'Value': 0.01005, '+Error': 0.0, '-Error': 0.0,'Prior': [0.01, CH_up_lim],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
            self.parameters = self.parameters.append({ 'Section': 'Extinction', 'Component': 'Ices','Name': "COa_x0",'Description': 'Ice Peak', 'Value': 4.6, '+Error': 0.0, '-Error': 0.0,'Prior': [4.55, 4.65],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
            self.parameters = self.parameters.append({ 'Section': 'Extinction', 'Component': 'Ices','Name': "COa_fwhm",'Description': 'Ice fwhm', 'Value': 0.1, '+Error': 0.0, '-Error': 0.0,'Prior': [0.05, 0.25],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
            self.parameters = self.parameters.append({ 'Section': 'Extinction', 'Component': 'Ices','Name': "COa_a",'Description': 'Ice Asym', 'Value': 4, '+Error': 0.0, '-Error': 0.0,'Prior': [-5.0, 10.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
           # self.parameters = self.parameters.append({ 'Section': 'Extinction', 'Component': 'Ices','Name': "COb_x0",'Description': 'Ice Peak', 'Value': 4.72, '+Error': 0.0, '-Error': 0.0,'Prior': [4.7, 4.75],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
           # self.parameters = self.parameters.append({ 'Section': 'Extinction', 'Component': 'Ices','Name': "COb_fwhm",'Description': 'Ice fwhm', 'Value': 0.1, '+Error': 0.0, '-Error': 0.0,'Prior': [0.05, 0.25],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
#
            self.parameters = self.parameters.append({ 'Section': 'Extinction', 'Component': 'Ices','Name': "COc_x0",'Description': 'Ice Peak', 'Value': 4.8, '+Error': 0.0, '-Error': 0.0,'Prior': [4.7, 4.9],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
            self.parameters = self.parameters.append({ 'Section': 'Extinction', 'Component': 'Ices','Name': "COc_fwhm",'Description': 'Ice fwhm', 'Value': 0.1, '+Error': 0.0, '-Error': 0.0,'Prior': [0.05, 0.35],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
            self.parameters = self.parameters.append({ 'Section': 'Extinction', 'Component': 'Ices','Name': "COc_a",'Description': 'Ice Asym', 'Value': -4, '+Error': 0.0, '-Error': 0.0,'Prior': [-10, .0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)

            if (Fit_CO == True):
                CO_up_lim = 5.0
            else:
                CO_up_lim = 0.0101
            print(CO_up_lim)
            self.parameters = self.parameters.append({ 'Section': 'Extinction', 'Component': 'Ices','Name': "tau_COa",'Description': 'CO Depth', 'Value': 0.02, '+Error': 0.0, '-Error': 0.0,'Prior': [0.01, CO_up_lim],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
            #self.parameters = self.parameters.append({ 'Section': 'Extinction', 'Component': 'Ices','Name': "tau_COa",'Description': 'Ice Asym', 'Value': 0.02, '+Error': 0.0, '-Error': 0.0,'Prior': [0.01, 0.03],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)

            #self.parameters = self.parameters.append({ 'Section': 'Extinction', 'Component': 'Ices','Name': "tau_COb",'Description': 'Ice Asym', 'Value': 1.0, '+Error': 0.0, '-Error': 0.0,'Prior': [0.01, 5.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
            self.parameters = self.parameters.append({ 'Section': 'Extinction', 'Component': 'Ices','Name': "tau_COc",'Description': 'CO Depth', 'Value': 0.02, '+Error': 0.0, '-Error': 0.0,'Prior': [0.01, CO_up_lim],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
            #self.parameters = self.parameters.append({ 'Section': 'Extinction', 'Component': 'Ices','Name': "tau_COc",'Description': 'Ice Asym', 'Value': 0.02, '+Error': 0.0, '-Error': 0.0,'Prior': [0.01, 0.03],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)

            # self.parameters = self.parameters.append({ 'Section': 'Extinction', 'Component': 'Ices','Name': "H2O_x0_NIR",'Description': 'Ice Peak', 'Value': 3.0, '+Error': 0.0, '-Error': 0.0,'Prior': [2.9, 3.1],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
            # self.parameters = self.parameters.append({ 'Section': 'Extinction', 'Component': 'Ices','Name': "H2O_fwhm_NIR",'Description': 'Ice fwhm', 'Value': 0.5, '+Error': 0.0, '-Error': 0.0,'Prior': [0.3, 0.7],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
            # self.parameters = self.parameters.append({ 'Section': 'Extinction', 'Component': 'Ices','Name': "H2O_a_NIR",'Description': 'Ice Asym', 'Value': -4, '+Error': 0.0, '-Error': 0.0,'Prior': [-10.0, -2.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)

            self.parameters = self.parameters.append({ 'Section': 'Extinction', 'Component': 'Ices','Name': "CO2_x0",'Description': 'Ice Peak', 'Value': 4.265, '+Error': 0.0, '-Error': 0.0,'Prior': [4.26, 4.275],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
            self.parameters = self.parameters.append({ 'Section': 'Extinction', 'Component': 'Ices','Name': "CO2_fwhm",'Description': 'Ice fwhm', 'Value': 0.035, '+Error': 0.0, '-Error': 0.0,'Prior': [0.005, 0.05],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
            self.parameters = self.parameters.append({ 'Section': 'Extinction', 'Component': 'Ices','Name': "CO2_a",'Description': 'Ice Asym', 'Value': 10, '+Error': 0.0, '-Error': 0.0,'Prior': [0.0, 30.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)



          #  self.parameters = self.parameters.append({ 'Section': 'Extinction', 'Component': 'Ices','Name': "CO_1",'Description': 'CO Opt Depth', 'Value': 1.0, '+Error': 0.0, '-Error': 0.0,'Prior': [0.01, 10.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
            # self.parameters = self.parameters.append({ 'Section': 'Extinction', 'Component': 'Sil','Name': "CO_2",'Description': 'CO Opt Depth', 'Value': 1.0, '+Error': 0.0, '-Error': 0.0,'Prior': [0.01, 10.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
            #  self.parameters = self.parameters.append({ 'Section': 'Extinction', 'Component': 'Sil','Name': "CO_3",'Description': 'CO Opt Depth', 'Value': 1.0, '+Error': 0.0, '-Error': 0.0,'Prior': [0.01, 10.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
               

        self.parameters = self.parameters.append({ 'Section': 'Extinction', 'Component': 'H2O','Name': "H2O_b",'Description': 'Ice Boxiness', 'Value': 2.0, '+Error': 0.0, '-Error': 0.0,'Prior': [1.5, 10.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
        self.parameters = self.parameters.append({ 'Section': 'Extinction', 'Component': 'H2O','Name': "H2O_x0",'Description': 'Ice Peak', 'Value': 6.05, '+Error': 0.0, '-Error': 0.0,'Prior': [6.0, 6.15],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
        self.parameters = self.parameters.append({ 'Section': 'Extinction', 'Component': 'H2O','Name': "H2O_fwhm",'Description': 'Ice fwhm', 'Value': 0.7, '+Error': 0.0, '-Error': 0.0,'Prior': [0.6, 1.5],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
        self.parameters = self.parameters.append({ 'Section': 'Extinction', 'Component': 'H2O','Name': "H2O_a",'Description': 'Ice Asym', 'Value': -4, '+Error': 0.0, '-Error': 0.0,'Prior': [-6.0, -2.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
        # self.parameters = self.parameters.append({ 'Section': 'Noise', 'Component': 'Noise Model','Name': "sig",'Description': 'Addidative Noise', 'Value': 0, '+Error': 0.0, '-Error': 0.0,'Prior': [0.0, 100.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
        # self.parameters = self.parameters.append({ 'Section': 'Noise', 'Component': 'Noise Model','Name': "logN/S",'Description': 'Noise to Signal', 'Value': 0, '+Error': 0.0, '-Error': 0.0,'Prior': [-4, 1],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)


        #Extract sampled parameters
        sampled_parameters = self.parameters.loc[(self.parameters["Fixed"] == False)]
        init_pos = sampled_parameters['Value'].to_numpy()
        
        self.Npar = sampled_parameters.shape[0]
        self.nchains = 5
        self.Nwalkers = int(2.0*self.Npar + 1.0)
        
        # Generate arrays and indices for passing into prob funcs
        self.ps = self.parameters["Value"].to_numpy()
        self.priors = self.parameters["Prior"].to_numpy()
        # Section indices
        self.stellar_indx = self.parameters.index[self.parameters["Section"] == "Stellar"].to_numpy()
        self.pah_indx = self.parameters.index[self.parameters["Section"] == "PAH"].to_numpy()
        self.cont_indx = self.parameters.index[self.parameters["Section"] == "Continuum"].to_numpy()
        self.ext_indx = self.parameters.index[self.parameters["Section"] == "Extinction"].to_numpy()
        self.scale_indx = self.parameters.index[self.parameters["Section"] == "ChnScales"].to_numpy()
        self.Nuc_indx = self.parameters.index[self.parameters["Section"] == "Nuclear"].to_numpy()


        self.sampled_indx = self.parameters.index[self.parameters["Fixed"] == False].to_numpy()
        
        self.fixed =self.parameters["Fixed"].to_numpy()
        
        self.pos=init_pos

        global T, tau



        T = jnp.logspace(jnp.log10(35), jnp.log10(1500), grid_size[0])
        tau = jnp.logspace(jnp.log(0.05), jnp.log(tau_lim), grid_size[1], base=jnp.exp(1))

        #Calculate blackbody normilisation over data wavelength range
        #norm[-1] = np.trapz(np.interp(x, Stellar_x, Stellar_y))
        global norm, norm_, norm2, norm1D
        norm1D = np.zeros(len(T))
        norm = np.empty((grid_size[1], grid_size[0]))
        norm2 = np.empty((grid_size[1], grid_size[0]))
        x = np.linspace(1.5, 28.0, 1000)
        x_ =x# np.linspace(4.8, 28.0, 1000)
        Tau = np.interp(x, ls, S) #+ np.interp(x, lss, SS)
        for i in range(len(T)):
            for j in range(len(tau)):
                B = B_nu(x, 1.0, T[i], 0.01)
                ext = np.exp(-tau[j]*Tau)
                norm[j, i] = np.trapz(B*ext, x_)
                norm2[j, i] = np.trapz(B, x)

            norm1D[i] = np.trapz(B, x)

        norm_ = np.concatenate((-1.0*norm[::-1, :], norm), axis=0)



        self.norm_ratio = norm/norm2
        
        self.data = [lam, flux, flux_err]
        self.flux_orig = flux_orig
        self.width_limits = mfs
        self.samples_flat = None



        
        #print(self.pah_indx[::4])
####################################################################### Functions ###############################################################




def complexe_modulo(z):
    a = z.real
    b = z.imag
    return a**2+b**2
    




        
        
drude_params = [[3.05, 0.45], [4.265, 0.033], [4.55, 0.2], [4.82, 0.2]]#, [4.67, 0.2]]
def Ext(lam, ext_params, cont_params, jax=False):
   # tau_ice =1.0# ext_params[0]*ext_params[1]
    #tau_CH = 1.0#ext_params[0]*ext_params[1]
    tau_ice =ext_params[0]#*ext_params[1]
    tau_CH =ext_params[1]#*ext_params[1]
    #delta_p = ext_params[2]
    # tau_H2O = ext_params[1]
    # tau_CO2 = ext_params[2]
    # tau_CO_1 = ext_params[3]
    # tau_CO_2 = ext_params[4]
    # tau_CO_3 = ext_params[5]



    ice = ModifiedGauss2(lam, ext_params[-3], ext_params[-2],  1.0, ext_params[-1], ext_params[-4], jax=True)
    #ice = jnp.interp(lam, ls_H20_Template, H20_Template)/jnp.interp(6.0, ls_H20_Template, H20_Template)


   # alpha = ext_params[5]
   # beta = ext_params[6]

   # tau_98 = -1.0*jnp.log(ReturnExt(jnp.array([9.8]), cont_params))
    #tau_ice *= tau_98
    tau_CH *= tau_ice
    #tau_9 = ext_params[0]
    tau_H2O = ext_params[2]

    ext_ice = jnp.exp(-tau_ice*ice)
    ext_CH = jnp.exp(-tau_CH*jnp.interp(lam, ls_C, CH)/jnp.interp(6.85, ls_C, CH))
    #ext_H2O = jnp.exp(-tau_H2O*jnp.interp(lam, ls_H, H)/jnp.interp(13.5, ls_H, H))

    ext_NIR=1.0
    # for i in range(int(len(ext_params)-3)):
    #     ext_NIR*=jnp.exp(-ext_params[int(i+2)]*Drude(lam, drude_params[i][0], drude_params[i][1], 1.0))

    #if (len(ext_params)>5):
    if (len(ext_params)>6):

        # ext_NIR*=jnp.exp(-ext_params[2]*Drude(lam, drude_params[0][0], drude_params[0][1], 1.0))
        # ext_NIR*=jnp.exp(-ext_params[3]*Drude(lam, drude_params[1][0], drude_params[1][1], 1.0))
        # ext_NIR*=jnp.exp(-ext_params[4]*Drude(lam, drude_params[2][0], drude_params[2][1], 1.0))
        # ext_NIR*=jnp.exp(-ext_params[4]*Drude(lam, drude_params[3][0], drude_params[3][1], 1.0))

        ext_NIR *= jnp.exp(-ext_params[2]*jnp.interp(lam, ls_NIR_Ice, NIR_Ice))
        #ext_NIR *= jnp.exp(-ext_params[2]*ModifiedGauss(lam, ext_params[13], ext_params[14],  1.0, ext_params[15], jax=True))
        ext_NIR *= jnp.exp(-ext_params[3]*jnp.interp(lam, ls_NIR_CO2, NIR_CO2))
        #ext_NIR *= jnp.exp(-ext_params[3]*ModifiedGauss(lam, ext_params[13], ext_params[14],  1.0, ext_params[15], jax=True))


        ext_NIR *= jnp.exp(-ext_params[4]*jnp.interp(lam, ls_C_NIR, CH_NIR))

        ext_NIR*= jnp.exp(-ext_params[11]*ModifiedGauss(lam, ext_params[5], ext_params[6],  1.0, ext_params[7], jax=True))
        #ext_NIR*= jnp.exp(-ext_params[13]*ModifiedGauss(lam, ext_params[7], ext_params[8],  1.0, 0.0, jax=True))
        ext_NIR*= jnp.exp(-ext_params[12]*ModifiedGauss(lam, ext_params[8], ext_params[9],  1.0, ext_params[10], jax=True))

        
    # ext_H2O = jnp.exp(-tau_H2O*Drude(lam, 3.05, 0.45, 1.0))
    # ext_CO2 = jnp.exp(-tau_CO2*Drude(lam, 4.265, 0.033, 1.0))
    # ext_CO = jnp.exp(-tau_CO_1*Drude(lam, 4.55, 0.2, 1.0))*jnp.exp(-tau_CO_2*Drude(lam, 4.82, 0.2, 1.0))*jnp.exp(-tau_CO_3*Drude(lam, 4.67, 0.2, 1.0))



   #  As = jnp.linspace(0.01, 10.0, 100)
   #  A_dist = ((As/tau_9)**alpha + (As/tau_9)**(-beta))**(-1.0)
   #  #A_dist = jnp.exp(-0.5*((As-tau_9)/alpha)**2)
   #  #A_dist = jnp.exp(-0.5*((As-tau_9)/0.01)**2)
   #  #A_dist /= jnp.trapz(A_dist, As)
  
   #  op_depth =jnp.interp(lam, ls, S) + jnp.interp(lam, lss, SS)

   #  #ext_S =jnp.exp(-op_depth*tau_9)
   #  #n = jnp.average(jnp.exp(-op_depth*As.reshape(len(As),1)), weights = A_dist, axis=0)
   # # ext_S = jnp.average(jnp.exp(-op_depth*As.reshape(len(As),1)), weights = A_dist, axis=0)
   # # ext_S = beta*jnp.exp(-op_depth*tau_9) + (1.0-beta)*jnp.exp(-op_depth*alpha)
   #  if (jax==True):
   #      ext_S = jnp.sum(A_dist*jnp.exp(-op_depth*As))/jnp.sum(A_dist)
   #  else:
   #      ext_S = jnp.average(jnp.exp(-op_depth*As.reshape(len(As),1)), weights = A_dist, axis=0)

   #  #ext_S = jnp.sum(A_dist*jnp.exp(-op_depth*As.reshape(len(As),1)), axis=0)/jnp.sum(A_dist)
   #  #jaxx.debug.print(" ext_S = {x}", x=ext_S)
   #  #jaxx.debug.print(" ext_ice = {x}", x=ext_ice)


    full_ext = ext_ice*ext_CH*ext_NIR#ext_H2O*ext_CO2*ext_CO
    #full_ext = ext_S
    return full_ext, 1.0#, ext_S#, [As, A_dist]#, tau_9*(1. - beta)*SilProfile, op_depth#, tau_9*(1. - beta), tau_ice/(ice_np(6.0)+CH_np(6.0)), Psi




# def Nuclear(lam, nuc_params):
#     tau_ice = nuc_params[0]
#     tau_CH = nuc_params[0]
#     tau_9 = nuc_params[1]
#     ext_ice = jnp.exp(-tau_ice*jnp.interp(lam, ls_i, ice))
#     ext_CH = jnp.exp(-tau_CH*jnp.interp(lam, ls_C, CH))
#     #op_depth =tau_9*(jnp.interp(lam, ls, S) + jnp.interp(lam, lss, SS))
#     op_depth = tau_9*jnp.interp(lam, ls, S)
#     ext_S =jnp.exp(-op_depth)



#     Amps = nuc_params[2:]

#    # xknots = np.linspace(4.8, 25.0, len(Amps))
#    # xknots = np.logspace(np.log10(4.8), np.log10(25.0), len(Amps))
 
#     #try:
#        # cont = IUS(xknots, Amps, k=3)(lam)
#     #except:
#         #cont = IUSScipy(xknots, Amps, k=3)(lam)
#     cont = DustCont2(lam, Amps, jax=True)


#     return cont*ext_S*ext_ice*ext_CH




# def Ext2(lam, ext_params, jax=False):
#    # tau_ice = ext_params[1]
#     #tau_CH = ext_params[1]
#     tau_9 = ext_params[0]

#     if (jax==False):
#         if (tau_9 == 0.0):
#             full_ext = np.ones(len(lam))
#         else:
#             full_ext = (1.0 - np.exp(-tau_9*S_np(lam)))/(tau_9*S_np(lam))
#         #full_ext = np.exp(-tau_9*S_np(lam))
#         return full_ext#, ext_ice*ext_CH, ext_S, tau_9*(1. - beta), tau_ice/(ice_np(6.0)+CH_np(6.0)), Psi
#     else:
#         full_ext = (1.0 - jnp.exp(-tau_9*jnp.interp(lam, ls, S)))/(tau_9*jnp.interp(lam, ls, S))
#         #full_ext = np.exp(-tau_9*S_np(lam))
#         return full_ext#, ext_ice*ext_CH, ext_S, tau_9*(1. - beta), tau_ice/(ice_np(6.0)+CH_np(6.0)), Psi
        
        
        
        
    
    

def ScaleModule(l, f, e, params, SL2Mask):
    scale = params[-1]
    fl = np.empty(len(l))
    el = np.empty(len(l))
    fl[SL2Mask] = scale*f[SL2Mask]
    el[SL2Mask] = scale*e[SL2Mask]
    fl[~SL2Mask] = f[~SL2Mask]
    el[~SL2Mask] = e[~SL2Mask]

    return fl, el







  
def Gauss(X,  l0, fwhm, A, jax=False):
    sig = fwhm/2.355
    
    if (jax==False):
        return A*np.exp(-0.5*((X-l0)/sig)**2)
    else:
        return A*jnp.exp(-0.5*((X-l0)/sig)**2)


def ModifiedGauss(lam, x0, fwhm,  Amp, a, jax=False):

    if (jax == False):
        fwhm = 2.0*fwhm/(1.0+np.exp(a*(lam-x0)))
        sig = fwhm/2.355
        return Amp*np.exp(-0.5*((lam-x0)/sig)**2)
    else:
        fwhm = 2.0*fwhm/(1.0+jnp.exp(a*(lam-x0)))
        sig = fwhm/2.355
        return Amp*jnp.exp(-0.5*((lam-x0)/sig)**2)

def ModifiedGauss2(lam, x0, fwhm,  Amp, a, b, jax=False):

    if (jax == False):
        fwhm = 2.0*fwhm/(1.0+np.exp(a*(lam-x0)))
        sig = fwhm/2.355
        return Amp*np.exp(-0.5*(abs(lam-x0)/sig)**b)
    else:
        fwhm = 2.0*fwhm/(1.0+jnp.exp(a*(lam-x0)))
        sig = fwhm/2.355
        return Amp*jnp.exp(-0.5*(abs(lam-x0)/sig)**b)


def Drude(lam, x0, fwhm, A):
    gamma = fwhm/x0
    
    return A*(gamma**2)/((((lam/x0)- (x0/lam))**2) + gamma**2)

def ModifiedDrude(lam, x0, w, A, a):

    fwhm = 2.0*w/(1.0+jnp.exp(a*(lam-x0)))
    gamma = fwhm/x0
    return A*(gamma**2)/((((lam/x0)- (x0/lam))**2) + gamma**2)#/((gamma**2)#/((((9.8/x0)- (x0/9.8))**2) + gamma**2))
    

# Normalise Emissivity components
# x_ = np.linspace(1.5, 30, 1000)
# em1 = (x_**(-2))/np.trapz(y=x_**(-2), x=x_)
# em2 = (np.interp(x_, lsss, SSS))/np.trapz(y=np.interp(x_, lsss, SSS), x=x_)



# def B_nu(x, A, T,alpha_sil,jax=False):

    
#     c=299792458.0
#     h=6.62607004e-34
#     k = 1.38064852e-23

        
#     x = x*1e-6 #micron to metres
#     l_peak = 2.897771e-3/T
#     nu = c/x
#     nu_peak = c/l_peak

#     if (jax==False):
#         norm = ((nu_peak**3)/(np.exp(h*nu_peak/(k*T))-1.0))*((1./l_peak)**2)
#         return (A* (nu**3)/(np.exp(h*nu/(k*T))-1.0))*((1./x)**2)/norm # Return in Jy
#     else:
#         norm = ((nu_peak**3)/(jnp.exp(h*nu_peak/(k*T))-1.0))*((1./l_peak)**2)
#         return (A* (nu**3)/(jnp.exp(h*nu/(k*T))-1.0))*((1./x)**2)/norm # Return in J


def B_nu(x, A, T, alpha_sil, jax=False):
    
    c=299792458.0
    h=6.62607004e-34
    k = 1.38064852e-23

   # em = (1.0-alpha_sil)*jnp.interp(x, x_, em1) + alpha_sil*jnp.interp(x, x_, em2)
    em = jnp.interp(x, ls_Em, Em)

    x = x*1e-6 #micron to metres
    l_peak = 2.897771e-3/T
    nu = c/x
    nu_peak = c/l_peak


    if (jax==False):
        normmm = ((nu_peak**3)/(np.exp(h*nu_peak/(k*T))-1.0))
        return (A* (nu**3)/(np.exp(h*nu/(k*T))-1.0))/normmm   * em
    else:
        normmm = ((nu_peak**3)/(jnp.exp(h*nu_peak/(k*T))-1.0)) 
        return (A* (nu**3)/(jnp.exp(h*nu/(k*T))-1.0))/normmm * em 


def B_n(x, A, T, jax=False):
    
    c=299792458.0
    h=6.62607004e-34
    k = 1.38064852e-23

        
    x = x*1e-6 #micron to metres
    l_peak = 2.897771e-3/T
    nu = c/x
    nu_peak = c/l_peak

    if (jax==False):
        normmm = ((nu_peak**3)/(np.exp(h*nu_peak/(k*T))-1.0))
        return (A* (nu**3)/(np.exp(h*nu/(k*T))-1.0))/normmm # Return in Jy
    else:
        normmm = ((nu_peak**3)/(jnp.exp(h*nu_peak/(k*T))-1.0))
        return (A* (nu**3)/(jnp.exp(h*nu/(k*T))-1.0))/normmm # Return in Jy
    
    


def MixedCont(lam, cont_params, ExtType, ext_params, jax=False):


    alpha_sil = ext_params[-3]

    if (ExtType==0.0):
        #Psi = jnp.reshape(jnp.exp(cont_params[:-1]), (grid_size[1], grid_size[0]+1))
        Psi = jnp.reshape(jnp.exp(cont_params[:-1]), (grid_size[1], grid_size[0]))


        # Add dummy Psi at constant minimum value with negative ext. to balance

        # Psi_dummy = jnp.full((20, 20), jnp.exp(-10.0))
        # tau_dummy = tau[::-1]
        # Psi = jnp.concatenate((Psi_dummy, Psi), axis=0)

        # tau_ = jnp.concatenate((tau_dummy, tau))

        if (jax==True):

            #B = jnp.zeros(len(T)+1)
            B = jnp.zeros(len(T))

            #B = B.at[:len(T)].set(B_nu(lam, 1.0, T, alpha_sil, jax=True)/jnp.interp(T, Ts[:-1], norm[:-1]))
            B = B.at[:len(T)].set(B_nu(lam, 1.0, T, alpha_sil, jax=True))

            #B = B.at[-1].set( B_n(lam, 1.0, 5000.0, jax=True)/norm[-1])
           # B = B.at[-1].set( jnp.interp(lam, Stellar_x, Stellar_y)/norm[-1])


            Tau = jnp.interp(lam, ls, S) #+ jnp.interp(lam, lss, SS)
            ext = jnp.exp(-tau*Tau)

            product = jnp.einsum('i,k->ki',B,ext)
            cont = jnp.average(product/norm ,weights = Psi/jnp.sum(Psi))

        else:
            #B = np.zeros((len(T)+1, len(lam)))
            B = np.zeros((len(T), len(lam)))

            B[:len(T),:] = B_nu(lam, 1.0, T.reshape(len(T),1), alpha_sil)

            #B[-1,:] =  B_n(lam, 1.0, 5000.0)/norm[-1]
           # B[-1,:] = np.interp(lam, Stellar_x, Stellar_y)/norm[-1]



            Tau = np.interp(lam, ls, S) #+ np.interp(lam, lss, SS)
            ext = np.exp(-tau.reshape(len(tau),1)*Tau)

            product = np.einsum('ij,kj->kij',B,ext)

            cont = np.empty(len(lam))
            for i in range(len(lam)):
                cont[i] = np.average(product[:,:,i]/norm, weights = Psi/jnp.sum(Psi))




        # Calculate cont at 15 microns for normalisation
        # B = jnp.zeros(len(T))
        # B = B.at[:len(T)].set(B_nu(15.0, 1.0, T, alpha_sil, jax=True))
        # Tau = jnp.interp(15.0, ls, S) #+ jnp.interp(lam, lss, SS)
        # ext = jnp.exp(-tau*Tau)
        # product = jnp.einsum('i,k->ki',B,ext)
        # cont_norm = jnp.average(product/norm ,weights = Psi/jnp.sum(Psi))

        return cont*cont_params[-1]


    # elif (ExtType==-1.0):




    elif (ExtType == 1.0):

        if (jax==True):
            BB_Amps = cont_params[:-1]
            tau_98 = cont_params[-1]

            Tau = jnp.interp(lam, ls, S) #+ jnp.interp(lam, lss, SS)
            ext = jnp.exp(-tau_98*Tau)
            #B = jnp.zeros(len(T)+1)
            B = jnp.zeros(len(T))
            B = B.at[:len(T)].set(B_nu(lam, 1.0, T, alpha_sil , jax=True)/jnp.interp(T, T, norm1D))

            #B = B.at[-1].set( B_n(lam, 1.0, 5000.0, jax=True)/norm[-1])
            #B = B.at[-1].set( jnp.interp(lam, Stellar_x, Stellar_y)/norm[-1])

            cont = ext*jnp.sum(B*BB_Amps)

        else:
            BB_Amps = cont_params[:-1]
            tau_98 = cont_params[-1]

            Tau = np.interp(lam, ls, S) #+ np.interp(lam, lss, SS)
            ext = np.exp(-tau_98*Tau)

            #B = np.zeros((len(T)+1, len(lam)))
            B = np.zeros((len(T), len(lam)))

            B[:len(T),:] = B_nu(lam, 1.0, T.reshape(len(T),1), alpha_sil )/(jnp.interp(T, T, norm1D).reshape(len(T),1))

            #B[-1,:] =  B_n(lam, 1.0, 5000.0)/norm[-1]
           # B[-1,:] = np.interp(lam, Stellar_x, Stellar_y)/norm[-1]



            cont = np.empty(len(lam))
            for i in range(len(lam)):
                cont[i] = ext[i]*jnp.sum(B[:,i]*BB_Amps)


        return cont

    elif (ExtType == 2.0):

        if (jax==True):
            BB_Amps = cont_params[:-1]
            tau_98 = cont_params[-1]

            Tau = jnp.interp(lam, ls, S) #+ jnp.interp(lam, lss, SS)

            ext = (1.0 - jnp.exp(-tau_98*Tau))/(tau_98*Tau)

            #B = jnp.zeros(len(T)+1)
            B = jnp.zeros(len(T))

            B = B.at[:len(T)].set(B_nu(lam, 1.0, T, alpha_sil ,jax=True)/jnp.interp(T, T, norm1D))

            #B = B.at[-1].set( B_n(lam, 1.0, 5000.0, jax=True)/norm[-1])
           # B = B.at[-1].set( jnp.interp(lam, Stellar_x, Stellar_y)/norm[-1])
                        #product = B*ext#jnp.einsum('i,k->ki',B,ext)
            cont = ext*jnp.sum(B*BB_Amps)

        else:
            BB_Amps = cont_params[:-1]
            tau_98 = cont_params[-1]

            Tau = np.interp(lam, ls, S) #+ np.interp(lam, lss, SS)
            ext = (1.0 - np.exp(-tau_98*Tau))/(tau_98*Tau)

            #B = np.zeros((len(T)+1, len(lam)))
            B = np.zeros((len(T), len(lam)))

            B[:len(T),:] = B_nu(lam, 1.0, T.reshape(len(T),1), alpha_sil )/(jnp.interp(T, T, norm1D).reshape(len(T),1))

           # B[-1,:] =  B_n(lam, 1.0, 5000.0)/norm[-1]
          #  B[-1,:] = np.interp(lam, Stellar_x, Stellar_y)/norm[-1]

          #  product = np.einsum('ij,j->ij',B,ext)


            cont = np.empty(len(lam))
            for i in range(len(lam)):
                cont[i] = ext[i]*jnp.sum(B[:,i]*BB_Amps)


        return cont

 
def ReturnExt(lamm, cont_params, ExtType, ext_params, stellar_parameters, jax=False):

    alpha_sil = ext_params[-3]

    #lam = np.logspace(np.log10(min(lamm)), np.log10(max(lamm)), 1000)
    lam = np.linspace(min(lamm), max(lamm), 1000)
    if (ExtType==0.0):

        #Psi = jnp.reshape(jnp.exp(cont_params[:-1]), (grid_size[1], grid_size[0]+1))
        Psi = jnp.reshape(jnp.exp(cont_params[:-1]), (grid_size[1], grid_size[0]))


        Psi_dummy = np.full((grid_size[1], grid_size[0]), np.exp(-10.0))
        tau_dummy = tau[::-1]
        Psi = np.concatenate((Psi_dummy, Psi), axis=0)

        tau_ = np.concatenate((tau_dummy, tau))
        #Calculate Unobscured continuum to obscured continuum ratio to get ext. factor

        #B = np.zeros((len(T)+1, len(lam)))
        B = np.zeros((len(T), len(lam)))

        B[:len(T),:] = B_nu(lam, 1.0, T.reshape(len(T),1), alpha_sil)#/(jnp.interp(T, Ts[:-1], norm[:-1]).reshape(len(T),1))

        #B[-1,:] =  B_n(lam, 1.0, 5000.0)/norm[-1]
       # B[-1,:] = np.interp(lam, Stellar_x, Stellar_y)/norm[-1]



        Tau = np.interp(lam, ls, S)# + np.interp(lam, lss, SS)
        ext = np.exp(-tau_.reshape(len(tau_),1)*Tau)

        product_unobscured = np.einsum('ij,kj->kij',B,ext/ext)
        product_obscured = np.einsum('ij,kj->kij',B,ext)

        cont_obscured = np.empty(len(lam))
        cont_unobscured = np.empty(len(lam))

        for i in range(len(lam)):
            cont_obscured[i] = np.average(product_obscured[:,:,i]/norm_, weights = Psi/jnp.sum(Psi))
            cont_unobscured[i] = np.average(product_unobscured[:,:,i]/norm_, weights = Psi/jnp.sum(Psi))



        tau_98 = stellar_parameters[-1]
        ext = jnp.exp(-tau_98*Tau)

        scale1 = stellar_parameters[0]
        scale2 = stellar_parameters[1]

        s_int = scale1*jnp.interp(lam, Stellar_x1, Stellar_y_smooth1) + scale2*jnp.interp(lam, Stellar_x2, Stellar_y_smooth2)
        s_obsc = ext*s_int



        return jnp.interp(lamm, lam, (cont_obscured*cont_params[-1]+s_obsc)/(cont_unobscured*cont_params[-1] +s_int))





    elif (ExtType == 1.0):
        tau_98 = cont_params[-1]
        Tau = np.interp(lam, ls, S)# + np.interp(lam, lss, SS)
        ext = np.exp(-tau_98*Tau)
        return jnp.interp(lamm, lam, ext)
    elif (ExtType == 2.0):
        tau_98 = cont_params[-1]
        Tau = np.interp(lam, ls, S)# + np.interp(lam, lss, SS)
        ext = (1.0 - np.exp(-tau_98*Tau))/(tau_98*Tau)
        return jnp.interp(lamm, lam, ext)





def ReturnIceExt(lamm, cont_params, ExtType, ext_params, stellar_parameters, jax=False):

    alpha_sil = ext_params[-3]

    #lam = np.logspace(np.log10(min(lamm)), np.log10(max(lamm)), 1000)
    lam = np.linspace(min(lamm), max(lamm), 1000)
    if (ExtType==0.0):

        #Psi = jnp.reshape(jnp.exp(cont_params[:-1]), (grid_size[1], grid_size[0]+1))
        Psi = jnp.reshape(jnp.exp(cont_params[:-1]), (grid_size[1], grid_size[0]))

        Psi_dummy = np.full((grid_size[1], grid_size[0]), np.exp(-10.0))
        tau_dummy = tau[::-1]
        Psi = np.concatenate((Psi_dummy, Psi), axis=0)

        tau_ = np.concatenate((tau_dummy, tau))

        #Calculate Unobscured continuum to obscured continuum ratio to get ext. factor

        #B = np.zeros((len(T)+1, len(lam)))
        B = np.zeros((len(T), len(lam)))

        B[:len(T),:] = B_nu(lam, 1.0, T.reshape(len(T),1), alpha_sil)#/(jnp.interp(T, Ts[:-1], norm[:-1]).reshape(len(T),1))

        #B[-1,:] =  B_n(lam, 1.0, 5000.0)/norm[-1]
       # B[-1,:] = np.interp(lam, Stellar_x, Stellar_y)/norm[-1]



        Tau = np.interp(lam, ls, S) #+ np.interp(lam, lss, SS)
        ext = np.exp(-tau_.reshape(len(tau_),1)*Tau)

        #product_unobscured = np.einsum('ij,kj->kij',B,ext/ext)
        product = np.einsum('ij,kj->kij',B,ext)

        cont= np.empty(len(lam))
       # cont_unobscured = np.empty(len(lam))

        for i in range(len(lam)):
            cont[i] = np.average(product[:,:,i]/norm_, weights = Psi/jnp.sum(Psi))
            #cont_unobscured[i] = np.average(product_unobscured[:,:,i]/norm, weights = Psi/jnp.sum(Psi))



        tau_98 = stellar_parameters[-1]
        ext = jnp.exp(-tau_98*Tau)

        scale1 = stellar_parameters[0]
        scale2 = stellar_parameters[1]

        s_int = scale1*jnp.interp(lam, Stellar_x1, Stellar_y_smooth1) + scale2*jnp.interp(lam, Stellar_x2, Stellar_y_smooth2)
        s = ext*s_int


        s_ice = np.copy(s)
        s_ice *= jnp.exp(-ext_params[2]*stellar_parameters[2]*jnp.interp(lam, ls_NIR_Ice, NIR_Ice))
        s_ice *= jnp.exp(-ext_params[3]*stellar_parameters[2]*jnp.interp(lam, ls_NIR_CO2, NIR_CO2))


        cont_ice = np.copy(cont)
        cont_ice *= jnp.exp(-ext_params[2]*jnp.interp(lam, ls_NIR_Ice, NIR_Ice))
        cont_ice *= jnp.exp(-ext_params[3]*jnp.interp(lam, ls_NIR_CO2, NIR_CO2))

        return jnp.interp(lamm, lam, (cont_ice*cont_params[-1]+s_ice)/(cont*cont_params[-1] +s))

    elif (ExtType == 1.0 or ExtType == 2.0):
        return jnp.interp(lamm, lam, jnp.exp(-ext_params[2]*jnp.interp(lam, ls_NIR_Ice, NIR_Ice))*jnp.exp(-ext_params[3]*jnp.interp(lam, ls_NIR_CO2, NIR_CO2)))








def ReturnEffTau98(lamm, cont_params, ext_params, stellar_parameters, jax=False):


    alpha_sil = ext_params[-3]

    lam = jnp.logspace(jnp.log10(jnp.min(lamm)*1.05), jnp.log10(jnp.max(lamm)*0.95), 20)

    #Psi = jnp.reshape(jnp.exp(cont_params[:-1]), (grid_size[1], grid_size[0]+1))
    Psi = jnp.reshape(jnp.exp(cont_params[:-1]), (grid_size[1], grid_size[0]))


    Psi_dummy = jnp.full((grid_size[1], grid_size[0]), jnp.exp(-10.0))
    tau_dummy = tau[::-1]
    Psi = jnp.concatenate((Psi_dummy, Psi), axis=0)

    tau_ = jnp.concatenate((tau_dummy, tau))

    #Calculate Unobscured continuum to obscured continuum ratio to get ext. factor

    #B = np.zeros((len(T)+1, len(lam)))
    B = jnp.zeros((len(T), len(lam)))

    #B[:len(T),:] = B_nu(lam, 1.0, T.reshape(len(T),1), alpha_sil, jax=True)/(jnp.interp(T, Ts[:-1], norm[:-1]).reshape(len(T),1))
    B = B.at[:len(T), :].set(B_nu(lam, 1.0, T.reshape(len(T),1), alpha_sil, jax=True))#/(jnp.interp(T, Ts[:-1], norm[:-1]).reshape(len(T),1)))

    #B[-1,:] =  B_n(lam, 1.0, 5000.0)/norm[-1]
   # B[-1,:] = np.interp(lam, Stellar_x, Stellar_y)/norm[-1]



    Tau = jnp.interp(lam, ls, S) #+ jnp.interp(lam, lss, SS)
    ext = jnp.exp(-tau_.reshape(len(tau_),1)*Tau)

    product_unobscured = jnp.einsum('ij,kj->kij',B,ext/ext)
    product_obscured = jnp.einsum('ij,kj->kij',B,ext)

    cont_obscured = jnp.empty(len(lam))
    cont_unobscured = jnp.empty(len(lam))

    for i in range(len(lam)):
        #cont_obscured[i] = jnp.average(product_obscured[:,:,i], weights = Psi/jnp.sum(Psi))
        #cont_unobscured[i] = jnp.average(product_unobscured[:,:,i], weights = Psi/jnp.sum(Psi))
        cont_obscured = cont_obscured.at[i].set(jnp.average(product_obscured[:,:,i]/norm_, weights = Psi/jnp.sum(Psi)))
        cont_unobscured = cont_unobscured.at[i].set(jnp.average(product_unobscured[:,:,i]/norm_, weights = Psi/jnp.sum(Psi)))


    tau_98 = stellar_parameters[-1]
    ext = jnp.exp(-tau_98*Tau)

    scale1 = stellar_parameters[0]
    scale2 = stellar_parameters[1]

    s_int = scale1*jnp.interp(lam, Stellar_x1, Stellar_y_smooth1) + scale2*jnp.interp(lam, Stellar_x2, Stellar_y_smooth2)
    s_obsc = ext*s_int



    ext_factor = (cont_obscured*cont_params[-1]+s_obsc)/(cont_unobscured*cont_params[-1] +s_int)
    #ext_factor = cont_obscured/cont_unobscured

    return -jnp.log(ext_factor)/Tau






def ReturnCont(lam, scale, Psi, ext_params):



    # Psi_dummy = np.full((20, 20), np.exp(-10.0))
    # tau_dummy = tau[::-1]
    # Psi = np.concatenate((Psi_dummy, Psi), axis=0)

    # tau_ = np.concatenate((tau_dummy, tau))

    alpha_sil = ext_params[-3]
    B = np.zeros((len(T), len(lam)))
    B[:len(T),:] = B_nu(lam, 1.0, T.reshape(len(T),1), alpha_sil )#/(jnp.interp(T, Ts[:-1], norm[:-1]).reshape(len(T),1))


    Tau = np.interp(lam, ls, S) #+ np.interp(lam, lss, SS)
    ext = np.exp(-tau.reshape(len(tau),1)*Tau)
    product = np.einsum('ij,kj->kij',B,ext)
    cont = np.empty(len(lam))
    for i in range(len(lam)):
        cont[i] = np.average(product[:,:,i]/norm, weights = Psi/jnp.sum(Psi))
    return cont*scale


# def ReturnStellar(lam, scale, Psi):

#         B = np.zeros((1, len(lam)))
#         B[-1,:] =  B_n(lam, 1.0, 5000.0)/norm[-1]

#         Tau = np.interp(lam, ls, S) + np.interp(lam, lss, SS)
#         ext = np.exp(-tau.reshape(len(tau),1)*Tau)
#         product = np.einsum('ij,kj->kij',B,ext)
#         cont = np.empty(len(lam))
#         for i in range(len(lam)):
#             cont[i] = np.average(product[:,:,i], weights = Psi/jnp.sum(Psi))
#         return cont*scale





def ReturnExtDiff(lamm,  Psi, ext_params, stellar_parameters, cont_scale):


    Psi[Psi==0.0]=np.exp(-10.0)

    Psi_dummy = np.full((grid_size[1], grid_size[0]), np.exp(-10.0))
    tau_dummy = tau[::-1]
    Psi = np.concatenate((Psi_dummy, Psi), axis=0)

    tau_ = np.concatenate((tau_dummy, tau))

    alpha_sil = ext_params[-3]
    lam = np.linspace(min(lamm), max(lamm), 1000)


        #Psi = jnp.reshape(jnp.exp(cont_params[:-1]), (grid_size[1], grid_size[0]+1))
        #Psi = jnp.reshape(jnp.exp(cont_params[:-1]), (grid_size[1], grid_size[0]))

        #Calculate Unobscured continuum to obscured continuum ratio to get ext. factor

        #B = np.zeros((len(T)+1, len(lam)))
    B = np.zeros((len(T), len(lam)))

    B[:len(T),:] = B_nu(lam, 1.0, T.reshape(len(T),1), alpha_sil)#/(jnp.interp(T, Ts[:-1], norm[:-1]).reshape(len(T),1))

        #B[-1,:] =  B_n(lam, 1.0, 5000.0)/norm[-1]
       # B[-1,:] = np.interp(lam, Stellar_x, Stellar_y)/norm[-1]



    Tau = np.interp(lam, ls, S) #+ np.interp(lam, lss, SS)
    ext = np.exp(-tau_.reshape(len(tau_),1)*Tau)

    product_unobscured = np.einsum('ij,kj->kij',B,ext/ext)
    product_obscured = np.einsum('ij,kj->kij',B,ext)

    cont_obscured = np.empty(len(lam))
    cont_unobscured = np.empty(len(lam))

    for i in range(len(lam)):
        cont_obscured[i] = np.average(product_obscured[:,:,i]/norm_, weights = Psi/jnp.sum(Psi))
        cont_unobscured[i] = np.average(product_unobscured[:,:,i]/norm_, weights = Psi/jnp.sum(Psi))



    tau_98 = stellar_parameters[-1]
    ext = jnp.exp(-tau_98*Tau)

    scale1 = stellar_parameters[0]
    scale2 = stellar_parameters[1]

    s_int = scale1*jnp.interp(lam, Stellar_x1, Stellar_y_smooth1) + scale2*jnp.interp(lam, Stellar_x2, Stellar_y_smooth2)


    s_obsc = ext*s_int


    return jnp.interp(lamm, lam, (cont_obscured*cont_scale+s_obsc)/(cont_unobscured*cont_scale +s_int))









def ReturnCont_Expanded(lam, cont_params, T, tau, Psi, ext_params, jax=False):



    # Psi_dummy = np.full((20, 20), np.exp(-10.0))
    # tau_dummy = tau[::-1]
    # Psi = np.concatenate((Psi_dummy, Psi), axis=0)

    # tau_ = np.concatenate((tau_dummy, tau))

    B = np.zeros((len(T), len(lam)))
    B[:len(T),:] = B_nu(lam, 1.0, T.reshape(len(T),1), alpha_sil )#/(jnp.interp(T, Ts_full, norm_full).reshape(len(T),1))

   # B[-1,:] = np.interp(lam, Stellar_x, Stellar_y)/norm[-1]



    Tau = np.interp(lam, ls, S) #+ np.interp(lam, lss, SS)
    ext = np.exp(-tau.reshape(len(tau),1)*Tau)

    product = np.einsum('ij,kj->kij',B,ext)

    cont = np.empty(len(lam))
    for i in range(len(lam)):
        cont[i] = np.average(product[:,:,i]/norm, weights = Psi/jnp.sum(Psi))




    return cont*cont_params[-1]





    
# Stellar Component

def Stellar(lam, stellar_parameters, cont_params, ExtType, ext_params, jax = False):
        Tau = jnp.interp(lam, ls, S) #+ jnp.interp(lam, lss, SS)
        #Disp = stellar_parameters[-2]



        if (ExtType == 0.0):
            tau_98 = stellar_parameters[-1]
            ext = jnp.exp(-tau_98*Tau)
            if (len(ext_params)>5):
                ext *= jnp.exp(-ext_params[2]*stellar_parameters[2]*jnp.interp(lam, ls_NIR_Ice, NIR_Ice))
                #ext *= jnp.exp(-ext_params[2]*stellar_parameters[2]*ModifiedGauss(lam, ext_params[13], ext_params[14],  1.0, ext_params[15], jax=True))

                ext *= jnp.exp(-ext_params[3]*stellar_parameters[2]*jnp.interp(lam, ls_NIR_CO2, NIR_CO2))
                #ext *= jnp.exp(-ext_params[3]*stellar_parameters[2]*ModifiedGauss(lam, ext_params[13], ext_params[14],  1.0, ext_params[15], jax=True))


                # ext*= jnp.exp(-ext_params[11]*stellar_parameters[2]*ModifiedGauss(lam, ext_params[5], ext_params[6],  1.0, ext_params[7], jax=True))
                # ext*= jnp.exp(-ext_params[12]*stellar_parameters[2]*ModifiedGauss(lam, ext_params[8], ext_params[9],  1.0, ext_params[10], jax=True))

        elif (ExtType == 1.0):
            tau_98 = stellar_parameters[-1]
            ext = jnp.exp(-tau_98*Tau) 
            if (len(ext_params)>5):
                ext *= jnp.exp(-ext_params[2]*stellar_parameters[2]*jnp.interp(lam, ls_NIR_Ice, NIR_Ice))
                #ext *= jnp.exp(-ext_params[2]*stellar_parameters[2]*ModifiedGauss(lam, ext_params[13], ext_params[14],  1.0, ext_params[15], jax=True))

                ext *= jnp.exp(-ext_params[3]*stellar_parameters[2]*jnp.interp(lam, ls_NIR_CO2, NIR_CO2))
                #ext *= jnp.exp(-ext_params[3]*stellar_parameters[2]*ModifiedGauss(lam, ext_params[13], ext_params[14],  1.0, ext_params[15], jax=True))

        elif (ExtType == 2.0):
            tau_98 = stellar_parameters[-1]
            ext = jnp.exp(-tau_98*Tau) #(1.0 - jnp.exp(-tau_98*Tau) )/(tau_98*Tau)                  
            if (len(ext_params)>5):
                ext *= jnp.exp(-ext_params[2]*stellar_parameters[2]*jnp.interp(lam, ls_NIR_Ice, NIR_Ice))
                #ext *= jnp.exp(-ext_params[2]*stellar_parameters[2]*ModifiedGauss(lam, ext_params[13], ext_params[14],  1.0, ext_params[15], jax=True))

                ext *= jnp.exp(-ext_params[3]*stellar_parameters[2]*jnp.interp(lam, ls_NIR_CO2, NIR_CO2))
                #ext *= jnp.exp(-ext_params[3]*stellar_parameters[2]*ModifiedGauss(lam, ext_params[13], ext_params[14],  1.0, ext_params[15], jax=True))
        scale1 = stellar_parameters[0]
        scale2 = stellar_parameters[1]

        stars = scale1* Stellar_y1 + scale2*  Stellar_y2

        # x__ = jnp.linspace(-Disp*5, Disp*5, 20)
        # d_vel =  jnp.exp(-0.5*(x__/Disp)**2)
        # d_vel /= jnp.sum(d_vel)

        # stars_new = jnp.zeros(len(Stellar_x1))
        # for i in range(len(x__)):
        #     stars_new += jnp.interp(Stellar_x1, Stellar_x1*(1.0 + x__[i]/3e5), stars)*d_vel[i]


        stars_new = jnp.interp(lam, Stellar_x1, stars)

        # x__ = x__[:, jnp.newaxis]
        # d_vel = d_vel[:, jnp.newaxis]

        # # Calculate the shifted wavelengths
        # lam_shifted = lam * (1.0 + x__ / 3e5)

        # # Interpolate 'stars' for each shifted wavelength
        # stars_shifted = jnp.interp(lam_shifted,lam, stars)

        # # Compute the weighted sum of 'stars_shifted' using d_vel
        # stars_new = jnp.sum(stars_shifted * d_vel, axis=0)

        return ext*(stars_new)



    



def PAH2(lam, pah_parameters, jax = False):

    if (jax == False):
        model = 0.0
        PAHS = np.empty((int(len( pah_parameters)/3.0), len(lam)))
        strengths = np.empty((int(len( pah_parameters)/3.0)))
        for i in range(int(len( pah_parameters)/3.0)):
            a = int(3.0*i) # Index for amps
            c =  int(3.0*i + 1.0) # Index for centres
            w =  int(3.0*i + 2.0) # Index for widths

            model+=Drude(lam,  pah_parameters[c],  pah_parameters[w], pah_parameters[a],)
            PAHS[i, :] =Drude(lam,  pah_parameters[c],  pah_parameters[w], pah_parameters[a])
            strengths[i] = 2.9979246e14 * 0.5 * pah_parameters[a]*pah_parameters[w]*1.0e-9*np.pi/(pah_parameters[c])
        return model, PAHS#, ratioCheck

    else:
        model = jnp.zeros(jnp.shape(lam))
        for i in range(int(len( pah_parameters)/3.0)):
            a = int(3.0*i) # Index for amps
            c = int(3.0*i + 1.0) # Index for centres
            w =  int(3.0*i + 2.0) # Index for widths

            model+=Drude(lam,  pah_parameters[c],  pah_parameters[w], pah_parameters[a])
        return model

def PAH(lam, pah_parameters, jax = False):

    if (jax == False):
        model = 0.0
        PAHS = np.empty((int(len( pah_parameters)/4.0), len(lam)))
      #  strengths = np.empty((int(len( pah_parameters)/4.0)))
        for i in range(int(len( pah_parameters)/4.0)):
            a = int(4.0*i) # Index for amps
            c =  int(4.0*i + 1.0) # Index for centres
            w =  int(4.0*i + 2.0) # Index for widths
            asym =  int(4.0*i + 3.0) # Index for asymm
            #if ((pah_parameters[c]>12.73) & (pah_parameters[c]<12.8)):
              #  pah_parameters[a] = pah_parameters[a]*pah_parameters[int(4.0*(i-1))] #Restrictions on 12.7 PAH

            model+=ModifiedDrude(lam,  pah_parameters[c],  pah_parameters[w], pah_parameters[a],pah_parameters[asym])
            PAHS[i, :] = ModifiedDrude(lam,  pah_parameters[c],  pah_parameters[w], pah_parameters[a],pah_parameters[asym])
         #   strengths[i] = 2.9979246e14 * 0.5 * pah_parameters[a]*pah_parameters[w]*1.0e-9*np.pi/(pah_parameters[c])
        return model, PAHS#, ratioCheck

    else:
        model = jnp.zeros(jnp.shape(lam))
        for i in range(int(len( pah_parameters)/4.0)):
            a = int(4.0*i) # Index for amps
            c = int(4.0*i + 1.0) # Index for centres
            w =  int(4.0*i + 2.0) # Index for widths
            asym =  int(4.0*i + 3.0) # Index for asymm


            # if ((pah_parameters[c]>12.73) & (pah_parameters[c]<12.8)):
            #     pah_parameters[a] = pah_parameters[a]*pah_parameters[int(4.0*(i-1))] #Restrictions on 12.7 PAH

            #pah_parameters =  pah_parameters.at[a].set(jnp.where((pah_parameters[c]>12.73) & (pah_parameters[c]<12.8), pah_parameters[a]*pah_parameters[int(4.0*(i-1))], pah_parameters[a]))

            model+=ModifiedDrude(lam,  pah_parameters[c],  pah_parameters[w], pah_parameters[a], pah_parameters[asym])
        return model




def SilEmission(lam, ext_params):
    
    tau_hot = ext_params[-6]
    tau_cold = ext_params[-5]
    Cf = ext_params[-4]
    Amp = ext_params[-3]
    Temp = ext_params[-2]
    
    ext_hot = jnp.exp(-tau_hot*jnp.interp(lam, lsss, SSS) )
    #ext_cold = jnp.exp(-tau_cold*jnp.interp(lam, ls, S) + jnp.interp(lam, lss, SS))
    ext_cold = jnp.exp(-tau_cold*jnp.interp(lam, ls, S))

    B =B_n(lam,  1.0, Temp, jax =True)
    
    
    model = (1.-Cf)*B*(1.0-ext_hot) + Cf*B*(1.-ext_hot)*ext_cold

    return model*Amp




def ScaleChannels(lam, scale_params):


   #  #l =int(ChnLengths[0])
   # # ChnScale =jnp.ones(l)
    ChnScale = jnp.ones(jnp.shape(lam))
    # if (len(scale_params)!=0):

    #     ChnScale = jnp.where(lam<4.05, scale_params, 1.0)
   # # ChnScale = ChnScale.at[0:ChnLengths[0]].set(scale_params[0])

   #  l =0#ChnLengths[0]
   #  for i in range(0, len(ChnLengths)-1):
   #      #ChnScale = jnp.concatenate((ChnScale, scale_params[i]*jnp.ones((ChnLengths[i]))))
   #      #ChnScale = ChnScale.at[l:l+ChnLengths[i]].set(scale_params[i-1])

   #      ChnScale = ChnScale.at[l:l+ChnLengths[i]].set(scale_params[i])
   #      l+=ChnLengths[i]
   # # ChnScale = lam/lam



    return ChnScale



def ErrorScale(lam, ext_params):


    e1 = jnp.exp(ext_params[-1])
    e2 = jnp.exp(ext_params[-2])

    ErrScale = jnp.ones(jnp.shape(lam))
    ErrScale = jnp.where(lam<4.5, e1, e2)
    #ErrScale = jnp.where(lam>4.5, e2, 1.0)



    return ErrScale




