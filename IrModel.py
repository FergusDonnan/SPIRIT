import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec#
import math
from astropy.modeling.models import BlackBody
from astropy import units as u

from astropy.io import fits
from astropy.io import ascii

try:
    plt.style.use(['science','ieee', 'no-latex'])
except:
    plt.style.use(['default'])
from matplotlib.colors import LogNorm, PowerNorm
import scipy.optimize as opt

from astropy.modeling import models, fitting
from astropy.modeling import models, fitting
import time
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.interpolate import interp1d, interp2d
from tqdm import tqdm

from scipy.interpolate import CubicSpline
#from scipy.misc import derivative
#from scipy.special import gamma, factorial, legendre
#import dynesty
#from dynesty import NestedSampler
#from dynesty.utils import resample_equal

from scipy.stats import uniform, loguniform
from jax import random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value, HMCECS, init_to_median, init_to_uniform

import astropy.units as u
from astropy.utils import data
from spectral_cube import SpectralCube, Projection
#import aplpy
import astropy
import jaxopt
import jax
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLaplaceApproximation

import SetupFit

import sys
import matplotlib.ticker as mticker
    
from astropy.table import Table
import pandas as pd
import os
#from jax.config import config



#config.update("jax_enable_x64", True)
import jax
jax.config.update("jax_enable_x64", True)
#platform = "gpu"
#XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4"

dir_path = os.path.dirname(os.path.realpath(__file__))


numpyro.set_platform("cpu")
#print(config)
print('N_cpu=', jax.device_count())
#assert platform in ["cpu", "gpu", "tpu"]


from jax import local_device_count,default_backend,devices
#print(local_device_count(),default_backend(),devices())

try:
    jnp.trapz = jnp.trapezoid
except:
    pass


#############################################################################################################################################################
import warnings
warnings.filterwarnings("ignore")


# plt.style.use(['science','ieee', 'no-latex'])
plt.rcParams["font.family"] = "sans serif"


x_PAH, y_PAH = np.loadtxt(dir_path + '/PAH_template2.txt', unpack=True)

# Function defining model for MCMC using Numpyro
def numpyro_model(parameters, data, flux_orig, stellar_indx, pah_indx, cont_indx, ext_indx, scale_indx, sampled_indx,  ExtType, priors, fixed,  priors_lower, priors_upper, pr_mean, priors_mean, priors_sig, useMCMC, ps):
    x_data, y_data, err = data
    
    # Set up parameters to be sampled and which are fixed
    if (useMCMC==True):
        names=[]
        ps=[]
        for i in range(len(parameters)):
            name = parameters["Name"][i]

            value = parameters["Value"][i]
            
            pr_type = parameters['Prior Type'][i]
            
            # if (len(pr_mean)>1):
            #     value = numpyro.sample(name, dist.TruncatedNormal(loc = pr_mean[i], scale = (priors_upper[i]-priors_lower[i])/10.0, low=priors_lower[i], high=priors_upper[i]))
            

            if (name =='Ice Frac'): # Ice frac has a normal dist. to default to 1
                value = numpyro.sample(name, dist.TruncatedNormal(loc = 1.0 , scale =0.2 , low = priors_lower[i], high=priors_upper[i]))

            # elif (pr_type == 'TruncatedNormal'):
            #     value = numpyro.sample(name, dist.TruncatedNormal(loc = priors_mean[i] , scale =priors_sig[i], low = priors_lower[i], high=priors_upper[i]))


            else:
                value = numpyro.sample(name, dist.Uniform(low = priors_lower[i], high = priors_upper[i]))


                
            #print(value)
            ps.append(value)
        ps = jnp.array(ps)
    else:
        ps[sampled_indx] = parameters
    #     # Define model

    model = jax.vmap(model_function, in_axes=(None, None, None,None,None, None, None, None, 0))


    Y = y_data
    E = err
    if (ExtType == 0.0):
        #grid_size=[20,20]
        Psi  = jnp.reshape(jnp.exp(ps[cont_indx][:-1]), (grid_size[1], grid_size[0]))#[:,:grid_size[0]]
        Psi /=jnp.sum(Psi)
        column_left = Psi[:,0:1]
        column_right = Psi[:,-1:]
        row_up = Psi[0:1, :]
        row_low = Psi[-1:, :]
        Psi_1 = Psi
        Psi_2 = Psi

        smoothness = jnp.sum((Psi_1[:, :-2] - 2.0*Psi_1[:, 1:-1]+Psi_1[:, 2:])**2) + jnp.sum((Psi_2[:-2, :] - 2.0*Psi_2[1:-1, :]+Psi_2[ 2:, :])**2)
        tau_p = jnp.arange(1, int(grid_size[1]+1),1)

        tau = jnp.logspace(jnp.log(0.05), jnp.log(tau_lim), grid_size[1], base=jnp.exp(1))
        tau98_eff = SetupFit.ReturnEffTau98(x_data, ps[cont_indx], ps[ext_indx], ps[stellar_indx], jax=True)
        #Convert to pixel
        tau98_eff = jnp.interp(tau98_eff, tau, tau_p)  
        d_tau_means  =jnp.diff(tau98_eff)
        d_tau_prior = jnp.sum(jnp.log(jnp.exp(-0.5*(d_tau_means/6.0)**2))) #sig = 1 pixel


        #dust = SetupFit.MixedCont(1.5, ps[cont_indx],  ExtType, ps[ext_indx],jax=True)
        stars1 = SetupFit.Stellar(1.0, ps[stellar_indx], ps[cont_indx], ExtType, ps[ext_indx], jax=True)
        total_model1 = model_function(stellar_indx, pah_indx, cont_indx, ext_indx,  ExtType, scale_indx, None,  ps, 1.0)
        stellar_ratio1 = stars1/total_model1
        stars2 = SetupFit.Stellar(1.5, ps[stellar_indx], ps[cont_indx], ExtType, ps[ext_indx], jax=True)
        total_model2 = model_function(stellar_indx, pah_indx, cont_indx, ext_indx,  ExtType, scale_indx, None,  ps, 1.5)
        stellar_ratio2 = stars2/total_model2
        #s_prior = jnp.log(jnp.exp(-0.5*((stellar_ratio1 - 1.0)/0.2)**2)) + jnp.log(jnp.exp(-0.5*((stellar_ratio2 - 1.0)/0.4)**2))
        s_prior = jnp.log(jnp.exp(-0.5*((stellar_ratio2 - 1.0)/0.4)**2))




        # Prior to fix bug with continuum not appearing
        total_model3 = model_function(stellar_indx, pah_indx, cont_indx, ext_indx,  ExtType, scale_indx, None, ps, 15.0)
        far_ir_cont = SetupFit.MixedCont(15.0, ps[cont_indx],  ExtType, ps[ext_indx],jax=True)
        far_ir_prior = jnp.log(jnp.exp(-0.5*((far_ir_cont/total_model3 - 1.0)/0.5)**2))

    else:
        entropy=0.0
        smoothness = 0.0
        d_tau_prior=0.0
        smoothness2 =0.0
        s_prior=0.0


    lims = [3, 4, 5.1, 5.5, 5.9, 6.6, 7.2, 8.2, 9.0, 10.5, 11.7, 12.2, 13.0, 15.0, 18]
    pah_prior = 0.0
    for i in range(len(lims)-1):
        x_PAH_1 = jnp.linspace(lims[i], lims[i+1], 100)
        model_PAHs1 =  SetupFit.PAH(x_PAH_1, ps[pah_indx], jax=True)
        model_PAHs1 /= jnp.trapz(model_PAHs1, x_PAH_1)
        y_PAH_1 = jnp.interp(x_PAH_1, x_PAH, y_PAH)
        y_PAH_1 /= jnp.trapz(y_PAH_1, x_PAH_1)
        pah_prior +=  jnp.sum(model_PAHs1 - y_PAH_1  - model_PAHs1*jnp.log(model_PAHs1/y_PAH_1 ))/len(x_PAH_1) #jnp.sum(((y_PAH_1 - model_PAHs1)/(0.5*y_PAH_1))**2)


    PAH11 = jnp.trapz(SetupFit.PAH(jnp.linspace(10.5, 11.7, 100), ps[pah_indx], jax=True),  jnp.linspace(10.5, 11.7, 100))
    PAH12 = jnp.trapz(SetupFit.PAH(jnp.linspace(11.7, 13.0, 100), ps[pah_indx], jax=True),  jnp.linspace(11.7, 13.0, 100))

    PAH17 = jnp.trapz(SetupFit.PAH(jnp.linspace(15.0, 18.0, 100), ps[pah_indx], jax=True),  jnp.linspace(15.0, 18.0, 100))


    pah_prior += -0.5*(( PAH17/PAH11 - 0.9)/0.5)**2
    pah_prior += -0.5*(( PAH11/PAH12 - 0.86)/0.2)**2


    #prob = numpyro.sample("obs", dist.Normal(model(stellar_indx, pah_indx, cont_indx, ext_indx,  ExtType, scale_indx, None, ps, x_data), E), obs=Y)


    #pah_prior = 0.0
    #smoothness = 0.0
    with numpyro.plate("N", Y.shape[0], subsample_size=2000):
        Y_sub_sample = numpyro.subsample(Y, event_dim=0)
        E_sub_sample = numpyro.subsample(E, event_dim=0)
        x_sub_sample = numpyro.subsample(x_data, event_dim=0)
        #lambd = len(Y_sub_sample)*1e4
        lambd = RStrength#/10.0#/1e3#len(Y_sub_sample)#*10

        M = 1.0

        prob = numpyro.sample("obs", dist.Normal(model(stellar_indx, pah_indx, cont_indx, ext_indx,  ExtType, scale_indx, None, ps, x_sub_sample), E_sub_sample), obs=Y_sub_sample)
       # jax.debug.print(" prob = {x}", x=prob)
        # jax.debug.print(" sum(prob) = {x}", x=jnp.sum(prob))
        # jax.debug.print(" smooth = {x}", x=smoothness*lambd)
        # jax.debug.print(" smooth2 = {x}", x=smoothness2*0.1)
        # jax.debug.print(" sum(Total) = {x}", x=jnp.sum(prob - smoothness*lambd -smoothness2*0.1))


        p_t = numpyro.factor("Penalty", prob - smoothness*lambd +d_tau_prior+s_prior + pah_prior+far_ir_prior)#-smoothness2*0.1)#)+d_tau_prior )# +rat_prior) # Change lambd??

    
# Function defining model     
def model_function(stellar_indx, pah_indx, cont_indx, ext_indx,  ExtType, scale_indx, Nuc_indx,  ps, x_data):

    #ps[sampled_indx] = parameters
    ext = SetupFit.Ext(x_data, ps[ext_indx], ps[cont_indx], jax=True)[0]
        # Line model
    lines =  SetupFit.PAH(x_data, ps[pah_indx], jax=True)
    

        # Continuum
    cont = SetupFit.MixedCont(x_data, ps[cont_indx],  ExtType, ps[ext_indx],jax=True)#SetupFit.DustCont(x_data, ps[cont_indx], jax=True)#SetupFit.contTemp(x_data, ps[cont_indx], jax=True)
    
    # Stellar Continuum
    stellar = SetupFit.Stellar(x_data, ps[stellar_indx], ps[cont_indx], ExtType, ps[ext_indx],jax=True)

   # Nuc = SetupFit.Nuclear(x_data, ps[Nuc_indx])
        # Define model
    model =  lines + cont*ext +stellar#*ext#+ Nuc

            # Lines + SF continuum + nuclear continuum

    return model
# Define logprob for quick/bootstrap methods
@jax.jit
def loss(parameters, data, flux_orig, stellar_indx, pah_indx, cont_indx, ext_indx, scale_indx, Nuc_indx, sampled_indx,  H2indices, ps):

    x_data, y_data, err = data
    ps = parameters

   # err = y_data*(10**(ps[-1])) + ps[-2]

    vmapped_mean_function = jax.vmap(model_function, in_axes=(None, None, None,None,None, None, None, None, 0))

    
    Y = y_data
    E = err
    if (ExtType == 0.0):
        #grid_size=[20,20]
        Psi  = jnp.reshape(jnp.exp(ps[cont_indx][:-1]), (grid_size[1], grid_size[0]))#[:,:grid_size[0]]
        Psi /=jnp.sum(Psi)

        column_left = Psi[:,0:1]
        column_right = Psi[:,-1:]
        row_up = Psi[0:1, :]
        row_low = Psi[-1:, :]
        Psi_1 = Psi#jnp.concatenate((column_left, column_left, Psi, column_right, column_right), axis=1)
        Psi_2 = Psi#jnp.concatenate((row_up, row_up, Psi, row_low, row_low), axis=0)
        smoothness = jnp.sum((Psi_1[:, :-2] - 2.0*Psi_1[:, 1:-1]+Psi_1[:, 2:])**2) + jnp.sum((Psi_2[:-2, :] - 2.0*Psi_2[1:-1, :]+Psi_2[ 2:, :])**2)


        T_p = jnp.arange(1, int(grid_size[0]+1),1)
        tau_p = jnp.arange(1, int(grid_size[1]+1),1)
        T_p_exp = jnp.arange(1, 27,1)

        tau = jnp.logspace(jnp.log(0.05), jnp.log(tau_lim), grid_size[1], base=jnp.exp(1))


        tau98_eff = SetupFit.ReturnEffTau98(x_data, ps[cont_indx], ps[ext_indx], ps[stellar_indx], jax=True)
        #Convert to pixel
        tau98_eff = jnp.interp(tau98_eff, tau, tau_p)

  
        d_tau_means  =jnp.diff(tau98_eff)


        d_tau_prior = jnp.sum(jnp.log(jnp.exp(-0.5*(d_tau_means/4.0)**2))) #sig = 4 pixel


        #Prior to ensure stellar continuum is fitted
        #dust = SetupFit.MixedCont(1.5, ps[cont_indx],  ExtType, ps[ext_indx],jax=True)
        stars1 = SetupFit.Stellar(1.0, ps[stellar_indx], ps[cont_indx], ExtType, ps[ext_indx], jax=True)
        total_model1 = model_function(stellar_indx, pah_indx, cont_indx, ext_indx,  ExtType, scale_indx, Nuc_indx,  ps, 1.0)
        stellar_ratio1 = stars1/total_model1
        stars2 = SetupFit.Stellar(1.5, ps[stellar_indx], ps[cont_indx], ExtType, ps[ext_indx], jax=True)
        total_model2 = model_function(stellar_indx, pah_indx, cont_indx, ext_indx,  ExtType, scale_indx, Nuc_indx,  ps, 1.5)
        stellar_ratio2 = stars2/total_model2
        s_prior =  jnp.log(jnp.exp(-0.5*((stellar_ratio2 - 1.0)/0.4)**2)) +jnp.log(jnp.exp(-0.5*((stellar_ratio1 - 1.0)/0.2)**2)) 


        #s_prior = 0.0

        if (len(stellar_indx)>3):
            ice_frac_prior = jnp.log(jnp.exp(-0.5*((ps[stellar_indx][2] - 1.0)/0.2)**2))
        else:
            ice_frac_prior = 0.0




        #Prior on PAH bandss
        x_grid = jnp.linspace(np.min(x_data), np.max(x_data), 1000)
        model_PAHs =  SetupFit.PAH(x_grid, ps[pah_indx], jax=True)

  
        # # # Prior to fix bug with continuum not appearing
        # total_model3 = model_function(stellar_indx, pah_indx, cont_indx, ext_indx,  ExtType, scale_indx, None, ps, 15.0)
        # far_ir_cont = SetupFit.MixedCont(15.0, ps[cont_indx],  ExtType, ps[ext_indx],jax=True)
        # far_ir_prior = jnp.log(jnp.exp(-0.5*((far_ir_cont/total_model3 - 1.0)/0.5)**2))

    else:
        entropy=0.0
        smoothness = 0.0
        smoothness2 = 0.0
        var=0.0
        ice_frac_prior = 0.0
        d_tau_prior=0.0
        s_prior=0.0
        far_ir_prior=0.0


    model_PAHs1 =  SetupFit.PAH(jnp.linspace(6.0, 6.5, 200), ps[pah_indx], jax=True)
    model_PAHs2 =  SetupFit.PAH(jnp.linspace(6.6, 7.3, 200), ps[pah_indx], jax=True)
    model_PAHs3 =  SetupFit.PAH(jnp.linspace(7.4, 8.2, 200), ps[pah_indx], jax=True)
    model_PAHs4 =  SetupFit.PAH(jnp.linspace(11.7, 12.2, 200), ps[pah_indx], jax=True)
    model_PAHs5 =  SetupFit.PAH(jnp.linspace(12.3, 13.0, 200), ps[pah_indx], jax=True)
    model_PAHs6 =  SetupFit.PAH(jnp.linspace(11.2, 11.5, 200), ps[pah_indx], jax=True)
    model_PAHs7 =  SetupFit.PAH(jnp.linspace(10.0, 11.1, 200), ps[pah_indx], jax=True)
    model_PAHs8 =  SetupFit.PAH(jnp.linspace(5.0, 6.0, 200), ps[pah_indx], jax=True)

    PAH_5 = jnp.trapz(model_PAHs8, jnp.linspace(5.0, 6.0, 200))
    PAH_62 = jnp.trapz(model_PAHs1, jnp.linspace(6.0, 6.5, 200))
    PAH_67 =jnp.trapz(model_PAHs2, jnp.linspace(6.6, 7.3, 200))
    PAH_77 = jnp.trapz(model_PAHs3,jnp.linspace(7.4, 8.2, 200) )
    PAH_12 = jnp.trapz(model_PAHs4,jnp.linspace(11.7, 12.2, 200))
    PAH_127 = jnp.trapz(model_PAHs5,jnp.linspace(12.3, 13.0, 200))
    PAH_11 = jnp.trapz(model_PAHs6,jnp.linspace(11.2, 11.5, 200))
    PAH_10 = jnp.trapz(model_PAHs7,jnp.linspace(10.0, 11.1, 200))


    pah_prior = 10.0*jnp.tanh(10.0*(1.0 - PAH_67/PAH_62 +1.0/10.0))-10.0
    pah_prior += 10.0*jnp.tanh(10.0*(0.25 - PAH_67/PAH_77+1.0/10.0))-10.0
    pah_prior += 10.0*jnp.tanh(10.0*(2.8 - PAH_127/PAH_12+1.0/10.0))-10.0
    pah_prior += 10.0*jnp.tanh(10.0*(1.0 - PAH_12/PAH_11+1.0/10.0))-10.0
    pah_prior += 10.0*jnp.tanh(10.0*(0.4 - PAH_10/PAH_11+1.0/10.0))-10.0
    pah_prior += 10.0*jnp.tanh(10.0*(0.3 - PAH_5/PAH_62+1.0/10.0))-10.0

    pah_prior += 10.0*jnp.tanh(10.0*(7.0 - PAH_77/PAH_62+1.0/10.0))-10.0
    pah_prior += 10.0*jnp.tanh(10.0*(3.3 - PAH_77/PAH_11+1.0/10.0))-10.0


    # pah_prior = 100.0*jnp.tanh(100.0*(1.0 - PAH_67/PAH_62))-100.0
    # pah_prior += 100.0*jnp.tanh(100.0*(0.25 - PAH_67/PAH_77))-100.0
    # pah_prior += 100.0*jnp.tanh(100.0*(2.8 - PAH_127/PAH_12))-100.0
    # pah_prior += 100.0*jnp.tanh(100.0*(1.0 - PAH_12/PAH_11))-100.0
    # pah_prior += 100.0*jnp.tanh(100.0*(0.4 - PAH_10/PAH_11))-100.0
    # pah_prior += 100.0*jnp.tanh(100.0*(0.3 - PAH_5/PAH_62))-100.0

    # pah_prior=0.0
    # pah_prior = -0.5*((PAH_67/PAH_62 - 0.95)/(0.11*5.0))**2 + 100.0*jnp.tanh(10.0*(1.0 - PAH_67/PAH_62 +2.0/10.0))-100.0
    # pah_prior += -0.5*((PAH_67/PAH_77 - 0.24)/(0.02*5.0))**2 + 100.0*jnp.tanh(10.0*(0.25 - PAH_67/PAH_77+2.0/10.0))-100.0
    # pah_prior += -0.5*((PAH_127/PAH_12 - 2.4)/(0.36*5.0))**2+ 100.0*jnp.tanh(10.0*(2.8 - PAH_127/PAH_12+2.0/10.0))-100.0
    # pah_prior += -0.5*((PAH_12/PAH_11 - 0.55)/(0.07*5.0))**2+ 100.0*jnp.tanh(10.0*(1.0 - PAH_12/PAH_11+2.0/10.0))-100.0
    # pah_prior += -0.5*((PAH_10/PAH_11 - 0.30)/(0.04*5.0))**2+ 100.0*jnp.tanh(10.0*(0.4 - PAH_10/PAH_11+2.0/10.0))-100.0
    # pah_prior += -0.5*((PAH_5/PAH_62 - 0.245)/(0.033*5.0))**2+ 100.0*jnp.tanh(10.0*(0.3 - PAH_5/PAH_62+2.0/10.0))-100.0


    # pah_prior = -0.5*((PAH_67/PAH_62 - 0.95)/(0.5*0.95))**2
    # pah_prior += -0.5*((PAH_67/PAH_77 - 0.24)/(0.5*0.24))**2
    # pah_prior += -0.5*((PAH_127/PAH_12 - 2.4)/(0.5*2.4))**2
    # pah_prior += -0.5*((PAH_12/PAH_11 - 0.55)/(0.5*0.55))**2
    # pah_prior += -0.5*((PAH_10/PAH_11 - 0.30)/(0.5*0.3))**2
    # pah_prior += -0.5*((PAH_5/PAH_62 - 0.245)/(0.5*0.245))**2

    lims = [3, 4, 5.1, 5.5, 5.9, 6.6, 7.2, 8.2, 9.0, 10.5, 11.7, 12.2, 13.0, 15.0, 18]
    pah_prior = 0.0
    for i in range(len(lims)-1):
        x_PAH_1 = jnp.linspace(lims[i], lims[i+1], 100)
        model_PAHs1 =  SetupFit.PAH(x_PAH_1, ps[pah_indx], jax=True)
        model_PAHs1 /= jnp.trapz(model_PAHs1, x_PAH_1)
        y_PAH_1 = jnp.interp(x_PAH_1, x_PAH, y_PAH)
        y_PAH_1 /= jnp.trapz(y_PAH_1, x_PAH_1)
        pah_prior +=  jnp.sum(model_PAHs1 - y_PAH_1  - model_PAHs1*jnp.log(model_PAHs1/y_PAH_1 ))/len(x_PAH_1) #jnp.sum(((y_PAH_1 - model_PAHs1)/(0.5*y_PAH_1))**2)


    PAH11 = jnp.trapz(SetupFit.PAH(jnp.linspace(10.5, 11.7, 100), ps[pah_indx], jax=True),  jnp.linspace(10.5, 11.7, 100))
    PAH12 = jnp.trapz(SetupFit.PAH(jnp.linspace(11.7, 13.0, 100), ps[pah_indx], jax=True),  jnp.linspace(11.7, 13.0, 100))
    PAH7 = jnp.trapz(SetupFit.PAH(jnp.linspace(7.2, 8.2, 100), ps[pah_indx], jax=True),  jnp.linspace(7.2, 8.2, 100))

    PAH17 = jnp.trapz(SetupFit.PAH(jnp.linspace(15.0, 18.0, 100), ps[pah_indx], jax=True),  jnp.linspace(15.0, 18.0, 100))


    pah_prior += -0.5*(( PAH17/PAH11 - 0.9)/1.0)**2 # 0.5
    pah_prior += -0.5*(( PAH11/PAH12 - 0.86)/1.0)**2 #0.2
    # PAH6 = jnp.trapz(SetupFit.PAH(jnp.linspace(6.0, 6.5, 100), ps[pah_indx], jax=True),  jnp.linspace(6.0, 6.5, 100))
    # pah_prior += -0.5*(( PAH11/PAH7 - 0.64)/0.5)**2 # 0.2
    # pah_prior += -0.5*(( PAH11/PAH6 - 2.94)/1.5)**2 # 0.2

    #pah_prior += -0.5*(( PAH11/PAH7 - 0.6)/0.3)**2
#    pah_prior += 100.0*jnp.tanh(100.0*(2.0 - PAH7/PAH11))-100.0
###
    # x_PAH_1 = jnp.linspace(5.0, 7.0, 200)
    # model_PAHs1 =  SetupFit.PAH(x_PAH_1, ps[pah_indx], jax=True)
    # model_PAHs1 /= jnp.trapz(model_PAHs1, x_PAH_1)#jnp.interp(6.3, x_PAH_1, model_PAHs1)
    # y_PAH_1 = jnp.interp(x_PAH_1, x_PAH, y_PAH)
    # y_PAH_1 /= jnp.trapz(y_PAH_1, x_PAH_1)
    # pah_prior +=  jnp.sum(model_PAHs1 - y_PAH_1  - model_PAHs1*jnp.log(model_PAHs1/y_PAH_1 ))/len(x_PAH_1) #jnp.sum(((y_PAH_1 - model_PAHs1)/(0.5*y_PAH_1))**2)

    # x_PAH_2 = jnp.linspace(7.0, 9.0, 200)
    # model_PAHs2 =  SetupFit.PAH(x_PAH_2, ps[pah_indx], jax=True)
    # model_PAHs2 /= jnp.trapz(model_PAHs2, x_PAH_2)#jnp.interp(6.3, x_PAH_1, model_PAHs1)
    # y_PAH_2 = jnp.interp(x_PAH_2, x_PAH, y_PAH)
    # y_PAH_2 /= jnp.trapz(y_PAH_2, x_PAH_2)
    # pah_prior +=  jnp.sum(model_PAHs2 - y_PAH_2  - model_PAHs2*jnp.log(model_PAHs2/y_PAH_2))/len(x_PAH_2) #jnp.sum(((y_PAH_1 - model_PAHs1)/(0.5*y_PAH_1))**2)


    # x_PAH_3 = jnp.linspace(10.5, 11.5, 200)
    # model_PAHs3 =  SetupFit.PAH(x_PAH_3, ps[pah_indx], jax=True)
    # model_PAHs3 /= jnp.trapz(model_PAHs3, x_PAH_3)#jnp.interp(6.3, x_PAH_1, model_PAHs1)
    # y_PAH_3 = jnp.interp(x_PAH_3, x_PAH, y_PAH)
    # y_PAH_3 /= jnp.trapz(y_PAH_3, x_PAH_3)
    # pah_prior +=  jnp.sum(model_PAHs3 - y_PAH_3  - model_PAHs3*jnp.log(model_PAHs3/y_PAH_3))/len(x_PAH_3) #jnp.sum(((y_PAH_1 - model_PAHs1)/(0.5*y_PAH_1))**2)



    # x_PAH_4 = jnp.linspace(12.5, 15.0, 200)
    # model_PAHs4 =  SetupFit.PAH(x_PAH_4, ps[pah_indx], jax=True)
    # model_PAHs4 /= jnp.trapz(model_PAHs4, x_PAH_4)#jnp.interp(6.3, x_PAH_1, model_PAHs1)
    # y_PAH_4 = jnp.interp(x_PAH_4, x_PAH, y_PAH)
    # y_PAH_4 /= jnp.trapz(y_PAH_4, x_PAH_4)
    # pah_prior += jnp.sum(model_PAHs4 - y_PAH_4  - model_PAHs4*jnp.log(model_PAHs4/y_PAH_4))/len(x_PAH_4) #jnp.sum(((y_PAH_1 - model_PAHs1)/(0.5*y_PAH_1))**2)




    # x_PAH_5 = jnp.linspace(15.0, 20.0, 200)
    # model_PAHs5 =  SetupFit.PAH(x_PAH_5, ps[pah_indx], jax=True)
    # model_PAHs5 /= jnp.trapz(model_PAHs5, x_PAH_5)#jnp.interp(6.3, x_PAH_1, model_PAHs1)
    # y_PAH_5 = jnp.interp(x_PAH_5, x_PAH, y_PAH)
    # y_PAH_5 /= jnp.trapz(y_PAH_5, x_PAH_5)
    # pah_prior +=   jnp.sum(model_PAHs5- y_PAH_5  - model_PAHs5*jnp.log(model_PAHs5/y_PAH_5))/len(x_PAH_5) #jnp.sum(((y_PAH_1 - model_PAHs1)/(0.5*y_PAH_1))**2)





 #   pah_prior = 0.0
    # jax.debug.print(" PAH67/62 = {x}", x=PAH_67/PAH_62)
    # jax.debug.print(" PAH67/77 = {x}", x=PAH_67/PAH_77)
    # jax.debug.print(" PAH5/62 = {x}", x=PAH_5/PAH_62)




    lambd = len(Y)*RStrength#1e4#*1e4#*1e4#1e4

    #lambd = 
    chi2 = jnp.sum(((Y-vmapped_mean_function(stellar_indx, pah_indx, cont_indx, ext_indx,  ExtType, scale_indx, Nuc_indx, ps, x_data) )/E)**2)
    jax.debug.print(" chi2 = {x}", x=chi2)
    jax.debug.print(" VarPen = {x}", x=jnp.sum(jnp.log((E)**2)))
    jax.debug.print(" ch2+VarPen = {x}", x=chi2+jnp.sum(jnp.log((E)**2)))

    jax.debug.print(" smooth = {x}", x=smoothness*lambd)
    jax.debug.print(" dTau Prior = {x}", x=d_tau_prior*len(Y))
    jax.debug.print(" ice frac Prior = {x}", x=ice_frac_prior*len(Y))
    jax.debug.print(" PAHPrior = {x}", x=pah_prior*len(Y))
    # jax.debug.print("")
    #pah_prior = 0.0
    jax.debug.print(" S Prior = {x}", x=s_prior*len(Y))
    # jax.debug.print(" PAH Prior = {x}", x=(10.0*jnp.tanh(10*(0.3 - PAH_5/PAH_62+1.0/10.0))-10.0))
    # jax.debug.print(" PAH 5/6.2 = {x}", x=PAH_5/PAH_62)

    #jax.debug.print(" PlnN = {x}", x=P*jnp.log(len(Y)))

    # jax.debug.print(" smooth2 = {x}", x=smoothness2*len(Y))
    # jax.debug.print(" total = {x}", x=chi2 + jnp.sum(jnp.log((E)**2)) + smoothness*lambd +smoothness2*len(Y))

    # out1.append(chi2)
    # out2.append(jnp.sum(jnp.log((E)**2)))
    # out3.append(smoothness*lambd)
    #smoothness=0.0
    # pah_prior *= 10
    return chi2 + jnp.sum(jnp.log((E)**2)) + smoothness*lambd - d_tau_prior*len(Y) -s_prior*len(Y) -ice_frac_prior*len(Y) - pah_prior*len(Y) #+P*jnp.log(len(Y)) #+smoothness2*len(Y)# -d_tau_prior*len(Y)  #+ var*lambd/1e5 #-rat_prior#+ std*lambd2#-entropy*lambd#+smooth_x*lambd #-entropy*lambd#+smoothness*lambd#-entropy*lambd #- entropy*(lambd/len(cont_indx)) # + 2.0*jnp.log(E**2))
    
    


    
    
out1=jnp.empty(200)
out2=[]
out3=[] 
    
def RunFit(objName, specdata, z, lam_range, binNo, useMCMC=True, ExtType_='Screen', Ices_6micron=False, InitialFit=False, BootStrap=False, N_bootstrap = 100,  HI_ratios = 'Case B', show_progress=True, N_MCMC = 5000, N_BurnIn = 15000, ExtCurve = 'D23ExtCurve', EmCurve = 'D24Emissivity', MIR_CH = 'CHExt_v3', NIR_CH = 'CH_NIR', Fit_NIR_CH = False, NIR_Ice = 'NIR_Ice', NIR_CO2 = 'NIR_CO2', RegStrength = 10000, Cont_Only = False, St_Cont = True, Extend = False, Fit_CO = False, spec_res = 'h'):
    setup = SetupFit.Fit(objName, specdata, z, lam_range,  ExtType_, Ices_6micron, ExtCurve, MIR_CH = MIR_CH, NIR_CH = NIR_CH, Fit_NIR_CH = Fit_NIR_CH ,NIR_Ice_ = NIR_Ice, NIR_CO2_ = NIR_CO2, Cont_Only = Cont_Only, St_Cont = St_Cont, Extend = Extend, Fit_CO = Fit_CO, spec_res = spec_res)
    ObjName = objName


    extended = Extend#False
    global grid_size, tau_lim
    if (extended == True):
        tau_lim = 67.29
        grid_size = [20, 25]
    else:
        grid_size = [20, 20]
        tau_lim = 15.0

    parameters = setup.parameters
    data = setup.data
    ps = setup.ps
    stellar_indx = setup.stellar_indx
    pah_indx = setup.pah_indx
    cont_indx = setup.cont_indx
    ext_indx = setup.ext_indx
    sampled_indx = setup.sampled_indx
    scale_indx = setup.scale_indx
    Nuc_indx = setup.Nuc_indx

    H2indices = setup.H2_indices
    ExtTypeOut=ExtType_
    global ExtType
    if (ExtType_ == 'Differential'):
        ExtType = 0.0
    if (ExtType_ == 'Differential_Parametric'):
        ExtType = -1.0
    elif (ExtType_ == 'Mixed'):
        ExtType = 2.0
    elif (ExtType_ =='Screen'):
        ExtType = 1.0


    global ext_curve, RStrength
    ext_curve = np.loadtxt(dir_path+'/Ext.Curves/'+ExtCurve+'.txt', unpack=True, usecols=[0,1])
    RStrength = RegStrength


    priors = setup.priors
    fixed = setup.fixed
    flux_orig = setup.flux_orig
    #SL2Mask = setup.SL2Mask
    SCALE = setup.scale
    print('N_data = ',len(data[0]))
    print('N_params = ',len(sampled_indx))
    #ps = setup.ps
            

    # Setup priors
    prs=setup.priors#[sampled_indx]
    bounds=[]
    priors_lower = np.empty(len(prs))
    priors_upper = np.empty(len(prs))
    priors_mean = np.empty(len(prs))
    priors_sig = np.empty(len(prs))
    for i in range(len(prs)):
        bounds.append((prs[i][0], prs[i][1]))
        priors_lower[i] = prs[i][0]
        priors_upper[i] = prs[i][1]
        if (len(prs[i])>2):
            priors_mean[i] = prs[i][2]
            priors_sig[i]  = prs[i][3]
   # if (PAHFIT != True):
    bounds = (jnp.array( priors_lower[sampled_indx]), jnp.array( priors_upper[sampled_indx]))

        
    # Read in data
    lam, flux, flux_err = data
    
    y_data=flux#.reshape(-1, 1)
    x_data=lam#.reshape(-1, 1)
    err = flux_err
    # plt.figure()
    # plt.plot(x_data, flux)
    # plt.show()
    gp = None
############


    if (useMCMC == True):
      #  kernel = NUTS(numpyro_model, dense_mass=False)


        if (InitialFit == True):
            try:
                solver = jaxopt.ScipyBoundedMinimize(fun=loss,  options={'disp': show_progress,'gtol': 1e-96*len(data[0]), 'maxfun': 100000, 'maxiter':100000})#, maxiter=200000)#
                soln = solver.run(jax.tree_util.tree_map(jnp.asarray, ps[sampled_indx].astype(float)), bounds, data, flux_orig, stellar_indx, pah_indx, cont_indx, ext_indx, scale_indx,  Nuc_indx, sampled_indx,  H2indices,  jax.tree_util.tree_map(jnp.asarray, ps.astype(float)))
            except:
                solver = jaxopt.ScipyBoundedMinimize(fun=loss,  options={'disp': show_progress,'gtol': 1e-9*len(data[0]), 'maxfun': 100000}, maxiter=100000)#
                soln = solver.run(jax.tree_util.tree_map(jnp.asarray, ps[sampled_indx].astype(float)), bounds, data, flux_orig, stellar_indx, pah_indx, cont_indx, ext_indx, scale_indx,  Nuc_indx, sampled_indx,   H2indices,  jax.tree_util.tree_map(jnp.asarray, ps.astype(float)))
            pr_mean = soln.params
        else:
            pr_mean = [0.0]
        # # print(f"Final negative log likelihood: {soln.state.fun_val}")
        # ps[sampled_indx] = soln.params

        # p_vals = ps

        # ref_params={}
        # for i in range(len(parameters)):
        #     name = parameters['Name'][i]
        #     prior = parameters['Prior'][i]

        #     f=1.0
        #     if (abs(p_vals[i]/prior[0])<1.05):
        #         f=1.2
        #     elif (abs(p_vals[i]/prior[1])>0.95):
        #         f=0.8
        #     ref_params[name]=p_vals[i]*f

        # print(ref_params)

        inner_kernel = NUTS(numpyro_model, dense_mass=True, init_strategy=init_to_uniform())#)init_to_value(values= ref_params))init_to_median()
        # note: if num_blocks=100, we'll update 10 index at each MCMC step
        # so it took 50000 MCMC steps to iterative the whole dataset
        kernel = HMCECS(inner_kernel, num_blocks=1)#, proxy=HMCECS.taylor_proxy(ref_params))
        #kernel = inner_kernel


        mcmc = MCMC(kernel, num_warmup=N_BurnIn, num_samples=N_MCMC, num_chains=1)
        rng_key = random.PRNGKey(int(np.random.uniform(0.0, 1e9)))
        mcmc.run(rng_key,  parameters, data, flux_orig, stellar_indx, pah_indx, cont_indx, ext_indx, scale_indx, sampled_indx,  ExtType, priors, fixed,  priors_lower, priors_upper, pr_mean, priors_mean, priors_sig, useMCMC, ps)

   
        
        mcmc.print_summary()
        samples = mcmc.get_samples()
    else:

        start = time.time()
        # if (SilicateEmission==True):
        #     solver = jaxopt.ScipyBoundedMinimize(fun=loss3,  options={'disp': True,'gtol': 1e-9*len(data[0]), 'maxfun': 200000, 'maxiter':200000})#, maxiter =200000)#
        # else:
        


        try:
            solver = jaxopt.ScipyBoundedMinimize(fun=loss,  options={'disp': show_progress,'gtol': 1e-9*len(data[0]), 'maxfun': 100000, 'maxiter':100000})#, maxiter=200000)#
            soln = solver.run(jax.tree_util.tree_map(jnp.asarray, ps[sampled_indx].astype(float)), bounds, data, flux_orig, stellar_indx, pah_indx, cont_indx, ext_indx, scale_indx,  Nuc_indx, sampled_indx,  H2indices,  jax.tree_util.tree_map(jnp.asarray, ps.astype(float)))
        except:
            solver = jaxopt.ScipyBoundedMinimize(fun=loss,  options={'disp': show_progress,'gtol': 1e-9*len(data[0]), 'maxfun': 100000}, maxiter=100000)#
            soln = solver.run(jax.tree_util.tree_map(jnp.asarray, ps[sampled_indx].astype(float)), bounds, data, flux_orig, stellar_indx, pah_indx, cont_indx, ext_indx, scale_indx,  Nuc_indx, sampled_indx,   H2indices,  jax.tree_util.tree_map(jnp.asarray, ps.astype(float)))

            


        print(f"Final negative log likelihood: {soln.state.fun_val}")
        ps[sampled_indx] = soln.params
        setup.parameters.loc[setup.parameters['Fixed'] == False, 'Value'] = ps[sampled_indx]
        end = time.time()
        print(end - start)
        #print(soln.info)
        # print(np.diag(soln.hess_inv.todense()))
            

        # Rerun after re-sampling data to get errors
        N_samps = N_bootstrap
        ps_samps = np.empty((N_samps, len(ps)))
        if (BootStrap==True):
            print('Running Bootstraps...')
            for i in tqdm(range(N_samps)):

                data_resampled = np.copy(data)
                data_resampled[1] = np.random.normal(data[1], data[2])

                #ps = np.random.normal(ps, 0.001*abs(ps))
                #ps[ps<priors_lower] = 1.001*priors_lower[ps<priors_lower]
                #ps[ps>priors_upper] = 0.999*priors_upper[ps>priors_upper]

                try:
                    solver = jaxopt.ScipyBoundedMinimize(fun=loss,  options={'disp': False,'gtol': 1e-9*len(data[0]), 'maxfun': 100000, 'maxiter':100000})#, maxiter=200000)#
                    soln = solver.run(jax.tree_util.tree_map(jnp.asarray, ps[sampled_indx].astype(float)), bounds, data_resampled, flux_orig, stellar_indx, pah_indx, cont_indx, ext_indx, scale_indx,  Nuc_indx, sampled_indx,  H2indices,  jax.tree_util.tree_map(jnp.asarray, ps.astype(float)))
                except:
                    solver = jaxopt.ScipyBoundedMinimize(fun=loss,  options={'disp': False,'gtol': 1e-9*len(data[0]), 'maxfun': 100000}, maxiter=100000)#
                    soln = solver.run(jax.tree_util.tree_map(jnp.asarray, ps[sampled_indx].astype(float)), bounds, data_resampled, flux_orig, stellar_indx, pah_indx, cont_indx, ext_indx, scale_indx,  Nuc_indx, sampled_indx,  H2indices,  jax.tree_util.tree_map(jnp.asarray, ps.astype(float)))

                ps_samps[i] = soln.params
        else:
            N_samps = 5
        
    params=[]
    uncs_up =[]
    uncs_low=[]
    #all_params = np.empty((setup.parameters.shape[0], len(samples[parameters["Name"][0]])))
    if (useMCMC == True):
        all_params = np.empty((setup.parameters.shape[0], len(samples[parameters["Name"][0]])))
        l =len(samples[parameters["Name"][0]])
    else:
        all_params = np.empty((setup.parameters.shape[0], N_samps))
        l=N_samps
    
    j=0

    for i in range(len(parameters)):
        name = parameters["Name"][i]

       # value = parameters["Value"][i]
        all_params[i] =np.full(l,  setup.parameters['Value'][i])

            
        if (useMCMC==True):
            p_samples = samples[name]
        elif (useMCMC==False and BootStrap == True):
            p_samples = ps_samps[:, i]
        else:
            p_samples = all_params[i]
            
        hist, bins = np.histogram(p_samples, bins=20, density=True)
        bin_cents = bins+(bins[1]-bins[0])/2.0
        mode = bin_cents[np.argmax(hist)]
        value=np.mean(p_samples)#mode
        params.append(value)
        #value = np.percentile(p_samples, [16, 50, 84])[1]
        #params.append(np.percentile(p_samples, [16, 50, 84])[1])
        uncs_up.append(np.percentile(p_samples, [16, 50, 84])[2] - np.percentile(p_samples, [16, 50, 84])[1])
        uncs_low.append(np.percentile(p_samples, [16, 50, 84])[1] - np.percentile(p_samples, [16, 50, 84])[0])
        all_params[i] = np.array(p_samples)

    # Generate Output Directory

    if(binNo == 0):
        fldr = 'Quick'
    elif(binNo == 1):
        fldr = 'MCMC'
    elif(binNo== 2):
        fldr = 'Bootstrap'

    resultsdir =dir_path+"/Results/"+ObjName+"/"+ExtTypeOut+"/"+fldr + "/"
    if not os.path.exists(dir_path+"/Results/"):
        os.mkdir(dir_path+"/Results/")
    if not os.path.exists(dir_path+"/Results/"+ObjName+"/"):
        os.mkdir(dir_path+"/Results/"+ObjName+"/")
    if not os.path.exists(dir_path+"/Results/"+ObjName+"/"+ExtTypeOut+"/"):
        os.mkdir(dir_path+"/Results/"+ObjName+"/"+ExtTypeOut+"/")   
    if not os.path.exists(resultsdir):
        os.mkdir(resultsdir)

    if not os.path.exists(resultsdir+"/Plots/"):
        os.mkdir(resultsdir+"/Plots/")
    if not os.path.exists(resultsdir+"/Model Components/"):
        os.mkdir(resultsdir+"/Model Components/")
    if not os.path.exists(resultsdir+"/Ext. Corrections/"):
        os.mkdir(resultsdir+"/Ext. Corrections/")


    setup.parameters.loc[setup.parameters['Fixed'] == False, 'Value'] = params
    setup.parameters.loc[setup.parameters['Fixed'] == False, '+Error'] = uncs_up
    setup.parameters.loc[setup.parameters['Fixed'] == False, '-Error'] = uncs_low

    
    setup.ps = setup.parameters["Value"].to_numpy()

    ps = setup.ps
    np.savetxt(resultsdir+'params.txt', np.transpose([np.ones(len(ps)),ps]))
      
        

        
        
        
    # Print best fit parameters
    print("")
    print("")
    print("BEST FIT PARAMETERS")
    print("")
    Table.pprint_all(Table.from_pandas(setup.parameters))

    ################################### PLOTTING ######################################################



    fig,ax = plt.subplots(figsize=(7, 2))

    wav = np.linspace(min(lam), max(lam), 50000)

    cont = SetupFit.MixedCont(wav, ps[cont_indx], ExtType, ps[ext_indx])#SetupFit.DustCont(wav, ps[cont_indx])[0]
    # cont = SetupFit.contTemp(wav, ps[cont_indx])
    ext = SetupFit.Ext(wav, ps[ext_indx], ps[cont_indx])[0]
    cont = cont*ext
    stellar = SetupFit.Stellar(wav, ps[stellar_indx], ps[cont_indx], ExtType, ps[ext_indx], jax=True)#*ext

    #Stellar = SetupFit.ReturnStellar(wav, ps[cont_indx], ExtType)




    E = flux_err#setup.flux_orig*(10**(ps[-1])) + ps[-2]
#* SetupFit.ErrorScale(x_data, ps[ext_indx])#* np.exp(2 * ps[ext_indx][-1])
    ax.errorbar(lam, setup.flux_orig*SCALE/lam, color="black", yerr=E*SCALE/lam, ls='none',marker='.', ms=.05, elinewidth=0.1, zorder=0)
  #  ax.errorbar(lam, setup.flux_orig*SCALE, color="grey", yerr=E*SCALE, ls='none',marker='none', ms=.05, elinewidth=0.1, zorder = 0)

    print(lam)
    print(SCALE)
    
    # Ext parameters
    ax.plot(wav, cont*SCALE/wav, color="goldenrod", ls="solid", lw=0.25, label="Dust Cont.", zorder = 1)
 #   ax.plot(wav, Stellar*SCALE, color="grey", ls="solid", lw=0.25, label="Stellar", zorder = 1)


    sil=0.0


    # ax.plot(wav, Nuc*SCALE, color='purple', ls="solid", lw=0.25, label="Nuclear Continuum", zorder = 1)

    ax.plot(wav, stellar*SCALE/wav, color="tab:blue", ls="solid", lw=0.25, label="Stellar Cont.", zorder = 1)



    # Plot PAHs




    pah =  SetupFit.PAH(wav, ps[pah_indx])[0]
    #ax.plot(wav,   ext[2]*pah*SCALE, color="green", ls="solid", lw=0.25, label="PAH's")
    ax.plot(wav,   pah*SCALE/wav, color="green", ls="solid", lw=0.25, label="PAH's")
    pahs= SetupFit.PAH(wav, ps[pah_indx])[1]


    for v in range(len(pahs)):
        ax.plot(wav,   pahs[v]*SCALE/wav, color="tab:purple", ls="solid", lw=0.15, zorder=0)

    ax.plot(wav,   pahs[-1]*SCALE/wav, color="tab:purple", ls="solid", lw=0.15, zorder=0, label='PAH Components')



    # Plot Full Model

    #total =  SetupFit.Lines(wav, ps[lines_indx]) + SetupFit.PAH(wav, ps[pah_indx])[0] +  ext[1]*cont
    total =  SetupFit.PAH(wav, ps[pah_indx])[0] + cont + sil + stellar# + Nuc

        
    # ax.plot(wav,  ext[2]*total*SCALE, color="red", ls="solid", lw=0.25, label="Full Model")
    ax.plot(wav,  total*SCALE/wav, color="red", ls="solid", lw=0.25, label="Full Model")



    ax.set_ylabel('$f_{\\nu}/\\lambda$ (Jy/$\mu$m)')
    #ax.set_xlabel("Rest Wavelength ($\mu m$)")
   # if (PAHFIT!=True):
      #  ext=np.ones(10)
      #  cont*=SCALE
  #  resid = (np.interp(lam, wav,  ext[2]*total*SCALE) - flux*SCALE)/(flux_err*SCALE)
  #  resid = (np.interp(lam, wav,  total*SCALE) - flux*SCALE)/(E*SCALE)

    #ax1.errorbar(lam, resid, color="black",  yerr = flux_err/flux_err,ls='none',marker='.', ms=.1, elinewidth=0.1)
    #ax1.hlines(0, min(lam)-0.5, max(lam)+0.5, color='black', ls='solid', lw=0.5)
    ax.set_xlabel("Rest Wavelength ($\mu m$)")
   # ax1.set_ylabel("$\chi$")
   # ax1.set_ylim(-6,6)
    
    extt = SetupFit.ReturnExt(wav, ps[cont_indx], ExtType, ps[ext_indx], ps[stellar_indx])

    ax.set_ylim( 0.05*np.percentile(setup.flux_orig*SCALE, 1) , 5.0*np.percentile(setup.flux_orig*SCALE, 90))
    ax.set_yscale('log')
    ax.set_xscale('log')

    if (min(lam)<4.0):
        ax.set_xticks([ 1.5, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25])
    else:
        ax.set_xticks([ 5, 6, 8, 10, 12, 15, 20, 25])



    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlim(min(lam)-0.05, max(lam)+0.5)
    ax.legend(frameon=True, prop={'size': 5}, loc='lower right')
    fig.savefig(resultsdir+"/Plots/"+ObjName+"_Plot.pdf")
    ax.set_yscale('linear')
    ax.set_xscale('linear')
 

    #plt.show()
    plt.close()



    if (ExtType == 0.0):
        fig,ax = plt.subplots(1, 2, figsize = (5,2))
        #P = np.reshape(np.exp(ps[cont_indx][:-1]),(setup.grid_size[1], setup.grid_size[0]+1))[:,:setup.grid_size[0]]
        P = np.reshape(jnp.exp(ps[cont_indx][:-1]),(grid_size[1], grid_size[0]))#[:,:setup.grid_size[0]]

        T = np.logspace(np.log10(35), np.log10(1500), grid_size[0])
        tau = np.logspace(np.log(0.05), np.log(tau_lim), grid_size[1], base=np.e)
        # P = np.reshape(ps[cont_indx][:int(-1-2.0*setup.grid_size)], (setup.grid_size, setup.grid_size+1))[:,:setup.grid_size]

        # T = 10**np.cumsum(ps[cont_indx][int(-1-2.0*setup.grid_size):int(-1-setup.grid_size)])
        # tau = np.exp(np.cumsum(ps[cont_indx][int(-1-setup.grid_size):-1]))


        dlogT = np.diff(np.log10(T))[0]

        T, tau = np.meshgrid(T, tau)
        # P_interp = interp2d(np.log10(T), np.log10(tau), P, kind='cubic')(np.linspace(np.log10(20), np.log10(1500), 1000), np.linspace(np.log(0.01), np.log(15), 1000))
        #P = np.transpose(P)
        #ax[1].imshow(P_interp, extent=[np.log10(20), np.log10(1500), np.log(15), np.log(0.01)], norm = PowerNorm(gamma=0.5))#, norm=LogNorm(vmin=0.001*np.max(P), vmax=np.max(P)))
        #pcol = ax[0].pcolormesh(T, tau, P, norm = PowerNorm(gamma=0.5), antialiased=True, linewidth=0, rasterized=True)
        pcol = ax[0].pcolormesh(T, tau, P*setup.norm_ratio,  norm = PowerNorm(gamma=0.5), antialiased=True, linewidth=0, rasterized=True)

        pcol.set_edgecolor('face')




       # TT, tautau = np.meshgrid(np.log10(T),np.log(tau))
        # ax[0].scatter(T, tau, s=10, c=P**0.5, cmap='viridis' )
        # ax[0].invert_yaxis()


      #  ax[0].set_aspect(.2)
        ax[0].set_xlabel('$\log_{10}(T$ (K))')
        ax[0].set_ylabel('$\ln(\\tau)$')



        ax[0].set_xscale('log')
        ax[0].set_yscale('log', base = np.e)
        ax[0].set_xlabel('$T$ (K)')
        ax[0].set_ylabel('$\\tau_{9.8}$ ')


        ax[0].set_xticks([ 35, 100, 1000])
        ax[0].get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        if (extended==True):
            ax[0].set_yticks([ 0.1, 1.0, 10.0, 50.0])
        else:
            ax[0].set_yticks([ 0.1, 1.0, 10.0])
        ax[0].get_yaxis().set_major_formatter(mticker.ScalarFormatter())
        ax[0].set_xlim(35, 1500)
        ax[0].set_ylim(0.05, tau_lim)
        ax[0].invert_yaxis()

        #ax[0].set_xlabel('$\log_{10}(T$ (K))')
       # ax[0].set_ylabel('$\ln(\\tau)$')

        ax[1].errorbar(lam, setup.flux_orig*SCALE, color="grey", yerr=flux_err*SCALE, ls='none',marker='.', ms=.05, elinewidth=0.1, zorder = 0)
        ax[1].plot(wav,  total*SCALE, color="tab:red", ls="solid", lw=0.5, label="Full Model")
        ax[1].plot(wav, SetupFit.MixedCont(wav, ps[cont_indx], ExtType, ps[ext_indx])*SCALE*ext, ls='solid', c='black', lw=0.5, label='Dust Cont.')
        ax[1].plot(wav, stellar*SCALE, ls='solid', c='tab:orange', lw=0.5, label='Stellar Cont.')

        ax[1].set_xlabel('Wavelength ($\mu$m)')
        ax[1].set_ylabel('$f_{\\nu}$ (Jy)')
        ax[1].set_yscale('log')
        ax[1].set_ylim( 0.5*np.min(setup.flux_orig*SCALE), 1.2*np.max(setup.flux_orig*SCALE))
        ax[1].set_xscale('log')
        ax[1].set_xticks([1.5,  3, 5, 10, 25])
        ax[1].get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        fig.suptitle(ObjName)
        plt.tight_layout()

        fig.savefig(resultsdir+"/Plots/"+ObjName+'_Psi.pdf')



    # fig.savefig(resultsdir+ObjName+'_Psi.pdf')
    # plt.close()
    #Save Components to file 
    np.savetxt(resultsdir+"/Model Components/"+ObjName+"_FullContinuum.txt", np.transpose([wav, cont*SCALE]))
#    np.savetxt(resultsdir+ObjName+"_IceCorrectedContinuum.txt", np.transpose([wav, cont*SCALE]))
   # np.savetxt(resultsdir+ObjName+"Lines.txt", np.transpose([wav,np.array(lin)*SCALE]))
    np.savetxt(resultsdir+"/Model Components/"+ObjName+"PAHs.txt", np.transpose([wav, np.array(pah)*SCALE]))
    np.savetxt(resultsdir+"/Model Components/"+ObjName+"PAHs_data.txt", np.transpose([lam, setup.flux_orig*SCALE - np.interp(lam, wav,cont*SCALE), E*SCALE]))
    np.savetxt(resultsdir+"/Model Components/"+ObjName+"FullModel.txt", np.transpose([wav,np.array(total)*SCALE]))
    np.savetxt(resultsdir+"/Model Components/"+ObjName+"Spectrum.txt", np.transpose([lam, setup.flux_orig*SCALE, E*SCALE]))
    np.savetxt(resultsdir+"/Model Components/"+ObjName+"Stellar.txt", np.transpose([wav, stellar*SCALE]))

#    np.savetxt(resultsdir+ObjName+"Extinction.txt", np.transpose([wav, extt*ext]))
#    np.savetxt(resultsdir+ObjName+"Extinction_IceCorr.txt", np.transpose([wav, extt]))
 #   np.savetxt(resultsdir+ObjName+"_UnObscuredContinuum.txt", np.transpose([wav, cont*SCALE/(extt*ext)]))
    np.savetxt(resultsdir+"/Model Components/"+ObjName+"PAHs_components.txt", np.transpose(pahs*SCALE))


    
    ################################## LINE PROPERTIES #######################################################
    # Output csv
    output = pd.DataFrame(columns=['Name', 'Rest Wavelength (micron)','Strength (10^-17 W/m^2)', 'S_err+','S_err-', 'Continuum (10^-17 W/m^2/um)', 'C_err+','C_err-','Eqw (micron)', 'E_err+','E_err-'])
    
    
    
    #Find Continuum samples
    cont_samps=[]
    sil_samps=[]
    sil_Psi_samps=[]
    ice_samps=[]
    print("Extracting Continuum Samples...")#
   # wav = np.linspace(min(lam), max(lam)+2, 1000)
    #wav_forIntegral = np.linspace(min(lam), 15.0, 1000)
    eqws_samples_lines = np.empty((setup.Nlines, len(all_params[0])))
    str_samples_lines  =  np.empty((setup.Nlines, len(all_params[0])))
    cont_samples_lines  = np.ones((setup.Nlines, len(all_params[0])))
    eqws_samples_pahs = np.empty((setup.Npah, len(all_params[0])))
    str_samples_pahs  =  np.empty((setup.Npah, len(all_params[0])))
    cont_samples_pahs  = np.ones((setup.Npah, len(all_params[0])))



    eqws_samples_33 = np.ones(len(all_params[0]))
    str_samples_33  =  np.ones(len(all_params[0]))
    cont_samples_33  = np.ones(len(all_params[0]))

    eqws_samples_52 = np.ones(len(all_params[0]))
    str_samples_52  =  np.ones(len(all_params[0]))
    cont_samples_52  = np.ones(len(all_params[0]))


    eqws_samples_57 = np.ones(len(all_params[0]))
    str_samples_57  =  np.ones(len(all_params[0]))
    cont_samples_57  = np.ones(len(all_params[0]))

    eqws_samples_6 = np.ones(len(all_params[0]))
    str_samples_6  =  np.ones(len(all_params[0]))
    cont_samples_6  = np.ones(len(all_params[0]))

    eqws_samples_7 = np.ones(len(all_params[0]))
    str_samples_7  =  np.ones(len(all_params[0]))
    cont_samples_7  = np.ones(len(all_params[0]))

    eqws_samples_8 = np.ones(len(all_params[0]))
    str_samples_8  =  np.ones(len(all_params[0]))
    cont_samples_8  = np.ones(len(all_params[0]))
    
    
    eqws_samples_11 = np.ones(len(all_params[0]))
    str_samples_11  =  np.ones(len(all_params[0]))
    cont_samples_11  = np.ones(len(all_params[0]))
    
    eqws_samples_12 = np.ones(len(all_params[0]))
    str_samples_12  =  np.ones(len(all_params[0]))
    cont_samples_12  = np.ones(len(all_params[0]))
    
    eqws_samples_17 = np.ones(len(all_params[0]))
    str_samples_17  =  np.ones(len(all_params[0]))
    cont_samples_17  = np.ones(len(all_params[0]))

    

    wav = lam
    wav_expanded = np.logspace(np.log10(1.0), np.log10(1000.0), 3000)
    nu = 2.9979246e14/wav
    if (ExtType == 0):
        dlogT = np.diff(np.log10(np.logspace(np.log10(35), np.log10(1500), setup.grid_size[0])))[0]
        T = np.logspace(np.log10(35), np.log10(1500), setup.grid_size[0])
        tau_pix = np.linspace(1, len(T),  setup.grid_size[0])
        tau_pix_smooth = np.linspace(1, len(T), 100)

        tau = np.logspace(np.log(0.05), np.log(tau_lim), setup.grid_size[1], base=np.e)
        Psi_samples = np.empty((len(all_params[0]), setup.grid_size[1], setup.grid_size[0]))


    AGN_conts = np.empty((len(all_params[0]), len(wav)))
    SF_conts = np.empty((len(all_params[0]), len(wav)))
    SF_ext_fac = np.empty((len(all_params[0]), len(wav)))
    Full_ext_fac = np.empty((len(all_params[0]), len(wav)))
    StellarCont_ext_fac = np.empty((len(all_params[0]), len(wav)))
    Ice_ext_fac = np.empty((len(all_params[0]), len(wav)))


    mean_T = np.empty(len(all_params[0]))
    mean_tau = np.empty(len(all_params[0]))

    mean_T2 = np.empty(len(all_params[0]))
    mean_tau2 = np.empty(len(all_params[0]))
    mean_T3 = np.empty(len(all_params[0]))
    mean_tau3 = np.empty(len(all_params[0]))

    AGN_fracs = np.empty(len(all_params[0]))
    cuttoffs = np.empty(len(all_params[0]))
    PAH_to_SF_ratio = np.random.normal(0.12, 0.03, len(all_params[0]))
    tau_means_samples = np.empty((len(all_params[0]), 20))

    logistic_samples = np.empty((len(all_params[0]), 100))
    Cumulative_mean_T_samples = np.empty((len(all_params[0]), 20))

    St_Cont_ext = np.empty(len(all_params[0]))
    # ext_curve = np.loadtxt('D23ExtCurve.txt', unpack=True, usecols=[0,1])
    # lss, SS = np.loadtxt('SilProfile.txt', unpack=True, usecols=[0,1])
    ps_mean = np.copy(ps)
    for i in tqdm(range(len(all_params[0]))):


        ps = all_params[:,i]
        #ext = 1.0# SetupFit.Ext(wav, ps[ext_indx])[2]
        cont = SetupFit.MixedCont(wav, ps[cont_indx], ExtType, ps[ext_indx])*SCALE+sil*SCALE
        PAHss = SetupFit.PAH(wav, ps[pah_indx])[0]*SCALE
        ext = SetupFit.ReturnExt(wav, ps[cont_indx], ExtType, ps[ext_indx], ps[stellar_indx])
        stellar = SetupFit.Stellar(wav, ps[stellar_indx], ps[cont_indx], ExtType, ps[ext_indx])*SCALE


        Full_ext_fac[i, :] = ext
        Ice_ext_fac[i, :] = SetupFit.ReturnIceExt(wav, ps[cont_indx], ExtType, ps[ext_indx], ps[stellar_indx])


        Tau = np.interp(wav, ext_curve[0], ext_curve[1])
        StellarCont_ext_fac[i, :] = np.exp(-ps[stellar_indx][-1]*Tau)
        St_Cont_ext[i] = ps[stellar_indx][-1]

        # Psi
        if (ExtType == 0.0):

            #Psi_fit = np.reshape(jnp.exp(ps[cont_indx][:-1]),(setup.grid_size[1], setup.grid_size[0]+1))[:,:setup.grid_size[0]]
            start = time.time()

            Psi_fit = np.reshape(jnp.exp(ps[cont_indx][:-1]),(setup.grid_size[1], setup.grid_size[0]))#[:,:setup.grid_size[0]]



            
            lam_ = np.linspace(5, 25.0, 1000)

            cont_intr = SetupFit.MixedCont(lam_,  ps[cont_indx], ExtType, ps[ext_indx])*SCALE
            PAH_flux = SetupFit.PAH(lam_, ps[pah_indx])[0]*SCALE
            cont_flux = np.trapz(y=cont_intr, x=lam_)

            #PAH_flux = np.trapz(y=PAH_flux, x=lam_)

            #cont_fluxes=np.zeros(np.shape(Psi_fit)[0])
            # Compute the cumulative sum up to each column
            Psi_fit/=np.sum(Psi_fit)
           # figggg, a = plt.subplots()


            #tau_means_samples[i, :] = np.interp(tau_means_samples[i,:], tau_p, tau )
            tau_means_samples[i, :]  = SetupFit.ReturnEffTau98(x_data, ps[cont_indx], ps[ext_indx], ps[stellar_indx], jax=True)

            Psi_samples[i, :, :] = Psi_fit

            w_x = np.sum(Psi_fit*setup.norm_ratio, axis=0)
            w_y = np.sum(Psi_fit*setup.norm_ratio, axis=1)
            if (np.all(w_x==0.0)):
                mean_T3[i] = np.nan
                mean_tau3[i] = np.nan
            else:
                mean_T3[i] = np.average(T, weights = w_x)
                mean_tau3[i] = np.average(tau, weights = w_y)


        for j in range(setup.Nlines):
            flux_interp_onto_model = np.interp(wav, lam, setup.flux_orig*SCALE)
            #line =  np.array((flux_interp_onto_model - ( PAHss + stellar+(cont)*SetupFit.Ext(wav, ps[ext_indx], ps[cont_indx])[0]))/ext)  #SetupFit.Lines(wav, ps[lines_indx][int(6*j):int(6*j+6)])*SCALE/ext#[2]
            line =  np.array((flux_interp_onto_model - ( PAHss + stellar+(cont)*SetupFit.Ext(wav, ps[ext_indx], ps[cont_indx])[0])))  #SetupFit.Lines(wav, ps[lines_indx][int(6*j):int(6*j+6)])*SCALE/ext#[2]


            # Make small continuum correction

            ll = setup.linecents[j]-setup.width_limits[j]
            ul = setup.linecents[j]+setup.width_limits[j]

            medl = np.median(line[(wav>ll)&(wav<ul)][:10] )
            medu = np.median(line[(wav>ll)&(wav<ul)][-10:] )
            medl_x = np.mean(wav[(wav>ll)&(wav<ul)][:10])
            medu_x = np.mean(wav[(wav>ll)&(wav<ul)][-10:])

            corr_ = np.interp(wav[(wav>ll)&(wav<ul)], [medl_x, medu_x], [medl, medu])

            line[(wav>ll)&(wav<ul)] -=corr_




            # if (i==1):
            #     # plt.plot(wav, flux_interp_onto_model, lw=0.2, color='tab:purple', ls='solid')
            #     # plt.plot(wav, PAHss + cont, lw=0.2, color='tab:green', ls='solid')


            #     plt.plot(wav, line, lw=0.2, color='black', ls='solid', zorder=2)
            #     plt.plot(wav[(wav>(setup.linecents[j]-setup.width_limits[j])) & (wav<(setup.linecents[j]+setup.width_limits[j]))], line[(wav>(setup.linecents[j]-setup.width_limits[j])) & (wav<(setup.linecents[j]+setup.width_limits[j]))], color='tab:red', lw=0.3, ls='dashed', zorder=3)

            #Create plots of each emission line
            if (i==0):
                plt.figure(figsize = (3,3))
                #plt.errorbar(lam, setup.flux_orig*SCALE - np.interp(lam, wav, ) , color="grey", yerr=flux_err*SCALE, ls='none',marker='.', ms=.05, elinewidth=0.1, zorder = 0)
                plt.errorbar(lam[(wav>ll)&(wav<ul)], (setup.flux_orig*SCALE - np.interp(lam, wav, PAHss + stellar+(cont)*SetupFit.Ext(wav, ps[ext_indx], ps[cont_indx])[0]))[(wav>ll)&(wav<ul)] , color="black", yerr=(flux_err*SCALE)[(wav>ll)&(wav<ul)], ls='solid',marker='.', ms=.1, elinewidth=0.1, zorder = 0, lw=0.5)
                plt.plot(wav[(wav>ll)&(wav<ul)], corr_, color='tab:red', lw=0.3, ls='dashed', zorder=3)

                #plt.plot(wav, flux_interp_onto_model, lw=0.2, color='black', ls='solid')
               # plt.plot(wav,  PAHss + stellar+(cont)*SetupFit.Ext(wav, ps[ext_indx], ps[cont_indx])[0], lw=0.2, color='tab:green', ls='solid')
                plt.axvline(ll, ls='dashed', lw=0.5, color='black', alpha=0.5)
                plt.axvline(ul, ls='dashed', lw=0.5, color='black', alpha=0.5)
                plt.xlabel('Rest Wavelength ($\\mu$m')
                plt.ylabel('Flux')
                plt.title(setup.linenames[j])


                #plt.xlim(0.99*ll, 1.01*ul)
                #plt.ylim( 0.99*min(line), 1.01*max(line))


                if not os.path.exists(resultsdir+"/Plots/EmissionLines/"):
                    os.mkdir(resultsdir+"/Plots/EmissionLines/")
        
                plt.savefig(resultsdir+"/Plots/EmissionLines/"+setup.linenames[j]+"("+str(np.round(setup.linecents[j], 3))+" mu).pdf")



            #eqws_samples_lines[j, i] = np.trapz(y = (line*ext/(stellar+cont))[(wav>(setup.linecents[j]-setup.width_limits[j])) & (wav<(setup.linecents[j]+setup.width_limits[j]))], x=wav[(wav>(setup.linecents[j]-setup.width_limits[j])) & (wav<(setup.linecents[j]+setup.width_limits[j]))])
            eqws_samples_lines[j, i] = np.trapz(y = (line/(stellar+cont*SetupFit.Ext(wav, ps[ext_indx], ps[cont_indx])[0]))[(wav>(setup.linecents[j]-setup.width_limits[j])) & (wav<(setup.linecents[j]+setup.width_limits[j]))], x=wav[(wav>(setup.linecents[j]-setup.width_limits[j])) & (wav<(setup.linecents[j]+setup.width_limits[j]))])

            line = line[(wav>(setup.linecents[j]-setup.width_limits[j])) & (wav<(setup.linecents[j]+setup.width_limits[j]))]
            e_line = np.interp(wav, lam, setup.data[2]*SCALE)[(wav>(setup.linecents[j]-setup.width_limits[j])) & (wav<(setup.linecents[j]+setup.width_limits[j]))]
            line = np.random.normal(line, e_line) # Resample based on error due to cont. correction

            nu_line = 2.9979246e14/(wav[(wav>(setup.linecents[j]-setup.width_limits[j])) & (wav<(setup.linecents[j]+setup.width_limits[j]))])
            str_samples_lines[j, i] = np.trapz(y = line[::-1], x=nu_line[::-1])
            cont_samples_lines[j, i] = np.interp(setup.linecents[j], wav, stellar+cont*SetupFit.Ext(wav, ps[ext_indx], ps[cont_indx])[0])


            #######
        # if (i==1):
        #     plt.show()

            ########
        cont = stellar+cont*SetupFit.Ext(wav, ps[ext_indx], ps[cont_indx])[0]
        # PAHs
        for j in range(setup.Npah):
            #line =  SetupFit.PAH(wav, ps[pah_indx][int(4*j):int(4*j+4)])[0]*SCALE/ext#[2]
            line =  SetupFit.PAH(wav, ps[pah_indx][int(4*j):int(4*j+4)])[0]*SCALE#/ext#[2]

            # # Only use points within 1% of max
            # wav_line = wav[line>0.01*np.max(line)]
            # nu_line = nu[line>0.01*np.max(line)]
            # ext_line = ext[line>0.01*np.max(line)]
            # line = line[line>0.01*np.max(line)]
            # cont_line = cont[line>0.01*np.max(line)]

            #eqws_samples_pahs[j, i] = np.trapz(y = line*ext/cont, x=wav)
            eqws_samples_pahs[j, i] = np.trapz(y = line/cont, x=wav)

            str_samples_pahs[j, i] = np.trapz(y = line[::-1], x=nu[::-1])
            cont_samples_pahs[j, i] = np.interp(ps[pah_indx][int(4*j+1)], wav, cont)
        
        


      
        # PAH 6.2 complex
        line=0.0
        pah33 = False
        if (min(lam)<3.0 and max(lam)>3.5):
            pah33 = True
            for j in range(setup.Npah):
            
             #   if (setup.pahcents[j] == 6.18 or setup.pahcents[j] == 6.30):
                if (setup.pahcents[j] == 3.29):

                    line +=  SetupFit.PAH(wav, ps[pah_indx][int(4*j):int(4*j+4)])[0]*SCALE#/ext#[2]
                    

            eqws_samples_33[i] = np.trapz(y = line/cont, x=wav)
            str_samples_33[i] = np.trapz(y = line[::-1], x=nu[::-1])
            cont_samples_33[i] = np.interp(3.3, wav, cont)
            



        # PAH 5.2 complex
        line=0.0
        pah52 = False
        if (min(lam)<5.0 and max(lam)>5.4):
            pah52 = True
            for j in range(setup.Npah):
            
             #   if (setup.pahcents[j] == 6.18 or setup.pahcents[j] == 6.30):
                if (setup.pahcents[j] == 5.175 or setup.pahcents[j]== 5.24):

                    line +=  SetupFit.PAH(wav, ps[pah_indx][int(4*j):int(4*j+4)])[0]*SCALE#/ext#[2]
                    

            eqws_samples_52[i] = np.trapz(y = line/cont, x=wav)
            str_samples_52[i] = np.trapz(y = line[::-1], x=nu[::-1])
            cont_samples_52[i] = np.interp(5.25, wav, cont)
            

        # PAH 5.7 complex
        line=0.0
        pah57 = False
        if (min(lam)<5.5 and max(lam)>5.9):
            pah57 = True
            for j in range(setup.Npah):
            
             #   if (setup.pahcents[j] == 6.18 or setup.pahcents[j] == 6.30):
                if (setup.pahcents[j] == 5.64 or setup.pahcents[j]== 5.7 or setup.pahcents[j]==5.76):

                    line +=  SetupFit.PAH(wav, ps[pah_indx][int(4*j):int(4*j+4)])[0]*SCALE#/ext#[2]
                    

            eqws_samples_57[i] = np.trapz(y = line/cont, x=wav)
            str_samples_57[i] = np.trapz(y = line[::-1], x=nu[::-1])
            cont_samples_57[i] = np.interp(5.7, wav, cont)

   
        # PAH 6.2 complex
        line=0.0
        pah62 = False
        if (min(lam)<6.0 and max(lam)>6.3):
            pah62 = True
            for j in range(setup.Npah):
            
             #   if (setup.pahcents[j] == 6.18 or setup.pahcents[j] == 6.30):
                if (setup.pahcents[j] == 6.2):

                    line +=  SetupFit.PAH(wav, ps[pah_indx][int(4*j):int(4*j+4)])[0]*SCALE#/ext#[2]
                    

            eqws_samples_6[i] = np.trapz(y = line/cont, x=wav)
            str_samples_6[i] = np.trapz(y = line[::-1], x=nu[::-1])
            cont_samples_6[i] = np.interp(6.2, wav, cont)
            
            
        # PAH 7.7 complex
        line=0.0
        pah77 = False
        if (min(lam)<7.0 and max(lam)>7.9):
            pah77 = True
            for j in range(setup.Npah):
            
            #    if (setup.pahcents[j] == 7.42 or setup.pahcents[j]  == 7.60 or setup.pahcents[j]  == 7.85):
                #if (setup.pahcents[j] == 7.42 or setup.pahcents[j] == 7.55 or setup.pahcents[j]  == 7.61 or setup.pahcents[j]  == 7.82):
                if (setup.pahcents[j] == 7.42 or setup.pahcents[j] == 7.55 or setup.pahcents[j]  == 7.61 or setup.pahcents[j]  == 7.82):

                    line +=  SetupFit.PAH(wav, ps[pah_indx][int(4*j):int(4*j+4)])[0]*SCALE#/ext#[2]
                    

            eqws_samples_7[i] = np.trapz(y = line/cont, x=wav)
            str_samples_7[i] = np.trapz(y = line[::-1], x=nu[::-1])
            cont_samples_7[i] = np.interp(7.7, wav, cont)
            
            
        # PAH 8.6 complex
        line=0.0
        pah86 = False
        if (min(lam)<8.5 and max(lam)>8.7):
            pah86 = True
            for j in range(setup.Npah):
            
                if (setup.pahcents[j] == 8.5 or setup.pahcents[j] == 8.61 ):
                    line +=  SetupFit.PAH(wav, ps[pah_indx][int(4*j):int(4*j+4)])[0]*SCALE#/ext#[2]
                    

            eqws_samples_8[i] = np.trapz(y = line/cont, x=wav)
            str_samples_8[i] = np.trapz(y = line[::-1], x=nu[::-1])
            cont_samples_8[i] = np.interp(8.6, wav, cont)
        
        
        # PAH 11.3 complex
        line=0.0
        pah113 = False
        if (min(lam)<11.0 and max(lam)>11.3):
            pah113 = True
            for j in range(setup.Npah):
            
             #   if (setup.pahcents[j] == 11.20  or setup.pahcents[j] == 11.22  or setup.pahcents[j] == 11.25 or setup.pahcents[j]  == 11.33):
                if (setup.pahcents[j] == 11.20  or setup.pahcents[j] == 11.26 or setup.pahcents[j] == 11.25):

                    line +=  SetupFit.PAH(wav, ps[pah_indx][int(4*j):int(4*j+4)])[0]*SCALE#/ext#[2]
                    

            eqws_samples_11[i] = np.trapz(y = line/cont, x=wav)
            str_samples_11[i] = np.trapz(y = line[::-1], x=nu[::-1])
            cont_samples_11[i] = np.interp(11.3, wav, cont)
  
        
        # PAH 12.7 complex
        line=0.0
        pah127 = False
        if (min(lam)<12.5 and max(lam)>13.0):
            pah127 = True
            for j in range(setup.Npah):
            
               # if (setup.pahcents[j] == 12.75 or setup.pahcents[j]  == 12.69):
                if (setup.pahcents[j] == 12.6 or setup.pahcents[j]  == 12.77):

                    line +=  SetupFit.PAH(wav, ps[pah_indx][int(4*j):int(4*j+4)])[0]*SCALE#/ext#[2]
            
            eqws_samples_12[i] = np.trapz(y = line/cont, x=wav)
            str_samples_12[i] = np.trapz(y = line[::-1], x=nu[::-1])
            cont_samples_12[i] = np.interp(12.7, wav, cont)
        
        
        # PAH 17 complex
        line=0.0
        pah17 = False
        if (min(lam)<16.4 and max(lam)>17.9):
            pah17 = True
            for j in range(setup.Npah):
            
                if (setup.pahcents[j] == 16.45 or setup.pahcents[j]  == 17.04 or  setup.pahcents[j]  == 17.375):
                    line +=  SetupFit.PAH(wav, ps[pah_indx][int(4*j):int(4*j+4)])[0]*SCALE#/ext#[2]
                    

            eqws_samples_17[i] = np.trapz(y = line/cont, x=wav)
            str_samples_17[i] = np.trapz(y = line[::-1], x=nu[::-1])
            cont_samples_17[i] = np.interp(17.0, wav, cont)
        


    np.savetxt(resultsdir+"/Ext. Corrections/"+ObjName+'Full_ext_fac.txt', np.transpose([wav, np.mean(Full_ext_fac, axis = 0), np.std(Full_ext_fac, axis = 0)]))
    np.savetxt(resultsdir+"/Ext. Corrections/"+ObjName+'NIR_Ice_ext_fac.txt', np.transpose([wav, np.mean(Ice_ext_fac, axis = 0), np.std(Ice_ext_fac, axis = 0)]))

    if (ExtType == 0.0):
        np.savetxt(resultsdir+"/Ext. Corrections/"+ObjName+'StellarCont._ext_fac.txt', np.transpose([wav, np.mean(StellarCont_ext_fac, axis = 0), np.std(StellarCont_ext_fac, axis = 0)]))



    #Lines
    lam_ = np.copy(lam)
    #line_params = all_params[lines_indx]
    H2_fluxes=[]
    H2_lams=[]
    HI_fluxes=[]
    H2_linecents=[]
    for i in range(setup.Nlines):

        
        lam = setup.linecents[i] #line_params[c]

        strr = np.percentile(str_samples_lines[i]*1e-9, [16,50,84])#
        str_med = strr[1]
        str_l = strr[1]-strr[0]
        str_u = strr[2]-strr[1]
        # Upper limits
       # if (str_med/str_l < 3.0):
         #   str_med = "<" + str(np.max(strength))
         #   str_l = "-"
         #   str_u = "-"
            

        #continuum =  SetupFit.RunningOptimalAverageSamples(data[0], line_subtracted, line_subtracted_err, deltas, lam)[1]*1.0e-9*2.9979246e14/(lam**2) # 10^-17 W/m^2/um
       # continuum = np.empty(len(lam))
       # for j in range(len(lam)):
           # continuum[j] =  cont_samps[j](lam[j])*1.0e-9*2.9979246e14/(lam[j]**2) # 10^-17 W/m^2/um#
        
        cont = np.percentile(cont_samples_lines[i]*1.0e-9*2.9979246e14/(lam**2), [16,50,84]) # 10^-17 W/m^2/um#
        cont_med = cont[1]
        cont_l = cont[1]-cont[0]
        cont_u = cont[2]-cont[1]
        # Upper limits
       # if (cont_med/cont_l < 3.0):
           # cont_med = "<" + str(np.max(continuum))
         #   cont_l = "-"
          #  cont_u = "-"
            
            
     #   eqw = strength/continuum # micron
        Eqw = np.percentile(eqws_samples_lines[i], [16,50,84])#
        Eqw_med = Eqw[1]
        Eqw_l = Eqw[1]-Eqw[0]
        Eqw_u = Eqw[2]-Eqw[1]
        # Upper limits
      #  if (Eqw_med/Eqw_l < 3.0):
      #      Eqw_med = "<" + str(np.max(eqw))
       #     Eqw_l = "-"
        #    Eqw_u = "-"#
        
        if (setup.linenames[i].startswith("BrGamma") or setup.linenames[i].startswith("BrBeta") or setup.linenames[i].startswith("PfGamma")or setup.linenames[i].startswith("HuAlpha")):
            HI_fluxes.append(str_samples_lines[i]*1e-9)#*np.interp(lam, wav, ext))


        output =pd.concat([output, pd.DataFrame([{ 'Name': setup.linenames[i], 'Rest Wavelength (micron)': setup.linecents[i],'Strength (10^-17 W/m^2)': str_med,'S_err+': str_u, 'S_err-': str_l, 'Continuum (10^-17 W/m^2/um)': cont_med, 'C_err+': cont_u,'C_err-': cont_l,'Eqw (micron)': Eqw_med, 'E_err+': Eqw_u, 'E_err-': Eqw_l}])], ignore_index=True)

        if (setup.linenames[i].startswith("H2") and setup.linecents[i]>5.0):
            H2_fluxes.append(str_samples_lines[i]*1e-9)#*np.interp(lam, wav, ext))
            H2_linecents.append(setup.linecents[i])

    H2_fluxes = np.stack(H2_fluxes)
   # print(np.shape(H2_fluxes))
    H2_fluxes = np.transpose(H2_fluxes)
    H2_linecents = np.array(H2_linecents)

    #print(np.shape(H2_fluxes), np.shape(H2_linecents))

    A = np.array([32360.0, 20000.0, 11400.0, 5880.0, 2640.0, 980.0, 275.0, 47.6])
    g = np.array([21.0, 57.0, 17.0, 45.0, 13.0, 33.0, 9.0, 21.0])
    H2_Eu = np.array([8677.0, 7196.0, 5828.0, 4585.0, 3473.0, 2503.0, 1681.0, 1015.0])
    H2_l0 = np.array([5.053, 5.511, 6.109, 6.910, 8.025, 9.665, 12.279, 17.035])
    H2_labels = np.array(['S(8)', 'S(7)', 'S(6)', 'S(5)', 'S(4)', 'S(3)', 'S(2)', 'S(1)'])

    if (max(lam_)<17.0):
        A = np.array([32360.0, 20000.0, 11400.0, 5880.0, 2640.0, 980.0, 275.0])
        g = np.array([21.0, 57.0, 17.0, 45.0, 13.0, 33.0, 9.0])
        H2_Eu = np.array([8677.0, 7196.0, 5828.0, 4585.0, 3473.0, 2503.0, 1681.0])
        H2_l0 = np.array([5.053, 5.511, 6.109, 6.910, 8.025, 9.665, 12.279])
        H2_labels = np.array(['S(8)', 'S(7)', 'S(6)', 'S(5)', 'S(4)', 'S(3)', 'S(2)'])

    if (max(lam_)<12.2):
        A = np.array([32360.0, 20000.0, 11400.0, 5880.0, 2640.0, 980.0])
        g = np.array([21.0, 57.0, 17.0, 45.0, 13.0, 33.0])
        H2_Eu = np.array([8677.0, 7196.0, 5828.0, 4585.0, 3473.0, 2503.0])
        H2_l0 = np.array([5.053, 5.511, 6.109, 6.910, 8.025, 9.665])
        H2_labels = np.array(['S(8)', 'S(7)', 'S(6)', 'S(5)', 'S(4)', 'S(3)'])

    #print(np.shape(A), np.shape(g))
    A = A[H2_l0 == H2_linecents]
    g = g[H2_l0 == H2_linecents]
    H2_Eu  = H2_Eu[H2_l0 == H2_linecents]
    H2_labels   = H2_labels[H2_l0 == H2_linecents]
    H2_l0 = H2_l0[H2_l0 == H2_linecents]
    #print(np.shape(A), np.shape(g))

    f = jnp.log10(abs(np.array(H2_fluxes)/(A*g)))

    # ext_curve = np.loadtxt('CustomExtCurve3.txt', unpack=True, usecols=[0,1])
    # lss, SS = np.loadtxt('SilProfile.txt', unpack=True, usecols=[0,1])

    #D03_x, D03_y = np.loadtxt(dir_path+'/Ext.Curves/D03MWRV31.txt', unpack=True, usecols=[0,1])
    taus = np.linspace(-2.0, 10, 999)
    Tau = np.interp(H2_l0, ext_curve[0], ext_curve[1])
    #Tau = np.interp(H2_l0, D03_x, D03_y )

    smooth = []
    for i in range(len(taus)):
        ext_cor = np.exp(-taus[i]*Tau)
        f = np.log10(abs(np.array(H2_fluxes/ext_cor)/(A*g)))#, axis=0)#[4:]
        f = f[:, np.argsort(H2_Eu)]
        smooth.append(np.sum((f[:, :-2] - 2.0*f[:, 1:-1]+f[:, 2:])**2, axis=1))
    # plt.plot(taus, smooth)
    smooth = np.stack(smooth)

    tau_s=np.empty(np.shape(smooth)[1])
    for l in range(np.shape(smooth)[1]):
        if (len( taus[smooth[:,l]==np.min(smooth[:,l])])> 1 or len( taus[smooth[:,l]==np.min(smooth[:,l])])==0 ):
            tau_s[l] = np.nan
        else:
            tau_s[l] = taus[smooth[:,l]==np.min(smooth[:,l])]

    tau_err = np.nanstd(tau_s)
    #tau = np.nanmean(tau_s)
    tau = np.nanpercentile(tau_s, [16, 50, 84])
    tau_l = tau[1]-tau[0]
    tau_u = tau[2]-tau[1]
    tau_m = tau[1]

    print('H2_tau=',tau, '+/-', tau_err)
     # plt.savefig(resultsdir+ObjName+'H2Smoothness.pdf')
     # plt.close()
     #plt.show()

    f_org = np.mean(np.log10(abs(np.array(H2_fluxes)/(A*g))), axis=0)#[4:]
    f_org_err = np.std(np.log10(abs(np.array(H2_fluxes)/(A*g))), axis=0)
    f_0 = f_org[-1]
    ext_cor = np.exp(-tau_m*Tau)

    f_samples = np.log10(abs(np.array(H2_fluxes/ext_cor)/(A*g)))
    f = np.mean(f_samples, axis=0)#[4:]
    f_err = np.std(f_samples, axis=0)#[4:]


    temp_s = np.empty(np.shape(smooth)[1])
    for l in range(np.shape(smooth)[1]):
        gradient = np.diff(f_samples[l])/np.diff(H2_Eu)
        Temps = -1.*np.log10(np.e)/gradient
        temp_s[l] = np.median(Temps)#[2:])
    #med_temp = np.mean(temp_s)
    med_temp_err = np.std(temp_s)
    med_temp = np.nanpercentile(temp_s, [16, 50, 84])
    med_temp_l = med_temp[1] - med_temp[0]
    med_temp_u = med_temp[2] - med_temp[1]
    med_temp_m = med_temp[1]

    print('H_2 Temperature = ', med_temp_m, "+/-", med_temp_err)
     # ax[0].scatter(np.log10(med_temp), np.log(tau), s=5, c='tab:red', marker='x')
     # ax[0].annotate('H$_2$', (np.log10(med_temp), np.log(tau)), color='tab:red')
    # ax[0].scatter(med_temp, tau, s=5, c='tab:red', marker='x')
    # ax[0].errorbar(med_temp, tau, ms=2, color='tab:red', marker='.', xerr = med_temp_err, yerr=tau_err)

    # ax[0].annotate('H$_2$', (med_temp, tau), color='tab:red')
    if (ExtType == 0.0):

        ax[0].hlines(tau_m, 35, 55, colors='tab:pink', lw = 1, label='H$_2$', alpha = 0.8)
        if (tau_err!=0.0):
            ax[0].fill_between([35, 55], (tau_m + tau_u)*np.ones(2), (tau_m - tau_l)*np.ones(2), color='tab:pink', alpha=0.5, linewidth=0)

    output =pd.concat([output, pd.DataFrame([{ 'Name': 'H2 Ext.', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': tau_m,'S_err+': tau_u, 'S_err-': tau_l, 'Continuum (10^-17 W/m^2/um)': 0.0, 'C_err+': 0.0,'C_err-': 0.0,'Eqw (micron)': 0.0, 'E_err+': 0.0, 'E_err-': 0.0}])], ignore_index=True)
    output =pd.concat([output, pd.DataFrame([{ 'Name': 'H2 Temp', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': med_temp_m,'S_err+': med_temp_u, 'S_err-': med_temp_l, 'Continuum (10^-17 W/m^2/um)': 0.0, 'C_err+': 0.0,'C_err-': 0.0,'Eqw (micron)': 0.0, 'E_err+': 0.0, 'E_err-': 0.0}])], ignore_index=True)

    Tau = np.interp(wav, ext_curve[0], ext_curve[1])
    np.savetxt(resultsdir+"/Ext. Corrections/"+ObjName+'H2_ext_fac.txt', np.transpose([wav, np.mean(np.exp(-tau_s[:, np.newaxis]*Tau), axis = 0), np.std(np.exp(-tau_s[:, np.newaxis]*Tau), axis = 0)]))

    plt.figure()
    plt.plot(taus, np.median(smooth, axis = 1))
    plt.fill_between(taus, np.nanpercentile(smooth, 16,axis = 1), np.nanpercentile(smooth, 84,axis = 1), color='black', alpha = 0.5, linewidth=0.0)

    plt.axvline(tau_m, ls='solid', c='tab:red', alpha =0.7, zorder=3)
    plt.axvline(tau_m-tau_l, ls='dashed', c='tab:red', alpha =0.7, zorder=3)
    plt.axvline(tau_m+tau_u, ls='dashed', c='tab:red', alpha =0.7, zorder=3)
    plt.xlabel('$\\tau_{9.8}$')
    plt.ylabel('Smoothness Penalty')
    plt.savefig(resultsdir+"/Plots/"+ObjName+'H2Smoothness.pdf')
    plt.close()

    plt.figure()
    plt.errorbar(H2_Eu, f_org-f_0, yerr=f_org_err, ls='none', marker='.', c='black', label='Original')
    plt.errorbar(H2_Eu, f-f_0, yerr=f_err, ls='none', marker='.', c='tab:red', label='Ext. Corrected')

    for s in range(len(H2_labels)):
        plt.annotate(H2_labels[s], xy = (1.05*H2_Eu[s], 1.05*(f-f_0)[s]), color='black' )

   # plt.show()

    plt.xlabel('$E_{u}/k$ (K)')
    plt.ylabel('$\log(N/g)$ (Rel. units)')
    #plt.annotate('$T = $'+str(np.round(med_temp, 0))+"$\pm$"+str(np.round(med_temp_err, 0))+' K, $\\tau = $'+str(np.round(tau,2))+"$\pm$"+str(np.round(tau_err,2)), xy = (0.2, 0.2), xycoords = 'axes fraction')
    plt.annotate('$\\tau_{9.8} = $'+str(np.round(tau_m,2))+"$\pm$"+str(np.round(tau_err,2)), xy = (0.05, 0.2), xycoords = 'axes fraction')
    plt.xlim(0.0, 10000.0)
    plt.legend(frameon=True)
    plt.savefig(resultsdir+"/Plots/"+ObjName+'H2 Rotation Plot.pdf')
    plt.close()



        #ax[0].annotate('H$_2$', xy = ( 1500,tau), color='tab:purple', va='center')
    Full_ext_fac = np.nanpercentile(Full_ext_fac, [16, 50, 84], axis=0)
    Full_ext_fac_m = Full_ext_fac[1]
    Full_ext_fac_u = Full_ext_fac[2] - Full_ext_fac[1]
    Full_ext_fac_l = Full_ext_fac[1] - Full_ext_fac[0]

    if (ExtType == 0.0):
        tau_means = np.nanpercentile(tau_means_samples, [16, 50, 84], axis=0)
        tau_means_m = tau_means[1]
        tau_means_u = tau_means[2] - tau_means[1]
        tau_means_l = tau_means[1] - tau_means[0]

        Psi_mean = np.nanmean(Psi_samples, axis=0)
        Psi_std = np.nanstd(Psi_samples, axis=0)

##############################################################
        figgggg,axxxxx = plt.subplots(1, 2, figsize = (5,2))

        T = np.logspace(np.log10(35), np.log10(1500), setup.grid_size[0])
        tau = np.logspace(np.log(0.05), np.log(tau_lim), setup.grid_size[1], base=np.e)
        dlogT = np.diff(np.log10(T))[0]
        T, tau = np.meshgrid(T, tau)
        pcol = axxxxx[0].pcolormesh(T, tau, Psi_mean*setup.norm_ratio,  norm = PowerNorm(gamma=0.5), antialiased=True, linewidth=0, rasterized=True)
        pcol.set_edgecolor('face')
        axxxxx[0].set_xlabel('$\log_{10}(T$ (K))')
        axxxxx[0].set_ylabel('$\ln(\\tau)$')



        axxxxx[0].set_xscale('log')
        axxxxx[0].set_yscale('log', base = np.e)
        axxxxx[0].set_xlabel('$T$ (K)')
        axxxxx[0].set_ylabel('$\\tau_{9.8}$ ')


        axxxxx[0].set_xticks([ 35, 100, 1000])
        axxxxx[0].get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        if (extended==True):
            axxxxx[0].set_yticks([ 0.1, 1.0, 10.0, 50.0])
        else:
            axxxxx[0].set_yticks([ 0.1, 1.0, 10.0])
        axxxxx[0].get_yaxis().set_major_formatter(mticker.ScalarFormatter())
        axxxxx[0].set_xlim(35, 1500)
        axxxxx[0].set_ylim(0.05, 15)
        axxxxx[0].invert_yaxis()


        pcol = axxxxx[1].pcolormesh(T, tau, Psi_std*setup.norm_ratio,  norm = PowerNorm(gamma=0.5), antialiased=True, linewidth=0, rasterized=True)
        pcol.set_edgecolor('face')
        axxxxx[1].set_xlabel('$\log_{10}(T$ (K))')
        axxxxx[1].set_ylabel('$\ln(\\tau)$')
        Psi_error_flat = Psi_std.flatten()

        np.savetxt(resultsdir+ObjName+'Psi_error.txt', np.transpose([np.ones(len(Psi_error_flat)), Psi_error_flat]))

        axxxxx[1].set_xscale('log')
        axxxxx[1].set_yscale('log', base = np.e)
        axxxxx[1].set_xlabel('$T$ (K)')
        axxxxx[1].set_ylabel('$\\tau_{9.8}$ ')


        axxxxx[1].set_xticks([ 35, 100, 1000])
        axxxxx[1].get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        if (extended==True):
            axxxxx[1].set_yticks([ 0.1, 1.0, 10.0, 50.0])
        else:
            axxxxx[1].set_yticks([ 0.1, 1.0, 10.0])
        axxxxx[1].get_yaxis().set_major_formatter(mticker.ScalarFormatter())
        axxxxx[1].set_xlim(35, 1500)
        axxxxx[1].set_ylim(0.05, tau_lim)
        axxxxx[1].invert_yaxis()
        figgggg.savefig(resultsdir+"/Plots/"+ObjName+'Psi_ErrorMap.pdf')

        #np.savetxt(resultsdir+ObjName+"Extinction_IceCorr_weighted.txt", np.transpose([wav, ext_weighted]))



##################################################################
        T = np.logspace(np.log10(35), np.log10(1500), setup.grid_size[0])

        ax[1].legend(frameon=True, prop={'size': 5})


        mean_T3 = np.nanpercentile(mean_T3, [16, 50, 84])
        mean_T3_m = mean_T3[1]
        mean_T3_u = mean_T3[2] - mean_T3[1]
        mean_T3_l = mean_T3[1] - mean_T3[0]

        mean_tau3 = np.nanpercentile(mean_tau3, [16, 50, 84])
        mean_tau3_m = mean_tau3[1]
        mean_tau3_u = mean_tau3[2] - mean_tau3[1]
        mean_tau3_l = mean_tau3[1] - mean_tau3[0]


        #ax[0].errorbar(mean_T_m, mean_tau_m, yerr = [[mean_tau_l],[mean_tau_u]], xerr = [[mean_tau_l], [mean_tau_u]], marker = '.', ms = 1.5, color = 'white')
        fig.subplots_adjust(wspace=0.45)
        ax[0].legend(frameon=True, prop={'size': 5}, bbox_to_anchor=(1.1, 1.1))
        fig.savefig(resultsdir+"/Plots/"+ObjName+'_Psi_+H2.pdf')
   # np.savetxt(resultsdir+ObjName+"Extinction_IceCorr.txt", np.transpose([wav, Full_ext_fac_m, Full_ext_fac_l, Full_ext_fac_u]))




    if (ExtType==0.0):
        plt.figure()

        lam_bins = jnp.logspace(jnp.log10(jnp.min(x_data)*1.05), jnp.log10(jnp.max(x_data)*0.95), 20)
        plt.errorbar(lam_bins, tau_means_m, yerr = [tau_means_l, tau_means_u], marker = '.', ms = 1.5, color = 'black', ls='none')

        d_tau_means  =jnp.diff(tau_means_m)
        plt.errorbar(lam_bins[:-1], d_tau_means, marker = '.', ms = 1.5, color = 'tab:red', ls='none')




        plt.xlabel('Wavelength ($\mu$m)')
        plt.ylabel('Eff. $\\tau_{9.8}$')
        plt.savefig(resultsdir+"/Plots/"+ObjName+'Effective Ext Plot.pdf')

        plt.close()


        plt.figure()


        tau_p = jnp.arange(1, int(setup.grid_size[1]+1),1)
        tau = jnp.logspace(jnp.log(0.05), jnp.log(tau_lim), setup.grid_size[1], base=jnp.exp(1))
        
        tau98_eff = jnp.interp(tau_means_m, tau, tau_p)
        d_tau_means  =jnp.diff(tau98_eff)
        plt.errorbar(lam_bins[:-1], d_tau_means, marker = '.', ms = 1.5, color = 'tab:red', ls='none')
        plt.xlabel('Wavelength ($\mu$m)')
        plt.ylabel('Eff. $\\tau_{9.8}$ (Pixel Units)')
        plt.savefig(resultsdir+"/Plots/"+ObjName+'Effective Ext Plot Pixel Units.pdf')

    #PAH's
   # eqws_samples=[]
   # str_samples = []
    #cont_samples=[]
    pah_params = all_params[pah_indx]
    for i in range(setup.Npah):
        a = int(4.0*i) # Index for amps
        c =  int(4.0*i + 1.0) # Index for centres
        w =  int(4.0*i + 2.0) # Index for widths
        
        lam = pah_params[c]
      #  strength = SCALE*2.9979246e14 * 0.5 * pah_params[a]*pah_params[w]*1.0e-9*np.pi/(pah_params[c]) # 10^-17 W/m^2
       # if (setup.pahcents[i] == 6.22 or setup.pahcents[i] == 11.23 or setup.pahcents[i] == 12.62):
            #for j in tqdm(range(len(pah_params[a]))):
               # strength[j] = quad(SetupFit.PAHIntegrand, pah_params[c][j] - 10.0*pah_params[w][j], pah_params[c][j] + 10.0*pah_params[w][j], args=( pah_params[a][j],pah_params[c][j], pah_params[w][j], all_params[ext_indx[1]][j]))[0]
               # print(strength[j])
            
        strr = np.percentile(str_samples_pahs[i]*1e-9, [16,50,84])
        str_med = strr[1]
        str_l = strr[1]-strr[0]
        str_u = strr[2]-strr[1]
        # Upper limits
     #   if (str_med/str_l < 3.0):
          #  str_med = "<" + str(np.max(strength))
          #  str_l = "-"
          #  str_u = "-"
        

       # continuum = np.empty(len(lam))
      #  for j in range(len(lam)):
        #    continuum[j] =   cont_samps[j](lam[j])*1.0e-9*2.9979246e14/(lam[j]**2) # 10^-17 W/m^2/um#
        
        
        
        cont = np.percentile(cont_samples_pahs[i]*1.0e-9*2.9979246e14/(lam**2), [16,50,84])
        cont_med = cont[1]
        cont_l = cont[1]-cont[0]
        cont_u = cont[2]-cont[1]
        # Upper limits
        #if (cont_med/cont_l < 3.0):
       #     cont_med = "<" + str(np.max(continuum))
        #    cont_l = "-"
         #   cont_u = "-"
        
        #eqw =  strength/continuum # micron
        Eqw = np.percentile(eqws_samples_pahs[i], [16,50,84])
        Eqw_med = Eqw[1]
        Eqw_l = Eqw[1]-Eqw[0]
        Eqw_u = Eqw[2]-Eqw[1]
        # Upper limits
       # if (Eqw_med/Eqw_l < 3.0):
            #Eqw_med = "<" + str(np.max(eqw))
            #Eqw_l = "-"
            #Eqw_u = "-"
        
        output =pd.concat([output, pd.DataFrame([{ 'Name': 'PAH', 'Rest Wavelength (micron)': setup.pahcents[i],'Strength (10^-17 W/m^2)': str_med,'S_err+': str_u, 'S_err-': str_l, 'Continuum (10^-17 W/m^2/um)': cont_med, 'C_err+': cont_u,'C_err-': cont_l,'Eqw (micron)': Eqw_med, 'E_err+': Eqw_u, 'E_err-': Eqw_l}])], ignore_index=True)


    if (pah33==True):
        strr = np.percentile(str_samples_33*1e-9, [16,50,84])
        str_med = strr[1]
        str_l = strr[1]-strr[0]
        str_u = strr[2]-strr[1]

        lam = 3.3
        
        cont = np.percentile(cont_samples_33*1.0e-9*2.9979246e14/(lam**2), [16,50,84])
        cont_med = cont[1]
        cont_l = cont[1]-cont[0]
        cont_u = cont[2]-cont[1]

        Eqw = np.percentile(eqws_samples_33, [16,50,84])
        Eqw_med = Eqw[1]
        Eqw_l = Eqw[1]-Eqw[0]
        Eqw_u = Eqw[2]-Eqw[1]
    
        output =pd.concat([output, pd.DataFrame([{ 'Name': 'PAH 3.3 Complex', 'Rest Wavelength (micron)': 3.3,  'Strength (10^-17 W/m^2)': str_med,'S_err+': str_u, 'S_err-': str_l, 'Continuum (10^-17 W/m^2/um)': cont_med, 'C_err+': cont_u,'C_err-': cont_l,'Eqw (micron)': Eqw_med, 'E_err+': Eqw_u, 'E_err-': Eqw_l}])], ignore_index=True)




    if (pah52==True):
        strr = np.percentile(str_samples_52*1e-9, [16,50,84])
        str_med = strr[1]
        str_l = strr[1]-strr[0]
        str_u = strr[2]-strr[1]

        lam = 5.25
        
        cont = np.percentile(cont_samples_52*1.0e-9*2.9979246e14/(lam**2), [16,50,84])
        cont_med = cont[1]
        cont_l = cont[1]-cont[0]
        cont_u = cont[2]-cont[1]

        Eqw = np.percentile(eqws_samples_52, [16,50,84])
        Eqw_med = Eqw[1]
        Eqw_l = Eqw[1]-Eqw[0]
        Eqw_u = Eqw[2]-Eqw[1]
    
        output =pd.concat([output, pd.DataFrame([{ 'Name': 'PAH 5.2 Complex', 'Rest Wavelength (micron)': 5.25,  'Strength (10^-17 W/m^2)': str_med,'S_err+': str_u, 'S_err-': str_l, 'Continuum (10^-17 W/m^2/um)': cont_med, 'C_err+': cont_u,'C_err-': cont_l,'Eqw (micron)': Eqw_med, 'E_err+': Eqw_u, 'E_err-': Eqw_l}])], ignore_index=True)
        
    if (pah57==True):
        strr = np.percentile(str_samples_57*1e-9, [16,50,84])
        str_med = strr[1]
        str_l = strr[1]-strr[0]
        str_u = strr[2]-strr[1]

        lam = 5.7
        
        cont = np.percentile(cont_samples_57*1.0e-9*2.9979246e14/(lam**2), [16,50,84])
        cont_med = cont[1]
        cont_l = cont[1]-cont[0]
        cont_u = cont[2]-cont[1]

        Eqw = np.percentile(eqws_samples_57, [16,50,84])
        Eqw_med = Eqw[1]
        Eqw_l = Eqw[1]-Eqw[0]
        Eqw_u = Eqw[2]-Eqw[1]
    
        output =pd.concat([output, pd.DataFrame([{ 'Name': 'PAH 5.7 Complex', 'Rest Wavelength (micron)': 5.7,  'Strength (10^-17 W/m^2)': str_med,'S_err+': str_u, 'S_err-': str_l, 'Continuum (10^-17 W/m^2/um)': cont_med, 'C_err+': cont_u,'C_err-': cont_l,'Eqw (micron)': Eqw_med, 'E_err+': Eqw_u, 'E_err-': Eqw_l}])], ignore_index=True)
        


    if (pah62==True):
        strr = np.percentile(str_samples_6*1e-9, [16,50,84])
        str_med = strr[1]
        str_l = strr[1]-strr[0]
        str_u = strr[2]-strr[1]

        lam = 6.2
        
        cont = np.percentile(cont_samples_6*1.0e-9*2.9979246e14/(lam**2), [16,50,84])
        cont_med = cont[1]
        cont_l = cont[1]-cont[0]
        cont_u = cont[2]-cont[1]

        Eqw = np.percentile(eqws_samples_6, [16,50,84])
        Eqw_med = Eqw[1]
        Eqw_l = Eqw[1]-Eqw[0]
        Eqw_u = Eqw[2]-Eqw[1]
    
        output =pd.concat([output, pd.DataFrame([{ 'Name': 'PAH 6.2 Complex', 'Rest Wavelength (micron)': 6.2,  'Strength (10^-17 W/m^2)': str_med,'S_err+': str_u, 'S_err-': str_l, 'Continuum (10^-17 W/m^2/um)': cont_med, 'C_err+': cont_u,'C_err-': cont_l,'Eqw (micron)': Eqw_med, 'E_err+': Eqw_u, 'E_err-': Eqw_l}])], ignore_index=True)
        
    if (pah86==True):
        strr = np.percentile(str_samples_8*1e-9, [16,50,84])
        str_med = strr[1]
        str_l = strr[1]-strr[0]
        str_u = strr[2]-strr[1]

        lam = 8.6
        
        cont = np.percentile(cont_samples_8*1.0e-9*2.9979246e14/(lam**2), [16,50,84])
        cont_med = cont[1]
        cont_l = cont[1]-cont[0]
        cont_u = cont[2]-cont[1]

        Eqw = np.percentile(eqws_samples_8, [16,50,84])
        Eqw_med = Eqw[1]
        Eqw_l = Eqw[1]-Eqw[0]
        Eqw_u = Eqw[2]-Eqw[1]
    
        output =pd.concat([output, pd.DataFrame([{ 'Name': 'PAH 8.6 Complex', 'Rest Wavelength (micron)': 8.6,  'Strength (10^-17 W/m^2)': str_med,'S_err+': str_u, 'S_err-': str_l, 'Continuum (10^-17 W/m^2/um)': cont_med, 'C_err+': cont_u,'C_err-': cont_l,'Eqw (micron)': Eqw_med, 'E_err+': Eqw_u, 'E_err-': Eqw_l}])], ignore_index=True)

    if (pah77==True):
        strr = np.percentile(str_samples_7*1e-9, [16,50,84])
        str_med = strr[1]
        str_l = strr[1]-strr[0]
        str_u = strr[2]-strr[1]

        lam = 7.7
        
        cont = np.percentile(cont_samples_7*1.0e-9*2.9979246e14/(lam**2), [16,50,84])
        cont_med = cont[1]
        cont_l = cont[1]-cont[0]
        cont_u = cont[2]-cont[1]

        Eqw = np.percentile(eqws_samples_7, [16,50,84])
        Eqw_med = Eqw[1]
        Eqw_l = Eqw[1]-Eqw[0]
        Eqw_u = Eqw[2]-Eqw[1]
    
        output =pd.concat([output, pd.DataFrame([{ 'Name': 'PAH 7.7 Complex', 'Rest Wavelength (micron)': 7.7, 'Strength (10^-17 W/m^2)': str_med,'S_err+': str_u, 'S_err-': str_l, 'Continuum (10^-17 W/m^2/um)': cont_med, 'C_err+': cont_u,'C_err-': cont_l,'Eqw (micron)': Eqw_med, 'E_err+': Eqw_u, 'E_err-': Eqw_l}])], ignore_index=True)
        
        
    if (pah113==True):
        strr = np.percentile(str_samples_11*1e-9, [16,50,84])
        str_med = strr[1]
        str_l = strr[1]-strr[0]
        str_u = strr[2]-strr[1]

        
        lam = 11.3
        cont = np.percentile(cont_samples_11*1.0e-9*2.9979246e14/(lam**2), [16,50,84])
        cont_med = cont[1]
        cont_l = cont[1]-cont[0]
        cont_u = cont[2]-cont[1]

        Eqw = np.percentile(eqws_samples_11, [16,50,84])
        Eqw_med = Eqw[1]
        Eqw_l = Eqw[1]-Eqw[0]
        Eqw_u = Eqw[2]-Eqw[1]
    
        output =pd.concat([output, pd.DataFrame([{ 'Name': 'PAH 11.3 Complex', 'Rest Wavelength (micron)': 11.3,'Strength (10^-17 W/m^2)': str_med,'S_err+': str_u, 'S_err-': str_l, 'Continuum (10^-17 W/m^2/um)': cont_med, 'C_err+': cont_u,'C_err-': cont_l,'Eqw (micron)': Eqw_med, 'E_err+': Eqw_u, 'E_err-': Eqw_l}])], ignore_index=True)
            
    if (pah127==True):
        strr = np.percentile(str_samples_12*1e-9, [16,50,84])
        str_med = strr[1]
        str_l = strr[1]-strr[0]
        str_u = strr[2]-strr[1]

        
        lam = 12.7
        cont = np.percentile(cont_samples_12*1.0e-9*2.9979246e14/(lam**2), [16,50,84])
        cont_med = cont[1]
        cont_l = cont[1]-cont[0]
        cont_u = cont[2]-cont[1]

        Eqw = np.percentile(eqws_samples_12, [16,50,84])
        Eqw_med = Eqw[1]
        Eqw_l = Eqw[1]-Eqw[0]
        Eqw_u = Eqw[2]-Eqw[1]
    
        output =pd.concat([output, pd.DataFrame([{ 'Name': 'PAH 12.7 Complex', 'Rest Wavelength (micron)': 12.7,'Strength (10^-17 W/m^2)': str_med,'S_err+': str_u, 'S_err-': str_l, 'Continuum (10^-17 W/m^2/um)': cont_med, 'C_err+': cont_u,'C_err-': cont_l,'Eqw (micron)': Eqw_med, 'E_err+': Eqw_u, 'E_err-': Eqw_l}])], ignore_index=True)
            
            
    if (pah17==True):
        strr = np.percentile(str_samples_17*1e-9, [16,50,84])
        str_med = strr[1]
        str_l = strr[1]-strr[0]
        str_u = strr[2]-strr[1]

        
        lam = 17.0
        cont = np.percentile(cont_samples_17*1.0e-9*2.9979246e14/(lam**2), [16,50,84])
        cont_med = cont[1]
        cont_l = cont[1]-cont[0]
        cont_u = cont[2]-cont[1]

        Eqw = np.percentile(eqws_samples_17, [16,50,84])
        Eqw_med = Eqw[1]
        Eqw_l = Eqw[1]-Eqw[0]
        Eqw_u = Eqw[2]-Eqw[1]
    
        output =pd.concat([output, pd.DataFrame([{ 'Name': 'PAH 17.0 Complex', 'Rest Wavelength (micron)': 17.0,'Strength (10^-17 W/m^2)': str_med,'S_err+': str_u, 'S_err-': str_l, 'Continuum (10^-17 W/m^2/um)': cont_med, 'C_err+': cont_u,'C_err-': cont_l,'Eqw (micron)': Eqw_med, 'E_err+': Eqw_u, 'E_err-': Eqw_l}])], ignore_index=True)
        
        
        
    if (pah62 == True and pah113 == True):
        strr = np.percentile(str_samples_6/str_samples_11, [16,50,84])
        str_med = strr[1]
        str_l = strr[1]-strr[0]
        str_u = strr[2]-strr[1]

        
        cont = np.percentile((cont_samples_6*1.0e-9*2.9979246e14/(6.2**2))/(cont_samples_11*1.0e-9*2.9979246e14/(11.3**2)), [16,50,84])
        cont_med = cont[1]
        cont_l = cont[1]-cont[0]
        cont_u = cont[2]-cont[1]

        Eqw = np.percentile(eqws_samples_6/eqws_samples_11, [16,50,84])
        Eqw_med = Eqw[1]
        Eqw_l = Eqw[1]-Eqw[0]
        Eqw_u = Eqw[2]-Eqw[1]
    
        output =pd.concat([output, pd.DataFrame([{ 'Name': '6.2/11.3', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': str_med,'S_err+': str_u, 'S_err-': str_l, 'Continuum (10^-17 W/m^2/um)': cont_med, 'C_err+': cont_u,'C_err-': cont_l,'Eqw (micron)': Eqw_med, 'E_err+': Eqw_u, 'E_err-': Eqw_l}])], ignore_index=True)
        
    if (pah127 == True and pah113 == True):
        strr = np.percentile(str_samples_12/str_samples_11, [16,50,84])
        str_med = strr[1]
        str_l = strr[1]-strr[0]
        str_u = strr[2]-strr[1]

        
        cont = np.percentile((cont_samples_12*1.0e-9*2.9979246e14/(12**2))/(cont_samples_11*1.0e-9*2.9979246e14/(11.3**2)), [16,50,84])
        cont_med = cont[1]
        cont_l = cont[1]-cont[0]
        cont_u = cont[2]-cont[1]

        Eqw = np.percentile(eqws_samples_12/eqws_samples_11, [16,50,84])
        Eqw_med = Eqw[1]
        Eqw_l = Eqw[1]-Eqw[0]
        Eqw_u = Eqw[2]-Eqw[1]
    
        output =pd.concat([output, pd.DataFrame([{ 'Name': '12.7/11.3', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': str_med,'S_err+': str_u, 'S_err-': str_l, 'Continuum (10^-17 W/m^2/um)': cont_med, 'C_err+': cont_u,'C_err-': cont_l,'Eqw (micron)': Eqw_med, 'E_err+': Eqw_u, 'E_err-': Eqw_l}])], ignore_index=True)




        # PAH Ext.
        Tau = np.interp([12.7,11.3], ext_curve[0], ext_curve[1])
        mu, sig, mn = [-1.4259723729502929, 0.8979984992680757, 0.5204031444083982]
        #instr_ratio = np.random.lognormal(mu, sig, len(str_samples_12))+mn #0.89#np.random.normal(0.89, 0.097, len(str_samples_12))
        #instr_ratio = np.random.normal(0.45, 0.23)
        instr_ratio = np.random.normal(0.72, 0.08, len(str_samples_12))
        res_rat = np.empty((len(taus), len(str_samples_12)))
        for i in range(len(taus)):
            ext_cor = np.exp(-taus[i]*Tau)
            #res_rat.append(0.922*ext_cor[0]/ext_cor[1])
            res_rat[i, :] = instr_ratio*ext_cor[0]/ext_cor[1]

        res_rat = np.array(res_rat)
        tau_PAH_s = np.empty(len(str_samples_12))
        for i in range(len(str_samples_12)): 
            tau_PAH_s[i] = np.interp((str_samples_12[i])/(str_samples_11[i]), res_rat[:, i][np.argsort(res_rat[:, i])] , taus[np.argsort(res_rat[:, i])])
            #if (tau_PAH_s[i]<0.0):
              #  tau_PAH_s[i] = 0.0
        tau_PAH_err = np.std(tau_PAH_s)
        #tau_PAH = np.mean(tau_PAH)
        #print('tau_PAH = ,' + str(np.round(tau_PAH, 2)), '+/-', str(np.round(tau_PAH_err, 2)))
        tau_PAH = np.nanpercentile(tau_PAH_s, [16,50,84])
        tau_PAH_u = tau_PAH[2]-tau_PAH[1]
        tau_PAH_l = tau_PAH[1]-tau_PAH[0]
        tau_PAH = tau_PAH[1]

        output =pd.concat([output, pd.DataFrame([{ 'Name': 'PAH Ext.', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': tau_PAH,'S_err+': tau_PAH_u, 'S_err-': tau_PAH_l, 'Continuum (10^-17 W/m^2/um)': 0.0, 'C_err+': 0.0,'C_err-': 0.0,'Eqw (micron)': 0.0, 'E_err+': 0.0, 'E_err-': 0.0}])], ignore_index=True)

        Tau = np.interp(wav, ext_curve[0], ext_curve[1])
        np.savetxt(resultsdir+"/Ext. Corrections/"+ObjName+'PAH_ext_fac.txt', np.transpose([wav, np.mean(np.exp(-tau_PAH_s[:, np.newaxis]*Tau), axis = 0), np.std(np.exp(-tau_PAH_s[:, np.newaxis]*Tau), axis = 0)]))



        # ax[0].hlines(np.log(tau_PAH), np.log10(20), np.log10(1500), colors='tab:red')
        # ax[0].annotate('PAHs', xy = ( 0.8*np.log10(1500),np.log(tau_PAH)), color='tab:red')
        # if (tau_PAH<0.01):
        #     tau_PAH = 0.01

        if (ExtType == 0.0):
            ax[0].hlines(tau_PAH, 35, 55, colors='tab:cyan', lw = 1, label='PAHs', alpha = 0.8)
            if (tau_PAH_err!=0.0):
                ax[0].fill_between([35, 55], (tau_PAH + tau_PAH_u)*np.ones(2), (tau_PAH - tau_PAH_l)*np.ones(2), color='tab:cyan', alpha=0.5, linewidth=0)

            #ax[0].annotate('PAHs', xy = ( 1500,tau_PAH), color='tab:red', va='center')
            ax[0].legend(frameon=True, prop={'size': 5}, bbox_to_anchor=(1.1, 1.1))
            fig.savefig(resultsdir+"/Plots/"+ObjName+'_Psi_+H2+PAH.pdf')

        

################################################ CLOUDY HI ratios ####
    n = np.array([5,6,7,8,9, 10])#,9,10,11])
    Brg_Brb = np.array([0.626, 0.626, 0.615, 0.792, 0.790, 0.788])#, 0.743, 0.820, 0.834])
    Pfg_Brb = np.array([0.23, 0.23, 0.21, 0.23, 0.22, 0.514])#, 0.15, 0.19, 0.367])
    Brb_Pfg = 1.0/Pfg_Brb
    Pfg_Brg = Pfg_Brb/Brg_Brb
    n_grid = np.linspace(5, 11, 100)
    Pfg_Hua = np.array([1.26, 1.23, 1.25, 4.945, 10.2, 8.22])
    Brb_Bra = np.array([0.603, 0.601, 0.573, 1.28, 0.984, 0.785])#, 0.740, 0.524, 1.0])
     
     
    Brb_Pfg = interp1d(n,Brb_Pfg, fill_value='extrapolate', kind='quadratic')
    Pfg_Brg= interp1d(n, Pfg_Brg, fill_value='extrapolate', kind='quadratic')
    Brg_Brb = interp1d(n, Brg_Brb, fill_value='extrapolate', kind='quadratic')
    Pfg_Hua = interp1d(n, Pfg_Hua, fill_value='extrapolate', kind='quadratic')
     
    gas_dens = np.linspace(5, 10, 50)

#############


    # HI Extinction
    if (len(HI_fluxes)>3):

        if (HI_ratios == 'CLOUDY'):
            print('Using CLOUDY HI line ratios')

            #Loop over gas densities
            ratio1_samps = np.empty((len(gas_dens), len(HI_fluxes[0])))
            ratio2_samps = np.empty((len(gas_dens), len(HI_fluxes[0])))

            for k in range(len(gas_dens)):



                #Tau = np.interp([2.63, 3.74], ext_curve[0], ext_curve[1]+SS)
                line_wavs = np.array([2.1655, 2.6252, 3.7395,    12.368])
                Tau = np.interp(line_wavs, ext_curve[0], ext_curve[1])

                res_rat1 = []
                res_rat2 = []
                for i in range(len(taus)):
                    ext_cor = np.exp(-taus[i]*Tau)
                    #res_rat.append(4.2981*ext_cor[0]/ext_cor[1])
                    res_rat1.append(Brg_Brb(gas_dens[k])*ext_cor[0]/ext_cor[1])
                    res_rat2.append(Pfg_Hua(gas_dens[k])*ext_cor[2]/ext_cor[3])

                res_rat1 = np.array(res_rat1)
                res_rat2 = np.array(res_rat2)

                BrGamma = np.array(HI_fluxes[0])#np.array(output['Strength (10^-17 W/m^2)'])[output['Name']=='BrBeta']
                BrBeta = np.array(HI_fluxes[1])#np.array(output['Strength (10^-17 W/m^2)'])[output['Name']=='PfGamma']
                PfGamma = np.array(HI_fluxes[2])#np.array(output['Strength (10^-17 W/m^2)'])[output['Name']=='PfGamma']
                HuAlpha = np.array(HI_fluxes[3])#np.array(output['Strength (10^-17 W/m^2)'])[output['Name']=='PfGamma']

                #fig.savefig(resultsdir+ObjName+'_Psi.pdf')

                #tau_HI = np.interp((BrBeta*np.interp(2.63, wav, ext))/(PfGamma*np.interp(3.74, wav, ext)), res_rat[np.argsort(res_rat)], taus[np.argsort(res_rat)])

                #List of samples for given gas density
                # Loop over samples
                tau_HI_1 = np.empty(len(BrGamma))
                tau_HI_2 = np.empty(len(BrGamma))
                for i in range(len(BrGamma)):            
                    tau_HI_1 = np.interp((BrGamma[i])/(BrBeta[i]), np.random.normal(res_rat1[np.argsort(res_rat1)], 0.02*res_rat1[np.argsort(res_rat1)]), taus[np.argsort(res_rat1)])

                    ratio1_samps[k,i] = tau_HI_1
                #tau_HI_2 = np.interp((PfGamma*np.interp(line_wavs[2], wav, ext))/(HuAlpha*np.interp(line_wavs[3], wav, ext)), res_rat2[np.argsort(res_rat2)], taus[np.argsort(res_rat2)])
                    tau_HI_2 = np.interp((PfGamma[i])/(HuAlpha[i]), np.random.normal(res_rat2[np.argsort(res_rat2)], 0.1*res_rat2[np.argsort(res_rat2)]), taus[np.argsort(res_rat2)])

                    ratio2_samps[k,i] = tau_HI_2

            # Find intersection per sample
            tau_HI_samples = np.empty(len(HI_fluxes[0]))
            gas_den_samples = np.empty(len(HI_fluxes[0]))
            for k in tqdm(range(len(HI_fluxes[0]))):
                idx = np.argwhere(np.diff(np.sign(ratio1_samps[:,k] - ratio2_samps[:,k])) != 0)


                if (len(idx)>1): # If there are mulyiple intersections, take average 
                    tau_HI_s = []
                    gas_dens_s = []
                    for i in range(len(idx)):
                        tau_HI_s.append(np.interp(gas_dens[idx[i]], gas_dens, ratio1_samps[:, k]))
                        gas_dens_s.append(gas_dens[idx[i]])

                    tau_HI_samples[k] = np.average(tau_HI_s)
                    gas_den_samples[k] =  np.average(gas_dens_s)
                elif(len(idx)==1):

                    tau_HI_samples[k] = np.interp(gas_dens[idx], gas_dens, ratio1_samps[:, k])
                    gas_den_samples[k] = gas_dens[idx]
                else:
                    tau_HI_samples[k] = np.nan
                    gas_den_samples[k] = np.nan    

            # Assume Brg/Brb value if no solution is found
            tau_HI_samples = np.full(np.shape(tau_HI_samples), np.nan)
            if (np.all(np.isnan(tau_HI_samples))):
                tau_HI_samples = ratio1_samps[0, :]


            plt.figure()
            plt.plot(gas_dens, np.mean(ratio1_samps, axis=1), label='Brg/Brb', color='tab:red', ls='solid')
            plt.fill_between(gas_dens, np.mean(ratio1_samps, axis=1)-np.std(ratio1_samps, axis=1), np.mean(ratio1_samps, axis=1)+np.std(ratio1_samps, axis=1), color='tab:red', alpha=0.5, linewidth=0)

            plt.plot(gas_dens, np.mean(ratio2_samps, axis=1), label='Pfg/Hua', color='tab:orange', ls='solid')
            plt.fill_between(gas_dens, np.mean(ratio2_samps, axis=1)-np.std(ratio2_samps, axis=1), np.mean(ratio2_samps, axis=1)+np.std(ratio2_samps, axis=1), color='tab:orange', alpha=0.5, linewidth=0)

            plt.legend(frameon=True)
            plt.xlabel('$\\log n$ cm$^{-2}$')
            plt.ylabel('Extinction, $\\tau_{9.8}$')

            plt.savefig(resultsdir+"/Plots/"+ObjName+'HI Ext.pdf')
            plt.close()
            


            tau_HI_err = np.nanstd(tau_HI_samples)#[0]
            # try:
            tau_HI_l = np.nanpercentile(tau_HI_samples, 50.0) - np.nanpercentile(tau_HI_samples, 16.0)
            tau_HI_u = np.nanpercentile(tau_HI_samples, 84.0) - np.nanpercentile(tau_HI_samples, 50.0)
            tau_HI = np.nanmean(tau_HI_samples)#[0]
           # print(tau_HI)
            #print(tau_HI_err)
            #print('tau_HI = ', str(np.round(tau_HI_1, 2)), '+/-', str(np.round(tau_HI_err, 2)))
            output =pd.concat([output, pd.DataFrame([{ 'Name': 'HI Ext.', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': tau_HI,'S_err+': tau_HI_u, 'S_err-': tau_HI_l, 'Continuum (10^-17 W/m^2/um)': 0.0, 'C_err+': 0.0,'C_err-': 0.0,'Eqw (micron)': 0.0, 'E_err+': 0.0, 'E_err-': 0.0}])], ignore_index=True)

            Tau = np.interp(wav, ext_curve[0], ext_curve[1])
            np.savetxt(resultsdir+"/Ext. Corrections/"+ObjName+'HI_ext_fac.txt', np.transpose([wav, np.mean(np.exp(-tau_HI_samples[:, np.newaxis]*Tau), axis = 0), np.std(np.exp(-tau_HI_samples[:, np.newaxis]*Tau), axis = 0)]))

            # Tau = np.interp(wav, ext_curve[0], ext_curve[1]+SS)
            # np.savetxt(resultsdir+"/Ext. Corrections/"+ObjName+'StellarCont._ext_fac.txt', np.transpose([wav, np.mean(np.exp(-tau_HI_samples[:, np.newaxis]*Tau), axis = 0), np.std(np.exp(-tau_HI_samples[:, np.newaxis]*Tau), axis = 0)]))



            gas_den_err = np.nanstd(gas_den_samples)#[0]
            # try:
            gas_den_l = np.nanpercentile(gas_den_samples, 50.0) - np.nanpercentile(gas_den_samples, 16.0)
            gas_den_u = np.nanpercentile(gas_den_samples, 84.0) - np.nanpercentile(gas_den_samples, 50.0)
            gas_den = np.nanmean(gas_den_samples)#[0]
           # print(tau_HI)
            #print(tau_HI_err)
            #print('tau_HI = ', str(np.round(tau_HI_1, 2)), '+/-', str(np.round(tau_HI_err, 2)))
            output =pd.concat([output, pd.DataFrame([{ 'Name': 'log n (HI cm^-3)', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': gas_den,'S_err+': gas_den_u, 'S_err-': gas_den_l, 'Continuum (10^-17 W/m^2/um)': 0.0, 'C_err+': 0.0,'C_err-': 0.0,'Eqw (micron)': 0.0, 'E_err+': 0.0, 'E_err-': 0.0}])], ignore_index=True)
        else:
            print('Using Case B')

            ratio1_samps = np.empty((len(HI_fluxes[0])))
            ratio2_samps = np.empty((len(HI_fluxes[0])))

            #Tau = np.interp([2.63, 3.74], ext_curve[0], ext_curve[1]+SS)
            line_wavs = np.array([2.1655, 2.6252, 3.7395, 12.368])
            Tau = np.interp(line_wavs, ext_curve[0], ext_curve[1])

            res_rat1 = []
            for i in range(len(taus)):
                ext_cor = np.exp(-taus[i]*Tau)
                #res_rat.append(4.2981*ext_cor[0]/ext_cor[1])
                res_rat1.append(0.6152*ext_cor[0]/ext_cor[1])

            res_rat1 = np.array(res_rat1)

            BrGamma = np.array(HI_fluxes[0])#np.array(output['Strength (10^-17 W/m^2)'])[output['Name']=='BrBeta']
            BrBeta = np.array(HI_fluxes[1])#np.array(output['Strength (10^-17 W/m^2)'])[output['Name']=='PfGamma']
            PfGamma = np.array(HI_fluxes[2])#np.array(output['Strength (10^-17 W/m^2)'])[output['Name']=='PfGamma']
            HuAlpha = np.array(HI_fluxes[3])#np.array(output['Strength (10^-17 W/m^2)'])[output['Name']=='PfGamma']

           # Loop over samples
            tau_HI_1 = np.empty(len(BrGamma))
            for i in range(len(BrGamma)):            
                tau_HI_1 = np.interp((BrGamma[i])/(BrBeta[i]), np.random.normal(res_rat1[np.argsort(res_rat1)], 0.02*res_rat1[np.argsort(res_rat1)]), taus[np.argsort(res_rat1)])
                ratio1_samps[i] = tau_HI_1

            tau_HI_samples = ratio1_samps
            tau_HI_err = np.nanstd(tau_HI_samples)#[0]
            # try:
            tau_HI_l = np.nanpercentile(tau_HI_samples, 50.0) - np.nanpercentile(tau_HI_samples, 16.0)
            tau_HI_u = np.nanpercentile(tau_HI_samples, 84.0) - np.nanpercentile(tau_HI_samples, 50.0)
            tau_HI = np.nanmean(tau_HI_samples)#[0]
            output =pd.concat([output, pd.DataFrame([{ 'Name': 'HI Ext.', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': tau_HI,'S_err+': tau_HI_u, 'S_err-': tau_HI_l, 'Continuum (10^-17 W/m^2/um)': 0.0, 'C_err+': 0.0,'C_err-': 0.0,'Eqw (micron)': 0.0, 'E_err+': 0.0, 'E_err-': 0.0}])], ignore_index=True)

            Tau = np.interp(wav, ext_curve[0], ext_curve[1])
            np.savetxt(resultsdir+"/Ext. Corrections/"+ObjName+'HI_ext_fac.txt', np.transpose([wav, np.mean(np.exp(-tau_HI_samples[:, np.newaxis]*Tau), axis = 0), np.std(np.exp(-tau_HI_samples[:, np.newaxis]*Tau), axis = 0)]))





        tau_stellar = np.nanmean(St_Cont_ext)
        tau_stellar_l = np.nanpercentile(St_Cont_ext, 50.0) - np.nanpercentile(St_Cont_ext, 16.0)
        tau_stellar_u = np.nanpercentile(St_Cont_ext, 84.0) - np.nanpercentile(St_Cont_ext, 50.0)
        #np.savetxt(resultsdir+ObjName+"HI Extinction.txt", np.transpose([wav, np.exp(-tau_HI*np.interp(wav, ext_curve[0], ext_curve[1]+SS))]))


            # ax[0].hlines(np.log(tau_PAH), np.log10(20), np.log10(1500), colors='tab:red')
            # ax[0].annotate('PAHs', xy = ( 0.8*np.log10(1500),np.log(tau_PAH)), color='tab:red')
        # if (tau_HI<0.01 and tau_HI>=0.0):
        #     tau_HI = 0.01
        if (ExtType == 0.0):

            if tau_HI:
                ax[0].hlines(tau_HI, 35, 55, colors='tab:orange', label='HI', alpha = 0.8)
                #ax[0].annotate('HI Br$\\gamma$/Br$\\beta$', xy = ( 1500,tau_HI_1), color='tab:red', va='center')
                if (tau_HI_u!=0.0):
                    ax[0].fill_between([35, 55], (tau_HI + tau_HI_u)*np.ones(2), (tau_HI - tau_HI_l)*np.ones(2), color='tab:orange', alpha=0.5, linewidth=0)
            
            if tau_stellar:
                ax[0].hlines(tau_stellar, 35, 55, colors='tab:red', label='Stellar Cont.', alpha = 0.8)
                #ax[0].annotate('HI Br$\\gamma$/Br$\\beta$', xy = ( 1500,tau_HI_1), color='tab:red', va='center')
                if (tau_stellar_u!=0.0):
                    ax[0].fill_between([35, 55], (tau_stellar + tau_stellar_u)*np.ones(2), (tau_stellar - tau_stellar_l)*np.ones(2), color='tab:red', alpha=0.5, linewidth=0)
            # if tau_HI_2:
            #     ax[0].hlines(tau_HI_2, 35, 55, colors='tab:orange', label = 'HI (Pf$\\gamma$/Hu$\\alpha$)')
            #     #ax[0].annotate('HI Pf$\\gamma$/Hu$\\alpha$', xy = ( 1500,tau_HI_2), color='tab:orange', va='center')
            #     if (tau_HI_u_2!=0.0):
            #         ax[0].fill_between([35, 55], (tau_HI_2 + tau_HI_u_2)*np.ones(2), (tau_HI_2 - tau_HI_l_2)*np.ones(2), color='tab:orange', alpha=0.5, linewidth=0)
            
            # except:
            #     a=0

            ax[0].legend(frameon=True, prop={'size': 5}, bbox_to_anchor=(1.1, 1.1))
            fig.savefig(resultsdir+"/Plots/"+ObjName+'_Psi_+H2+PAH+HI.pdf')
    # except:
    #     a=0

    plt.close()

    tau_stellar = np.nanmean(St_Cont_ext)
    tau_stellar_l = np.nanpercentile(St_Cont_ext, 50.0) - np.nanpercentile(St_Cont_ext, 16.0)
    tau_stellar_u = np.nanpercentile(St_Cont_ext, 84.0) - np.nanpercentile(St_Cont_ext, 50.0)
        
    if (pah86 == True and pah113 == True):
        strr = np.percentile(str_samples_8/str_samples_11, [16,50,84])
        str_med = strr[1]
        str_l = strr[1]-strr[0]
        str_u = strr[2]-strr[1]

        
        cont = np.percentile((cont_samples_8*1.0e-9*2.9979246e14/(8.6**2))/(cont_samples_11*1.0e-9*2.9979246e14/(11.3**2)), [16,50,84])
        cont_med = cont[1]
        cont_l = cont[1]-cont[0]
        cont_u = cont[2]-cont[1]

        Eqw = np.percentile(eqws_samples_8/eqws_samples_11, [16,50,84])
        Eqw_med = Eqw[1]
        Eqw_l = Eqw[1]-Eqw[0]
        Eqw_u = Eqw[2]-Eqw[1]
    
        output =pd.concat([output, pd.DataFrame([{ 'Name': '8.6/11.3', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': str_med,'S_err+': str_u, 'S_err-': str_l, 'Continuum (10^-17 W/m^2/um)': cont_med, 'C_err+': cont_u,'C_err-': cont_l,'Eqw (micron)': Eqw_med, 'E_err+': Eqw_u, 'E_err-': Eqw_l}])], ignore_index=True)

    if (pah62 == True and pah77 == True):
        strr = np.percentile(str_samples_6/str_samples_7, [16,50,84])
        str_med = strr[1]
        str_l = strr[1]-strr[0]
        str_u = strr[2]-strr[1]

        
        cont = np.percentile((cont_samples_6*1.0e-9*2.9979246e14/(6.2**2))/(cont_samples_7*1.0e-9*2.9979246e14/(7.7**2)), [16,50,84])
        cont_med = cont[1]
        cont_l = cont[1]-cont[0]
        cont_u = cont[2]-cont[1]

        Eqw = np.percentile(eqws_samples_6/eqws_samples_7, [16,50,84])
        Eqw_med = Eqw[1]
        Eqw_l = Eqw[1]-Eqw[0]
        Eqw_u = Eqw[2]-Eqw[1]
    
        output =pd.concat([output, pd.DataFrame([{ 'Name': '6.2/7.7', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': str_med,'S_err+': str_u, 'S_err-': str_l, 'Continuum (10^-17 W/m^2/um)': cont_med, 'C_err+': cont_u,'C_err-': cont_l,'Eqw (micron)': Eqw_med, 'E_err+': Eqw_u, 'E_err-': Eqw_l}])], ignore_index=True)

    if (pah113 == True and pah77 == True):
        strr = np.percentile(str_samples_11/str_samples_7, [16,50,84])
        str_med = strr[1]
        str_l = strr[1]-strr[0]
        str_u = strr[2]-strr[1]

        
        cont = np.percentile((cont_samples_11*1.0e-9*2.9979246e14/(11.3**2))/(cont_samples_7*1.0e-9*2.9979246e14/(7.7**2)), [16,50,84])
        cont_med = cont[1]
        cont_l = cont[1]-cont[0]
        cont_u = cont[2]-cont[1]

        Eqw = np.percentile(eqws_samples_11/eqws_samples_7, [16,50,84])
        Eqw_med = Eqw[1]
        Eqw_l = Eqw[1]-Eqw[0]
        Eqw_u = Eqw[2]-Eqw[1]
    
        output =pd.concat([output, pd.DataFrame([{ 'Name': '11.3/7.7', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': str_med,'S_err+': str_u, 'S_err-': str_l, 'Continuum (10^-17 W/m^2/um)': cont_med, 'C_err+': cont_u,'C_err-': cont_l,'Eqw (micron)': Eqw_med, 'E_err+': Eqw_u, 'E_err-': Eqw_l}])], ignore_index=True)
    
    if (pah17 == True and pah62 == True):
        strr = np.percentile(str_samples_6/str_samples_17, [16,50,84])
        str_med = strr[1]
        str_l = strr[1]-strr[0]
        str_u = strr[2]-strr[1]

        
        cont = np.percentile((cont_samples_6*1.0e-9*2.9979246e14/(6.2**2))/(cont_samples_17*1.0e-9*2.9979246e14/(17.0**2)), [16,50,84])
        cont_med = cont[1]
        cont_l = cont[1]-cont[0]
        cont_u = cont[2]-cont[1]

        Eqw = np.percentile(eqws_samples_6/eqws_samples_17, [16,50,84])
        Eqw_med = Eqw[1]
        Eqw_l = Eqw[1]-Eqw[0]
        Eqw_u = Eqw[2]-Eqw[1]
    
        output =pd.concat([output, pd.DataFrame([{ 'Name': '6.2/17.0', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': str_med,'S_err+': str_u, 'S_err-': str_l, 'Continuum (10^-17 W/m^2/um)': cont_med, 'C_err+': cont_u,'C_err-': cont_l,'Eqw (micron)': Eqw_med, 'E_err+': Eqw_u, 'E_err-': Eqw_l}])], ignore_index=True)
   



    if (pah113 == True and pah33 == True):
        strr = np.percentile(str_samples_11/str_samples_33, [16,50,84])
        str_med = strr[1]
        str_l = strr[1]-strr[0]
        str_u = strr[2]-strr[1]

        
        cont = np.percentile((cont_samples_11*1.0e-9*2.9979246e14/(11.3**2))/(cont_samples_33*1.0e-9*2.9979246e14/(3.3**2)), [16,50,84])
        cont_med = cont[1]
        cont_l = cont[1]-cont[0]
        cont_u = cont[2]-cont[1]

        Eqw = np.percentile(eqws_samples_11/eqws_samples_33, [16,50,84])
        Eqw_med = Eqw[1]
        Eqw_l = Eqw[1]-Eqw[0]
        Eqw_u = Eqw[2]-Eqw[1]
    
        output =pd.concat([output, pd.DataFrame([{ 'Name': '11.3/3.3', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': str_med,'S_err+': str_u, 'S_err-': str_l, 'Continuum (10^-17 W/m^2/um)': cont_med, 'C_err+': cont_u,'C_err-': cont_l,'Eqw (micron)': Eqw_med, 'E_err+': Eqw_u, 'E_err-': Eqw_l}])], ignore_index=True)

    if (pah17 == True and pah33 == True):
        strr = np.percentile(str_samples_33/str_samples_17, [16,50,84])
        str_med = strr[1]
        str_l = strr[1]-strr[0]
        str_u = strr[2]-strr[1]

        
        cont = np.percentile((cont_samples_33*1.0e-9*2.9979246e14/(3.3**2))/(cont_samples_17*1.0e-9*2.9979246e14/(17.0**2)), [16,50,84])
        cont_med = cont[1]
        cont_l = cont[1]-cont[0]
        cont_u = cont[2]-cont[1]

        Eqw = np.percentile(eqws_samples_33/eqws_samples_17, [16,50,84])
        Eqw_med = Eqw[1]
        Eqw_l = Eqw[1]-Eqw[0]
        Eqw_u = Eqw[2]-Eqw[1]
    
        output =pd.concat([output, pd.DataFrame([{ 'Name': '3.3/17.0', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': str_med,'S_err+': str_u, 'S_err-': str_l, 'Continuum (10^-17 W/m^2/um)': cont_med, 'C_err+': cont_u,'C_err-': cont_l,'Eqw (micron)': Eqw_med, 'E_err+': Eqw_u, 'E_err-': Eqw_l}])], ignore_index=True)
   

    if (pah62 == True and pah33 == True):
        strr = np.percentile(str_samples_6/str_samples_33, [16,50,84])
        str_med = strr[1]
        str_l = strr[1]-strr[0]
        str_u = strr[2]-strr[1]

        
        cont = np.percentile((cont_samples_6*1.0e-9*2.9979246e14/(6.2**2))/(cont_samples_33*1.0e-9*2.9979246e14/(3.3**2)), [16,50,84])
        cont_med = cont[1]
        cont_l = cont[1]-cont[0]
        cont_u = cont[2]-cont[1]

        Eqw = np.percentile(eqws_samples_6/eqws_samples_33, [16,50,84])
        Eqw_med = Eqw[1]
        Eqw_l = Eqw[1]-Eqw[0]
        Eqw_u = Eqw[2]-Eqw[1]
    
        output =pd.concat([output, pd.DataFrame([{ 'Name': '6.2/3.3', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': str_med,'S_err+': str_u, 'S_err-': str_l, 'Continuum (10^-17 W/m^2/um)': cont_med, 'C_err+': cont_u,'C_err-': cont_l,'Eqw (micron)': Eqw_med, 'E_err+': Eqw_u, 'E_err-': Eqw_l}])], ignore_index=True)
   


    # Extinction
    #output =pd.concat([output, pd.DataFrame([{ 'Name': 'tau_9.8 ', 'Rest Wavelength (micron)': 9.8,'Strength (10^-17 W/m^2)': -1.0*jnp.log(SetupFit.ReturnExt(jnp.array([9.8]), ps[cont_indx], ExtType)),'S_err+': 0.0, 'S_err-': 0.0, 'Continuum (10^-17 W/m^2/um)': 0.0, 'C_err+': 0.0,'C_err-': 0.0,'Eqw (micron)': 0.0, 'E_err+': 0.0, 'E_err-': 0.0}])], ignore_index=True)
    if (ExtType ==0.0):
        output =pd.concat([output, pd.DataFrame([{ 'Name': 'Full_mean_tau', 'Rest Wavelength (micron)': 9.8,'Strength (10^-17 W/m^2)': mean_tau3_m,'S_err+': mean_tau3_u, 'S_err-': mean_tau3_l, 'Continuum (10^-17 W/m^2/um)': 0.0, 'C_err+': 0.0,'C_err-': 0.0,'Eqw (micron)': 0.0, 'E_err+': 0.0, 'E_err-': 0.0}])], ignore_index=True)
        output =pd.concat([output, pd.DataFrame([{ 'Name': 'Full_mean_T', 'Rest Wavelength (micron)': 9.8,'Strength (10^-17 W/m^2)': mean_T3_m,'S_err+': mean_T3_u, 'S_err-': mean_T3_l, 'Continuum (10^-17 W/m^2/um)': 0.0, 'C_err+': 0.0,'C_err-': 0.0,'Eqw (micron)': 0.0, 'E_err+': 0.0, 'E_err-': 0.0}])], ignore_index=True)

        output =pd.concat([output, pd.DataFrame([{ 'Name': 'Stellar Ext.', 'Rest Wavelength (micron)': 9.8,'Strength (10^-17 W/m^2)': tau_stellar,'S_err+': tau_stellar_u, 'S_err-': tau_stellar_l, 'Continuum (10^-17 W/m^2/um)': 0.0, 'C_err+': 0.0,'C_err-': 0.0,'Eqw (micron)': 0.0, 'E_err+': 0.0, 'E_err-': 0.0}])], ignore_index=True)

            

    
    
    #print(output )
    output.to_csv(resultsdir+setup.ObjName+"Output.csv",  index=False)
    
    print("")
    print("")
    print("Line and PAH properties")
    Table.pprint_all(Table.from_pandas(output))
    
    
    return output
    
    
    
    

   # setup.samples_flat = samples_flat
   # setup.CornerPlot    ()
from os import listdir
from os.path import isfile, join

if __name__ == '__main__':

    
    useMCMC= False
    Ices_6micron = False
    ExtType_ = 'Differential' #Screen, Mixed, Differential
    HI_ratios = 'Case B' #Case B or CLOUDY

    BootStrap = False
    N_bootstrap = 100
    InitialFit = False
    lam_range=[1.5, 28.0]
    show_progress = True
    z=0.0

    skip=True

    objs = ['NGC 3256_Nuc1_SF1','NGC 3256_Nuc1_SF2', 'NGC 3256_Nuc1_SF3','NGC 3256_Nuc1_SF4','NGC 3256_Nuc1_SF5','NGC 3256_Nuc1_SF6','NGC 3256_Nuc1_Nuc',
     'NGC 7469_SF1', 'NGC 7469_SF2','NGC 7469_SF3', 'NGC 7469_SF4','NGC 7469_SF5','NGC 7469_SF6']
    objs = ['NGC 3256_Nuc1_SF1', 'NGC 7469_Nuc']
    objs = ['IIZw96', 'ESO137-G034_Nuc', 'MCG-05-23-016_Nuc']
    objs = ['Mock_Test2_2']
    objs = ['Mock_Test2_2', 'Mock_Test2_3']#, 'sp_ngc5728_fringed_', 'sp_ngc7172_fringed_']#, 'NGC 3256_Nuc1_SF2']
    #objs = ['NGC 3256_Nuc2_Nuc']
    objs = [ 'NGC 3256_Nuc1_SF1']#, 'sp_ngc7172_fringed_']

   # objs=['c2_template_large_2022_v1']
    for i in range(len(objs)):
        objName = objs[i]
        print(objName) 
        try:
            lams, flux, flux_err = np.loadtxt('./Data/'+objName+'.txt', unpack=True, usecols=[0,1,2])
        except:
            try:
                lams, flux, flux_err = np.loadtxt('./Data/'+objName+'.dat', unpack=True, usecols=[0,1,2])
            except:
                lams, flux, flux_err = np.loadtxt('./Data/'+objName+'.tbl', unpack=True, usecols=[0,1,2])
                # flux *=1e-3
                # flux_err *=1e-3


        flux_err = flux_err[~np.isnan(flux)]#/10.0
        lams = lams[~np.isnan(flux)]
        flux = flux[~np.isnan(flux)]

        lams = lams[flux_err!=0.0]
        flux = flux[flux_err!=0.0]
        flux_err = flux_err[flux_err!=0.0]
        lams = lams[flux_err>0.0]
        flux = flux[flux_err>0.0]
        flux_err = flux_err[flux_err>0.0]


        specdata=[lams, flux, flux_err]

        if (skip==False or useMCMC==False):
            binNo = 0
            if (BootStrap == True):
                binNo = 2
            output = RunFit(objName, specdata, z, lam_range, binNo,  useMCMC=False,   ExtType_=ExtType_, Ices_6micron = Ices_6micron,  BootStrap = BootStrap, N_bootstrap = N_bootstrap, HI_ratios = HI_ratios, show_progress=show_progress  )
        
        if (useMCMC==True):        
            binNo = 1
            output = RunFit(objName, specdata, z, lam_range, binNo,  useMCMC=useMCMC,ExtType_=ExtType_, Ices_6micron = Ices_6micron,   InitialFit = InitialFit,  HI_ratios = HI_ratios , show_progress=show_progress )



