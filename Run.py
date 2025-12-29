import SPIRIT as ModelGUI
import numpy as np



# objs = ['ESO137-G034_reg1_scaled']

ModelGUI.RunModel(objs, Dust_Geometry = 'Differential', HI_ratios = 'Case B', Ices_6micron = False, BootStrap = False, N_bootstrap = 100, 
    useMCMC = False, InitialFit = False, lam_range=[1.5, 28.0], show_progress = True, N_MCMC = 5000, N_BurnIn = 15000, 
    ExtCurve = 'D23ExtCurve', EmCurve = 'D24Emissivity', MIR_CH = 'CHExt_v3', NIR_CH = 'CH_NIR', Fit_NIR_CH = False, 
    NIR_Ice = 'NIR_Ice', NIR_CO2 = 'NIR_CO2', Fit_CO = False, spec_res = 'h')