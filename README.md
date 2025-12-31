<div align="center">

![SPIRIT_Logo](https://github.com/user-attachments/assets/0fdd6af0-3090-47dd-b334-697f7079cf15)


# SPIRIT

**S**pectral **I**nfra**R**ed **I**nference **T**ool

A powerful Python tool for modeling JWST NIRSpec and MIRI spectra using differential extinction continuum model and flexible PAH profiles. 



</div>

---

## 🌟 Overview

SPIRIT is a novel and flexible tool designed to analyze JWST infrared spectra enabling one to infer Polycyclic Aromatic Hydrocarbon (PAH) fluxes and characterise the nature of the dust emission/extinction. The tool was first presented in [Donnan+24a](https://ui.adsabs.harvard.edu/abs/2024MNRAS.529.1386D/abstract). An example fit to an obscured galaxy nucleus is shown below with the inferred 2D dust distribution.


![ExampleFit](https://github.com/user-attachments/assets/202e9c9a-da95-46df-b8df-263a0a015252)



### Key Features

- **Differential Extinction Continuum Model**: The main novelty of this tool is the differential extinction model, which models the dust continuum as a 2D weighted average of modified black bodies at a variety of temperatures and extinctions. This is therefore a generalisation of the geometry of the dust where instead of assuming a simple screen or mixed model, the fit will infer a 2D distribution of dust extinction and temperature. This provides not only the flexibility to fit highly obscured environments where other codes fail but also provides physical constraints on teh nature of the obscuring dust.
- **Flexible PAH Profiles**: Inspired by PAHFIT (Smith+07), the PAH features are modelled as a series of Drude profiles where the shape is allowed to vary to allow accurate PAH flux measurements in a variety of environments. We also use a prior on the PAH band shapes to prevent overfitting in cases of low signal to noise.
- **PAH and Line Fluxes**: The code will outut PAH fluxes as well as emission line fluxes, where the latter are inferred by integrating continuum subtracted line regions.
- **Easy to use GUI**: Simple user interface to quickly fit spectra with a variety of options to swap out extinction curves, ice templates etc.
- **Extinction Estimates**: The code estimates the exitinction through a variety of ways: continuum from the differential extinction model, H2 lines via the rotational diagram, HI lines, stellar continuum and PAHs. For more details see [Donnan+24a](https://ui.adsabs.harvard.edu/abs/2024MNRAS.529.1386D/abstract).
- **Different Fitting Methods enabled by JAX**: The fitting can be performed in three ways, a simple maximum likelihood (quickest), bootstraping and MCMC using NUMPYRO. 

---

## 📋 Requirements

Run pip install -r requirements.txt to install all the required packages.
```
astropy==6.0.1
jax==0.4.6
jax-cosmo==0.1.0
jaxlib==0.4.6
jaxopt==0.4.2
jdaviz==2.7.1
matplotlib==3.9.4
numpy==1.24.4
numpyro==0.11.0
pandas==2.3.2
SciencePlots==2.1.1
scipy==1.9.0
tabulate==0.9.0
tk==0.1.0
tqdm==4.67.1

```

---

## 🚀 Installation

Download the files from GitHub and place into a suitable directory. To install enter the SPIRIT directory and run 
```
pip install -r requirements.txt
```
to install the required dependencies. To avoid messing up current installs you can create a new conda environment first using 
```
conda create -n SPIRIT_env python=3.9.23
```
and activate the environment with 
```
conda activate SPIRIT_env
```
That's all to install as everything is ran from this directory.

---

## 💡 Quick Start

### Basic Usage

The easiest way to run SPIRIT is with the GUI which is opened by running SPIRIT.py in the terminal:
```
python SPIRIT.py
```
This will produce a popup with a variety of options before running the code. First, place a .txt file of the spectrum you would like to fit in the /Data directory. The file should have three columns, wavlength (micron) flux (Jy) error (Jy). The error values don't matter too much as the code will calculate the appropriate error values first before fitting. Select the spectrum you would like to fit from the drop down menu.

A fitting method then needs to be selected. By default the "Quick" method is a maximimum probability fit, "Boootstrap" will repeat the maximum probability fit N times to obtain uncertanties on the best fit parameters. "MCMC" using NUMPYRO NUTS sampling to sample the posterior probability to obtain uncertanties.
After clicking run, there will be a long pause while it compiles before running the fit. The Results folder will be populated once completed. It may take some time to run depending on your device, a quick fit for a NIRSpec+MIRI spectrum should take at a few hours to run.


<img width="703" height="730" alt="GUI" src="https://github.com/user-attachments/assets/5704808c-5139-4702-be67-ae6f0539ff17" />


Alternatively, the code can be ran in a python file or jupyter notebook by calling the RunModel function from SPIRIT.py. The Run.py file has an example of this which is also shown below. This can be good to run multiple fits in sequence, specifiying the spectra name in the objs array.


```python
import SPIRIT as SPIRIT
import numpy as np



objs = ['VV114_NE']

SPIRIT.RunModel(objs, Dust_Geometry = 'Differential', HI_ratios = 'Case B', Ices_6micron = False, BootStrap = False, N_bootstrap = 50, 
    useMCMC = False, InitialFit = False, lam_range=[1.5, 28.0], show_progress = True, N_MCMC = 2000, N_BurnIn = 10000, 
    ExtCurve = 'D23ExtCurve_New', EmCurve = 'D24Emissivity', MIR_CH = 'CHExt_v3', NIR_CH = 'CH_NIR', Fit_NIR_CH = False,
    NIR_Ice = 'NIR_Ice', NIR_CO2 = 'NIR_CO2', RegStrength = 1000,  Cont_Only = False, St_Cont = False, Extend = False, Fit_CO = False, spec_res = 'h')



```





---

## 📊 Fitting Output

All the output files are located in /Results folder. Within the reuslts directory there are a varierty of files:

### Plots
A variety of plots are generated including:

*_Plot.pdf: Contains a simple plot of the best fit model and its constituent components.

*_Psi.pdf: Shows the inferred 2D dust distribution.

*H2 Rotation Plot.pdf: H2 rotational plot with the inferred H2 extinction value.

*H2Smoothness.pdf:

*Effective Ext Plot.pdf: The effective tau_9.8 as a function of wavelength for the differential extinction model. 


### Fluxes
Fluxes of the PAH features and emission lines are stored within the *Output.csv file. This file also containes equivalent widths and the continuum values at the corresponding wavelengths.

### Model Components
This folder contains the various components of the best fit model which can be used for your own plotting, including the original spectrum with the error values updated using the values calculated by the code before fitting.



### Ext. Corrections
This folder contains the extinction correction factors as a function of wavelength, which is of the form F_obs/F_intr, for the different estimates of extinction from the best fit. To calculate extinction corrected fluxes, take the extinction correction you would like to use and divide the observed flux by the value of the extinction correction factor at the corresponding wavelength. I would reccomend the H2 or HI estimated extinction for correcting the flux of emission features as the continuum can often be more buried than the source of emission.


---

##  Options

There are a variety different options when running the code. These are presented in the GUI when running SPIRIT.py. There are a variety of drop down menus to swap out ice templates, extinction curves and the emissivity curve. To get your custom files to appear in the drop down menus, place your files in the appropriate directory.


---

## Additional Analysis

It is possible to infer the temperature and extinction of different components of the dust distribution by splitting the dust distribution at various points as was done in [Donnan+24a](https://ui.adsabs.harvard.edu/abs/2024MNRAS.529.1386D/abstract). This is useful if the dust distribution shows distinct seperate components. The file ExtractComponents.py contains code to do this. 



---

## 🔬 Science Applications

SPIRIT has already been used in numerous works such as 

- <!-- Science application 1 -->
- <!-- Science application 2 -->
- <!-- Science application 3 -->
- <!-- Science application 4 -->
---

## 📄 Citation

If you use SPIRIT in your research, please cite:

```bibtex
@ARTICLE{2024MNRAS.529.1386D,
       author = {{Donnan}, F.~R. and {Garc{\'\i}a-Bernete}, I. and {Rigopoulou}, D. and {Pereira-Santaella}, M. and {Roche}, P.~F. and {Alonso-Herrero}, A.},
        title = "{Peeling back the layers of extinction of dusty galaxies in the era of JWST: modelling joint NIRSpec + MIRI spectra at rest-frame 1.5-28 {\ensuremath{\mu}}m}",
      journal = {\mnras},
     keywords = {techniques: spectroscopic, galaxies: evolution, galaxies: nuclei, Astrophysics - Astrophysics of Galaxies},
         year = 2024,
        month = apr,
       volume = {529},
       number = {2},
        pages = {1386-1404},
          doi = {10.1093/mnras/stae612},
archivePrefix = {arXiv},
       eprint = {2402.17479},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024MNRAS.529.1386D},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}


```



---

## 📝 To do

While the bootstrap and MCMC methods provide uncertanties on the PAH fluxes, the uncertanties appear very small due to the high signal to noise of JWST spectra. A better method of estimating uncertanties needs to be added that accounts for sytematic errors in the model choices/assumptions rather than just errors based on the quality of the data. 



---



---

## 📬 Contact
Any issues please contact me:
- **Email**: fdonnan@ucsd.edu

---



---

<div align="center">

</div>
