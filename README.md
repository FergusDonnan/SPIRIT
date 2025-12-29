<div align="center">

![SPIRIT_Logo](https://github.com/user-attachments/assets/0fdd6af0-3090-47dd-b334-697f7079cf15)


# SPIRIT

**S**pectral **I**nfra**R**ed **I**nference **T**ool

A powerful Python tool for modeling JWST NIRSpec and MIRI spectra using differential extinction continuum model and flexible PAH profiles.



</div>

---

## 🌟 Overview

SPIRIT is a novel and flexible tool designed to analyze JWST infrared spectra enabling one to infer Polycyclic Aromatic Hydrocarbon (PAH) fluxes and characterise the nature of the dust emission/extinction. The tool was first presented in [Donnan+24a](https://ui.adsabs.harvard.edu/abs/2024MNRAS.529.1386D/abstract). 



### Key Features

- **Differential Extinction Continuum Model**: The main novelty of this tool is the differential extinction model, which models the dust continuum as a 2D weighted average of modified black bodies at a variety of temperatures and extinctions. This is therefore a generalisation of the geometry of the dust where instead of assuming a simple screen or mixed model, the fit will infer a 2D distribution of dust extinction and temperature. This provides not only the flexibility to fit highly obscured environments where other codes fail but also provides physical constraints on teh nature of the obscuring dust.
- **Flexible PAH Profiles**: Inspired by PAHFIT (Smith+07), the PAH features are modelled as a series of Drude profiles where the shape is allowed to vary to allow accurate PAH flux measurements in a variety of environments. We also use a prior on the PAH band shapes to prevent overfitting in cases of low signal to noise.
- **PAH and Line Fluxes**: The code will outut PAH fluxes as well as emission line fluxes, where the latter are inferred by integrating continuum subtracted line regions.
- **Easy to use GUI**: Simple user interface to quickly fit spectra with a variety of options to swap out extinction curves, ice templates etc.
- **Extinction Estimates**: The code estimates the exitinction through a variety of ways: continuum from the differential extinction model, H2 lines via the rotational diagram, HI lines, stellar continuum and PAHs. For more details see [Donnan+24a](https://ui.adsabs.harvard.edu/abs/2024MNRAS.529.1386D/abstract).
- **Different Fitting Methods enabled by JAX**: The fitting can be performed in three ways, a simple maximum likelihood (quickest), bootstraping and MCMC using NUMPYRO. 

---

## 📋 Requirements

```
python >= 3.8
numpy
scipy
astropy
matplotlib
<!-- Additional dependencies -->
```

---

## 🚀 Installation

Download the files from GitHub and place into a suitable directory. To install enter the SPIRIT directory and run 
```
pip install -r requirements.txt
```
to install the required dependencies. To avoid messing up current installs you can create a new conda environment first using 
```
conda create -n SPIRIT_env python=3.6.3
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
After clicking run, the fit will run and populate the Results folder once completed. It may take some time to run depending on your device, a quick fit for a NIRSpec+MIRI spectrum should take at least a few hours to run.


<img width="703" height="730" alt="GUI" src="https://github.com/user-attachments/assets/5704808c-5139-4702-be67-ae6f0539ff17" />


Alternatively, the code can be ran in a python file or jupyter notebook by calling the RunModel function from SPIRIT.py. The Run.py file has an example of this which is also shown below. This can be good to run multiple fits in sequence, specifiying the spectra name in the objs array.


```python
import SPIRIT as SPIRIT
import numpy as np



# objs = ['VV114_NE']

SPIRIT.RunModel(objs, Dust_Geometry = 'Differential', HI_ratios = 'Case B', Ices_6micron = False, BootStrap = False, N_bootstrap = 100, 
    useMCMC = False, InitialFit = False, lam_range=[1.5, 28.0], show_progress = True, N_MCMC = 5000, N_BurnIn = 15000, 
    ExtCurve = 'D23ExtCurve', EmCurve = 'D24Emissivity', MIR_CH = 'CHExt_v3', NIR_CH = 'CH_NIR', Fit_NIR_CH = False, 
    NIR_Ice = 'NIR_Ice', NIR_CO2 = 'NIR_CO2', Fit_CO = False, spec_res = 'h')
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



### Ext. Corrections



---

##  Options


---

## Additional Analysis

<!-- 
Section for example plots/figures showing: 
- Input spectrum
- Model fit
- PAH flux measurements
- Residuals
-->



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

### Related Publications

- <!-- Publication 1 -->
- <!-- Publication 2 -->

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---



---

## 📬 Contact
Any issues please contact me:
- **Email**: fdonnan@ucsd.edu

---



---

<div align="center">

</div>
