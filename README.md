<div align="center">

![SPIRIT_Logo](https://github.com/user-attachments/assets/0fdd6af0-3090-47dd-b334-697f7079cf15)


# SPIRIT

**S**pectral **I**nfra**R**ed **I**nference **T**ool

A powerful Python package for modeling JWST NIRSpec and MIRI spectra using differential extinction continuum models and flexible PAH profiles.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen. svg)](docs/)

[Installation](#installation) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Citation](#citation)

</div>

---

## 🌟 Overview

SPIRIT is a novel and flexible tool designed to analyze JWST infrared spectra with sophisticated physical models.  The package enables precise inference of Polycyclic Aromatic Hydrocarbon (PAH) fluxes and characterization of dust emission and extinction properties.

### Key Features

- **Differential Extinction Continuum Model**: <!-- Description of the continuum modeling approach -->
- **Flexible PAH Profiles**:  <!-- Description of PAH profile flexibility and parameterization -->
- **JWST Instrument Support**: Native support for NIRSpec and MIRI observations
- **Bayesian Inference**: <!-- Description of inference methodology -->
- **Customizable Components**: <!-- Description of model customization options -->

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

### Via pip (recommended)

```bash
pip install spirit-jwst
```

### From source

```bash
git clone https://github.com/FergusDonnan/SPIRIT.git
cd SPIRIT
pip install -e .
```

---

## 💡 Quick Start

### Basic Usage

```python
import spirit

# Load your JWST spectrum
spectrum = spirit.load_spectrum('path/to/your/spectrum.fits')

# Initialize the model
model = spirit.Model(
    # Model configuration parameters
)

# Fit the spectrum
results = model.fit(spectrum)

# Visualize results
spirit.plot_results(results)
```

### Advanced Example

```python
# Example with custom PAH profiles and extinction model
# Code example here
```

---

## 📖 Documentation

### Model Components

#### Differential Extinction Continuum

<!-- Detailed description of the continuum model -->

#### PAH Profiles

<!-- Detailed description of PAH modeling approach -->

#### Dust Emission

<!-- Detailed description of dust emission component -->

### Tutorials

- [Tutorial 1: Fitting a single NIRSpec spectrum](#)
- [Tutorial 2: Working with MIRI MRS data](#)
- [Tutorial 3: Custom PAH profile configuration](#)
- [Tutorial 4: Batch processing multiple spectra](#)

Full documentation available at:  <!-- Documentation URL -->

---

## 🔬 Science Applications

SPIRIT has been developed to address key science questions including:

- <!-- Science application 1 -->
- <!-- Science application 2 -->
- <!-- Science application 3 -->
- <!-- Science application 4 -->

---

## 📊 Example Results

<!-- 
Section for example plots/figures showing: 
- Input spectrum
- Model fit
- PAH flux measurements
- Residuals
-->

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/FergusDonnan/SPIRIT.git
cd SPIRIT

# Create development environment
# Instructions here

# Run tests
# Test commands here
```

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

## 👥 Authors and Acknowledgments

**Lead Developer**: <!-- Name and affiliation -->

**Contributors**:
- <!-- Contributor list -->

**Acknowledgments**: 
- <!-- Acknowledgment of funding, collaborators, etc. -->

---

## 📬 Contact

- **Issues**: [GitHub Issues](https://github.com/FergusDonnan/SPIRIT/issues)
- **Discussions**: [GitHub Discussions](https://github.com/FergusDonnan/SPIRIT/discussions)
- **Email**: <!-- Contact email -->

---

## 🗺️ Roadmap

- [ ] <!-- Planned feature 1 -->
- [ ] <!-- Planned feature 2 -->
- [ ] <!-- Planned feature 3 -->
- [ ] Integration with other JWST analysis tools

---

<div align="center">

**SPIRIT** is developed for the astronomical community working with JWST infrared spectroscopy. 

*Made with ❤️ for better PAH science*

</div>
