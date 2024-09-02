# Machine Learning Approaches to Predict the El Niño-Southern Oscillation

## Project Objectives
This work aims to advance the understanding and application of Machine Learning approaches in predicting the El Niño-Southern Oscillation by focusing on the following key objectives:
1. Investigate Graph Structure Generation Methods: Explore and evaluate various methods for generating graph structures that can effectively capture and represent the complex relationships within and between climate fields [[*Tsonis, Swanson, and Roebber, 2006*](https://doi.org/10.1175/BAMS-87-5-585)].
2. Select the Optimal Earth System Models (ESMs) for Transfer Learning: Apply intuition from modeling studies [[*Rivera Tello, Takahashi, and Karamperidou, 2023*](https://doi.org/10.1038/s41598-023-45739-3)] to select ESMs whose modes of variability most accurately represent the diversity of El Niño events, with the aim of improving predictive performance.
3. Apply GNNs to the prediction of various ENSO flavors: Use data derived from the selected ESM outputs to train GNNs to more accurately predict the Eastern and Central Pacific El Niño events.
4. Enhance the Interpretability of Trained Models: Use explainability techniques [[*Samek et al., 2019*](https://doi.org/10.1007/978-3-030-28954-6)] to better understand the underlying factors influencing the predictions made by GNN models.


## Credentials
- **Author**
  - [Francis Murray](https://github.com/francis-murray), EPFL
- **Supervisors**
  - [Prof. Julien Emile-Geay](https://dornsife.usc.edu/profile/julien-emile-geay/), USC
  - [Prof. Sam Silva](https://dornsife.usc.edu/profile/sam-silva/), USC
  - [Prof. Devis Tuia](https://people.epfl.ch/devis.tuia), EPFL
- **Hosting Lab**
  - Climate Dynamics Lab, [Department of Earth Sciences](https://dornsife.usc.edu/earth/research/climate-science/), [University of Southern California](https://www.usc.edu/)


## Data
- All CMIP6 datasets used in this study are publicly available through the Earth System Grid Federation (ESGF) federated nodes at https://esgf-node.llnl.gov/projects/esgf-llnl/.
- The NCEP Global Ocean Data Assimilation System (GODAS) reanalysis dataset is available at https://psl.noaa.gov/data/gridded/data.godas.html.
- The Hadley Centre Global Sea Ice and Sea Surface Temperature (HadISST) observational dataset is available at https://www.metoffice.gov.uk/hadobs/hadisst/data/download.html.