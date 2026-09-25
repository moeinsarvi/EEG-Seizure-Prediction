# Prediction of Epileptic Seizures Using EEG Signals

This repository contains the code and reports for a project completed as part of an undergraduate Signals and Systems course. The project represents an initial, hands-on exploration into biomedical signal processing and baseline classification techniques.

## Project Context and Scope
The primary objective of this project was to apply theoretical concepts from the Signals and Systems curriculum to real-world continuous data. While the overarching theme is seizure prediction, this work precedes advanced coursework in machine learning or deep learning. Consequently, the focus is on building a fundamental pipeline—from raw signal preprocessing to basic feature extraction and baseline classification—rather than achieving state-of-the-art predictive accuracy. The modest classification results serve as a proof-of-concept for the applied signal processing techniques and provided a valuable learning experience in handling noisy, complex physiological data.

## Dataset
The data used in this project is sourced from the publicly available CHB-MIT Scalp EEG Database on PhysioNet. 
Note: To comply with repository size limits and standard practices, the raw `.edf` data files are not included in this repository. To execute the code, the relevant patient data must be downloaded independently and placed in the working directory.

## Methodology

The project was conducted in two main phases:

### Phase 1: Signal Preprocessing (EEGLAB)
The first phase focused on cleaning the raw EEG datasets using the EEGLAB toolbox in MATLAB.
* Filtering: Application of a 1 Hz high-pass filter to remove baseline drifts and a 50 Hz notch filter to mitigate power line interference.
* Re-referencing: Implementation of the average reference method to balance signal amplitudes across channels.
* Artifact Removal: Use of Independent Component Analysis (ICA) to decompose the signals and isolate physiological artifacts (e.g., muscle activity, eye blinks, cardiac noise).
* Source Localization: Basic localization performed using the dipfit plugin and the MNI head model.

### Phase 2: Feature Extraction and Classification
The second phase involved extracting manageable features from the cleaned, continuous EEG signals (segmented into 16-second epochs) to train baseline classifiers.
* Spectral and Temporal Features: Extraction of Power Spectral Density (PSD) via Fast Fourier Transform (FFT), Shannon Entropy to measure signal complexity, and basic statistical measures (mean, standard deviation, min, max).
* Feature Selection: A statistical t-test (Alpha = 0.001) was utilized to identify variables showing significant differences.
* Baseline Classification: The extracted features were evaluated using baseline models, specifically a Support Vector Machine (SVM) with a linear kernel and a K-Nearest Neighbor (KNN) classifier (K=5).

## Repository Structure
* Phase1_Preprocessing: Contains the project report detailing the theoretical foundations and EEGLAB preprocessing steps.
* Phase2_Classification: Contains the MATLAB scripts for feature extraction and classification, alongside the final project report.

## Documentation
Detailed methodologies, mathematical formulations, and visual outputs (including frequency spectrums and ICA dipole locations) can be found in the attached PDF reports within each phase's directory.
