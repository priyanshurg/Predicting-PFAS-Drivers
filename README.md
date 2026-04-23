Predicting PFAS Drivers in Public Water Systems
Overview

This repository contains the full modeling workflow used to identify industrial and geospatial drivers of PFAS detections in U.S. Public Water Systems (PWS).

The project integrates:

Geospatial feature engineering (e.g., industry distributions around water systems)
Machine learning models (e.g., Random Forests)
Interpretability tools (e.g., SHAP)

to uncover relationships between industrial activity and PFAS occurrence.

Objectives

The predictive framework is designed to:

Identify regions at elevated risk of PFAS contamination
Highlight key industrial sectors associated with detections
Provide interpretable insights to support monitoring and policy decisions
Data Access

Due to GitHub file size limitations, the dataset is hosted externally:
https://utexas.app.box.com/folder/366623207667

Instructions
Download the dataset from the link above
Extract the contents (if compressed)
Store the folder locally on your machine
Note the full directory path — you will need this during setup
Repository Structure (Conceptual)
Model-0 → Based on combined industry density in the HUC-8s
Model-1 → Based on AFFF users (airports, firefighting training stations, and military bases)
Model-2 → Based on only individual industry counts (25 NAICS codes)
Model-3 → Based on only individual industry counts (25 NAICS codes) [Duplicated, needs to be replaced]
Model-4 → 25 NAICS codes + 3 AFFF users
Model-5 → Interpretation and analysis (e.g., SHAP insights)
