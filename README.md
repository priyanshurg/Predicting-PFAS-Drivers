# Predicting PFAS Drivers in Public Water Systems: Overview

This repository contains the full modeling workflow used to identify **industrial and geospatial drivers** of PFAS detections in U.S. Public Water Systems (PWS).

The project integrates:
* **Geospatial feature engineering** (e.g., industry distributions around water systems)
* **Machine learning models** (e.g., Random Forests)
* **Interpretability tools** (e.g., SHAP)

to uncover relationships between industrial activity and PFAS occurrence.

---

### **Objectives**
The predictive framework is designed to:
* **Identify regions** at elevated risk of PFAS contamination.
* **Highlight key industrial sectors** associated with detections.
* **Provide interpretable insights** to support monitoring and policy decisions.

---

### **Data Access**
Due to GitHub file size limitations, the dataset is hosted externally:  
👉 [**Download Dataset Here**](https://utexas.app.box.com/folder/366623207667)
Link: https://utexas.app.box.com/folder/366623207667

#### **Instructions**
1.  **Download** the dataset from the link above.
2.  **Extract** the contents (if compressed).
3.  **Store** the folder locally on your machine.
4.  **Note the full directory path** — you will need this during setup.

---

### **Repository Structure**
* **`Model-0`** → Based on combined industry density in the HUC-8s.
* **`Model-1`** → Based on AFFF users (airports, firefighting training stations, and military bases).
* **`Model-2`** → Based on only individual industry counts (25 NAICS codes).
* **`Model-3`** → Based on only individual industry counts (25 NAICS codes). *[Duplicated, needs to be replaced]*
* **`Model-4`** → 25 NAICS codes + 3 AFFF users.
* **`Model-5`** → Interpretation and analysis (e.g., SHAP insights).


## How to Run

Follow the steps below to reproduce the full pipeline:

### 1. Clone or Download the Repository
Run the following in your terminal:
```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Create and Activate a Virtual Environment
It is recommended to isolate dependencies to avoid version conflicts.
```bash
python -m venv pfas_env
source pfas_env/bin/activate      # Linux / Mac
pfas_env\Scripts\activate         # Windows
```
If you're using conda:
```conda create -n pfas_env python=3.10
conda activate pfas_env
```

Install the dependencies using:
```bash
pip install -r requirements.txt
```

### 3. Update File Paths in Code
Locate the variable cwd in your script:
```python
cwd = "path_to_source_data_files"
```
### 4. Run the Models
Choose the model you want to run from the Repository Structure and run it. Example, if you want to run Model-5:
```bash
python Model-5.py
```

### 5. User Interface
<img width="1220" height="653" alt="image" src="https://github.com/user-attachments/assets/1f86a7c3-4605-4e59-b805-1e2fdf6aa884" />

1.  Select your model architecture of choice from the **dropdown**.
2.  Click **Train & Evaluate** to train the model using 5-Fold cross validation.
3.  This will plot the ROC-AUC curve as well as the different performance metrics in the white boxes.
4.  Use the **Threshold** scroll bar to adjust the classification threshold and view the impact on the performance metrics.
Other Buttons:

A.                **Threshold Metrics:** plots the model performance metrics versus the threshold choice between 0 and 1.

B.                **SHAP summary:** will plot the beeswarm plot of SHAP impacts for the trained model.

C.                **Mean SHAP importance:** will plot the mean of absolute SHAP impacts for the top 10 most important features of the model.

D.                **Save:** to save the model and current threshold values as a pickle, _.pkl_, file.

E.                **Load:** load a saved model and threshold saved as a _.pkl_ file.

Exemplar Interface after training a Random Forest Model and optimizing the threshold for maximizing F1-score.
<img width="1220" height="653" alt="image" src="https://github.com/user-attachments/assets/c48bef26-5ff0-4399-8f9c-135ae3730f2f" />

