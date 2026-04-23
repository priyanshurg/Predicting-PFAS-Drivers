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

#### **Instructions**
1.  **Download** the dataset from the link above.
2.  **Extract** the contents (if compressed).
3.  **Store** the folder locally on your machine.
4.  **Note the full directory path** — you will need this during setup.

---

### **Repository Structure (Conceptual)**
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
```python -m venv pfas_env
source pfas_env/bin/activate      # Linux / Mac
pfas_env\Scripts\activate         # Windows
```
If you're using conda:
```conda create -n pfas_env python=3.10
conda activate pfas_env
```

Install the dependencies using:
```pip install -r requirements.txt
```

