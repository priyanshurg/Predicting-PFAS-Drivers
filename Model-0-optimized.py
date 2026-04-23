
import geopandas as gpd
import pandas as pd
import numpy as np
import pickle
from shapely.geometry import Point
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import os


# --- Configuration & Paths ---
cwd = "/scr/Student-Folders/PRG/PROJECTS/PFAS-Project2/PFAS-Project"
contaminants_of_interest = ["PFPeA","PFBA", "PFHxA", "PFBS", "PFOS", "PFOA", "PFHxS", "PFHpA"]
states_to_exclude = ['AK', 'HI', 'PR', 'VI', 'AS', 'GU', 'NN', 'MP', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

# 1. Load Watershed Geometries
gdf = gpd.read_file(f"{cwd}/source_data_files/HUC8_CONUS/HUC8_US.shp").to_crs(epsg=4326)
gdf = gdf[['HUC8','geometry', 'NAME']]

# 2. Load and Filter UCMR5 PFAS Data
ucmr_all = pd.read_csv(f"{cwd}/source_data_files/UCMR-data-Oct-2024/UCMR5_All.txt", sep='\t', low_memory=False)
ucmr_filtered = ucmr_all[
    ucmr_all['Contaminant'].isin(contaminants_of_interest) & 
    ~ucmr_all['State'].isin(states_to_exclude)
].copy()

# 3. Load PWS Geolocations
pws_geolocations = pd.read_csv(f"{cwd}/source_data_files/generated-data-files/representative_pws_geolocations.csv")
pws_geo_gdf = gpd.GeoDataFrame(
    pws_geolocations, 
    geometry=gpd.points_from_xy(pws_geolocations.longitude, pws_geolocations.latitude),
    crs="EPSG:4326"
)

# --- Vectorized Spatial Join for PWS mapping ---
# Mapping Public Water Systems to HUC8 Watersheds
pws_mapped = gpd.sjoin(pws_geo_gdf, gdf[['HUC8', 'geometry']], how="left", predicate="within")

# Identify systems with detections (AnalyticalResultsSign == "=")
detect_pwsids = ucmr_filtered.loc[ucmr_filtered["AnalyticalResultsSign"] == "=", "PWSID"].unique()

# Separate into Detect and Non-Detect DataFrames
pws_detects_df = pws_mapped[pws_mapped["PWSID"].isin(detect_pwsids)].copy()
pws_non_detects_df = pws_mapped[~pws_mapped["PWSID"].isin(detect_pwsids)].copy()

# Identify Watersheds for Analysis
huc8_with_detects = set(pws_detects_df['HUC8'].dropna().unique())
huc8_with_any_data = set(pws_mapped['HUC8'].dropna().unique())
huc8_clean_controls = huc8_with_any_data - huc8_with_detects

# Filter Main GDF to only include watersheds with UCMR data
gdf_in_ucmr = gdf[gdf['HUC8'].isin(huc8_with_any_data)].reset_index(drop=True)

print("--- DATA PROCESSING SUMMARY ---")
print(f"Total PWS records analyzed: {len(pws_mapped)}")
print(f"Watersheds with detections: {len(huc8_with_detects)}")
print(f"Watersheds with zero detections (controls): {len(huc8_clean_controls)}")

# --- Industrial Data Processing ---
epa_naics = pd.read_csv(f"{cwd}/source_data_files/Facilities-coordinates/EPA-NAICS-Geolocations.csv")
naics_codes = [
    313320, 325510, 322220, 313210, 322121, 332813, 324110, 325612,
    334413, 326113, 332812, 333318, 334419, 562212, 325199, 323111,
    313110, 314110, 316110, 325211, 324191, 325998, 562211, 562213,
    313310, 322219, 323120, 313220, 313230, 322130, 332999, 424690,
    314910, 326112, 335999, 562112, 562219, 325611
]

# Filter for NAICS of interest and drop duplicates for speed
industry_filtered = epa_naics[epa_naics['naics_code'].isin(naics_codes)].dropna(subset=['latitude83', 'longitude83'])
industry_filtered = industry_filtered.drop_duplicates(subset=['latitude83', 'longitude83'])

# Select high-frequency NAICS (Count >= 500)
naics_counts = industry_filtered['naics_code'].value_counts()
top_naics = naics_counts[naics_counts >= 500].index.tolist()
industry_final = industry_filtered[industry_filtered['naics_code'].isin(top_naics)]

# Convert to GeoDataFrame
industry_gdf = gpd.GeoDataFrame(
    industry_final, 
    geometry=gpd.points_from_xy(industry_final.longitude83, industry_final.latitude83),
    crs="EPSG:4326"
)

# --- Vectorized Industrial Count ---
# Spatial join industry points to watersheds
industry_joined = gpd.sjoin(industry_gdf, gdf_in_ucmr[['HUC8', 'geometry']], how="inner", predicate="within")
huc_counts = industry_joined['HUC8'].value_counts()

# Map counts back to main dataframe and fill NaNs (watersheds with 0 industry) with 0
gdf_in_ucmr['Industry_Count'] = gdf_in_ucmr['HUC8'].map(huc_counts).fillna(0)

# --- Feature Engineering (Density & Labels) ---
# Project to Equal Area (EPSG:5070) for accurate area calculation (units: meters^2)
gdf_proj = gdf_in_ucmr.to_crs(epsg=5070)
gdf_in_ucmr['density'] = gdf_in_ucmr['Industry_Count'] / gdf_proj['geometry'].area

# Set Violation Status (1 = detection in watershed, 0 = clean control)
gdf_in_ucmr['violation_status'] = gdf_in_ucmr['HUC8'].isin(huc8_with_detects).astype(int)

print("\n--- GEOSPATIAL FEATURE ENGINEERING COMPLETE ---")
print(f"Total Watersheds in Final Dataset: {len(gdf_in_ucmr)}")
print(f"Average Industrial Density: {gdf_in_ucmr['density'].mean():.2e} facilities/m^2")
print(f"Label Distribution: {gdf_in_ucmr['violation_status'].value_counts().to_dict()}")
print("-" * 30)


# # MODEL B - Training a 10-fold cross validated Logistic Regression Model on Generic Industrial Density

# In[197]:


X = gdf_in_ucmr[['density']].values
y = gdf_in_ucmr['violation_status'].values

import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_score, recall_score, f1_score

class ModelEvaluationUI:
    def __init__(self, root, X, y):
        self.root = root
        self.root.title("PFAS Violation Model Explorer")
        self.X, self.y = X, y
        
        # Data storage for results
        self.last_y_test = None
        self.last_y_proba = None
        
        # --- UI Layout ---
        control_frame = ttk.Frame(root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # Model Dropdown
        ttk.Label(control_frame, text="Select Model:").pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar(value="RandomForest")
        self.model_selector = ttk.Combobox(control_frame, textvariable=self.model_var)
        self.model_selector['values'] = ("RandomForest", "LogisticRegression", "MLP")
        self.model_selector.pack(side=tk.LEFT, padx=5)

        # Run Button
        self.run_btn = ttk.Button(control_frame, text="Train & Evaluate", command=self.run_pipeline)
        self.run_btn.pack(side=tk.LEFT, padx=10)

        # Threshold Slider
        ttk.Label(control_frame, text="Threshold:").pack(side=tk.LEFT, padx=5)
        self.threshold_slider = tk.Scale(control_frame, from_=0.0, to=1.0, resolution=0.01, 
                                         orient=tk.HORIZONTAL, command=self.update_metrics)
        self.threshold_slider.set(0.5)
        self.threshold_slider.pack(side=tk.LEFT, padx=5)

        # Metrics Output
        self.stats_label = ttk.Label(root, text="Select a model and click Run", font=('Helvetica', 10, 'bold'))
        self.stats_label.pack(side=tk.TOP, pady=5)

        # Matplotlib Figure
        self.fig, (self.ax_roc, self.ax_metrics) = plt.subplots(1, 2, figsize=(12, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def get_selected_model(self):
        m = self.model_var.get()
        if m == "RandomForest": return RandomForestClassifier()
        if m == "LogisticRegression": return LogisticRegression()
        if m == "MLP": return MLPClassifier(max_iter=500)
        return RandomForestClassifier()

    def run_pipeline(self):
        model = self.get_selected_model()
        kf = KFold(n_splits=5, shuffle=True)
        scaler = StandardScaler()
        
        tprs, aucs = [], []
        mean_fpr = np.linspace(0, 1, 100)
        
        self.ax_roc.clear()

        # Simple loop for ROC Cross-Val
        for train_idx, test_idx in kf.split(self.X):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model.fit(X_train_scaled, y_train)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc(fpr, tpr))
            self.ax_roc.plot(fpr, tpr, alpha=0.3, lw=1)
            
            # Save the last fold for the threshold slider demo
            self.last_y_test = y_test
            self.last_y_proba = y_proba

        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        
        self.ax_roc.plot(mean_fpr, mean_tpr, color='b', lw=2, label=f'Mean ROC (AUC={mean_auc:.2f})')
        self.ax_roc.plot([0,1],[0,1], 'r--')
        self.ax_roc.set_title("ROC Curve (K-Fold)")
        self.ax_roc.legend()
        
        self.update_metrics()

    def update_metrics(self, event=None):
        if self.last_y_test is None: return
        
        thresh = self.threshold_slider.get()
        y_pred = (self.last_y_proba >= thresh).astype(int)
        
        acc = accuracy_score(self.last_y_test, y_pred)
        pre = precision_score(self.last_y_test, y_pred, zero_division=0)
        rec = recall_score(self.last_y_test, y_pred, zero_division=0)
        f1  = f1_score(self.last_y_test, y_pred, zero_division=0)
        
        self.stats_label.config(text=f"Current Metrics at Threshold {thresh:.2f} -> Accuracy: {acc:.2f} | Precision: {pre:.2f} | Recall: {rec:.2f} | F1: {f1:.2f}")
        
        # Update Metric bars or just redraw
        self.ax_metrics.clear()
        names = ['Accuracy', 'Precision', 'Recall', 'F1']
        values = [acc, pre, rec, f1]
        self.ax_metrics.bar(names, values, color=['blue', 'orange', 'green', 'red'])
        self.ax_metrics.set_ylim(0, 1.1)
        self.ax_metrics.set_title(f"Performance at Threshold {thresh}")
        
        self.canvas.draw()
root = tk.Tk()
app = ModelEvaluationUI(root, X, y)
root.mainloop()