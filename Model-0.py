

# In[1]:
cwd = "/scr/Student-Folders/PRG/PROJECTS/PFAS-Project2/PFAS-Project"

# filename = 
import geopandas as gpd
import folium
import json
import pandas as pd
import numpy as np
import pickle
from shapely.geometry import Point
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import pprint
import os
pp = pprint.PrettyPrinter(indent=0)


contaminants_of_interest = ["PFPeA","PFBA", "PFHxA", "PFBS", "PFOS", "PFOA", "PFHxS", "PFHpA"]
states_i_want_to_exclude = ['AK', 'HI', 'PR', 'VI', 'AS', 'GU', 'NN', 'MP', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

shapefile_path = f"{cwd}/source_data_files/HUC8_CONUS/HUC8_US.shp"
gdf = gpd.read_file(shapefile_path)
gdf = gdf.to_crs(epsg=4326)
gdf = gdf[['HUC8','geometry', 'NAME']]

# PWS Geospatial Data
pws_geolocations = pd.read_csv(f"{cwd}/source_data_files/generated-data-files/representative_pws_geolocations.csv", header=0)
print("Number of PWS geolocations:", pws_geolocations.shape[0], "with unique PWSIDs:", pws_geolocations.PWSID.nunique())
ucmr_all_file_path = f"{cwd}/source_data_files/UCMR-data-Oct-2024/UCMR5_All.txt"
ucmr_all_df = pd.read_csv(ucmr_all_file_path, sep='\t', header=0)
print("Number of UCMR5 records:", ucmr_all_df.shape[0], "with unique PWSIDs:", ucmr_all_df.PWSID.nunique())
# Filter UCMR5 data for contaminants of interest and then for Contiguous United States
ucmr_filtered_df = ucmr_all_df[ucmr_all_df['Contaminant'].isin(contaminants_of_interest)]
print("Number of UCMR5 records after filtering for contaminants of interest:", ucmr_filtered_df.PWSID.nunique(), " unique PWSIDs.\nNumber of PWSIDs that did not report a contaminant of interest:", ucmr_all_df.PWSID.nunique() - ucmr_filtered_df.PWSID.nunique())
ucmr_filtered_df = ucmr_filtered_df[ucmr_filtered_df['State'].isin(states_i_want_to_exclude) == False]
print("Number of UCMR5 records after filtering for Contiguous United States:", ucmr_filtered_df.PWSID.nunique(), "unique PWSIDs.")
print("Column Names in UCMR5 data after filtering:", ucmr_filtered_df.columns)


# In[185]:


pws_without_geolocation = []
pws_detects = ucmr_filtered_df[ucmr_filtered_df["AnalyticalResultsSign"]=="="][["PWSID", "Contaminant"]]
pws_detects_df = pd.DataFrame(pws_detects.groupby("PWSID")["Contaminant"].apply(list)).reset_index()
for index, row in pws_detects_df.iterrows():
    geolocation = pws_geolocations[pws_geolocations["PWSID"] == row["PWSID"]][["latitude", "longitude"]].values
    if geolocation.size == 0:
        pws_without_geolocation.append(row["PWSID"])
        pws_detects_df.at[index, "HUC8"] = np.nan
        continue
    else:
        latitude, longitude = geolocation[0]
        point = Point(longitude, latitude)
        pws_detects_df.at[index, "geometry"] = point
        matching_huc8 = gdf[gdf.geometry.contains(point)]
        pws_detects_df.at[index, "HUC8"] = matching_huc8["HUC8"].values[0] if not matching_huc8.empty else None
# Explanatory Summary for Detects
print("-" * 30)
print("GEOSPATIAL JOIN SUMMARY: PUBLIC WATER SYSTEMS (PWS) WITH DETECTS")
print(f"Total unique PWSIDs with at least one contaminant detection: {pws_detects_df.PWSID.nunique()}")
print(f"Number of detected systems missing geolocation or HUC8 mapping: {pws_detects_df['HUC8'].isna().sum()}")
print(f"DataFrame attributes created: {', '.join(pws_detects_df.columns.tolist())}")

pws_non_detects = ucmr_filtered_df[ucmr_filtered_df["PWSID"].isin(pws_detects_df["PWSID"]) == False]
pws_non_detects_df = pd.DataFrame(pws_non_detects.groupby("PWSID")["Contaminant"].apply(list)).reset_index()
for index, row in pws_non_detects_df.iterrows():
    geolocation = pws_geolocations[pws_geolocations["PWSID"] == row["PWSID"]][["latitude", "longitude"]].values
    if geolocation.size == 0:
        pws_without_geolocation.append(row["PWSID"])
        pws_non_detects_df.at[index, "HUC8"] = np.nan
        continue
    else:
        latitude, longitude = geolocation[0]
        point = Point(longitude, latitude)
        pws_non_detects_df.at[index, "geometry"] = point
        matching_huc8 = gdf[gdf.geometry.contains(point)]
        pws_non_detects_df.at[index, "HUC8"] = matching_huc8["HUC8"].values[0] if not matching_huc8.empty else None
# Explanatory Summary for Non-Detects
print("\n" + "-" * 30)
print("GEOSPATIAL JOIN SUMMARY: PUBLIC WATER SYSTEMS (PWS) WITHOUT DETECTS")
print(f"Total unique PWSIDs with zero detections (all measurements below threshold): {pws_non_detects_df.PWSID.nunique()}")
print(f"Number of non-detect systems missing geolocation or HUC8 mapping: {pws_non_detects_df['HUC8'].isna().sum()}")

# Final Totals
print("\n" + "=" * 30)
print("FINAL DATA INTEGRITY CHECK")
print(f"Total PWSIDs excluded due to missing geolocation data: {len(pws_without_geolocation)}")
print("=" * 30)

# In[186]:

# --- Aggregating Watersheds with Confirmed Detections ---
# We filter for systems where a HUC8 was successfully mapped and a contaminant was detected
df_of_huc8_with_detections = pws_detects_df[pws_detects_df["HUC8"].notna()][["HUC8", "Contaminant"]]

print("-" * 30)
print("WATERSHED ANALYSIS: POSITIVE DETECTIONS")
print(f"Number of HUC8 watersheds with at least one PWS detection: {df_of_huc8_with_detections['HUC8'].nunique()}")
print(f"Total system-level detection records mapped to HUC8 units: {df_of_huc8_with_detections.shape[0]}")

# --- Aggregating Watersheds with Zero Detections (Control Group) ---
# First, identify all systems that had no detections and were successfully geocoded
df_of_huc8_with_atleast_one_non_detections = pws_non_detects_df[pws_non_detects_df["HUC8"].notna()][["HUC8", "Contaminant"]]

# Crucially, we isolate HUC8 units where NO systems reported detections.
# This excludes HUC8s that might have a mix of clean and contaminated systems.
df_of_huc8_with_all_non_detects = df_of_huc8_with_atleast_one_non_detections[
    ~df_of_huc8_with_atleast_one_non_detections["HUC8"].isin(df_of_huc8_with_detections["HUC8"])
]

print("\n" + "-" * 30)
print("WATERSHED ANALYSIS: UNCONTAMINATED CONTROLS")
print(f"Number of 'Clean' HUC8 watersheds (zero detections across all associated PWS): {df_of_huc8_with_all_non_detects['HUC8'].nunique()}")
print(f"Total system-level non-detect records in these 'clean' watersheds: {df_of_huc8_with_all_non_detects.shape[0]}")
print("-" * 30)

# In[187]:


epa_naics_geolocations = pd.read_csv(f"{cwd}/source_data_files/Facilities-coordinates/EPA-NAICS-Geolocations.csv")
prefix2 = f"{cwd}/source_data_files/Facilities-coordinates/"
airports = pickle.load(open(f"{prefix2}major_airport_coordinates.pkl", "rb"))
military_bases = pickle.load(open(f"{prefix2}military_coordinates.pkl", "rb"))
fire_stations = pickle.load(open(f"{prefix2}fire-training-coordinates.pkl", "rb"))



# In[190]:


naics_codes = [
    313320, 325510, 322220, 313210, 322121, 332813, 324110, 325612,
    334413, 326113, 332812, 333318, 334419, 562212, 325199, 323111,
    313110, 314110, 316110, 325211, 324191, 325998, 562211, 562213,
    313310, 322219, 323120, 313220, 313230, 322130, 332999, 424690,
    314910, 326112, 335999, 562112, 562219, 325611
]
naics_points = []
for naics_code in naics_codes:
    naics_code_xy_df = epa_naics_geolocations[epa_naics_geolocations['naics_code'] == naics_code].reset_index(drop=True)
    naics_code_xy_data = naics_code_xy_df[['latitude83', 'longitude83']].dropna(axis=0, how='any').reset_index(drop=True)
    unique_xy_data = []
    seen = set()

    for lst in naics_code_xy_data.values:
        t = tuple(lst)  # Convert to tuple (hashable)
        if t not in seen:
            seen.add(t)
            unique_xy_data.append(lst)
    naics_points.append(len(unique_xy_data))

naics_count_df = pd.DataFrame({"NAICS_CODE": naics_codes, "COUNT": naics_points})
naics_count_df.sort_values(by="COUNT", ascending=False, inplace=True)
naics_count_df.reset_index(drop=True, inplace=True) 
NAICS_CODES_OF_INTEREST = naics_count_df[naics_count_df["COUNT"]>=500]['NAICS_CODE'].tolist()


# In[192]:


industry_xy_data = []
for naics_code in tqdm(NAICS_CODES_OF_INTEREST, desc="Processing NAICS Codes"):
    naics_code_xy_df = epa_naics_geolocations[epa_naics_geolocations['naics_code'] == naics_code].reset_index(drop=True)
    naics_code_xy_data = naics_code_xy_df[['latitude83', 'longitude83']].dropna(axis=0, how='any').reset_index(drop=True)
    unique_xy_data = []
    seen = set()

    for lst in naics_code_xy_data.values:
        t = tuple(lst)  
        if t not in seen:
            seen.add(t)
            unique_xy_data.append(lst)
    industry_xy_data.append(unique_xy_data)


    if len(naics_code_xy_data) == 0:
        print(f"NAICS code {naics_code} has no geolocations")
        break



# In[193]:


huc8s_in_ucmr = pd.concat([df_of_huc8_with_detections["HUC8"], df_of_huc8_with_all_non_detects["HUC8"]], ignore_index=True).unique()
gdf_in_ucmr = gdf[gdf['HUC8'].isin(huc8s_in_ucmr)].reset_index(drop=True)

naics_points = []
naics_ids = []
for group_id, group in enumerate(industry_xy_data):
    for lat, lon in group:
        naics_points.append(Point(lon, lat))
        naics_ids.append(NAICS_CODES_OF_INTEREST[group_id])  # optional if you want to retain NAICS group info

gdf_points = gpd.GeoDataFrame({'naics_group': naics_ids}, geometry=naics_points, crs="EPSG:4326")
gdf_polygons = gdf_in_ucmr.to_crs(epsg=4326)
joined = gpd.sjoin(gdf_points, gdf_polygons[['HUC8', 'geometry']], how="left", predicate="within")
huc_industry_count = joined['HUC8'].value_counts().to_dict()


# In[194]:


for index, row in gdf_in_ucmr.iterrows():
    huc8 = row['HUC8']
    count = joined[joined['HUC8'] == huc8].shape[0]
    gdf_in_ucmr.at[index, 'Industry_Count'] = count
gdf_in_ucmr.head()


# In[195]:

gdf_proj = gdf_in_ucmr.to_crs(epsg=5070)

# 2. Calculate density using vectorized series operations
# .area on a projected GeoSeries returns values in square meters (for EPSG:5070)
gdf_in_ucmr['density'] = gdf_proj['Industry_Count'] / gdf_proj['geometry'].area

# 3. Create a set of HUC8s with detections for O(1) lookup speed
detected_huc8s = set(df_of_huc8_with_detections['HUC8'].unique())

# 4. Vectorized status assignment using .isin()
gdf_in_ucmr['violation_status'] = gdf_in_ucmr['HUC8'].isin(detected_huc8s).astype(int)

# --- Explanatory Summary for the Paper ---
print("-" * 30)
print("GEOSPATIAL FEATURE ENGINEERING SUMMARY")
print(f"Industrial Density: Calculated for {len(gdf_in_ucmr)} units based on Albers Equal Area projection (EPSG:5070).")
print(f"Violation Status: {gdf_in_ucmr['violation_status'].sum()} watersheds labeled as 'Detected' (1), "
      f"{len(gdf_in_ucmr) - gdf_in_ucmr['violation_status'].sum()} labeled as 'Non-Detect' (0).")
print("-" * 30)


# # MODEL B - Training a 10-fold cross validated Logistic Regression Model on Generic Industrial Density

# In[197]:


X = gdf_in_ucmr[['density']].values
y = gdf_in_ucmr['violation_status'].values
kf = KFold(n_splits=10, shuffle=True)
mean_fpr = np.linspace(0, 1, 100)
scaler = StandardScaler()

roc_auc_list, accuracy_list, fpr_list, tpr_list, tprs   = [], [], [], [], []

model = RandomForestClassifier()
# Perform K-fold cross-validation
for train_index, test_index in kf.split(X):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    roc_auc_list.append(roc_auc)
    accuracy_list.append(accuracy)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(roc_auc_list)

plt.figure(figsize=(8, 6))
for i, roc_auc in enumerate(roc_auc_list):
    plt.plot(fpr_list[i], tpr_list[i], '--y', alpha=0.3)

plt.plot(mean_fpr, mean_tpr, color='b', label=f"Mean ROC (AUC = {mean_auc:.2f}+/- {std_auc:.2f})", lw=2)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()

print("Mean accuracy: ", np.mean(accuracy_list),'+/-', np.std(accuracy_list))


# ##### Tuning the threshold

# In[198]:


thresholds_for_tuning = np.arange(0.0, 1.01, 0.01)
accuracy_store, f1_score_store, precision_store, recall_store, sensitivity_store, specificity_store = [], [], [], [], [], []
for item in thresholds_for_tuning:
    y_pred_thresholded = (y_pred_proba >= item).astype(int)
    accuracy = accuracy_score(y_test, y_pred_thresholded)
    f1_score = (2 * np.sum(y_test * y_pred_thresholded)) / (np.sum(y_test) + np.sum(y_pred_thresholded))
    precision = np.sum(y_test * y_pred_thresholded) / np.sum(y_pred_thresholded)
    recall = np.sum(y_test * y_pred_thresholded) / np.sum(y_test)
    sensitivity = recall
    specificity = np.sum((1 - y_test) * (1 - y_pred_thresholded)) / np.sum(1 - y_test)
    accuracy_store.append(accuracy)
    f1_score_store.append(f1_score)
    precision_store.append(precision)
    recall_store.append(recall)
    sensitivity_store.append(sensitivity)
    specificity_store.append(specificity)
plt.figure(figsize=(10, 6))
plt.plot(thresholds_for_tuning, accuracy_store, label='Accuracy', color='blue')
plt.plot(thresholds_for_tuning, f1_score_store, label='F1 Score', color='green')
plt.plot(thresholds_for_tuning, precision_store, label='Precision', color='orange')
plt.plot(thresholds_for_tuning, recall_store, label='Recall', color='red')
plt.plot(thresholds_for_tuning, sensitivity_store, label='Sensitivity', color='purple')
plt.plot(thresholds_for_tuning, specificity_store, label='Specificity', color='brown')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Threshold Tuning for Classification Metrics')
plt.legend()
plt.grid()
plt.show()

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