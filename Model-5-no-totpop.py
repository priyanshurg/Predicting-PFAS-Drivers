
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

prefix2 = f"{cwd}/source_data_files/Facilities-coordinates/"
airports = pickle.load(open(f"{prefix2}major_airport_coordinates.pkl", "rb"))
military_bases = pickle.load(open(f"{prefix2}military_coordinates.pkl", "rb"))
fire_stations = pickle.load(open(f"{prefix2}fire-training-coordinates.pkl", "rb"))

airports_lat_long = airports[['latitude', 'longitude']].dropna(axis=0, how='any').reset_index(drop=True)
military_bases_lat_lon = military_bases.rename(columns={'Y': 'latitude', 'X': 'longitude'})
fire_stations_lat_lon = fire_stations.rename(columns={'Y': 'latitude', 'X': 'longitude'})


# In[200]:


for index, row in airports_lat_long.iterrows():
    point = Point(row['longitude'], row['latitude'])
    airports_lat_long.at[index, 'matching_HUC'] = gdf_in_ucmr[gdf_in_ucmr.contains(point)]["HUC8"].values[0] if not gdf_in_ucmr[gdf_in_ucmr.contains(point)].empty else None

for index, row in military_bases_lat_lon.iterrows():
    point = Point(row['longitude'], row['latitude'])
    military_bases_lat_lon.at[index, 'matching_HUC'] = gdf_in_ucmr[gdf_in_ucmr.contains(point)]["HUC8"].values[0] if not gdf_in_ucmr[gdf_in_ucmr.contains(point)].empty else None

for index, row in fire_stations_lat_lon.iterrows():
    point = Point(row['longitude'], row['latitude'])
    fire_stations_lat_lon.at[index, 'matching_HUC'] = gdf_in_ucmr[gdf_in_ucmr.contains(point)]["HUC8"].values[0] if not gdf_in_ucmr[gdf_in_ucmr.contains(point)].empty else None

huc8_afff_users_df = gdf_in_ucmr[["HUC8",'violation_status']].drop_duplicates().reset_index(drop=True)
print("Shape of huc8_afff_users_df:", huc8_afff_users_df.shape)
for index, row in huc8_afff_users_df.iterrows():
    huc_id = row['HUC8']
    airports_count = airports_lat_long[airports_lat_long['matching_HUC'] == huc_id].shape[0]
    military_bases_count = military_bases_lat_lon[military_bases_lat_lon['matching_HUC'] == huc_id].shape[0]
    fire_stations_count = fire_stations_lat_lon[fire_stations_lat_lon['matching_HUC'] == huc_id].shape[0]
    huc8_afff_users_df.at[index, 'Airports_Count'] = airports_count
    huc8_afff_users_df.at[index, 'Military_Bases_Count'] = military_bases_count
    huc8_afff_users_df.at[index, 'Fire_Stations_Count'] = fire_stations_count
print("\nShape of huc8_afff_users_df after adding AFFF users:", huc8_afff_users_df.shape)
print("Column Names in huc8_afff_users_df:", huc8_afff_users_df.columns)

# Step 1: Group and count once
counts = (
    industry_joined[industry_joined['naics_code'].isin(naics_codes)]
    .groupby(['HUC8', 'naics_code'])
    .size()
    .reset_index(name='count')
)

# Step 2: Pivot so each naics_code is a column
pivoted = counts.pivot(index='HUC8', columns='naics_code', values='count').fillna(0).astype(int)

# Step 3: Merge with gdf_in_ucmr on HUC8
gdf_in_ucmr_model_C = gdf_in_ucmr.merge(pivoted, how='left', on='HUC8')

# Step 4: Fill missing counts with 0
gdf_in_ucmr_model_C[top_naics] = gdf_in_ucmr_model_C[top_naics].fillna(0).astype(int)

# # Model D - 10-fold cross validated Random Forest Model of Industry Data + AFFF Foam users

# In[206]:


gdf_in_ucmr_model_D = gdf_in_ucmr_model_C.copy()
for index, row in gdf_in_ucmr_model_D.iterrows():
    huc8 = row['HUC8']
    airport_count = huc8_afff_users_df[huc8_afff_users_df.HUC8 == huc8]['Airports_Count'].values[0]
    military_base_count = huc8_afff_users_df[huc8_afff_users_df.HUC8 == huc8]['Military_Bases_Count'].values[0]
    fire_station_count = huc8_afff_users_df[huc8_afff_users_df.HUC8 == huc8]['Fire_Stations_Count'].values[0]
    area = row['geometry'].area
    airport_density = airport_count / area if area != 0 else 0
    military_base_density = military_base_count / area if area != 0 else 0
    fire_station_density = fire_station_count / area if area != 0 else 0
    gdf_in_ucmr_model_D.at[index, 'Airport_Density'] = airport_density
    gdf_in_ucmr_model_D.at[index, 'Military_Base_Density'] = military_base_density
    gdf_in_ucmr_model_D.at[index, 'Fire_Station_Density'] = fire_station_density
print("\nShape of gdf_in_ucmr_model_D after adding AFFF users density:", gdf_in_ucmr_model_D.shape)
print("Column Names in gdf_in_ucmr_model_D:", gdf_in_ucmr_model_D.columns)

# # Model E- Training 10-fold cross validated Random Forest Model on Industry + AFFF + Sociodemographic Data

# ##### Loading the socio demographic data

# In[209]:


from shapely import wkt
zcta_data_with_geolocations = pd.read_csv(f"{cwd}/source_data_files/generated-data-files/zcta_data_with_geolocations.csv")
zcta_data_with_geolocations = zcta_data_with_geolocations.drop(columns=["latitude","longitude"])
zcta_data_with_geolocations['geometry'] = zcta_data_with_geolocations['geometry'].apply(wkt.loads)


# In[210]:


# Ensure both GeoDataFrames are in the same CRS
zcta_data_with_geolocations = gpd.GeoDataFrame(zcta_data_with_geolocations, geometry='geometry', crs="EPSG:4326")
# zcta_data_with_geolocations = zcta_data_with_geolocations.to_crs(gdf_in_ucmr.crs)
# Spatial join: assign HUC8 from polygons to points
zcta_with_huc8 = gpd.sjoin(
    zcta_data_with_geolocations,
    gdf_in_ucmr[['HUC8', 'geometry']],
    how='left',
    predicate='within'  # or 'contains', but 'within' is better when matching point-in-polygon
)
# Drop the spatial index column if you don't need it
zcta_with_huc8 = zcta_with_huc8.drop(columns='index_right')

print("Shape of zcta_with_huc8:", zcta_with_huc8.shape)
print("Unique HUC8s in zcta_with_huc8:", zcta_with_huc8['HUC8'].nunique())
print("Column Names in zcta_with_huc8:", zcta_with_huc8.columns)


# In[211]:


rpl_themes_by_huc8 = zcta_with_huc8[['HUC8','RPL_THEMES']].groupby('HUC8').mean().reset_index()
rpl_themes_by_huc8["HUC8"] = rpl_themes_by_huc8["HUC8"].astype(str)
rpl_themes_by_huc8.index = rpl_themes_by_huc8.index.astype(str)


# In[212]:


gdf_in_ucmr_model_E = gdf_in_ucmr_model_D.copy()
gdf_in_ucmr_model_E = gpd.GeoDataFrame(gdf_in_ucmr_model_E, geometry='geometry', crs="EPSG:4326")
for index, row in gdf_in_ucmr_model_E.iterrows():
    huc8 = row['HUC8']
    if huc8 in rpl_themes_by_huc8['HUC8'].values:
        rpl_themes = rpl_themes_by_huc8[rpl_themes_by_huc8['HUC8'] == huc8]['RPL_THEMES'].values[0]
        gdf_in_ucmr_model_E.at[index, 'RPL_THEMES'] = rpl_themes
    else:
        gdf_in_ucmr_model_E.at[index, 'RPL_THEMES'] = np.nan
gdf_in_ucmr_model_E.rename(columns={'RPL_THEMES': 'Mean_SVI'}, inplace=True)
print("\nShape of gdf_in_ucmr_model_E after adding RPL_THEMES:", gdf_in_ucmr_model_E.shape)
print("Column Names in gdf_in_ucmr_model_E:", gdf_in_ucmr_model_E.columns)



# In[213]:


# Data Source: https://nanda.isr.umich.edu/project/socioeconomic-status-and-demographic-characteristics/
nanda_data_filename = f"{cwd}/source_data_files/generated-data-files/nanda_data_with_geolocations.csv"
nanda_data = pd.read_csv(nanda_data_filename)
nanda_data['geometry'] = nanda_data['geometry'].apply(wkt.loads)
nanda_data = gpd.GeoDataFrame(nanda_data, geometry='geometry', crs="EPSG:4326")
gdf_in_ucmr_model_E = gdf_in_ucmr_model_E.to_crs(epsg=4326)  # Ensure both GeoDataFrames are in the same CRS

# Spatial join nanda_data and gdf_in_ucmr_model_E
nanda_with_huc8 = gpd.sjoin(
    gdf_in_ucmr_model_E[['HUC8', 'geometry']],
    nanda_data,
    how='left',
    predicate='contains',  # or 'contains', but 'within' is better when matching point-in-polygon

)
nanda_with_huc8.drop(columns=['index_right','geometry','ZCTA'], inplace=True)
cols_to_fix = ["AFFLUENCE", "DISADVANTAGE"]

for col in cols_to_fix:
    # errors='coerce' automatically turns ' ' or other non-numeric strings into np.nan
    nanda_with_huc8[col] = pd.to_numeric(nanda_with_huc8[col], errors='coerce')
for col in cols_to_fix:
    nan_count = nanda_with_huc8[col].isna().sum()

nanda_with_huc8["AFFLUENCE"] = nanda_with_huc8.AFFLUENCE.astype(float)
nanda_with_huc8["DISADVANTAGE"] = nanda_with_huc8.DISADVANTAGE.astype(float)
nanda_pop_data = nanda_with_huc8[["HUC8","TOTPOP"]].groupby("HUC8").sum().reset_index()
nanda_demograph_data = nanda_with_huc8[["HUC8","AFFLUENCE", "DISADVANTAGE"]].groupby("HUC8").mean().reset_index()
gdf_in_ucmr_model_E = gdf_in_ucmr_model_E.merge(nanda_pop_data, on="HUC8", how="left", suffixes=('', '_pop'))
gdf_in_ucmr_model_E = gdf_in_ucmr_model_E.merge(nanda_demograph_data, on="HUC8", how="left", suffixes=('', '_demograph'))


# #### Training the model

# In[225]:

naics_names = pd.read_excel(f"{cwd}/source_data_files/2017-NAICS-2-3-4-5-6-Digit-Codes-Listed-Numerically.xlsx", sheet_name="6 Digit NAICS", header=0)
naics_names = naics_names[['Six Digit NAICS Codes', '2017 NAICS Title (USA)']].dropna().reset_index(drop=True)
naics_names.columns = ['NAICS_Code', 'NAICS_Name']
naics_names_filtered = naics_names[naics_names['NAICS_Code'].isin(naics_codes)].reset_index(drop=True)
# Create a dictionary for mapping: {313320: 'Fabric Coating Mills', ...}
naics_map = dict(zip(naics_names_filtered['NAICS_Code'], naics_names_filtered['NAICS_Name']))

# Function to truncate long titles for better plotting
def clean_label(label):
    # Truncate to 50 characters and add ellipsis if needed
    return (label[:47] + '..') if len(str(label)) > 50 else label

# Apply truncation to the values in our map
naics_map = {code: clean_label(name) for code, name in naics_map.items()}

# gdf_in_ucmr_model_E.dropna(inplace=True)
X_with_labels = gdf_in_ucmr_model_E.drop(columns=['TOTPOP', 'HUC8', 'geometry','Industry_Count','NAME','density', 'violation_status', 'Mean_SVI']) 
X = X_with_labels.values
y = gdf_in_ucmr_model_E['violation_status'].values

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pickle
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_score, recall_score, f1_score

class ModelEvaluationUI:
    def __init__(self, root, X, y, feature_names, naics_map):
        self.root = root
        self.root.title("PFAS Violation Model Explorer")
        
        # Data
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.naics_map = naics_map # Mapping dict: {code: "Name"}
        
        # State Variables
        self.last_y_test = None
        self.last_y_proba = None
        self.current_model = None
        self.current_scaler = StandardScaler()
        self.last_X_scaled = None
        
        # --- UI Setup ---
        control_frame = ttk.Frame(root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(control_frame, text="Algorithm:").pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar(value="RandomForest")
        self.model_selector = ttk.Combobox(control_frame, textvariable=self.model_var)
        self.model_selector['values'] = ("RandomForest", "LogisticRegression", "MLP")
        self.model_selector.pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="🚀 Train & Evaluate", command=self.run_pipeline).pack(side=tk.LEFT, padx=10)

        action_frame = ttk.Frame(root, padding="5")
        action_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(action_frame, text="📊 Threshold Metrics", command=self.open_threshold_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="🧬 SHAP Summary", command=self.open_shap_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="📈 Mean SHAP Importance", command=self.open_mean_shap_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="💾 Save", command=self.save_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="📂 Load", command=self.load_model).pack(side=tk.LEFT, padx=5)

        ttk.Label(action_frame, text="Cut-off:").pack(side=tk.LEFT, padx=(20, 5))
        self.threshold_slider = tk.Scale(action_frame, from_=0.0, to=1.0, resolution=0.01, 
                                         orient=tk.HORIZONTAL, command=self.update_live_metrics)
        self.threshold_slider.set(0.5)
        self.threshold_slider.pack(side=tk.LEFT, padx=5)

        self.stats_label = ttk.Label(root, text="Model Status: Waiting", font=('Helvetica', 10, 'bold'))
        self.stats_label.pack(side=tk.TOP, pady=5)

        self.fig, (self.ax_roc, self.ax_metrics) = plt.subplots(1, 2, figsize=(12, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def run_pipeline(self):
        m_type = self.model_var.get()
        if m_type == "RandomForest": self.current_model = RandomForestClassifier(n_estimators=100)
        elif m_type == "LogisticRegression": self.current_model = LogisticRegression()
        else: self.current_model = MLPClassifier(max_iter=500)

        kf = KFold(n_splits=5, shuffle=True)
        tprs = []
        mean_fpr = np.linspace(0, 1, 100)
        self.ax_roc.clear()

        for train_idx, test_idx in kf.split(self.X):
            X_tr, X_te = self.X[train_idx], self.X[test_idx]
            y_tr, y_te = self.y[train_idx], self.y[test_idx]
            
            X_tr_scaled = self.current_scaler.fit_transform(X_tr)
            X_te_scaled = self.current_scaler.transform(X_te)
            
            self.current_model.fit(X_tr_scaled, y_tr)
            y_proba = self.current_model.predict_proba(X_te_scaled)[:, 1]
            
            fpr, tpr, _ = roc_curve(y_te, y_proba)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            self.ax_roc.plot(fpr, tpr, alpha=0.2, lw=1)
            
            self.last_y_test, self.last_y_proba, self.last_X_scaled = y_te, y_proba, X_te_scaled

        mean_tpr = np.mean(tprs, axis=0)
        self.ax_roc.plot(mean_fpr, mean_tpr, color='b', lw=2, label=f'Mean ROC (AUC={auc(mean_fpr, mean_tpr):.2f})')
        self.ax_roc.plot([0,1],[0,1], 'r--')
        self.ax_roc.set_title(f"ROC Curve: {m_type}")
        self.ax_roc.legend()
        self.update_live_metrics()

    def update_live_metrics(self, event=None):
        if self.last_y_test is None: return
        t = self.threshold_slider.get()
        y_p = (self.last_y_proba >= t).astype(int)
        
        acc = accuracy_score(self.last_y_test, y_p)
        pre = precision_score(self.last_y_test, y_p, zero_division=0)
        rec = recall_score(self.last_y_test, y_p, zero_division=0)
        f1  = f1_score(self.last_y_test, y_p, zero_division=0)
        
        self.stats_label.config(text=f"Threshold {t:.2f} | Acc: {acc:.2f} | Pre: {pre:.2f} | Rec: {rec:.2f} | F1: {f1:.2f}")
        
        self.ax_metrics.clear()
        self.ax_metrics.bar(['Accuracy', 'Precision', 'Recall', 'F1'], [acc, pre, rec, f1], color=['#2E4053', '#A04000', '#1D8348', '#922B21'])
        self.ax_metrics.set_ylim(0, 1.1)
        self.canvas.draw()

    def open_threshold_analysis(self):
        if self.last_y_test is None: return
        pop = tk.Toplevel(self.root)
        pop.title("Full Threshold Tuning")
        thresholds = np.arange(0.0, 1.01, 0.01)
        accs, f1s, pres, sens, specs, youden = [], [], [], [], [], []

        for t in thresholds:
            y_p = (self.last_y_proba >= t).astype(int)
            rec = recall_score(self.last_y_test, y_p, zero_division=0)
            spec = np.sum((y_p == 0) & (self.last_y_test == 0)) / np.sum(self.last_y_test == 0)
            accs.append(accuracy_score(self.last_y_test, y_p))
            f1s.append(f1_score(self.last_y_test, y_p, zero_division=0))
            pres.append(precision_score(self.last_y_test, y_p, zero_division=0))
            sens.append(rec); specs.append(spec); youden.append(rec + spec - 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(thresholds, accs, label='Accuracy'); ax.plot(thresholds, f1s, label='F1')
        ax.plot(thresholds, pres, label='Precision'); ax.plot(thresholds, specs, label='Specificity')
        ax.plot(thresholds, sens, label='Sensitivity'); ax.plot(thresholds, youden, label="Youden J", lw=2, ls='--')
        ax.legend(loc='lower left', fontsize='x-small'); ax.grid(alpha=0.3)
        canvas = FigureCanvasTkAgg(fig, master=pop); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True); canvas.draw()

    def open_shap_analysis(self):
        if self.current_model is None: return
        pop = tk.Toplevel(self.root)
        pop.title("SHAP Feature Summary")
        load_lbl = ttk.Label(pop, text="Calculating SHAP values... please wait.", padding=20); load_lbl.pack()
        self.root.update()

        try:
            X_df = pd.DataFrame(self.last_X_scaled, columns=self.feature_names)
            X_df.rename(columns=self.naics_map, inplace=True)
            
            if isinstance(self.current_model, RandomForestClassifier):
                explainer = shap.TreeExplainer(self.current_model)
                vals = explainer.shap_values(X_df)
                display_vals = vals[1] if isinstance(vals, list) else (vals[:,:,1] if len(vals.shape) > 2 else vals)
            else:
                explainer = shap.Explainer(self.current_model, X_df)
                display_vals = explainer(X_df)

            load_lbl.destroy()
            fig = plt.figure(figsize=(12, 10))
            shap.summary_plot(display_vals, X_df, show=False)
            plt.title("Feature Impact on Detection Probability")
            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=pop); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True); canvas.draw()
        except Exception as e:
            messagebox.showerror("SHAP Error", f"Calculation failed: {e}")

    def open_mean_shap_analysis(self):
        if self.current_model is None: return
        pop = tk.Toplevel(self.root)
        pop.title("Mean Absolute SHAP")

        try:
            X_df = pd.DataFrame(self.last_X_scaled, columns=self.feature_names)
            if isinstance(self.current_model, RandomForestClassifier):
                vals = shap.TreeExplainer(self.current_model).shap_values(X_df)
                shap_matrix = vals[1] if isinstance(vals, list) else (vals[:,:,1] if len(vals.shape) > 2 else vals)
            else:
                shap_matrix = shap.Explainer(self.current_model, X_df)(X_df).values

            mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
            importance_df = pd.DataFrame({'Raw': self.feature_names, 'Importance': mean_abs_shap})
            importance_df['Display'] = importance_df['Raw'].map(self.naics_map).fillna(importance_df['Raw'])
            
            # Filter Top 10 + TOTPOP
            top_df = importance_df.sort_values(by='Importance', ascending=False).head(10).copy()
            if 'TOTPOP' in importance_df['Raw'].values and 'TOTPOP' not in top_df['Raw'].values:
                top_df = pd.concat([top_df, importance_df[importance_df['Raw'] == 'TOTPOP']])
            
            top_df = top_df.sort_values(by='Importance', ascending=True)

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(top_df['Display'], top_df['Importance'], color='skyblue', edgecolor='navy')
            
            # FIXED: Removed 'padx' which caused your error
            for bar in bars:
                width = bar.get_width()
                ax.text(width + (max(top_df['Importance'])*0.01), bar.get_y() + bar.get_height()/2, 
                        f'{width:.4f}', va='center', ha='left', fontsize=9)

            ax.set_xlabel('Mean Absolute SHAP Value')
            ax.set_title('Top Drivers of PFAS Detection')
            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=pop); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True); canvas.draw()
        except Exception as e:
            messagebox.showerror("SHAP Error", f"Mean calculation failed: {e}")

    def save_model(self):
        if self.current_model is None: return
        path = filedialog.asksaveasfilename(defaultextension=".pkl")
        if path:
            with open(path, 'wb') as f:
                pickle.dump({'model': self.current_model, 'scaler': self.current_scaler, 'threshold': self.threshold_slider.get(), 'features': self.feature_names}, f)
            messagebox.showinfo("Success", "Saved.")

    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
        if path:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.current_model, self.current_scaler = data['model'], data['scaler']
            self.threshold_slider.set(data['threshold'])
            self.feature_names = data['features']
            messagebox.showinfo("Success", "Loaded.")
    # Ensure X and y are NumPy arrays and feature_names is a list of strings
root = tk.Tk()
app = ModelEvaluationUI(root, X, y, X_with_labels.columns.tolist(),naics_map)
root.mainloop()

# %%
