import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LassoCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

# --- SAFE IMPORTS (Handle Library Conflicts) ---
# Try importing statsmodels (Handling scipy version conflict)
try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except (ImportError, AttributeError):
    STATSMODELS_AVAILABLE = False

# Try importing mlxtend
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Asian Economic Growth Models",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .success-box {
        padding: 15px;
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        border-radius: 5px;
    }
    .warning-box {
        padding: 15px;
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_main_data():
    try:
        # Loading the clustering result file which contains the main dataset
        df = pd.read_excel("data.xlsx")
        return df
    except FileNotFoundError:
        st.error("⚠️ File 'data.xlsx - Sheet1.csv' not found. Please place it in the same directory.")
        return None

@st.cache_data
def load_research_questions():
    try:
        df = pd.read_csv("cauhoinghiemcuu (1).xlsx - tạo ra bảng mới giúp tôi nhé.csv")
        return df
    except:
        return None

df = load_main_data()
questions_df = load_research_questions()

# --- SIDEBAR NAVIGATION ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2942/2942544.png", width=100)
st.sidebar.title("Project Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Overview", "Exploratory Data Analysis (EDA)", "Clustering Analysis (K-Means)", "Regression & Risk Analysis", "Association Rules Miner"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Project:** Data Mining & Growth Models of Asian Economies (2000-2020)\n\n"
    "**Tools:** Python, Scikit-Learn, Statsmodels, Streamlit, Mlxtend"
)

# --- PAGE 1: OVERVIEW ---
if page == "Overview":
    st.markdown('<p class="main-header">🌏 Asian Economic Growth Models (2000-2020)</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Introduction
    This project applies Data Mining techniques to analyze the economic development patterns of Asian countries over two decades. 
    By leveraging **Clustering, Association Rules, and Regression**, we aim to identify distinct growth models and the factors driving them.
    """)

    if questions_df is not None:
        st.markdown("### 📋 Research Framework")
        st.dataframe(questions_df, use_container_width=True)
    
    st.markdown("### 🏗️ Methodology Pipeline")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("#### 1. Data Collection")
        st.caption("World Bank Data (GDP, FDI, Trade, etc.)")
    with col2:
        st.markdown("#### 2. K-Means Clustering")
        st.caption("Identify economic states (Take-off vs Stagnation)")
    with col3:
        st.markdown("#### 3. Association Rules")
        st.caption("Find patterns (e.g., Debt vs. Growth)")
    with col4:
        st.markdown("#### 4. Regression & Risk")
        st.caption("Drivers & Efficiency Risk Labeling")

# --- PAGE 2: EDA ---
elif page == "Exploratory Data Analysis (EDA)":
    st.markdown('<p class="main-header">📊 Exploratory Data Analysis</p>', unsafe_allow_html=True)

    if df is not None:
        # Define numeric columns based on your code snippet
        user_numeric_cols = [
            'GDP_Growth', 'GDP_PC', 'Agr_Share', 'Ind_Share', 'Ser_Share',
            'Trade_Open', 'FDI', 'Capital_Form', 'Inflation', 'Labor_Force',
            'Unemployment', 'Urbanization', 'Internet', 'Mobile', 'Public_Debt', 
            'Current_Account', 'Electricity', 'Reserves'
        ]
        # Filter to ensure columns exist in the loaded dataframe
        valid_numeric_cols = [c for c in user_numeric_cols if c in df.columns]

        tabs = st.tabs(["Regional Overview", "Distributions (Hist/Box)", "Trends by Year", "Correlation", "Feature Selection"])

        # TAB 1: Regional Overview
        with tabs[0]:
            st.subheader("1. Regional Distribution")
            
            # User Task: Bar Chart of Countries per Sub-region
            if 'Tiểu khu vực' in df.columns:
                df_unique = df[['Quốc gia', 'Tiểu khu vực']].drop_duplicates()
                region_counts = df_unique['Tiểu khu vực'].value_counts().reset_index()
                region_counts.columns = ['Tiểu khu vực', 'So_Luong']
                
                fig_reg_count = plt.figure(figsize=(10, 6))
                ax = sns.barplot(data=region_counts, x='Tiểu khu vực', y='So_Luong', palette='viridis')
                
                # Annotate bars
                for p in ax.patches:
                    ax.annotate(f'{int(p.get_height())}', 
                                (p.get_x() + p.get_width() / 2., p.get_height()), 
                                ha='center', va='center', xytext=(0, 8), 
                                textcoords='offset points', fontsize=12, fontweight='bold', color='black')
                
                plt.title('Số lượng Quốc gia trong từng Tiểu khu vực', fontsize=15, fontweight='bold')
                plt.xlabel('Tiểu khu vực', fontsize=12)
                plt.ylabel('Số lượng Quốc gia', fontsize=12)
                plt.ylim(0, 15)
                plt.grid(axis='y', linestyle='--', alpha=0.5)
                st.pyplot(fig_reg_count)
            
            st.write("---")
            st.subheader("2. Regional Economic Deep Dive")
            st.markdown("Select a region to see the detailed line charts for all indicators (as per your function `ve_bieu_do_khu_vuc`).")
            
            selected_region = st.selectbox("Select Region:", df['Tiểu khu vực'].unique())
            
            if selected_region:
                # Adapted logic from your function
                data_subset = df[df['Tiểu khu vực'] == selected_region]
                ds_quoc_gia = sorted(data_subset['Quốc gia'].unique())
                
                if not data_subset.empty:
                    palette_raw = sns.color_palette("tab20", len(ds_quoc_gia))
                    map_mau = dict(zip(ds_quoc_gia, palette_raw))
                    
                    n_cols = 3
                    n_rows = math.ceil(len(valid_numeric_cols) / n_cols)
                    
                    fig_region, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
                    axes = axes.flatten()
                    
                    st.write(f"**Visualizing: {selected_region}**")
                    
                    for i, col in enumerate(valid_numeric_cols):
                        if i < len(axes):
                            sns.lineplot(
                                data=data_subset, x='Năm', y=col, hue='Quốc gia',
                                hue_order=ds_quoc_gia, palette=palette_raw, marker='o',
                                markersize=4, linewidth=2, legend=False, ax=axes[i]
                            )
                            axes[i].set_title(col, fontsize=10, fontweight='bold', color='#003366')
                            axes[i].grid(True, linestyle='--', alpha=0.5)
                            
                    for i in range(len(valid_numeric_cols), len(axes)):
                        fig_region.delaxes(axes[i])
                        
                    # Custom Legend
                    custom_lines = [Line2D([0], [0], color=map_mau[nuoc], lw=4) for nuoc in ds_quoc_gia]
                    fig_region.legend(custom_lines, ds_quoc_gia, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=min(len(ds_quoc_gia), 5))
                    plt.tight_layout()
                    st.pyplot(fig_region)

        # TAB 2: Distributions (Histograms & Boxplots)
        with tabs[1]:
            st.subheader("1. Histograms (Distributions)")
            # Replicating your Histogram Grid Logic
            n_cols = 3
            n_rows = int(np.ceil(len(valid_numeric_cols) / n_cols))
            fig_hist, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            axes = axes.flatten()

            for i, col in enumerate(valid_numeric_cols):
                sns.histplot(data=df, x=col, kde=True, ax=axes[i], color='skyblue', edgecolor='black')
                axes[i].set_title(f'Phân phối: {col}', fontsize=10, fontweight='bold')
                axes[i].set_xlabel('')
                axes[i].set_ylabel('Tần suất')

            for i in range(len(valid_numeric_cols), len(axes)):
                fig_hist.delaxes(axes[i])

            plt.tight_layout()
            st.pyplot(fig_hist)
            
            st.write("---")
            st.subheader("2. Boxplots (Outliers Detection)")
            # Replicating your Boxplot Grid Logic
            cols_per_row = 4
            rows = math.ceil(len(valid_numeric_cols) / cols_per_row)
            fig_box = plt.figure(figsize=(20, 5 * rows))

            for i, col in enumerate(valid_numeric_cols):
                plt.subplot(rows, cols_per_row, i + 1)
                sns.boxplot(y=df[col], color='skyblue')
                plt.title(f'Phân phối của {col}', fontsize=12)
                plt.ylabel('')
                plt.grid(True, linestyle='--', alpha=0.6)

            plt.tight_layout()
            st.pyplot(fig_box)

        # TAB 3: Trends by Year (Boxplots)
        with tabs[2]:
            st.subheader("Distribution Trends by Year")
            # Replicating Boxplots by Year Logic
            cols_per_row = 2
            rows = math.ceil(len(valid_numeric_cols) / cols_per_row)
            
            fig_year = plt.figure(figsize=(20, 6 * rows))

            for i, col in enumerate(valid_numeric_cols):
                plt.subplot(rows, cols_per_row, i + 1)
                sns.boxplot(data=df, x='Năm', y=col, palette="viridis")
                plt.title(f'Phân phối {col} theo Năm', fontsize=14)
                plt.xticks(rotation=45)
                plt.xlabel('')
                plt.grid(True, axis='y', linestyle='--', alpha=0.5)

            plt.tight_layout()
            st.pyplot(fig_year)

        # TAB 4: Correlation
        with tabs[3]:
            st.subheader("Correlation Matrix")
            # Replicating Heatmap Logic
            numeric_df = df.select_dtypes(include=['float64', 'int64'])
            # Filter for specific columns if desired or use all numeric
            correlation_matrix = numeric_df[valid_numeric_cols].corr()

            fig_corr = plt.figure(figsize=(16, 12))
            sns.heatmap(
                correlation_matrix,
                annot=True,
                cmap='RdBu_r',
                fmt=".2f",
                vmin=-1, vmax=1,
                linewidths=.5
            )
            plt.title('Ma trận Tương quan các biến Kinh tế', fontsize=18)
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig_corr)

        # TAB 5: Feature Selection
        with tabs[4]:
            st.subheader("Feature Selection: Analyzing Excluded Variables")
            st.markdown("""
            **Task:** Analyze variables that were excluded due to saturation (Electricity), high correlation (Mobile), or scale bias.
            """)
            
            # Replicating the logic from the EDA snippet provided
            excluded_features = ['Mobile', 'Electricity', 'Labor_Force', 'Reserves', 'Current_Account']
            available_excluded = [col for col in excluded_features if col in df.columns]
            included_features = ['Internet', 'GDP_PC', 'Urbanization', 'Trade_Open']
            
            if available_excluded:
                st.markdown("#### 1. Correlation of Excluded Features")
                corr_cols = available_excluded + included_features
                # Ensure columns exist before correlation
                valid_corr_cols = [c for c in corr_cols if c in df.columns]
                corr_matrix_ex = df[valid_corr_cols].corr()
                
                fig_corr_ex = px.imshow(
                    corr_matrix_ex,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title='Correlation Matrix: Excluded vs Included Features'
                )
                st.plotly_chart(fig_corr_ex, use_container_width=True)
                
                st.markdown("#### 2. Distributions (Why we removed them)")
                col_d1, col_d2 = st.columns(2)
                
                if 'Electricity' in df.columns:
                    with col_d1:
                        fig_elec = px.histogram(df, x='Electricity', nbins=20, title="Distribution of Electricity Access (%)")
                        st.plotly_chart(fig_elec, use_container_width=True)
                        st.caption("**Observation:** Data is skewed/saturated (mostly 100%). Low variance makes it poor for clustering.")
                        
                if 'Labor_Force' in df.columns:
                    with col_d2:
                        fig_labor = px.histogram(df, x='Labor_Force', nbins=20, log_y=True, title="Labor Force Distribution (Log Scale)")
                        st.plotly_chart(fig_labor, use_container_width=True)

# --- PAGE 3: CLUSTERING ---
elif page == "Clustering Analysis (K-Means)":
    st.markdown('<p class="main-header">🧩 Clustering Analysis (K-Means)</p>', unsafe_allow_html=True)
    
    if df is not None:
        st.markdown("""
        **Features used for Clustering:**
        `GDP_Growth`, `GDP_PC`, `Agr_Share`, `Ind_Share`, `Ser_Share`, `Trade_Open`, `FDI`, `Capital_Form`, `Inflation`, `Unemployment`, `Urbanization`, `Internet`, `Public_Debt`, `Labor_Force`
        """)
        
        # Prepare Data for Live Calculation
        features_clustering =['GDP_Growth', 'GDP_PC', 'Agr_Share', 'Ind_Share', 'Ser_Share',
                            'Trade_Open', 'FDI', 'Capital_Form', 'Inflation', 'Unemployment',
                            'Urbanization', 'Internet', 'Public_Debt','Labor_Force']
        
        # Filter and Dropna (as per snippet)
        X = df[features_clustering].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        tabs = st.tabs(["1. Optimal K (Elbow)", "2. PCA Visualization", "3. Snake Plot (Profiles)", "4. Heatmap (Timeline)"])

        with tabs[0]:
            st.subheader("Step 1: Determine Optimal Clusters (Elbow & Silhouette)")
            
            if st.button("Run Elbow Analysis"):
                inertia = []
                silhouette_scores = []
                K_range = range(2, 11)

                with st.spinner("Running K-Means for K=2 to 10..."):
                    for k in K_range:
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        kmeans.fit(X_scaled)
                        inertia.append(kmeans.inertia_)
                        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
                
                # Plotting Dual Axis (Inertia vs Silhouette)
                fig_dual = go.Figure()
                
                # Trace 1: Inertia (Elbow)
                fig_dual.add_trace(go.Scatter(
                    x=list(K_range), y=inertia, name="Inertia (Elbow)",
                    mode='lines+markers', line=dict(color='blue')
                ))
                
                # Trace 2: Silhouette
                fig_dual.add_trace(go.Scatter(
                    x=list(K_range), y=silhouette_scores, name="Silhouette Score",
                    mode='lines+markers', line=dict(color='red', dash='dash'),
                    yaxis='y2'
                ))
                
                fig_dual.update_layout(
                    title="Evaluation of Optimal K: Elbow Method & Silhouette Score",
                    xaxis_title="Number of Clusters (K)",
                    yaxis=dict(title="Inertia", titlefont=dict(color="blue")),
                    yaxis2=dict(title="Silhouette Score", titlefont=dict(color="red"), overlaying='y', side='right'),
                    legend=dict(x=0.1, y=1.1, orientation='h')
                )
                st.plotly_chart(fig_dual, use_container_width=True)
                
                best_k = list(K_range)[np.argmax(silhouette_scores)]
                st.success(f"**Suggestion:** The optimal number of clusters based on Silhouette Score is **K={best_k}**.")

        with tabs[1]:
            st.subheader("Step 2: PCA Visualization (K=4)")
            st.markdown("Projecting the high-dimensional data into 2D using PCA to visualize cluster separation.")
            
            # Run PCA
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(X_scaled)
            pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
            
            # Use existing clusters from file if available, otherwise recalculate K=4
            if 'Cluster_Name' in df.columns:
                # Ensure alignment of indices since we dropped NA
                pca_df['Cluster'] = df.loc[X.index, 'Cluster_Name'].values
            else:
                kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)
                pca_df['Cluster'] = clusters.astype(str)

            fig_pca = px.scatter(
                pca_df, x='PC1', y='PC2', color='Cluster',
                title='Cluster Distribution (PCA Projection)',
                template='plotly_white',
                opacity=0.8
            )
            st.plotly_chart(fig_pca, use_container_width=True)

        with tabs[2]:
            st.subheader("Step 3: Snake Plot (Cluster Characterization)")
            st.markdown("Comparing the standardized mean values (Z-Scores) of each feature across clusters.")
            
            # Using the clusters from the file/dataframe
            # We need to compute the means on the SCALED data to make them comparable
            
            # 1. Create a temporary df with scaled values and cluster labels
            X_scaled_df = pd.DataFrame(X_scaled, columns=features_clustering, index=X.index)
            X_scaled_df['Cluster'] = df.loc[X.index, 'Cluster_Name']
            
            # 2. Group by Cluster and take mean
            cluster_means = X_scaled_df.groupby('Cluster').mean()
            
            # 3. Melt for plotting
            cluster_melt = cluster_means.reset_index().melt(id_vars='Cluster', var_name='Feature', value_name='Z-Score')
            
            fig_snake = px.line(
                cluster_melt, 
                x='Feature', 
                y='Z-Score', 
                color='Cluster', 
                markers=True,
                title="Snake Plot: Standardized Characteristics of Each Cluster"
            )
            fig_snake.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
            st.plotly_chart(fig_snake, use_container_width=True)
            
            st.info("""
            **Interpretation:**
            * **Z-Score > 0:** The cluster is *above average* for this feature.
            * **Z-Score < 0:** The cluster is *below average* for this feature.
            """)

        with tabs[3]:
            st.subheader("Step 4: Evolution Heatmap")
            st.markdown("Tracking how countries change economic states over time.")
            
            # Pivot table: Index=Country Code, Columns=Year, Values=Cluster Name
            # Since Cluster Names are strings, we map them to numbers for color scaling, or use categorical heatmap
            
            # Prepare data
            heatmap_data = df.pivot(index='Code', columns='Năm', values='Cluster_Name')
            
            # For visualization purposes, let's map names to integers if needed, 
            # but Plotly imshow handles categories if we provide custom colors. 
            # However, mapping to Cluster_ID (if available) is safer for coloring.
            
            if 'Cluster_ID' in df.columns:
                heatmap_val = df.pivot(index='Code', columns='Năm', values='Cluster_ID')
                fig_heat = px.imshow(
                    heatmap_val,
                    color_continuous_scale='Viridis',
                    aspect="auto",
                    title="Heatmap: Economic State Evolution (By Cluster ID)"
                )
                st.plotly_chart(fig_heat, use_container_width=True)
                
                # Legend helper
                st.caption("Reference the Cluster Names in the 'Profiles' tab to interpret IDs.")

# --- PAGE 4: REGRESSION ---
elif page == "Regression & Risk Analysis":
    st.markdown('<p class="main-header">📈 Regression & Risk Analysis</p>', unsafe_allow_html=True)
    
    st.markdown("""
    This section implements the **Lasso Feature Selection**, **OLS Regression** by cluster, and **Efficiency Risk Analysis**.
    """)

    if df is not None:
        if not STATSMODELS_AVAILABLE:
            st.error("⚠️ Library `statsmodels` failed to load due to a system environment conflict (scipy vs statsmodels). Please run `pip install --upgrade statsmodels scipy` in your terminal to fix this. In the meantime, this analysis page is disabled to prevent crashing.")
        else:
            tabs = st.tabs(["1. Feature Selection (Lasso)", "2. Growth Drivers & Efficiency", "3. Detailed Results"])
            
            # --- TAB 1: LASSO ---
            with tabs[0]:
                st.subheader("Lasso Regression: Variable Selection")
                st.markdown("We use Lasso Regression to automatically select the most important variables and eliminate irrelevant ones.")
                
                # 2. PREPARE "FULL" CANDIDATE LIST (from snippet)
                candidates = [
                    'Capital_Form', 'Trade_Open', 'FDI', 'Public_Debt', 'Inflation',
                    'Labor_Force', 'Ind_Share', 'Agr_Share', 'Ser_Share', 
                    'Unemployment', 'Urbanization', 'Internet', 'Mobile', 'Electricity', 
                    'Reserves', 'Current_Account'
                ]
                
                # Clean data for Lasso
                # Ensure candidates exist in df
                valid_candidates = [c for c in candidates if c in df.columns]
                data_lasso = df[valid_candidates + ['GDP_Growth']].dropna()

                X_lasso = data_lasso[valid_candidates]
                y_lasso = data_lasso['GDP_Growth']

                # 3. STANDARDIZE
                scaler = StandardScaler()
                X_scaled_lasso = scaler.fit_transform(X_lasso)

                # 4. RUN LASSO CV
                lasso = LassoCV(cv=5, random_state=42).fit(X_scaled_lasso, y_lasso)

                # 5. EXTRACT & SORT
                coef_df = pd.DataFrame({
                    'Feature': valid_candidates,
                    'Importance': lasso.coef_
                })
                coef_df['Abs_Importance'] = coef_df['Importance'].abs()
                coef_df = coef_df.sort_values(by='Abs_Importance', ascending=False)

                # 6. VISUALIZE
                fig_lasso = plt.figure(figsize=(12, 6))
                colors = ['red' if x < 0 else 'green' for x in coef_df['Importance']]
                sns.barplot(data=coef_df, x='Importance', y='Feature', palette=colors)
                plt.title('LASSO REGRESSION: Which variables matter?', fontsize=14, fontweight='bold')
                plt.xlabel('Standardized Coefficient')
                plt.axvline(0, color='black', linestyle='--')
                plt.tight_layout()
                st.pyplot(fig_lasso)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Top 5 Selected:**")
                    st.dataframe(coef_df.head(5)[['Feature', 'Importance']])
                with col2:
                    st.write("**Rejected (Coeff ~ 0):**")
                    st.dataframe(coef_df[coef_df['Abs_Importance'] < 0.05][['Feature', 'Importance']])

            # --- TAB 2: COMPREHENSIVE ANALYSIS (4 CHARTS) ---
            with tabs[1]:
                st.subheader("Comprehensive Regression Dashboard")
                st.markdown("Visualizing the Genomic Heatmap of Growth Drivers, Efficiency Frontier, and Key Factor Distributions.")
                
                # Define Predictors (Lasso Selected / Hardcoded from snippet)
                target_col = 'GDP_Growth'
                predictors = ['Capital_Form', 'Trade_Open', 'FDI', 'Public_Debt', 'Inflation',
                            'Internet', 'Ser_Share', 'Unemployment', 'Reserves', 'Electricity']
                
                # Ensure cols exist
                valid_predictors = [c for c in predictors if c in df.columns]

                # 2. RUN REGRESSION FOR HEATMAP (Loop over clusters)
                cluster_results = []
                if 'Cluster_ID' in df.columns and 'Cluster_Name' in df.columns:
                    clusters = df['Cluster_ID'].unique()
                    clusters.sort()
                    
                    for cluster_id in clusters:
                        sub_df = df[df['Cluster_ID'] == cluster_id].dropna(subset=valid_predictors + [target_col])
                        if len(sub_df) > 10:
                            cluster_name = sub_df['Cluster_Name'].iloc[0]
                            X_clus = sub_df[valid_predictors]
                            y_clus = sub_df[target_col]
                            X_clus = sm.add_constant(X_clus)
                            model_clus = sm.OLS(y_clus, X_clus).fit()

                            for feature in valid_predictors:
                                cluster_results.append({
                                    'Cluster_Name': cluster_name,
                                    'Factor': feature,
                                    'Coefficient': model_clus.params[feature]
                                })
                    
                    res_df = pd.DataFrame(cluster_results)
                    heatmap_data = res_df.pivot(index='Factor', columns='Cluster_Name', values='Coefficient')
                else:
                    st.error("Cluster columns missing. Ensure K-Means has been run or file has cluster info.")
                    heatmap_data = pd.DataFrame()

                # 3. CALCULATE EFFICIENCY & RISK LABEL
                # Global Regression
                X_all = df['Capital_Form']
                y_all = df['GDP_Growth']
                X_all = sm.add_constant(X_all)
                
                # Drop NaN for specific regression
                mask_eff = ~np.isnan(X_all['Capital_Form']) & ~np.isnan(y_all)
                X_all_clean = X_all[mask_eff]
                y_all_clean = y_all[mask_eff]
                
                model_all = sm.OLS(y_all_clean, X_all_clean).fit()
                
                # Predict and calculate residuals (Efficiency)
                df.loc[mask_eff, 'Expected_Growth'] = model_all.predict(X_all_clean)
                df.loc[mask_eff, 'Efficiency_Score'] = df.loc[mask_eff, 'GDP_Growth'] - df.loc[mask_eff, 'Expected_Growth']
                df['Risk_Label'] = df['Efficiency_Score'].apply(lambda x: 1 if x < 0 else 0)

                # --- PLOTTING (4 CHARTS) ---
                if not heatmap_data.empty:
                    fig = plt.figure(figsize=(20, 18))
                    gs = fig.add_gridspec(3, 2)

                    # Chart 1: Heatmap
                    ax1 = fig.add_subplot(gs[0, 0])
                    sns.heatmap(heatmap_data, annot=True, cmap='RdBu_r', center=0, fmt=".3f", ax=ax1, linewidths=.5)
                    ax1.set_title('1. GROWTH GENOME MAP\n(Blue: Positive | Red: Negative)', fontsize=12, fontweight='bold')
                    ax1.set_xlabel('')

                    # Chart 2: Efficiency Frontier
                    ax2 = fig.add_subplot(gs[0, 1])
                    sns.scatterplot(data=df, x='Capital_Form', y='GDP_Growth', hue='Cluster_Name', style='Cluster_Name',
                                    palette='viridis', alpha=0.7, ax=ax2, s=60)
                    # Highlight Vietnam
                    vn_data = df[df['Code'] == 'VNM']
                    if not vn_data.empty:
                        ax2.scatter(vn_data['Capital_Form'], vn_data['GDP_Growth'], color='red', s=150, label='Việt Nam', edgecolors='black', marker='*', zorder=10)
                    sns.regplot(data=df, x='Capital_Form', y='GDP_Growth', scatter=False, ax=ax2, color='gray', line_kws={'linestyle':'--'})
                    ax2.set_title('2. EFFICIENCY FRONTIER\n(Vietnam: Moderate Investment - High Growth)', fontsize=12, fontweight='bold')
                    ax2.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize='small')

                    # Chart 3: Trade Openness Boxplot
                    ax3 = fig.add_subplot(gs[1, 0])
                    sns.boxplot(data=df, x='Cluster_Name', y='Trade_Open', palette='Set2', ax=ax3)
                    ax3.set_title('3. TRADE OPENNESS (Key Driver)', fontsize=12, fontweight='bold')
                    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=15, ha='right')
                    ax3.set_xlabel('')

                    # Chart 4: FDI Boxplot
                    ax4 = fig.add_subplot(gs[1, 1])
                    sns.boxplot(data=df, x='Cluster_Name', y='FDI', palette='Set3', ax=ax4)
                    ax4.set_title('4. FDI (Crucial Capital Flow)', fontsize=12, fontweight='bold')
                    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=15, ha='right')
                    ax4.set_xlabel('')

                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning("Insufficient data to generate regression dashboard.")

            # --- TAB 3: DETAILED RESULTS ---
            with tabs[2]:
                st.subheader("Regression Statistics per Cluster")
                
                # Re-running logic to get full stats (P-values etc.)
                full_cluster_results = []
                if 'Cluster_ID' in df.columns:
                    for cluster_id in clusters:
                        sub_df = df[df['Cluster_ID'] == cluster_id].dropna(subset=valid_predictors + [target_col])
                        if len(sub_df) > 10:
                            cluster_name = sub_df['Cluster_Name'].iloc[0]
                            X_c = sm.add_constant(sub_df[valid_predictors])
                            y_c = sub_df[target_col]
                            mod = sm.OLS(y_c, X_c).fit()
                            
                            for feature in valid_predictors:
                                pval = mod.pvalues[feature]
                                sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
                                full_cluster_results.append({
                                    'Cluster_Name': cluster_name,
                                    'Factor': feature,
                                    'Impact': f"{mod.params[feature]:.4f} {sig}"
                                })
                            
                            full_cluster_results.append({
                                'Cluster_Name': cluster_name,
                                'Factor': 'R-squared',
                                'Impact': f"{mod.rsquared:.4f}"
                            })
                    
                    results_df_full = pd.DataFrame(full_cluster_results)
                    pivot_results = results_df_full.pivot(index='Factor', columns='Cluster_Name', values='Impact')
                    factors_order = valid_predictors + ['R-squared']
                    pivot_results = pivot_results.reindex(factors_order)
                    
                    st.write("Coefficient (Significance: *** < 0.01, ** < 0.05, * < 0.1)")
                    st.dataframe(pivot_results)

# --- PAGE 5: ASSOCIATION RULES ---
elif page == "Association Rules Miner":
    st.markdown('<p class="main-header">🔗 Association Rules Miner</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Apriori Algorithm & Association Rules
    Uncovering hidden patterns and "if-then" relationships in the economic data using **Apriori Algorithm**.
    """)

    if not MLXTEND_AVAILABLE:
        st.error("⚠️ Library `mlxtend` is missing. Please run `pip install mlxtend` to use this feature.")
    elif df is not None:
        # --- PREPARATION: RE-CALCULATE RISK LABEL ---
        # We need Risk_Label to exist even if user skipped the Regression page
        # Logic: GDP_Growth ~ Capital_Form residuals
        if STATSMODELS_AVAILABLE:
            X_risk = df['Capital_Form']
            y_risk = df['GDP_Growth']
            X_risk = sm.add_constant(X_risk)
            mask_risk = ~np.isnan(X_risk['Capital_Form']) & ~np.isnan(y_risk)
            if mask_risk.sum() > 10:
                mod_risk = sm.OLS(y_risk[mask_risk], X_risk[mask_risk]).fit()
                expected = mod_risk.predict(X_risk[mask_risk])
                efficiency = y_risk[mask_risk] - expected
                # 1 = Inefficient (Risky), 0 = Efficient
                df.loc[mask_risk, 'Risk_Label'] = efficiency.apply(lambda x: 1 if x < 0 else 0)
        else:
             st.warning("Skipping Risk Label calculation because `statsmodels` is missing.")


        # --- CONFIGURATION ---
        selected_vars = [
            'Capital_Form', 'Trade_Open', 'FDI', 'Public_Debt', 'Inflation',
            'Internet', 'Ser_Share', 'Unemployment', 'Reserves'
        ]
        
        # Unit Mapping for better labels
        units_map = {
            'Capital_Form': '%', 'Trade_Open': '%', 'FDI': '%',
            'Public_Debt': '%', 'Inflation': '%', 'Internet': '%',
            'Ser_Share': '%', 'Unemployment': '%', 'Reserves': 'USD'
        }

        # --- HELPER FUNCTION: BINNING ---
        def bin_feature_equal_freq(df_input, col, q=3):
            try:
                bins = pd.qcut(df_input[col], q=q, duplicates='drop')
                unit = units_map.get(col, "")
                def make_label(interval):
                    left = f"{interval.left:.1f}"
                    right = f"{interval.right:.1f}"
                    return f"{col} {left}{unit}-{right}{unit}"
                return bins.apply(make_label)
            except Exception:
                return df_input[col]

        # --- TABS UI ---
        tabs = st.tabs(["1. Run Apriori Rules", "2. Grid Search Optimization"])

        # --- TAB 1: RUN APRIORI ---
        with tabs[0]:
            st.subheader("Mining Association Rules")
            
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                min_sup = st.slider("Minimum Support", 0.01, 0.2, 0.05, 0.01)
            with col_p2:
                min_conf = st.slider("Minimum Confidence", 0.1, 1.0, 0.5, 0.1)
                
            if st.button("Run Apriori Analysis"):
                with st.spinner("Processing Data (Binning & Encoding)..."):
                    # 1. Processing
                    df_pro = df.copy()
                    valid_cols = [c for c in selected_vars if c in df_pro.columns]
                    
                    # Binning
                    for col in valid_cols:
                        if pd.api.types.is_numeric_dtype(df_pro[col]):
                            df_pro[col] = bin_feature_equal_freq(df_pro, col, q=3)
                    
                    # One-hot
                    policy_items = pd.get_dummies(df_pro[valid_cols], prefix="", prefix_sep="")
                    
                    # Add Risk Label items
                    if 'Risk_Label' in df_pro.columns:
                        df_pro['Risk_Item'] = df_pro['Risk_Label'].apply(lambda x: 'Risk_Label=1' if x==1 else 'Risk_Label=0')
                        risk_items = pd.get_dummies(df_pro['Risk_Item'], prefix="", prefix_sep="")
                        item_matrix = pd.concat([policy_items, risk_items], axis=1).astype(bool)
                    else:
                        item_matrix = policy_items.astype(bool)

                with st.spinner(f"Running Apriori (Support={min_sup})..."):
                    frequent_itemsets = apriori(item_matrix, min_support=min_sup, use_colnames=True)
                
                if frequent_itemsets.empty:
                    st.error("No frequent itemsets found! Try lowering Min Support.")
                else:
                    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)
                    rules['ante_len'] = rules['antecedents'].apply(len)
                    rules = rules.sort_values(['lift', 'confidence'], ascending=False)
                    
                    st.success(f"Found {len(rules)} rules.")
                    
                    # Display Top Rules
                    st.markdown("#### 🏆 Top 15 Strongest Rules (by Lift)")
                    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(15))
                    
                    # Display Risky Rules
                    st.markdown("#### ⚠️ High Risk Rules (Consequent: Risk_Label=1)")
                    risky_rules = rules[rules['consequents'].apply(lambda x: 'Risk_Label=1' in list(x))]
                    risky_filtered = risky_rules[(risky_rules['lift'] > 1.1) & (risky_rules['ante_len'] >= 1)]
                    
                    if not risky_filtered.empty:
                        st.warning(f"Found {len(risky_filtered)} rules indicating conditions for Inefficiency:")
                        st.dataframe(risky_filtered[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(15))
                    else:
                        st.info("No strong rules found leading specifically to High Risk (Inefficiency).")

        # --- TAB 2: GRID SEARCH ---
        with tabs[1]:
            st.subheader("Grid Search: Optimal Parameters")
            st.markdown("Finding the 'Sweet Spot' between Support and Confidence to generate a meaningful number of rules.")
            
            if st.button("Run Grid Search"):
                with st.spinner("Processing & Calculating Matrix..."):
                     # 1. Processing (Repeat logic efficiently)
                    df_pro_grid = df.copy()
                    valid_cols_grid = [c for c in selected_vars if c in df_pro_grid.columns]
                    for col in valid_cols_grid:
                        if pd.api.types.is_numeric_dtype(df_pro_grid[col]):
                            df_pro_grid[col] = bin_feature_equal_freq(df_pro_grid, col, q=3)
                    
                    policy_items_g = pd.get_dummies(df_pro_grid[valid_cols_grid], prefix="", prefix_sep="")
                    if 'Risk_Label' in df_pro_grid.columns:
                        df_pro_grid['Risk_Item'] = df_pro_grid['Risk_Label'].apply(lambda x: 'Risk_Label=1' if x==1 else 'Risk_Label=0')
                        risk_items_g = pd.get_dummies(df_pro_grid['Risk_Item'], prefix="", prefix_sep="")
                        item_matrix_g = pd.concat([policy_items_g, risk_items_g], axis=1).astype(bool)
                    else:
                        item_matrix_g = policy_items_g.astype(bool)

                    # 2. Grid Search Loop
                    support_candidates = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1]
                    confidence_candidates = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
                    results = []
                    
                    # Pre-calc frequent items for lowest support to save time
                    freq_items_all = apriori(item_matrix_g, min_support=0.01, use_colnames=True)
                    
                    for s in support_candidates:
                        f_items_s = freq_items_all[freq_items_all['support'] >= s]
                        for c in confidence_candidates:
                            if f_items_s.empty:
                                n_rules = 0
                            else:
                                try:
                                    r_temp = association_rules(f_items_s, metric="confidence", min_threshold=c)
                                    n_rules = len(r_temp)
                                except:
                                    n_rules = 0
                            results.append({'Support': s, 'Confidence': c, 'Rule_Count': n_rules})
                    
                    # 3. Plotting
                    res_df = pd.DataFrame(results)
                    pivot_t = res_df.pivot(index='Support', columns='Confidence', values='Rule_Count')
                    
                    fig_heat = plt.figure(figsize=(10, 6))
                    sns.heatmap(pivot_t, annot=True, fmt='d', cmap='YlGnBu', cbar_kws={'label': '# Rules'})
                    plt.title('Number of Rules by Support & Confidence')
                    plt.gca().invert_yaxis()
                    st.pyplot(fig_heat)
                    
                    st.info("Tip: Choose parameters that yield a manageable number of rules (e.g., light blue/green areas).")

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit • 2024")