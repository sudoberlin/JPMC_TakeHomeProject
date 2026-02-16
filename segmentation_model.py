#!/usr/bin/env python3
"""
Segmentation Model - Customer Clustering
Groups customers into segments for targeted marketing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def load_data(data_path, columns_path):
    print("Loading census data...")
    
    with open(columns_path, 'r') as f:
        cols = [line.strip() for line in f if line.strip()]
    
    df = pd.read_csv(data_path, header=None)
    
    if len(cols) == df.shape[1]:
        df.columns = cols
    else:
        df.columns = cols[:df.shape[1]]
    
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    
    df = df.replace(' ?', np.nan)
    df = df.replace('?', np.nan)
    
    return df


def prepare_features(df):
    print("\nPreparing features for clustering...")
    
    drop_cols = ['weight']
    if df.columns[-1] in df.columns:
        drop_cols.append(df.columns[-1])
    drop_cols = [col for col in drop_cols if col in df.columns]
    
    X = df.drop(drop_cols, axis=1)
    
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    if num_cols:
        imp_num = SimpleImputer(strategy='median')
        X[num_cols] = imp_num.fit_transform(X[num_cols])
    
    if cat_cols:
        imp_cat = SimpleImputer(strategy='constant', fill_value='Unknown')
        X[cat_cols] = imp_cat.fit_transform(X[cat_cols])
    
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    return X, num_cols


def find_optimal_clusters(X_scaled):
    print("\nEvaluating optimal number of clusters...")
    
    inertias = []
    K_range = range(2, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.savefig('elbow_curve.png', dpi=300, bbox_inches='tight')
    print("Saved elbow curve as 'elbow_curve.png'")
    
    return 3


def create_segments(X, df_raw, num_cols, optimal_k):
    print(f"\nClustering with k={optimal_k}...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    df_analysis = df_raw.copy()
    df_analysis['Cluster'] = clusters
    
    print(f"\nCluster sizes:")
    print(df_analysis['Cluster'].value_counts().sort_index())
    
    print("\n" + "-"*60)
    print("Segment Profiles")
    print("-"*60)
    
    available_numeric = []
    for col in num_cols:
        if col in df_analysis.columns:
            if df_analysis[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                available_numeric.append(col)
    
    if available_numeric:
        profile_cols = available_numeric[:min(8, len(available_numeric))]
        profile = df_analysis.groupby('Cluster')[profile_cols].mean()
        print("\n", profile.to_string())
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters,
                         cmap='viridis', alpha=0.6, edgecolors='k')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('Customer Segments (PCA Visualization)')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.savefig('segmentation_viz.png', dpi=300, bbox_inches='tight')
    print("\nSaved visualization as 'segmentation_viz.png'")
    
    return kmeans, scaler, clusters


def main():
    DATA_FILE = 'census-bureau.data'
    COLS_FILE = 'census-bureau.columns'
    
    try:
        df_raw = load_data(DATA_FILE, COLS_FILE)
        X, num_cols = prepare_features(df_raw)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        optimal_k = find_optimal_clusters(X_scaled)
        model, scaler, clusters = create_segments(X, df_raw, num_cols, optimal_k)
        
        print("\n" + "="*60)
        print("Segmentation complete!")
        print("="*60)
        print("\nGenerated files:")
        print("  - elbow_curve.png")
        print("  - segmentation_viz.png")
        
    except FileNotFoundError:
        print("\n❌ Data files not found. Need:")
        print("  - census-bureau.data")
        print("  - census-bureau.columns")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
