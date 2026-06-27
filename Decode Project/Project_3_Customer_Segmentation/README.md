# Project 3: Customer Segmentation via Unsupervised Learning

## Project Overview
This project implements an end-to-end unsupervised learning pipeline designed to transform raw consumer behavioral tracking data into mathematically isolated, strategically actionable marketing segments. 

## Architectural Framework (IPO Model)
- **Input (Scale):** Handled spatial proximity distortion across 20+ features via Z-score standardization using `StandardScaler`.
- **Process (Compress):** Bypassed the Curse of Dimensionality using Principal Component Analysis (PCA) to extract orthogonal axes of maximum variance maintaining a $\ge 95\%$ cumulative variance threshold.
- **Process (Cluster):** Segmented data using the `K-Means++` algorithm, mathematically optimized via WCSS (Elbow Method) and Silhouette Score analysis.
- **Output (Translate):** Reverse-engineered abstract PCA coordinates back to real consumer metrics using inverse transformation mapping.

## Tech Stack
- **Language:** Python
- **Core Libraries:** Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn

## Results: Strategic Persona Matrix
- **Cluster 0: Affluent Conservatives** (High income, low spending score) -> *Action:* Loyalty focus & premium warranties.
- **Cluster 1: High-Value Trendsetters** (High income, high spending score) -> *Action:* Early access & exclusive perks.
- **Cluster 2: Budget-Conscious Explorers** (Low income, high spending score) -> *Action:* Flash sales & BNPL options.
- **Cluster 3: Conservative Minimisers** (Low income, low spending score) -> *Action:* Clear price utility values.

## 📸 Screenshots

![Screenshot 1](Customer_segmentation.png)
![Screenshot 2](PLOT_OUTPUT.png)
![Screenshot 3](3D_PLOT_OUTPUT.png)
