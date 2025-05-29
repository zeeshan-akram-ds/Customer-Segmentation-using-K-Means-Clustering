# Customer Segmentation App using K-Means Clustering

A fully interactive and user-friendly Streamlit web application for unsupervised customer segmentation using the K-Means algorithm. This project is designed to help businesses or analysts automatically discover natural groupings in customer data for marketing, personalization, or targeting strategies.

---

## Objective
The main objective of this project is to create a **complete clustering workflow** inside an intuitive web interface. Users can upload any customer dataset, handle preprocessing (including encoding, scaling, and missing value treatment), apply K-Means, and visualize results in a business-friendly format.

This tool is perfect for freelancers, analysts, and data scientists who want to:
- Segment customers without coding
- Quickly understand patterns and behaviors
- Present results to non-technical stakeholders

---

## Features

### File Upload & Preview
- Upload your own CSV file
- Preview uploaded data inside an expandable panel

### Preprocessing Options
- **Column Selection:**
  - Choose both **numerical** and **categorical** columns
- **Missing Values Handling:**
  - Drop rows with missing values
  - Or impute using:
    - Mean (for numeric)
    - Mode (for categorical)
- **Encoding Options:**
  - Apply **Label Encoding** or **One-Hot Encoding** to selected categorical columns
- **Feature Scaling:**
  - Choose between **StandardScaler** and **MinMaxScaler**

### KMeans Clustering Configuration
- Choose the number of clusters `k` using a slider (range: 2–10)

### Outputs
- **Cluster Assignments:**
  - View the first few rows of your dataset with a new `Cluster` column
- **Cluster Summary:**
  - Mean values per cluster displayed in a styled table
- **Elbow Curve:**
  - Visual aid to find the optimal `k`
- **Silhouette Score:**
  - Measures clustering quality (automatically shown)

### Visualizations
- **PCA-Based Cluster Plot (2D):**
  - Beautiful and interactive Plotly scatter plot
- **Cluster Centers:**
  - Written form in a styled table (inverse scaled)
  - Visual form as a line plot comparing each cluster’s average values

### Export Option
- Download the final clustered dataset as a CSV file with a single click

---

## How to Run Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run kmeans_app.py
```
---

## Folder Structure
```text
├── kmeans_app.py              # Main Streamlit app
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
```

---

## Tech Stack
- Python
- Pandas, NumPy
- scikit-learn
- Plotly
- Streamlit

---

## Contact
For questions or freelance inquiries:
📧 Email: [zeeshanakram1704@gmail.com]  
🔗 LinkedIn: [Linkedln](https://www.linkedin.com/in/zeeshan-akram-572bbb34a/)

---

> Built by Zeeshan Akram — certified data scientist and ML engineer.

---
