# Predictive Modeling for Cybersecurity Risk Assessment and Prioritization

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0.2-orange.svg)
![Pandas](https://img.shields.io/badge/pandas-1.4.2-blueviolet.svg)
![NLTK](https://img.shields.io/badge/NLTK-3.7-green.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This repository houses the project "Predictive Modeling for Cybersecurity Risk Assessment and Prioritization Using CVE and CWE Mappings." It presents a machine learning framework designed to classify software vulnerabilities with high accuracy, moving beyond traditional scoring systems to provide a more context-aware and actionable approach to risk management.

## 📋 Table of Contents
- [The Challenge](#-the-challenge)
- [Our Solution](#-our-solution)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Key Results](#-key-results)
- [How to Run This Project](#-how-to-run-this-project)
- [Authors](#-authors)
- [License](#-license)

## 🚨 The Challenge

In today's digital landscape, security teams are inundated with a massive volume of software vulnerabilities. Traditional methods for prioritizing these threats, like the Common Vulnerability Scoring System (CVSS), are essential but often insufficient. Relying solely on a severity score can overlook the underlying nature of the weakness (the "why") and the specific context described in the vulnerability report. This creates a critical gap in risk assessment, making it difficult to allocate resources effectively.

## 💡 Our Solution

This project introduces a robust machine learning pipeline that integrates three crucial data points to create a more holistic and predictive model for vulnerability prioritization:

1.  **CVE Descriptions (Textual Data):** Utilizes Natural Language Processing (NLP) to analyze the textual descriptions of vulnerabilities.
2.  **CWE Categories (Categorical Data):** Incorporates the "root cause" of the vulnerability by mapping it to a Common Weakness Enumeration (CWE) type.
3.  **CVSS Scores (Numerical Data):** Leverages the industry-standard severity score as a critical numerical feature.

By combining these multi-modal features, our model learns to classify vulnerabilities into **High, Medium, or Low** severity with exceptional accuracy, providing a more reliable foundation for triaging efforts.

## 📊 Dataset

The model was developed using the **"CVE and CWE Mapping Dataset"** from Kaggle, which aggregates data from the National Vulnerability Database (NVD).

- **Source:** NVD and CVE repositories.
- **Size:** A stratified sample of 50,000 records was used for modeling.
- **Features Used:** `CVE-ID`, `DESCRIPTION`, `CWE-ID`, `CVSS v2/v3 Scores`, and `Severity`.

## ⚙️ Methodology

The project follows a comprehensive and systematic pipeline to ensure robust and reproducible results.

1.  **Data Preprocessing:**
    * **Filtering:** Removed entries with invalid or missing CWE information.
    * **CVSS Unification:** Created a single, consistent CVSS score column, prioritizing v3 over v2.
    * **Text Cleaning:** Normalized CVE descriptions by lowercasing, tokenizing, removing stopwords, and standardizing security-specific terms using a custom synonym dictionary.
    * **Label Encoding:** Mapped severity labels (Low, Medium, High, Critical) to numerical values (0, 1, 2).

2.  **Feature Engineering:**
    * **Text Vectorization:** Converted cleaned CVE descriptions into numerical features using **TF-IDF** (max 1,000 features).
    * **Feature Combination:** Horizontally stacked the TF-IDF vectors, encoded CWE IDs, and numerical CVSS scores into a single feature matrix.

3.  **Modeling and Evaluation:**
    * **Class Imbalance:** Addressed the natural imbalance in severity classes using **SMOTE (Synthetic Minority Oversampling Technique)**.
    * **Dimensionality Reduction:** Applied **LDA (Linear Discriminant Analysis)** to reduce feature space complexity and improve the performance of linear models.
    * **Classifiers Tested:**
        * Random Forest
        * Logistic Regression
        * Naive Bayes (Multinomial and Gaussian)
        * Support Vector Machine (SVM)
    * **Evaluation:** Employed a 5-fold cross-validation strategy, measuring performance with **Accuracy, Precision, Recall, and F1-Score**.

## 🏆 Key Results

The models demonstrated outstanding predictive power. The **Random Forest classifier**, trained on the original, full feature set, emerged as the top-performing model, showcasing its ability to handle high-dimensional, mixed-type data effectively.

### Best Model Performance: Random Forest
| Metric         | Score           |
| :------------- | :-------------- |
| **F1-Score (Weighted Avg)** | **0.9928** |
| **Standard Deviation (F1)** | **0.0013** |

The confusion matrix for the Random Forest model showed near-perfect classification across all three severity levels. Additionally, linear models (SVM, Logistic Regression) also achieved strong results (F1 > 0.95) when enhanced with LDA, confirming the value of dimensionality reduction.

These results strongly indicate that integrating textual CVE descriptions and categorical CWE data with numerical CVSS scores leads to a significantly more accurate and reliable vulnerability prioritization system.

## 🚀 How to Run This Project

To set up and run this analysis on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/zx784/Predictive-Modeling-for-Cybersecurity-Risk-Assessment-and-Prioritization.git](https://github.com/zx784/Predictive-Modeling-for-Cybersecurity-Risk-Assessment-and-Prioritization.git)
    cd Predictive-Modeling-for-Cybersecurity-Risk-Assessment-and-Prioritization
    ```

2.  **Set up a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    The project requires several libraries. You can install them using the provided `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
    Key libraries include: `pandas`, `numpy`, `scikit-learn`, `nltk`, `matplotlib`, `seaborn`, and `imbalanced-learn`.

4.  **Execute the notebook:**
    The entire workflow is documented in a Jupyter Notebook. Launch it to see the code and results.
    ```bash
    jupyter notebook your_notebook_name.ipynb
    ```

## ✍️ Authors

- **Amro Shiek** (`amro.shiek@student.aiu.edu.my`)
- **Umar Faruk** (`umarfaruk.bello@student.aiu.edu.my`)

School of Computing and Informatics, Albukhary International University, Malaysia.

## 📄 License

This project is distributed under the MIT License. See the `LICENSE` file for more information.
