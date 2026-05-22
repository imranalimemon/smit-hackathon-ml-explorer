"""
sample_datasets.py — Generate sample CSV datasets for testing the app

Creates 3 sample datasets that users can download and upload to the app:
1. Iris-like classification dataset
2. Customer segmentation dataset
3. Student performance dataset
"""

import pandas as pd
import numpy as np
import os


def create_iris_dataset(output_dir):
    """Create a classification dataset similar to Iris."""
    np.random.seed(42)
    n_per_class = 50
    
    # Setosa
    setosa = pd.DataFrame({
        'sepal_length': np.random.normal(5.0, 0.35, n_per_class),
        'sepal_width': np.random.normal(3.4, 0.38, n_per_class),
        'petal_length': np.random.normal(1.5, 0.17, n_per_class),
        'petal_width': np.random.normal(0.2, 0.10, n_per_class),
        'species': 'setosa'
    })
    
    # Versicolor
    versicolor = pd.DataFrame({
        'sepal_length': np.random.normal(5.9, 0.52, n_per_class),
        'sepal_width': np.random.normal(2.8, 0.31, n_per_class),
        'petal_length': np.random.normal(4.3, 0.47, n_per_class),
        'petal_width': np.random.normal(1.3, 0.20, n_per_class),
        'species': 'versicolor'
    })
    
    # Virginica
    virginica = pd.DataFrame({
        'sepal_length': np.random.normal(6.6, 0.64, n_per_class),
        'sepal_width': np.random.normal(3.0, 0.32, n_per_class),
        'petal_length': np.random.normal(5.6, 0.55, n_per_class),
        'petal_width': np.random.normal(2.0, 0.27, n_per_class),
        'species': 'virginica'
    })
    
    df = pd.concat([setosa, versicolor, virginica], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.round(1)
    
    path = os.path.join(output_dir, 'sample_iris.csv')
    df.to_csv(path, index=False)
    print(f"  Created: {path} ({len(df)} rows)")


def create_customer_dataset(output_dir):
    """Create a customer segmentation dataset for clustering."""
    np.random.seed(123)
    n = 200
    
    df = pd.DataFrame({
        'annual_income_k': np.random.randint(15, 140, n),
        'spending_score': np.random.randint(1, 100, n),
        'age': np.random.randint(18, 70, n),
        'visits_per_month': np.random.randint(1, 30, n),
        'avg_purchase_value': np.random.randint(10, 500, n),
    })
    
    path = os.path.join(output_dir, 'sample_customers.csv')
    df.to_csv(path, index=False)
    print(f"  Created: {path} ({len(df)} rows)")


def create_student_dataset(output_dir):
    """Create a student performance classification dataset."""
    np.random.seed(99)
    n = 300
    
    study_hours = np.random.uniform(1, 12, n).round(1)
    attendance = np.random.uniform(40, 100, n).round(1)
    previous_grade = np.random.choice(['A', 'B', 'C', 'D', 'F'], n, p=[0.15, 0.30, 0.30, 0.15, 0.10])
    extracurricular = np.random.choice(['Yes', 'No'], n, p=[0.4, 0.6])
    
    # Generate pass/fail based on features
    score = study_hours * 5 + attendance * 0.3
    result = np.where(score > 45, 'Pass', 'Fail')
    
    df = pd.DataFrame({
        'study_hours_per_day': study_hours,
        'attendance_pct': attendance,
        'previous_grade': previous_grade,
        'extracurricular': extracurricular,
        'result': result
    })
    
    path = os.path.join(output_dir, 'sample_students.csv')
    df.to_csv(path, index=False)
    print(f"  Created: {path} ({len(df)} rows)")


if __name__ == "__main__":
    print("=" * 50)
    print("  Generating Sample Datasets")
    print("=" * 50)
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_data')
    os.makedirs(output_dir, exist_ok=True)
    
    create_iris_dataset(output_dir)
    create_customer_dataset(output_dir)
    create_student_dataset(output_dir)
    
    print("\n  Upload any of these CSVs to the Streamlit app to test!")
