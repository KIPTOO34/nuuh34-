#!/usr/bin/env python3
"""
Standalone training script for the Epidemic Prediction model.
Run this in VS Code (or any terminal) to retrain the model using your CSV files.
"""

import os
import sys

# Add the project root to the Python path (if needed)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import app  # imports all functions and global variables from app.py

def main():
    print("=" * 60)
    print("Epidemic Prediction Model Training")
    print("=" * 60)

    # Verify that data files exist
    data_dir = os.path.join(app.BASE, 'data')
    train_file = app.TRAIN
    test_file = app.TEST

    if not os.path.exists(train_file):
        print(f"ERROR: Training data not found at {train_file}")
        print("Please ensure 'training.csv' is inside the 'data/' folder.")
        sys.exit(1)
    if not os.path.exists(test_file):
        print(f"ERROR: Testing data not found at {test_file}")
        print("Please ensure 'testing.csv' is inside the 'data/' folder.")
        sys.exit(1)

    print(f"Using training data: {train_file}")
    print(f"Using testing data : {test_file}")

    # 1. Initialise the database (creates tables if they don't exist)
    print("\n[1/3] Initialising database...")
    app.init_db()

    # 2. Run the training pipeline (loads data, trains, evaluates, saves)
    print("[2/3] Starting model training...")
    try:
        acc, prec, rec, f1, cm = app.train_and_save()
    except Exception as e:
        print(f"\nERROR during training:\n{e}")
        sys.exit(1)

    # 3. Display results
    print("\n[3/3] Training completed successfully!")
    print("-" * 60)
    print(f"Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f}  ({prec*100:.2f}%)")
    print(f"Recall   : {rec:.4f}  ({rec*100:.2f}%)")
    print(f"F1-Score : {f1:.4f}  ({f1*100:.2f}%)")
    print("\nConfusion Matrix:")
    for row in cm:
        print(f"  {row}")

    print("\nModel artifacts saved to 'model/' directory:")
    print("  - knn.pkl   (trained KNN classifier)")
    print("  - scaler.pkl (fitted StandardScaler)")
    print("  - encoder.pkl (fitted LabelEncoder)")
    print("\nMetrics also stored in the database (model_runs table).")
    print("=" * 60)

if __name__ == "__main__":
    main()