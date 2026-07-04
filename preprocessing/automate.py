# import library
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def preprocess_data():
    # Path default
    input_file = "bank_raw/bank.csv"
    output_dir = "processed_data"

    print(f"Loading dataset dari {input_file}...")
    df = pd.read_csv(input_file)

    # Encode target
    if 'deposit' in df.columns:
        df['deposit'] = df['deposit'].map({'yes': 1, 'no': 0})

    # Pisahkan kolom kategorikal
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if 'deposit' in cat_cols:
        cat_cols.remove('deposit')

    # Label Encoding
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Pisahkan fitur dan target
    X = df.drop('deposit', axis=1)
    y = df['deposit']

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling numerik
    scaler = StandardScaler()
    num_cols = X.select_dtypes(include='number').columns.tolist()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Simpan hasil
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False, header=True)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False, header=True)

    print(f"Preprocessing selesai. File disimpan di folder: {output_dir}")

if __name__ == "__main__":
    preprocess_data()