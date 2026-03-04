import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

def load_and_preprocess_data(dataset_path='Smartphone_Usage_Productivity_Dataset_50000.csv', batch_size=64):
    if not os.path.exists(dataset_path):
        for dirname, _, filenames in os.walk('./'):
            for filename in filenames:
                path = os.path.join(dirname, filename)
                if "Smartphone_Usage_Productivity_Dataset_50000.csv" in filename:
                    dataset_path = path

    df = pd.read_csv(dataset_path)
    if 'User_ID' in df.columns: 
        df = df.drop(columns=['User_ID'])

    categorical_cols = ['Gender', 'Occupation', 'Device_Type']
    target_col = 'Work_Productivity_Score'
    df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df_processed.drop(columns=[target_col]).values
    y = df_processed[target_col].values.astype(float)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    # First split: 70% Train, 30% Temp (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(X_tensor, y_tensor, test_size=0.30, random_state=42)

    # Second split: Split the 30% Temp into equal halves (15% Val, 15% Test)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    # DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), train_loader, val_loader, test_loader, X_tensor.shape[1]
