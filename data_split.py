import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split

"""
Merges all sequences from .npz files into one dataset
and splits into train/valid/test 
"""

def load_all_classes(processed_dir):
    """
    Loads all .npz sequences data
    """
    all_files = glob.glob(os.path.join(processed_dir, "*.npz"))
    X_list, y_list = [], []

    for file in all_files:
        data = np.load(file, allow_pickle=True)
        X_list.append(data["X"])
        y_list.append(data["y"])
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y

def split_dataset(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Data splitting to train/valid/test sets
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratio sum must equal to 1"

    # train vs temp (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_ratio), stratify=y, random_state=random_state
    )

    # val vs test
    val_size = val_ratio / (val_ratio + test_ratio)  # proporcja w stosunku do temp
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_size), stratify=y_temp, random_state=random_state
    )

    return X_train, y_train, X_val, y_val, X_test, y_test

def save_splits(output_dir, X_train, y_train, X_val, y_val, X_test, y_test):
    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, "train.npz"), X=X_train, y=y_train)
    np.savez(os.path.join(output_dir, "val.npz"), X=X_val, y=y_val)
    np.savez(os.path.join(output_dir, "test.npz"), X=X_test, y=y_test)
    print(f"[INFO] Subsets saved to {output_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split processed NPZ files into train/val/test")
    parser.add_argument("--processed_dir", type=str, default="data/processed",
                        help="Input .npz sequences path")
    parser.add_argument("--output_dir", type=str, default="data/splits",
                        help="Output .npz subsets path")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--random_state", type=int, default=42)
    
    args = parser.parse_args()

    X, y = load_all_classes(args.processed_dir)
    print(f"[INFO] Loaded data: {X.shape[0]} sequences, {len(np.unique(y))} classes")

    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(
        X, y,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state
    )

    save_splits(args.output_dir, X_train, y_train, X_val, y_val, X_test, y_test)
