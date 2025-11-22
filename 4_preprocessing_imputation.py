"""
KNN imputation for missing beacon measurements.

Uses k=3 nearest neighbors with Euclidean distance.
"""

import numpy as np
from sklearn.impute import KNNImputer


def impute_missing_values(rssi_data, k=3):
    """
    Impute missing RSSI values using KNN.
    
    Args:
        rssi_data: Array of shape (n_samples, n_beacons)
        k: Number of neighbors for imputation
        
    Returns:
        Imputed data of same shape
    """
    imputer = KNNImputer(n_neighbors=k, weights='distance')
    imputed_data = imputer.fit_transform(rssi_data)
    return imputed_data


if __name__ == "__main__":
    # Example usage
    data = np.random.randn(100, 10)
    data[np.random.rand(100, 10) < 0.2] = np.nan  # Add missing values
    
    imputed = impute_missing_values(data, k=3)
    print(f"Imputation complete! Shape: {imputed.shape}")
