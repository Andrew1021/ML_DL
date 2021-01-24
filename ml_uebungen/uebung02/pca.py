import pandas as pd
import numpy as np
from sklearn import preprocessing


def pca(dataframe, r):
    n, d = dataframe.shape

    # Matrix zentrieren
    dataframe_centered = dataframe - dataframe.mean()

    # Matrix normalisieren, dass Varianz = 1
    standard_scaler = preprocessing.StandardScaler()
    dataframe_scaled = pd.DataFrame(standard_scaler.fit_transform(dataframe_centered))

    # Singulärwertzerlegung
    U, d_vector, Vt = np.linalg.svd(dataframe_scaled)

    # d_vector in dxd Form bringen
    D = np.zeros((n-d, d))
    D = np.concatenate((np.diag(d_vector), D))

    # Basisvektoren q_r
    q_r = Vt.T[:, :r]

    # Scores / Projected Data / a_i
    scores = np.matmul(U, D)[:, :r]
    scores_left = np.matmul(U, D)[:, r:]

    scores_var = pd.DataFrame(scores).var()
    scores_left_var = pd.DataFrame(scores_left).var()

    # Varianzen / Eigenwerte
    variances = np.power(d_vector, 2) / (n - 1)
    sum_variance = np.sum(variances)

    # Fehler in Prozent
    error_p = (np.sum(scores_left_var) / (np.sum(scores_var) + np.sum(scores_left_var)))*100

    # Erklärte Varianz
    exp_var = np.sum(scores_var) / sum_variance

    return q_r, scores, variances, sum_variance, scores_var, exp_var, error_p
