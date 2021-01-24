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
    V = Vt.T

    # d_vector in dxd Form bringen
    D = np.zeros((n-d, d))
    D = np.concatenate((np.diag(d_vector), D))

    # Or
    # D = np.zeros(dataframe.shape)
    # for i in range(len(d_vector)):
    #     D[i, i] = d_vector[i]

    # Basisvektoren q_r
    q_r = V[:, :r]

    # Scores
    scores = np.matmul(U, D)[:, :r]
    scores_left = np.matmul(U, D)[:, r:]

    variances = np.power(d_vector, 2) / (n - 1)
    sum_variance = np.sum(variances)  # Summe der Varianzen

    scores_df = pd.DataFrame(scores)
    scores_var = scores_df.var()

    scores_left_df = pd.DataFrame(scores_left)
    scores_left_var = scores_left_df.var()
    error_p = (np.sum(scores_left_var) / (np.sum(scores_var) + np.sum(scores_left_var)))*100 # Fehler in %

    # Erklärte Varianz
    exp_var = np.sum(scores_var) / sum_variance

    return q_r, scores, variances, sum_variance, scores_var, exp_var, error
