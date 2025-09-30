import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, det
from numpy.typing import NDArray
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier




def summarize_data(resumen):
    rows = []
    for n,(m,s) in resumen["LDA"].items():
        rows.append({"metodo":"LDA","n":n,"k":None,"mean":m,"std":s})
    for n,(m,s) in resumen["QDA"].items():
        rows.append({"metodo":"QDA","n":n,"k":None,"mean":m,"std":s})
    for (n,k),(m,s) in resumen["kNN"].items():
        rows.append({"metodo":"kNN","n":n,"k":k,"mean":m,"std":s})
    df = pd.DataFrame(rows)
    return df



# ---------------------------
# 1) Scenarios (p=2 para for multivariate normals)
# ---------------------------
def get_scenario(name:str):
    """This function will return the values for the requiered scenarios

    Args:
        name (str): The name of the scenario to get the parametro

    Raises:
        ValueError: If the name it's not in the defined options

    Returns:
        mu_0(NDArray): Mean vector for class 0
        mu_1(NDArray): Mean vector for class 1
        Sigma_0(NDArray): Covariance matrix for class 0
        Sigma_1(NDArray): Covariance matrix for class 1
        priors(tuple): Priors for each class (pi_0, pi_1)
    """
    if name == "lda_optimo":      # Σ0 = Σ1 → Bayes = LDA
        mu0 = np.array([0., 0.])
        mu1 = np.array([2.5, 2.5])
        Sigma0 = Sigma1 = np.eye(2)
        priors = (0.5, 0.5)

    elif name == "qda_optimo":    # Σ0 ≠ Σ1 → Bayes = QDA
        mu0 = np.array([0., 0.])
        mu1 = np.array([1.2, 1.2])
        Sigma0 = np.eye(2)
        Sigma1 = np.array([[2.0, 0.5],
                           [0.5, 1.2]])
        priors = (0.5, 0.5)

    elif name == "desbalance":    # priors unbalanced
        mu0 = np.array([0., 0.])
        mu1 = np.array([2.0, 2.0])
        Sigma0 = Sigma1 = np.eye(2)
        priors = (0.8, 0.2)       # π0=0.8, π1=0.2

    elif name == "correlacion_fuerte":  # cov correlated
        mu0 = np.array([0., 0.])
        mu1 = np.array([1.5, 1.5])
        Sigma0 = np.array([[1.0, 0.9],
                           [0.9, 1.0]])
        Sigma1 = np.array([[1.0, 0.7],
                           [0.7, 1.0]])
        priors = (0.5, 0.5)
    else:
        raise ValueError("Escenario no reconocido.")
    return mu0, mu1, Sigma0, Sigma1, priors

# ---------------------------
# 2) Generate data
# ---------------------------
def generate_data(n:int, mu0:NDArray, mu1:NDArray, Sigma0:NDArray, Sigma1:NDArray, 
                  priors:tuple, rng:np.random.Generator):
    """This will generate a dataset with n samples from two classes. This data 
       will be used to classify it using LDA, QDA and kNN.

    Args:
        n (int): Number of samples to generate
        mu0 (NDArray): Mean vector for class 0
        mu1 (NDArray): Mean vector for class 1
        Sigma0 (NDArray): Covariance matrix for class 0
        Sigma1 (NDArray): Covariance matrix for class 1
        priors (tuple): Priors for each class (pi_0, pi_1)
        rng (np.random.Generator): Random number generator to use.

    Returns:
        X(NDArray): Generated samples
        Y(NDArray): Class labels for each sample
    """
    pi0, pi1 = priors
    Y = rng.choice([0,1], size=n, p=[pi0, pi1])
    X = np.zeros((n, len(mu0)))
    for i in range(n):
        if Y[i] == 0:
            X[i] = rng.multivariate_normal(mu0, Sigma0)
        else:
            X[i] = rng.multivariate_normal(mu1, Sigma1)
    return X, Y

# ---------------------------
# 3) Bayes: score y clasificador
# ---------------------------
def bayes_scores(X:NDArray, mu:NDArray, Sigma:NDArray, pi:float):
    """This function will compute the bayes scores for each sample in X.

    Args:
        X (NDArray): Samples to compute the scores
        mu (NDArray): Mean vector for the class
        Sigma (NDArray): Covariance matrix for the class
        pi (float): Prior for the class

    Returns:
        float: Scores for each sample in X.
    """
    invS = inv(Sigma)
    logdet = np.log(det(Sigma))
    # δ_k(x) = log π    _k - 0.5 log|Σ_k| - 0.5 (x-μ_k)^T Σ_k^{-1} (x-μ_k)
    XC = X - mu
    quad = np.einsum('ij,jk,ik->i', XC, invS, XC)  # Cuadratic form for each row
    return np.log(pi) - 0.5*logdet - 0.5*quad

def bayes_pred(X:NDArray, mu0:NDArray, mu1:NDArray, S0:NDArray, S1:NDArray, pi0:float, pi1:float):
    """This function will compute the bayes predictions for each sample in X.

    Args:
        X (NDArray): Samples to compute the predictions
        mu0 (NDArray): Mean vector for class 0
        mu1 (NDArray): Mean vector for class 1
        S0 (NDArray): Covariance matrix for class 0
        S1 (NDArray): Covariance matrix for class 1
        pi0 (float): Prior for class 0
        pi1 (float): Prior for class 1

    Returns:
        int: Predicted class labels for each sample in X.
    """
    d0 = bayes_scores(X, mu0, S0, pi0)
    d1 = bayes_scores(X, mu1, S1, pi1)
    return (d1 > d0).astype(int)

def risk_bayes_montecarlo(mu0:NDArray, mu1:NDArray, S0:NDArray, S1:NDArray,
                          priors:tuple, N:int=200_000, 
                          rng:np.random.Generator=np.random.default_rng()):
    """This function will estimate the bayes risk using montecarlo simulation.

    Args:
        mu0 (NDArray): Mean vector for class 0
        mu1 (NDArray): Mean vector for class 1
        S0 (NDArray): Covariance matrix for class 0
        S1 (NDArray): Covariance matrix for class 1
        priors (tuple): Priors for each class (pi_0, pi_1)
        N (int, optional): The number number of values to generate for MC. Defaults to 200_000.
        rng (np.random.Generator): Random number generator to use.


    Returns:
        float: Estimated bayes risk.
    """
    pi0, pi1 = priors
    Y = rng.choice([0,1], size=N, p=[pi0, pi1])
    X = np.zeros((N, len(mu0)))
    idx0 = (Y==0); idx1 = ~idx0
    X[idx0] = rng.multivariate_normal(mu0, S0, size=idx0.sum())
    X[idx1] = rng.multivariate_normal(mu1, S1, size=idx1.sum())
    Yhat = bayes_pred(X, mu0, mu1, S0, S1, pi0, pi1)
    return np.mean(Yhat != Y)

# ---------------------------
# 4) Risk by CV
# ---------------------------
def risk_cv(clf:BaseEstimator, X:NDArray, Y:NDArray, cv:int=5):
    """This function will estimate the classification error using cross-validation.

    Args:
        clf (BaseEstimator): The classifier to evaluate
        X (NDArray): Samples to classify
        Y (NDArray): True class labels for each sample
        cv (int, optional): The number of cross validations to do. Defaults to 5.

    Returns:
        NDArray: Error for each fold in the cross-validation.
    """

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=None)
    # We use 1 - accuracy = error
    acc = cross_val_score(clf, X, Y, cv=skf, scoring="accuracy")
    return 1.0 - acc  # vector de errores por fold

# ---------------------------
# 5) Bucle experimental
# ---------------------------
def experiment_bucle(scenario_name:str,
                     n_list:list=[50,100,200,500],
                     k_list:list=[1,3,5,11,21],
                     R=20,
                     rng:np.random.Generator=np.random.default_rng()):
    """This function will run the experiments for the given scenario.

    Args:
        scenario_name (str): The name of the scenario to run the experiment
        n_list (list, optional): List with sample sizes. Defaults to [50,100,200,500].
        k_list (list, optional): Number of k to test in KNN. Defaults to [1,3,5,11,21].
        R (int, optional): Iterations by experment. Defaults to 20.
        rng (np.random.Generator): Random number generator to use.


    Returns:
        resultados(dict): Detailed results of the experiment
        resumen(dict): Summary of the results (mean and std)
    """
    mu0, mu1, S0, S1, priors = get_scenario(scenario_name)

    # True risk (Bayes)
    L_bayes = risk_bayes_montecarlo(mu0, mu1, S0, S1, priors, N=200_000,rng=rng)

    resultados = {
        "meta": dict(scenario=scenario_name, priors=priors, L_bayes=L_bayes,
                     n_list=n_list, k_list=k_list, R=R),
        "LDA": {n: [] for n in n_list},
        "QDA": {n: [] for n in n_list},
        "kNN": {(n,k): [] for n in n_list for k in k_list},
    }

    for n in n_list:
        for r in range(R):
            X, Y = generate_data(n, mu0, mu1, S0, S1, priors, rng=rng)

            # LDA / QDA with defined priors
            lda = LinearDiscriminantAnalysis()#priors=list(priors))
            qda = QuadraticDiscriminantAnalysis()#priors=list(priors))
            # The fit will be done inside risk_cv by cross_val_score

            err_lda = risk_cv(lda, X, Y).mean()
            err_qda = risk_cv(qda, X, Y).mean()
            resultados["LDA"][n].append(err_lda)
            resultados["QDA"][n].append(err_qda)

            # k-NN: iterating in k
            for k in k_list:
                knn = KNeighborsClassifier(n_neighbors=k)
                err_knn = risk_cv(knn, X, Y).mean()
                resultados["kNN"][(n,k)].append(err_knn)

    # Save mean and std of results
    resumen = {"LDA": {}, "QDA": {}, "kNN": {}}
    for n in n_list:
        v = np.array(resultados["LDA"][n]);  resumen["LDA"][n] = (v.mean(), v.std(ddof=1))
        v = np.array(resultados["QDA"][n]);  resumen["QDA"][n] = (v.mean(), v.std(ddof=1))
        for k in k_list:
            v = np.array(resultados["kNN"][(n,k)])
            resumen["kNN"][(n,k)] = (v.mean(), v.std(ddof=1))

    return resultados, resumen