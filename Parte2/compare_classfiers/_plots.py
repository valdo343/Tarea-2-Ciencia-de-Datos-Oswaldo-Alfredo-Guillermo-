import matplotlib.pyplot as plt
import numpy as np


def plot_L_vs_n(resumen:dict, meta:dict):
    """This function will plot the estimated risk L(g) vs n for LDA and QDA.

    Args:
        resumen (dict): Summary of the results (mean and std)
        meta (dict): Metadata of the experiment
    """
    n_list = meta["n_list"]

    plt.figure()
    # LDA
    lda_means = [resumen["LDA"][n][0] for n in n_list]
    lda_stds  = [resumen["LDA"][n][1] for n in n_list]
    plt.errorbar(n_list, lda_means, yerr=lda_stds, marker='o', label="LDA")

    # QDA
    qda_means = [resumen["QDA"][n][0] for n in n_list]
    qda_stds  = [resumen["QDA"][n][1] for n in n_list]
    plt.errorbar(n_list, qda_means, yerr=qda_stds, marker='s', label="QDA")
    # Fisher 1D
    if "Fisher1D" in resumen:
        fisher_means = [resumen["Fisher1D"][n][0] for n in n_list]
        fisher_stds  = [resumen["Fisher1D"][n][1] for n in n_list]
        plt.errorbar(n_list, fisher_means, yerr=fisher_stds, marker='^', label="Fisher1D")
    # Naive Bayes
    if "NaiveBayes" in resumen:
        gnb_means = [resumen["NaiveBayes"][n][0] for n in n_list]
        gnb_stds  = [resumen["NaiveBayes"][n][1] for n in n_list]
        plt.errorbar(n_list, gnb_means, yerr=gnb_stds, marker='x', label="Naive Bayes")

    plt.axhline(meta["L_bayes"], color="k", linestyle="--", label="Bayes")
    plt.xlabel("n (por clase)")
    plt.ylabel("Riesgo estimado L(g)")
    plt.title(f"L(g) vs n | Escenario: {meta['scenario']}")
    plt.legend()
    plt.show()


def plot_knn_vs_k(resumen:dict, meta:dict):
    """This function will plot the estimated risk L(kNN) vs k for different n.

    Args:
        resumen (dict): Summary of the results (mean and std)
        meta (dict): Metadata of the experiment
    """
    
    k_list = meta["k_list"]
    n_list = meta["n_list"]

    plt.figure()
    for n in n_list:
        means = [resumen["kNN"][(n,k)][0] for k in k_list]
        stds  = [resumen["kNN"][(n,k)][1] for k in k_list]
        plt.errorbar(k_list, means, yerr=stds, marker='o', label=f"n={n}")
    
    plt.axhline(meta["L_bayes"], color="k", linestyle="--", label="Bayes")
    plt.xlabel("Número de vecinos (k)")
    plt.ylabel("Riesgo estimado L(kNN)")
    plt.title(f"L(kNN) vs k | Escenario: {meta['scenario']}")
    plt.legend()
    plt.show()


def plot_gap_vs_n(resumen:dict, meta:dict):
    """
    This function will plot the gap L(g) - L(Bayes) vs n for LDA and QDA.

    Args:
        resumen (dict): Summary of the results (mean and std)
        meta (dict): Metadata of the experiment
    """

    n_list = meta["n_list"]
    Lb = meta["L_bayes"]

    lda_gap = [resumen["LDA"][n][0] - Lb for n in n_list]
    qda_gap = [resumen["QDA"][n][0] - Lb for n in n_list]
    naive_bayes_gap = [resumen["NaiveBayes"][n][0] - Lb for n in n_list] if "NaiveBayes" in resumen else None
    fisher_gap = [resumen["Fisher1D"][n][0] - Lb for n in n_list] if "Fisher1D" in resumen else None

    plt.figure()
    plt.plot(n_list, lda_gap, marker='o', label="LDA - Bayes")
    plt.plot(n_list, qda_gap, marker='s', label="QDA - Bayes")
    if naive_bayes_gap is not None:
        plt.plot(n_list, naive_bayes_gap, marker='x', label="Naive Bayes - Bayes")
    if fisher_gap is not None:
        plt.plot(n_list, fisher_gap, marker='^', label="Fisher1D - Bayes")
    plt.axhline(0, color="k", linestyle="--")
    plt.xlabel("n")
    plt.ylabel("Brecha de riesgo")
    plt.title(f"Brechas L(g) - L(Bayes) | Escenario: {meta['scenario']}")
    plt.legend()
    plt.show()

def plot_scatter_vs_bayes(resumen:dict, meta:dict, pad:float=0.01,
                          jitter:float=0.0, include_knn:bool=False):
    """This function will plot a scatter plot comparing the estimated risk by CV vs the true Bayes risk.

    Args:
        resumen (dict): Summary of the results (mean and std)
        meta (dict): Metadata of the experiment
        pad (float, optional): Extra margin added when setting the axis limits in the plot. Default is 0.01.
        jitter (float, optional): Amount of random horizontal noise added to the X-axis coordinates (Bayes risk). Default is 0.0 (no jitter).
        include_knn (bool, optional) Whether to include k-NN results in the plot in addition to LDA and QDA. Default is False.
    """
    Lb = meta["L_bayes"]

    xs, ys, labels, markers = [], [], [], []

    # LDA / QDA
    for n,(m,s) in resumen["LDA"].items():
        xs.append(Lb); ys.append(m); labels.append(f"LDA n={n}"); markers.append('o')
    for n,(m,s) in resumen["QDA"].items():
        xs.append(Lb); ys.append(m); labels.append(f"QDA n={n}"); markers.append('s')
    # Fisher 1D
    if "Fisher1D" in resumen:
        for n,(m,s) in resumen["Fisher1D"].items():
            xs.append(Lb); ys.append(m); labels.append(f"Fisher1D n={n}"); markers.append('^')
    # Naive Bayes
    if "NaiveBayes" in resumen:
        for n,(m,s) in resumen["NaiveBayes"].items():
            xs.append(Lb); ys.append(m); labels.append(f"Naive Bayes n={n}"); markers.append('x')

    # (opcional) kNN
    if include_knn:
        for (n,k),(m,s) in resumen["kNN"].items():
            xs.append(Lb); ys.append(m); labels.append(f"kNN k={k} n={n}"); markers.append('^')

    xs = np.array(xs, float)
    ys = np.array(ys, float)

    # Jitter horizontal para separar visualmente puntos con el mismo Lb
    if jitter > 0:
        xs = xs + np.random.uniform(-jitter, jitter, size=len(xs))

    plt.figure()
    for x,y,lab,mark in zip(xs,ys,labels,markers):
        plt.scatter(x, y, marker=mark, label=lab)

    # Línea y=x
    plt.plot([0,1],[0,1], 'k--', label="y=x")

    # ---- Límites automáticos de ejes ----
    y_min, y_max = ys.min(), ys.max()
    # mitad de rango X centrado en Lb = desvío vertical máximo respecto a Lb + pad
    half_span_x = max(abs(y_min - Lb), abs(y_max - Lb)) + pad
    x_left  = max(0.0, Lb - half_span_x)
    x_right = min(1.0, Lb + half_span_x)
    plt.xlim(x_left, x_right)

    # margen vertical
    y_pad = pad
    plt.ylim(max(0.0, y_min - y_pad), min(1.0, y_max + y_pad))

    plt.xlabel("Riesgo verdadero Bayes")
    plt.ylabel("Riesgo estimado CV")
    plt.title(f"Comparación validación vs Bayes | Escenario: {meta['scenario']}")
    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()
