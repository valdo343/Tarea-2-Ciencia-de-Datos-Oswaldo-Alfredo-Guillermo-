from ._functions import experiment_bucle, summarize_data
from ._plots import plot_L_vs_n, plot_knn_vs_k, plot_gap_vs_n, plot_scatter_vs_bayes

import pandas as pd
import numpy as np

class SimulationRunner:
    """
    This class will run the simulations for the given scenarios and store the results.
    """
    def __init__(self, escenarios, n_list, k_list, R=20, seed=16967):
        self.escenarios = escenarios
        self.n_list = n_list
        self.k_list = k_list
        self.R = R
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.df_results = pd.DataFrame()
        self.all_results = {}   # guarda resultados crudos por escenario
        self.all_summaries = {} # guarda resúmenes por escenario

    def run(self):
        for esc in self.escenarios:
            # aquí llamamos a tus funciones globales
            resultados, resumen = experiment_bucle(
                scenario_name=esc,
                n_list=self.n_list,
                k_list=self.k_list,
                R=self.R,
                rng=self.rng
            )

            meta = resultados["meta"]
            print(f"\n=== Escenario: {esc} | priors={meta['priors']} | L*(Bayes)≈ {meta['L_bayes']:.4f} ===")

            # imprimir tabla LDA/QDA
            for n, (m,s) in resumen["LDA"].items():
                mq, sq = resumen["QDA"][n]
                print(f" n={n:>3} | LDA: {m:.3f} ± {s:.3f} | QDA: {mq:.3f} ± {sq:.3f}")

            # imprimir fila de kNN (ejemplo n=200)
            n_show = 200
            row = " kNN (n=200): " + "  ".join(
                [f"k={k}: {resumen['kNN'][(n_show,k)][0]:.3f}±{resumen['kNN'][(n_show,k)][1]:.3f}"
                 for k in meta["k_list"]]
            )
            print(row)

            # plots (funciones globales)
            plot_L_vs_n(resumen, meta)
            plot_knn_vs_k(resumen, meta)
            plot_gap_vs_n(resumen, meta)
            plot_scatter_vs_bayes(resumen, meta, pad=0.01, jitter=0.002)

            # guardar resultados y resúmenes
            self.all_results[esc] = resultados
            self.all_summaries[esc] = resumen

            aux = summarize_data(resumen)
            aux['scenario'] = esc
            self.df_results = pd.concat([self.df_results, aux], ignore_index=True)

    def save_csv(self, filename="resultados_tarea3.csv"):
        self.df_results.to_csv(filename, index=False)
        print(f"Resultados guardados en {filename}")
