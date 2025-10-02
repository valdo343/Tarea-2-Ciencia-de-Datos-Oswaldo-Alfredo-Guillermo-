if __name__ == "__main__":
    from compare_classfiers import SimulationRunner
    escenarios = ["lda_optimo", "qda_optimo", "desbalance",
                  "correlacion_fuerte", "medias_cercanas"]
    runner = SimulationRunner(escenarios,
                              n_list=[50,100,200,500],
                              k_list=[1,3,5,11,21],
                              R=20)
    runner.run()
    runner.save_csv("resultados_tarea3.csv")
