import sys
sys.path.append('../../')

import pandas as pd
from utils.plots import plot_curves, plot_multicurves
from utils.enums import Datas, ModelTypes, Sets, main_data, teste_size
from utils.checkpoints import loadCheckPointData
from report_results.metrics_classification import evaluate_classification

path_reports = f"../../report_results/{Datas.HAR.value}/{main_data.value}_{teste_size}/"

ck = loadCheckPointData(path="../../best_models/", file_name=f"backbone_{main_data.value}", device="cpu")
print("Best Acuracy: ", ck['accuracy'])

# df = pd.read_csv(f"{path_reports}errors_{ModelTypes.PRETEXT.value}.dat", sep=" ")
# plot_curves(
#     data1=df[Sets.TRAIN.value], 
#     data2=df[Sets.VALIDATION.value], 
#     data1_legend="Treino", 
#     data2_legend="Validação", 
#     title="Treino x Validação"
# )

# df = pd.read_csv(f"{path_reports}predictions_{ModelTypes.PRETEXT.value}.dat", sep=" ")
# metrics = evaluate_classification(
#     y_pred= df[Sets.PREDICTION.value],
#     y_true= df[Sets.REAL.value]
# )

# for key, item in metrics.items():
#     if not key == "Matriz de Confusão":
#         print(f"{key}: {item}")
#     else:
#         print(f"{key}:\n{item}")

ck = loadCheckPointData(path="../../best_models/", file_name=f"model_{main_data.value}_{ModelTypes.DOWNSTREAM.value}", device="cpu")
print("Best Acuracy: ", ck['accuracy'])

# df = pd.read_csv(f"{path_reports}errors_{ModelTypes.DOWNSTREAM.value}.dat", sep=" ")
# plot_curves(
#     data1=df[Sets.TRAIN.value], 
#     data2=df[Sets.VALIDATION.value], 
#     data1_legend="Treino", 
#     data2_legend="Validação", 
#     title="Treino x Validação"
# )

# df = pd.read_csv(f"{path_reports}predictions_{ModelTypes.DOWNSTREAM.value}.dat", sep=" ")
# metrics = evaluate_classification(
#     y_pred= df[Sets.PREDICTION.value],
#     y_true= df[Sets.REAL.value]
# )

# for key, item in metrics.items():
#     if not key == "confusion_mat":
#         print(f"{key}: {item}")
#     else:
#         print(f"{key}:\n{item}")

percents = [1, 20, 40, 60, 80, 100]

main_path = "../../report_results/har/percent"
path_cnn = f"{main_path}/cnn"
path_var = f"{main_path}/vae"

results_cnn_bk = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1_score': []
}
results_cnn = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1_score': []
}
results_var_bk = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1_score': []
}
results_var = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1_score': []
}

for percent in percents:
    cnn_bk = pd.read_csv(f"{path_cnn}/predictions_{percent}_m1.dat", sep=" ")
    cnn = pd.read_csv(f"{path_cnn}/predictions_{percent}_m2.dat", sep=" ")

    var_bk = pd.read_csv(f"{path_var}/predictions_{percent}_m1.dat", sep=" ")
    var = pd.read_csv(f"{path_var}/predictions_{percent}_m2.dat", sep=" ")

    metrics_cnn_bk = evaluate_classification(
        y_pred= cnn_bk[Sets.PREDICTION.value],
        y_true= cnn_bk[Sets.REAL.value]
    )
    metrics_cnn = evaluate_classification(
        y_pred= cnn[Sets.PREDICTION.value],
        y_true= cnn[Sets.REAL.value]
    )
    metrics_var_bk = evaluate_classification(
        y_pred= var_bk[Sets.PREDICTION.value],
        y_true= var_bk[Sets.REAL.value]
    )
    metrics_var = evaluate_classification(
        y_pred= var[Sets.PREDICTION.value],
        y_true= var[Sets.REAL.value]
    )

    for key in results_cnn.keys():
        results_cnn[key].append(metrics_cnn[key])   
        results_var[key].append(metrics_var[key])
        results_cnn_bk[key].append(metrics_cnn_bk[key])
        results_var_bk[key].append(metrics_var_bk[key])

for key in results_cnn.keys():
    plot_multicurves(
        title=f"{key} performance",
        x_label="database_percent",
        y_label=key,
        x=percents,
        path=f"{main_path}/plots",
        datas=[results_cnn[key], results_cnn_bk[key], results_var[key], results_var_bk[key]],
        legends=["CNN", "CNN_pre-trained", "VAE", "VAE_pre-trained"]
    )
