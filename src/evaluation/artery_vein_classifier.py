import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import os
import glob
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from tqdm import tqdm

from src import settings
from src.utils.gather_pa_spectra_from_dataset import calculate_mean_spectrum

here = Path(__file__)
np.random.seed(141)


def load_data():

    test_real_data_root = "/home/kris/Work/Data/domain_adaptation_simulations/min_max_preprocessed_data_sqrt_ms/real_images/test"
    val_real_data_root = "/home/kris/Work/Data/domain_adaptation_simulations/min_max_preprocessed_data_sqrt_ms/real_images/validation"
    train_real_data_root = "/home/kris/Work/Data/domain_adaptation_simulations/min_max_preprocessed_data_sqrt_ms/real_images/training"
    sim_root = "/home/kris/Work/Data/domain_adaptation_simulations/min_max_preprocessed_data_sqrt_ms/good_simulations/test"
    gan_cinn_root = "/home/kris/Work/Data/DA_results/gan_cinn/2023_01_23_22_47_44/testing/training"
    unit_root = "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/unit/2023_01_21_20_14_58/testing/training"

    test_real_data = glob.glob(os.path.join(test_real_data_root, "*.npz"))
    val_real_data = glob.glob(os.path.join(val_real_data_root, "*.npz"))
    train_real_data = glob.glob(os.path.join(train_real_data_root, "*.npz"))
    sim_data = glob.glob(os.path.join(sim_root, "*.npz"))
    gan_cinn_data = glob.glob(os.path.join(gan_cinn_root, "*.npz"))
    unit_data = glob.glob(os.path.join(unit_root, "*.npz"))

    test_real_spectra = calculate_mean_spectrum(test_real_data)
    val_real_spectra = calculate_mean_spectrum(val_real_data)
    train_real_spectra = calculate_mean_spectrum(train_real_data)
    sim_spectra = calculate_mean_spectrum(sim_data[:len(test_real_data)])
    gan_cinn_spectra = calculate_mean_spectrum(gan_cinn_data[:len(test_real_data)])
    unit_spectra = calculate_mean_spectrum(unit_data[:len(test_real_data)])

    datasets = {
        "test_real_spectra": test_real_spectra,
        "val_real_spectra": val_real_spectra,
        "train_real_spectra": train_real_spectra,
        "sim_spectra": sim_spectra,
        "gan_cinn_spectra": gan_cinn_spectra,
        "unit_spectra": unit_spectra,
    }

    for key, value in datasets.items():
        labels = np.concatenate([
            np.ones(len(value["artery_spectra_all"])),
            np.zeros(len(value["vein_spectra_all"]))
        ])

        spectra = np.concatenate([
            value["artery_spectra_all"],
            value["vein_spectra_all"]
        ])

        indices = np.arange(len(labels))
        np.random.shuffle(indices)

        spectra, labels = spectra[indices], labels[indices]

        datasets[key] = (spectra, labels)

    results = dict(train=dict(x_real=datasets["train_real_spectra"][0], y_real=datasets["train_real_spectra"][1],
                              x_simulated=datasets["sim_spectra"][0], y_simulated=datasets["sim_spectra"][1],
                              x_gan_cinn=datasets["gan_cinn_spectra"][0], y_gan_cinn=datasets["gan_cinn_spectra"][1],
                              x_unit=datasets["unit_spectra"][0], y_unit=datasets["unit_spectra"][1]),
                   test=dict(x_real=datasets["test_real_spectra"][0], y_real=datasets["test_real_spectra"][1]))

    return results


def get_model(x: np.ndarray, y: np.ndarray, **kwargs):
    model = RandomForestClassifier(**kwargs)
    model.fit(x, y)
    return model


def eval_classification():
    stages = [
        'unit',  # train model on synthetic test set adapted to real domain via UNIT
        'gan_cinn',  # train model on synthetic test set adapted to real domain via INNs
        'real',  # train model on real train set sub sampled to have same size as test set
        'simulated',  # train model on synthetic test set generated by sampling wavelengths
    ]
    mapping = {0: "vein", 1: "artery"}
    labels = [int(k) for k in mapping]
    names = [mapping[k] for k in labels]
    data = load_data()
    for stage in tqdm(stages, desc="iterating stages"):
        train_data = data.get('train').get(f'x_{stage}')
        train_labels = data.get('train').get(f'y_{stage}')
        model = get_model(train_data, train_labels, n_jobs=-1, n_estimators=100)
        # compute score on test set of real data
        test_data = data.get('test').get('x_real')
        test_labels = data.get('test').get('y_real')
        score = model.score(test_data, test_labels)
        print(f'score {stage}: {score}')
        y_pred = model.predict(test_data)
        y_proba = model.predict_proba(test_data)
        report = classification_report(test_labels, y_pred, target_names=names, labels=labels, output_dict=True)
        print(report)
        ConfusionMatrixDisplay.from_predictions(test_labels, y_pred=y_pred, labels=labels, display_labels=names, normalize="true")
        plt.title(f"{stage}")
        plt.tight_layout()
        plt.show()

        results = pd.DataFrame(report)
        save_dir_path = settings.results_dir / 'rf_pa'
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path, exist_ok=True)
        results.to_csv(settings.results_dir / 'rf_pa' / f'rf_pa_classifier_report_{stage}.csv', index=True)

        matrix = confusion_matrix(test_labels, y_pred, labels=labels, normalize='pred')
        np.save(str(settings.results_dir / 'rf_pa' / f'rf_pa_classifier_train_x_{stage}.npy'), train_data)
        np.save(str(settings.results_dir / 'rf_pa' / f'rf_pa_classifier_train_y_{stage}.npy'), train_labels)
        np.save(str(settings.results_dir / 'rf_pa' / f'rf_pa_classifier_test_x_{stage}.npy'), test_data)
        np.save(str(settings.results_dir / 'rf_pa' / f'rf_pa_classifier_test_y_{stage}.npy'), test_labels)
        np.save(str(settings.results_dir / 'rf_pa' / f'rf_pa_classifier_test_y_pred_{stage}.npy'), y_pred)
        np.save(str(settings.results_dir / 'rf_pa' / f'rf_pa_classifier_test_y_proba_{stage}.npy'), y_proba)
        np.savez(str(settings.results_dir / 'rf' / f'rf_classifier_matrix_{stage}.npz'), matrix=matrix, labels=labels)
        joblib.dump(model, str(settings.results_dir / 'rf_pa' / f'rf_pa_classifier_{stage}.joblib'))


@click.command()
@click.option('--rf_pa', is_flag=True, help="evaluate random forest classifier")
def main(rf_pa: bool):
    if rf_pa:
        eval_classification()


if __name__ == '__main__':
    main()
