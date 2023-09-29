import os
import warnings
from itertools import cycle
from pathlib import Path
from typing import Any

import click
import joblib
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, balanced_accuracy_score, f1_score

from src import settings
from src.data.data_modules.semantic_module import SemanticDataModule, EnableTestData
from src.data.utils import get_label_mapping, IGNORE_CLASSES
from src.utils.susi import ExperimentResults

here = Path(__file__)
manual_seed = 42
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)

LABELS = [int(k) for k, i in settings.mapping.items() if i in settings.organ_labels and i not in IGNORE_CLASSES]


def balance_class_prevalence(x: torch.Tensor, y: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    creates a new dataset based on `x` and `y`. In the new data set, the prevalence of each class is the same for all
    unique classes in `y`. The samples in x are repeated (randomly), until the number of samples in the class with the
    maximum prevalence is reached.

    :param x: data samples
    :param y: data labels (1D)
    :return: (x_balanced, y_balanced)
    """
    unique_classes = np.unique(y.numpy())
    max_class_size = max([len(x[y == label]) for label in unique_classes])
    class_index = {k: np.random.permutation(np.where(y == k)[0]) for k in unique_classes}
    for k, v in class_index.items():
        class_index[k] = cycle(torch.tensor(v, dtype=torch.int64))
    balanced_ind = []
    data_size = len(unique_classes) * max_class_size
    for k in cycle(unique_classes):
        balanced_ind.append(int(next(class_index[k])))
        if len(balanced_ind) == data_size:
            break
    x_balanced = x[balanced_ind]
    y_balanced = y[balanced_ind]
    for k in unique_classes:
        assert int((y_balanced == k).sum()) == max_class_size, f"imbalanced {k}: {len(y_balanced == k)} != {max_class_size}"
    return x_balanced, y_balanced


def _load_da_results(dm: SemanticDataModule, n_samples: int, results_folder: str, balance_classes: bool) -> (torch.Tensor, torch.Tensor):
    """
    load synthetic adapted data. Teh precise data that is loaded can be defined through `results_folder`.
    Spectra (`spectra_ab`) adapted from domain_a (synthetic) to domain_b (real) is loaded, hence the label of the
    synthetic data is assigned to it.

    :param dm: data module used to query the data statistics used for normalization
    :param n_samples: number of samples randomly selected without repetition from dataset
    :param results_folder: folder where files are stored in `.npy` format
    :return:
    """
    data_stats = dm.train_dataloader().dataset.exp_config.data
    folder = settings.results_dir / results_folder
    files = list(folder.glob('*.npz'))
    data = []
    seg = []
    for file in files:
        tmp_data = np.load(file, allow_pickle=True)
        x = torch.tensor(tmp_data['spectra_ab'])
        # spectra adapted from synthetic to real should be normalized with the statistics of the real data set
        x = ((x / torch.linalg.norm(x, ord=2, dim=1).unsqueeze(dim=1)) - data_stats["mean_b"]) / data_stats["std_b"]
        y = tmp_data['seg_a']
        selector = np.any([y == i for i in LABELS], axis=0)
        y = torch.tensor(y)
        x = x[selector]
        y = y[selector]
        data.append(x)
        seg.append(y)
    data = torch.concatenate(data, dim=0)
    seg = torch.concatenate(seg)
    if balance_classes:
        data, seg = balance_class_prevalence(x=data, y=seg)
    else:
        if n_samples > data.shape[0]:
            index = np.random.choice(np.arange(data.shape[0]), size=data.shape[0], replace=False)
        else:
            index = np.random.choice(np.arange(data.shape[0]), size=n_samples, replace=False)
        data = data[index]
        seg = seg[index]
    return data, seg


def _load_synthetic_data(dm: SemanticDataModule, n_samples: int, target: Any, balance_classes: bool) -> (torch.Tensor, torch.Tensor):
    """
    loads synthetic data (no adaptation) from data module according to target specifier. A random sample of the
    corresponding data set is generated in order to return `n_samples`. This sampling is done without repetition.

    NOTE: The test set of corresponding to the synthetic data splits is loaded

    :param dm: data module used to query the data statistics used for normalization
    :param n_samples: number of samples randomly selected without repetition from dataset
    :param target: dummy value, always the test set of the synthetic data is loaded.
    :return:
    """
    with EnableTestData(dm):
        dl = dm.test_dataloader()
    data = dl.dataset.data_a
    data = ((data / torch.linalg.norm(data, ord=2, dim=1).unsqueeze(dim=1)) - dl.dataset.exp_config.data[
        "mean_a"]) / dl.dataset.exp_config.data["std_a"]
    seg = dl.dataset.seg_data_a
    if balance_classes:
        data, seg = balance_class_prevalence(x=data, y=seg)
    else:
        if n_samples > data.shape[0]:
            index = np.random.choice(np.arange(data.shape[0]), size=data.shape[0], replace=False)
        else:
            index = np.random.choice(np.arange(data.shape[0]), size=n_samples, replace=False)
        data = data[index]
        seg = seg[index]
    return data, seg


def _load_target_real_data(dm: SemanticDataModule, target: str, balance_classes: bool) -> (torch.Tensor, torch.Tensor):
    """
    loads real data from data module according to target specifier. The number of samples in this dataset define the
    number of samples drawn for all other loaders used during classifier training.

    :param dm: data module used to query the data statistics used for normalization
    :param target: defines the data set where the classifier is tested on, options are `val` and `test`
    :return:
    """
    # load real data from target data set
    if target == 'test':
        with EnableTestData(dm):
            test_dl = dm.test_dataloader()
    elif target == 'val':
        test_dl = dm.val_dataloader()
    else:
        raise ValueError(f"unknown target {target}")
    data = test_dl.dataset.data_b
    data = ((data / torch.linalg.norm(data, ord=2, dim=1).unsqueeze(dim=1)) - test_dl.dataset.exp_config.data[
        "mean_b"]) / test_dl.dataset.exp_config.data["std_b"]
    seg = test_dl.dataset.seg_data_b
    if balance_classes:
        data, seg = balance_class_prevalence(x=data, y=seg)
    n_samples = data.shape[0]
    return data, seg, n_samples


def _load_train_clf_real_data(dm: Any, n_samples: int, balance_classes: bool) -> (torch.Tensor, torch.Tensor):
    """
    loads real data that was used to generate the synthetic sampled data set. A random sample of the
    corresponding data set is generated in order to return `n_samples`. This sampling is done without repetition.
    To load this real data, a new data module is used with the target `real_source`.
    Normally real data would be loaded into `*_b` variables, but here we use a trick to load the real source data
    into `*_a` variables.

    :param dm: dummy variable, it is overwritten by correct data module
    :param n_samples: number of samples randomly selected without repetition from dataset
    :return:
    """
    if dm.exp_config.data.choose_spectra == 'sampled':
        choose_spectra = 'real_source'
    elif dm.exp_config.data.choose_spectra == 'unique':
        choose_spectra = 'real_source_unique'
    else:
        raise ValueError(f"unrecognized choose_spectra = {dm.exp_config.data.choose_spectra}")
    conf = dict(
        batch_size=100,
        shuffle=False,
        num_workers=1,
        normalization='standardize',
        data=dict(mean_a=None, std_a=None, mean_b=None, std_b=None, balance_classes=False,
                  dataset_version='semantic_v2', choose_spectra=choose_spectra),
        # data stats loaded internally by loader
        noise_aug=False,
        noise_aug_level=0
    )
    conf = DictConfig(conf)
    dm = SemanticDataModule(experiment_config=conf)
    dm.setup(stage="train")
    with EnableTestData(dm):
        tdl = dm.test_dataloader()
    data_stats = tdl.dataset.exp_config.data
    data = tdl.dataset.data_a
    data = ((data / torch.linalg.norm(data, ord=2, dim=1).unsqueeze(dim=1)) - data_stats["mean_b"]) / data_stats["std_b"]
    seg = tdl.dataset.seg_data_a
    if balance_classes:
        data, seg = balance_class_prevalence(x=data, y=seg)
    else:
        if n_samples > data.shape[0]:
            index = np.random.choice(np.arange(data.shape[0]), size=data.shape[0], replace=False)
        else:
            index = np.random.choice(np.arange(data.shape[0]), size=n_samples, replace=False)
        data = data[index]
        seg = seg[index]
    return data, seg


def load_data(target: str = 'val', balance_classes: bool = False) -> dict:
    """
    loads data that will be used for training and testing a classifier. Given the data structure below, the following
    sets are loaded:

    * `target=val`: (REAL, val), (SYNTHETIC, test), (DOMAIN ADAPTED, test), (REAL2, test)
    * `target=test`: (REAL, test), (SYNTHETIC, test), (DOMAIN ADAPTED, test), (REAL2, test)

     ======= =========== ================ ===================================
      REAL    SYNTHETIC   DOMAIN ADAPTED   REAL (used to generate synthetic)
     ======= =========== ================ ===================================
      train   train       train            train
      val     val         val              val
      test    test        test             test
     ======= =========== ================ ===================================

    :param target: defines the data set where the classifier is tested on, options are `val` and `test`
    :return: dictionary with data on each keyword
    """
    conf = dict(
        batch_size=100,
        shuffle=False,
        num_workers=1,
        normalization='standardize',
        data=dict(mean_a=None, std_a=None, mean_b=None, std_b=None, balance_classes=False,
                  dataset_version='semantic_v2', choose_spectra='unique'),
        # data stats loaded internally by loader
        noise_aug=False,
        noise_aug_level=0
    )
    conf = DictConfig(conf)
    dm = SemanticDataModule(experiment_config=conf)
    dm.ignore_classes = IGNORE_CLASSES
    dm.setup(stage='eval')
    # data_a refers to simulations while data_b refers to real data
    target_data_real, target_seg_real, n_samples = _load_target_real_data(dm=dm, target=target, balance_classes=balance_classes)

    synthetic_data, synthetic_seg = _load_synthetic_data(dm=dm, n_samples=n_samples, target=target, balance_classes=balance_classes)

    synthetic_real_source_data, synthetic_real_source_seg = _load_train_clf_real_data(dm=dm, n_samples=n_samples, balance_classes=balance_classes)

    inn_data, inn_seg = _load_da_results(dm=dm, n_samples=n_samples, results_folder='inn/generated_spectra_data', balance_classes=balance_classes)

    unit_data, unit_seg = _load_da_results(dm=dm, n_samples=n_samples, results_folder='unit/generated_spectra_data', balance_classes=balance_classes)

    for label in np.unique(synthetic_seg):
        n_synthetic = len(synthetic_seg[synthetic_seg == label])
        n_real_source = len(synthetic_real_source_seg[synthetic_real_source_seg == label])
        n_inn = len(inn_seg[inn_seg == label])
        n_unit = len(unit_seg[unit_seg == label])
        if not n_synthetic == n_real_source == n_inn == n_unit:
            warnings.warn(f"found missmatch of class prevalence: {n_synthetic}!={n_real_source}!={n_inn}!={n_unit}")

    if balance_classes:
        size_matches = synthetic_real_source_seg.size() == synthetic_seg.size() == inn_seg.size() == unit_seg.size()
        if not size_matches:
            warnings.warn(f"found data size missmatch: {synthetic_real_source_seg.size()}!={synthetic_seg.size()}!="
                          f"{inn_seg.size()}!={unit_seg.size()}")
    results = dict(train=dict(x_real=synthetic_real_source_data, y_real=synthetic_real_source_seg,
                              x_simulated=synthetic_data, y_simulated=synthetic_seg,
                              x_cINN=inn_data, y_cINN=inn_seg,
                              x_UNIT=unit_data, y_UNIT=unit_seg),
                   test=dict(x_real=target_data_real, y_real=target_seg_real))
    return results


def get_model(x: np.ndarray, y: np.ndarray, **kwargs) -> RandomForestClassifier:
    """
    instantiates a random forest classifier and trains it

    :param x: data used to train classifier
    :param y: labels used for training classifier
    :param kwargs: additional arguments parsed to classifier
    :return: sklearn classifier
    """
    model = RandomForestClassifier(**kwargs)
    model.fit(x, y)
    return model


def eval_classification(target: str, balance_classes: bool):
    """
    evaluates the performance of a classifier trained on different data source and tested on a held out data set.
    The results are stored in the folder defined by `settings.results_dir`

    :param target: defines the data set where the classifier is tested on, options are `val` and `test`
    :return: None
    """
    stages = [
        'real',  # train model on real train set sub sampled to have same size as test set
        'simulated',  # train model on synthetic test set generated by sampling wavelengths
        'UNIT',  # train model on synthetic test data set generated by integrating filter responses
        'cINN',  # train model on synthetic test set adapted to real domain via INNs
    ]
    mapping = get_label_mapping()
    labels = [int(k) for k in mapping]
    names = [mapping[str(k)] for k in labels]
    data = load_data(target=target, balance_classes=balance_classes)
    metrics = ExperimentResults()

    test_data = data.get('test').get('x_real')
    test_labels = data.get('test').get('y_real')

    n_estimators = 100
    calibration = True
    per_class_metrics = False

    # print(f"\nNumber of RF estimators: {n_estimators}")
    # print(f"Calibrate model: {calibration}")

    for stage in stages:
        train_data = data.get('train').get(f'x_{stage}')
        train_labels = data.get('train').get(f'y_{stage}')
        model = get_model(train_data, train_labels, n_jobs=-1, n_estimators=n_estimators)

        # compute score on test set of real data
        if calibration:
            model = CalibratedClassifierCV(model)
            model.fit(train_data, train_labels)
            print("#### With calibration ####")

        y_cal_proba = model.predict_proba(test_data)
        y_cal_pred = model.predict(test_data)

        report = classification_report(test_labels, y_cal_pred, target_names=names, labels=labels, output_dict=True)

        if per_class_metrics:
            conf_matrix = confusion_matrix(test_labels, y_cal_pred)
            per_class_accuracies = list()

            for n_idx, name in enumerate(names):
                true_negatives = np.sum(np.delete(np.delete(conf_matrix, n_idx, axis=0), n_idx, axis=1))
                true_positives = conf_matrix[n_idx, n_idx]
                per_class_accuracies.append((true_positives + true_negatives) / np.sum(conf_matrix))

            per_class_auroc = roc_auc_score(y_true=test_labels, y_score=y_cal_proba, multi_class="ovr", average=None)
            per_class_f1 = f1_score(y_true=test_labels, y_pred=y_cal_pred, average=None)
            metrics.append(name='per_class_accuracy', value=per_class_accuracies)
            metrics.append(name='per_class_auroc', value=per_class_auroc)
            metrics.append(name='per_class_f1', value=per_class_f1)
            metrics.append(name='data', value=[stage for _ in names])
            metrics.append(name='organ', value=names)


            print(f'f1 score {stage}: {per_class_f1:2f}')
            print(f"balanced_accuracy_score {stage}: {per_class_accuracies:2f}")
            print(f"roc_auc_score {stage}: {per_class_auroc:2f}")

        else:
            balanced_accuracy = balanced_accuracy_score(y_true=test_labels, y_pred=y_cal_pred)
            roc_auc = roc_auc_score(y_true=test_labels, y_score=y_cal_proba, multi_class="ovr")
            f1 = f1_score(y_true=test_labels, y_pred=y_cal_pred, average="weighted")
            metrics.append(name='balanced_accuracy', value=float(balanced_accuracy))
            metrics.append(name='auroc', value=float(roc_auc))
            metrics.append(name='f1', value=float(f1))
            metrics.append(name='data', value=stage)

            print(f"balanced_accuracy_score {stage}: {balanced_accuracy:2f}")
            print(f"roc_auc_score {stage}: {roc_auc:2f}")
            print(f'f1 score {stage}: {f1:2f}')

        results = pd.DataFrame(report)
        save_dir_path = settings.results_dir / 'rf'
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path, exist_ok=True)
        results.to_csv(settings.results_dir / 'rf' / f'rf_classifier_report_{stage}.csv', index=True)

        matrix = confusion_matrix(test_labels, y_cal_pred, labels=labels, normalize='true')
        np.save(str(settings.results_dir / 'rf' / f'rf_classifier_train_x_{stage}.npy'), train_data)
        np.save(str(settings.results_dir / 'rf' / f'rf_classifier_train_y_{stage}.npy'), train_labels)
        np.save(str(settings.results_dir / 'rf' / f'rf_classifier_test_x_{stage}.npy'), test_data)
        np.save(str(settings.results_dir / 'rf' / f'rf_classifier_test_y_{stage}.npy'), test_labels)
        np.save(str(settings.results_dir / 'rf' / f'rf_classifier_test_y_pred_{stage}.npy'), y_cal_pred)
        np.save(str(settings.results_dir / 'rf' / f'rf_classifier_test_y_proba_{stage}.npy'), y_cal_proba)
        np.savez(str(settings.results_dir / 'rf' / f'rf_classifier_matrix_{stage}.npz'), matrix=matrix, labels=labels)
        joblib.dump(model, str(settings.results_dir / 'rf' / f'rf_classifier_{stage}.joblib'))

    metrics = metrics.get_df()
    metrics.to_csv(settings.results_dir / 'rf' / f'rf_classifier_metrics.csv', index=True)


@click.command()
@click.option('--rf', is_flag=True, help="evaluate random forest classifier")
@click.option('--target', type=str, default='val', help="target data set used to compute classification results")
@click.option('--balance_classes', is_flag=True, help="balance classes during data loading. For each data source, the "
                                                      "maximum # samples per organ is selected, and the samples for"
                                                      "class are repeated until reaching such maximum #")
def main(rf: bool, target: str, balance_classes: bool):
    """
    evaluates a random forest classifier on the real data, simulation, and simulations adapted to real data. It does
    so taking into account the data splits:

    ======= =========== ================ ===================================
    REAL    SYNTHETIC   DOMAIN ADAPTED   REAL (used to generate synthetic)
    ======= =========== ================ ===================================
    train   train       train            train\n
    val     val         val              val\n
    test    test        test             test\n
    ======= =========== ================ ===================================

    """
    if rf:
        eval_classification(target=target, balance_classes=balance_classes)


if __name__ == '__main__':
    main()
