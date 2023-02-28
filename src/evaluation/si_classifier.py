import click
import numpy as np
import pandas as pd
import torch
import joblib
import os
from omegaconf import DictConfig
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, balanced_accuracy_score
from tqdm import tqdm

from src.data.data_modules.semantic_module import SemanticDataModule, EnableTestData
from src import settings
from src.data.utils import get_label_mapping, IGNORE_CLASSES

here = Path(__file__)
np.random.seed(100)

LABELS = [int(k) for k, i in settings.mapping.items() if i in settings.organ_labels and i not in IGNORE_CLASSES]


def load_data(target: str = 'val'):
    conf = dict(
        batch_size=100,
        shuffle=False,
        num_workers=1,
        normalization='standardize',
        data=dict(mean_a=None, std_a=None, mean_b=None, std_b=None),  # this is handled internally by the loader
        noise_aug=False,
        noise_aug_level=0
    )
    conf = DictConfig(conf)
    dm = SemanticDataModule(experiment_config=conf, target='sampled')
    dm.ignore_classes = IGNORE_CLASSES
    dm.setup(stage='eval')
    tdl = dm.train_dataloader()
    # data_a refers to simulations while data_b refers to real data
    # load real data from target data set
    with EnableTestData(dm):
        if target == 'test':
            test_dl = dm.test_dataloader()
        elif target == 'val':
            test_dl = dm.val_dataloader()
        else:
            raise ValueError(f"unknown target {target}")
    tdata_b = test_dl.dataset.data_b
    tdata_b = ((tdata_b / torch.linalg.norm(tdata_b, ord=2, dim=1).unsqueeze(dim=1)) - test_dl.dataset.exp_config.data["mean_b"]) / test_dl.dataset.exp_config.data["std_b"]
    tseg_b = test_dl.dataset.seg_data_b
    n_samples = tdata_b.shape[0]
    # load synthetic data from target data set
    tdata_a = test_dl.dataset.data_a
    tdata_a = ((tdata_a / torch.linalg.norm(tdata_a, ord=2, dim=1).unsqueeze(dim=1)) - test_dl.dataset.exp_config.data["mean_a"]) / test_dl.dataset.exp_config.data["std_a"]
    tseg_a = test_dl.dataset.seg_data_a
    index = np.random.choice(np.arange(tdata_a.shape[0]), size=n_samples, replace=False)
    tdata_a = tdata_a[index]
    tseg_a = tseg_a[index]

    # load real data from train data set
    data_b = tdl.dataset.data_b
    data_b = ((data_b / torch.linalg.norm(data_b, ord=2, dim=1).unsqueeze(dim=1)) - tdl.dataset.exp_config.data["mean_b"]) / tdl.dataset.exp_config.data["std_b"]
    seg_b = tdl.dataset.seg_data_b
    index = np.random.choice(np.arange(data_b.shape[0]), size=n_samples, replace=False)
    data_b = data_b[index]
    seg_b = seg_b[index]

    # load synthetic data adapted with INNs
    folder = settings.results_dir / 'gan_cinn_hsi/2023_02_17_19_27_05/testing/generated_spectra_data'
    files = list(folder.glob('*.npz'))
    data_c = []
    seg_c = []
    for file in files:
        tmp_data = np.load(file, allow_pickle=True)
        x = torch.tensor(tmp_data['spectra_ab'])
        # spectra adapted from synthetic to real should be normalized with the statistics of the real data set
        x = ((x / torch.linalg.norm(x, ord=2, dim=1).unsqueeze(dim=1)) - tdl.dataset.exp_config.data["mean_b"]) / tdl.dataset.exp_config.data["std_b"]
        y = tmp_data['seg_a']
        selector = np.any([y == i for i in LABELS], axis=0)
        y = torch.tensor(y)
        x = x[selector]
        y = y[selector]
        data_c.append(x)
        seg_c.append(y)
    data_c = torch.concatenate(data_c, dim=0)
    seg_c = torch.concatenate(seg_c)
    index = np.random.choice(np.arange(data_c.shape[0]), size=n_samples, replace=True)
    data_c = data_c[index]
    seg_c = seg_c[index]

    # load UNIT results
    folder = settings.results_dir / 'unit' / 'generated_spectra_data'
    files = list(folder.glob('*.npz'))
    data_d = []
    seg_d = []
    for file in files:
        tmp_data = np.load(file, allow_pickle=True)
        x = torch.tensor(tmp_data['spectra_ab'])
        # spectra adapted from synthetic to real should be normalized with the statistics of the real data set
        x = ((x / torch.linalg.norm(x, ord=2, dim=1).unsqueeze(dim=1)) - tdl.dataset.exp_config.data["mean_b"]) / \
            tdl.dataset.exp_config.data["std_b"]
        y = tmp_data['seg_a']
        selector = np.any([y == i for i in LABELS], axis=0)
        y = torch.tensor(y)
        x = x[selector]
        y = y[selector]
        data_d.append(x)
        seg_d.append(y)
    data_d = torch.concatenate(data_d, dim=0)
    seg_d = torch.concatenate(seg_d)
    index = np.random.choice(np.arange(data_d.shape[0]), size=n_samples, replace=True)
    data_d = data_d[index]
    seg_d = seg_d[index]

    assert seg_b.size() == tseg_a.size() == seg_c.size() == seg_d.size() == tseg_b.size(), 'missmatch in data set sizes'
    results = dict(train=dict(x_real=data_b, y_real=seg_b, x_sampled=tdata_a, y_sampled=tseg_a,
                              x_adapted_inn=data_c, y_adapted_inn=seg_c, x_unit=data_d, y_unit=seg_d),
                   test=dict(x_real=tdata_b, y_real=tseg_b))
    return results


def get_model(x: np.ndarray, y: np.ndarray, **kwargs):
    model = RandomForestClassifier(**kwargs)
    model.fit(x, y)
    return model


def eval_classification(target: str):
    stages = [
        'adapted_inn',  # train model on synthetic test set adapted to real domain via INNs
        'real',  # train model on real train set sub sampled to have same size as test set
        'sampled',  # train model on synthetic test set generated by sampling wavelengths
        'unit'  # train model on synthetic test data set generated by integrating filter responses
    ]
    mapping = get_label_mapping()
    labels = [int(k) for k in mapping]
    names = [mapping[str(k)] for k in labels]
    data = load_data(target=target)
    for stage in tqdm(stages, desc="iterating stages"):
        train_data = data.get('train').get(f'x_{stage}')
        train_labels = data.get('train').get(f'y_{stage}')
        model = get_model(train_data, train_labels, n_jobs=-1, n_estimators=10, random_state=13)
        # compute score on test set of real data
        test_data = data.get('test').get('x_real')
        test_labels = data.get('test').get('y_real')
        score = model.score(test_data, test_labels)
        print(f'score {stage}: {score}')
        y_pred = model.predict(test_data)
        y_proba = model.predict_proba(test_data)
        report = classification_report(test_labels, y_pred, target_names=names, labels=labels, output_dict=True)

        print(f"balanced_accuracy_score {stage} {balanced_accuracy_score(y_true=test_labels, y_pred=y_pred)}")
        print(f"roc_auc_score {stage} {roc_auc_score(y_true=test_labels, y_score=y_proba, multi_class='ovr')}")

        results = pd.DataFrame(report)
        save_dir_path = settings.results_dir / 'rf'
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path, exist_ok=True)
        results.to_csv(settings.results_dir / 'rf' / f'rf_classifier_report_{stage}.csv', index=True)

        matrix = confusion_matrix(test_labels, y_pred, labels=labels, normalize='true')
        np.save(str(settings.results_dir / 'rf' / f'rf_classifier_train_x_{stage}.npy'), train_data)
        np.save(str(settings.results_dir / 'rf' / f'rf_classifier_train_y_{stage}.npy'), train_labels)
        np.save(str(settings.results_dir / 'rf' / f'rf_classifier_test_x_{stage}.npy'), test_data)
        np.save(str(settings.results_dir / 'rf' / f'rf_classifier_test_y_{stage}.npy'), test_labels)
        np.save(str(settings.results_dir / 'rf' / f'rf_classifier_test_y_pred_{stage}.npy'), y_pred)
        np.save(str(settings.results_dir / 'rf' / f'rf_classifier_test_y_proba_{stage}.npy'), y_proba)
        np.savez(str(settings.results_dir / 'rf' / f'rf_classifier_matrix_{stage}.npz'), matrix=matrix, labels=labels)
        joblib.dump(model, str(settings.results_dir / 'rf' / f'rf_classifier_{stage}.joblib'))


@click.command()
@click.option('--rf', is_flag=True, help="evaluate random forest classifier")
@click.option('--target', type=str, default='val', help="target data set used to compute classification results")
def main(rf: bool, target: str):
    if rf:
        eval_classification(target=target)


if __name__ == '__main__':
    main()
