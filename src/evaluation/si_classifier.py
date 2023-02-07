import click
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

from src.data.data_modules.semantic_module import SemanticDataModule
from src import settings

here = Path(__file__)


def get_label_mapping():
    mapping = settings.mapping
    organ_labels = settings.organ_labels
    content = {k: i for k, i in mapping.items() if i in organ_labels}
    return content


def load_data(target: str):
    conf = dict(
        batch_size=100,
        shuffle=False,
        num_workers=1,
        normalization='standardize',
        data=dict(mean_a=None, std_a=None, mean_b=None, std_b=None), # this is handled internally by the loader
    )
    conf = DictConfig(conf)
    dm = SemanticDataModule(experiment_config=conf, target=target)
    dm.setup(stage='eval')
    tdl = dm.train_dataloader()
    data_a = tdl.dataset.data_a
    data_a = ((data_a / torch.linalg.norm(data_a, ord=2, dim=1).unsqueeze(dim=1)) - tdl.dataset.exp_config.data["mean_a"]) / tdl.dataset.exp_config.data["std_a"]
    data_b = tdl.dataset.data_b
    data_b = ((data_b / torch.linalg.norm(data_b, ord=2, dim=1).unsqueeze(dim=1)) - tdl.dataset.exp_config.data["mean_b"]) / tdl.dataset.exp_config.data["std_b"]
    seg_a = tdl.dataset.seg_data_a
    seg_b = tdl.dataset.seg_data_b
    vdl = dm.val_dataloader()
    vdata_a = vdl.dataset.data_a
    vdata_a = ((vdata_a / torch.linalg.norm(vdata_a, ord=2, dim=1).unsqueeze(dim=1)) - vdl.dataset.exp_config.data["mean_a"]) / vdl.dataset.exp_config.data["std_a"]
    vdata_b = vdl.dataset.data_b
    vdata_b = ((vdata_b / torch.linalg.norm(vdata_b, ord=2, dim=1).unsqueeze(dim=1)) - vdl.dataset.exp_config.data["mean_b"]) / vdl.dataset.exp_config.data["std_b"]
    vseg_a = vdl.dataset.seg_data_a
    vseg_b = vdl.dataset.seg_data_b
    results = dict(train=dict(x_a=data_a, y_a=seg_a, x_b=data_b, y_b=seg_b),
                   val=dict(x_a=vdata_a, y_a=vseg_a, x_b=vdata_b, y_b=vseg_b))
    return results


def get_model(x: np.ndarray, y: np.ndarray, **kwargs):
    model = RandomForestClassifier(**kwargs)
    model.fit(x, y)
    return model


def eval_classification(targets: list):
    for target in targets:
        mapping = get_label_mapping()
        labels = [int(k) for k in mapping]
        names = [mapping[str(k)] for k in labels]
        data = load_data(target=target)
        model = get_model(data.get('train').get('x_a'), data.get('train').get('y_a'), n_jobs=-1, n_estimators=10)
        score = model.score(data.get('val').get('x_b'), data.get('val').get('y_b'))
        print(f'score {target}: {score}')
        y_pred = model.predict(data.get('val').get('x_b'))
        report = classification_report(data.get('val').get('y_b'), y_pred, target_names=names, labels=labels, output_dict=True)

        results = pd.DataFrame(report)
        results.to_csv(settings.results_dir / 'rf' / f'rf_classifier_report_{target}.csv', index=True)

        matrix = confusion_matrix(data.get('val').get('y_b'), y_pred, labels=labels, normalize='pred')
        np.save(str(settings.results_dir / 'rf' / f'rf_classifier_matrix_{target}.npy'), matrix)
        np.save(str(settings.results_dir / 'rf' / f'rf_classifier_y_val_{target}.npy'), data.get('val').get('y_b'))
        np.save(str(settings.results_dir / 'rf' / f'rf_classifier_y_val_pred_{target}.npy'), y_pred)


@click.command()
@click.option('--rf', is_flag=True, help="evaluate random forest classifier")
@click.option('-t', '--targets', multiple=True, default=['sampled', 'adapted'], help="target datasets, to evaluate")
def main(rf: bool, targets: list):
    if rf:
        eval_classification(targets=targets)


if __name__ == '__main__':
    main()
