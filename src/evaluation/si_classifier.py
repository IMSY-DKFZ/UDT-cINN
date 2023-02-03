import json
import click
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

from src.data.data_modules.semantic_module import SemanticDataModule
from src import settings

here = Path(__file__)


def get_label_mapping():
    with open(str(settings.intermediates_dir / 'semantic' / 'mapping.json'), 'rb') as handle:
        content = json.load(handle)
    with open(str(here.parent.parent / 'data' / 'semantic_organ_labels.json'), 'rb') as handle:
        organ_labels = json.load(handle)['organ_labels']
    content = {k: i for k, i in content.items() if i in organ_labels}
    return content


def load_data(target: str):
    conf = dict(
        batch_size=100,
        shuffle=False,
        num_workers=1,
        normalization='standardize',
        data=dict(mean=0.1, std=0.1),
        target=target
    )
    conf = DictConfig(conf)
    dm = SemanticDataModule(experiment_config=conf)
    dm.setup(stage='eval')
    tdl = dm.train_dataloader()
    data = tdl.dataset.data
    seg = tdl.dataset.seg_data
    vdl = dm.val_dataloader()
    vdata = vdl.dataset.data
    vseg = vdl.dataset.seg_data
    results = dict(train=dict(x=data, y=seg), val=dict(x=vdata, y=vseg))
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
        model = get_model(data.get('train').get('x'), data.get('train').get('y'), n_jobs=-1 ,n_estimators=10)
        score = model.score(data.get('val').get('x'), data.get('val').get('y'))
        print(f'score {target}: {score}')
        y_pred = model.predict(data.get('val').get('x'))
        report = classification_report(data.get('val').get('y'), y_pred, target_names=names, labels=labels, output_dict=True)

        results = pd.DataFrame(report)
        results.to_csv(settings.results_dir / f'rf_classifier_report_{target}.csv', index=True)

        matrix = confusion_matrix(data.get('val').get('y'), y_pred, labels=labels, normalize='pred')
        np.save(str(settings.results_dir / f'rf_classifier_matrix_{target}.npy'), matrix)
        np.save(str(settings.results_dir / f'rf_classifier_y_val_{target}.npy'), data.get('val').get('y'))
        np.save(str(settings.results_dir / f'rf_classifier_y_val_pred_{target}.npy'), y_pred)


@click.command()
@click.option('--rf', is_flag=True, help="evaluate random forest classifier")
@click.option('-t', '--targets', multiple=True, default=['real', 'synthetic'], help="target datasets, to evaluate")
def main(rf: bool, targets: list):
    if rf:
        eval_classification(targets=targets)


if __name__ == '__main__':
    main()
