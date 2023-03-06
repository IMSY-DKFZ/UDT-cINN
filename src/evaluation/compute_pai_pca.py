import click
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
import joblib

from src import settings
from src.utils.susi import ExperimentResults
from src.utils.gather_pa_spectra_from_dataset import calculate_mean_spectrum


def load_data():
    sources = [
        ('simulated', "/home/kris/Work/Data/domain_adaptation_simulations/min_max_preprocessed_data_sqrt_ms/good_simulations/test"),
        ('real', '/home/kris/Work/Data/domain_adaptation_simulations/min_max_preprocessed_data_sqrt_ms/real_images/test'),
        ('inn', '/home/kris/Work/Data/DA_results/Ablation_Study/PAI/Domain_and_Tissue_labels_as_conditioning/cINN/2023_02_28_19_27_48/testing/training'),
        ('unit', '/home/kris/Work/Data/DA_results/Ablation_Study/PAI/Domain_and_Tissue_labels_as_conditioning/UNIT/2023_02_28_19_27_12/testing/training'),
    ]
    results = {}
    for name, path in sources:
        files = list(Path(path).glob('*.npz'))
        data = calculate_mean_spectrum(files)
        tissue_data = {'artery': data.get('artery_spectra_all'),
                       'vein': data.get('vein_spectra_all')}
        results[name] = tissue_data
    return results


def compute_pca():
    data = load_data()
    results = ExperimentResults()
    tissues = ['artery', 'vein']
    targets = list(data.keys())
    for tissue in tissues:
        pca = PCA(n_components=2)
        pca.fit(data['real'][tissue])
        for target in targets:
            x = data[target][tissue]
            pcs = pca.transform(x)
            results.append(name="data", value=[target for _ in range(pcs.shape[0])])
            results.append(name="tissue", value=[tissue for _ in range(pcs.shape[0])])
            results.append(name="pc_1", value=pcs[:, 0])
            results.append(name="pc_2", value=pcs[:, 1])
            joblib.dump(pca, str(settings.results_dir / 'pca' / f'pai_pca_{tissue}.joblib'))
    results = results.get_df()
    results.to_csv(settings.results_dir / 'pca' / f'pai_pca.csv', index=False)


@click.command()
@click.option('--pca', is_flag=True, help="compute principal component analysis of tissue structures")
def main(pca):
    if pca:
        compute_pca()


if __name__ == '__main__':
    main()
