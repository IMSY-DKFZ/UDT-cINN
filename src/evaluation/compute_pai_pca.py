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
        ('simulated', '/media/menjivar/Extreme SSD/MICCAI_23/results/photoacoustics/good_simulations/test'),
        ('real', '/media/menjivar/Extreme SSD/MICCAI_23/results/photoacoustics/real_images/validation'),
        ('inn', '/media/menjivar/Extreme SSD/MICCAI_23/results/photoacoustics/2023_01_23_22_47_44/testing/training'),
        ('unit', '/media/menjivar/Extreme SSD/MICCAI_23/results/photoacoustics/2023_02_18_15_05_06/testing/training'),
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
