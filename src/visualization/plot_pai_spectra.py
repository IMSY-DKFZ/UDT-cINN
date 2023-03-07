import click
import numpy as np
import pandas as pd
from pathlib import Path

from src import settings
from src.utils.susi import ExperimentResults
from src.utils.gather_pa_spectra_from_dataset import calculate_mean_spectrum
from src.visualization.plot import line
from src.visualization.templates import cmap_qualitative, cmap_qualitative_diff


def load_data() -> pd.DataFrame:
    sources = [
        ('simulated', "/home/kris/Work/Data/domain_adaptation_simulations/min_max_preprocessed_data_sqrt_ms/good_simulations/test"),
        ('real', '/home/kris/Work/Data/domain_adaptation_simulations/min_max_preprocessed_data_sqrt_ms/real_images/test'),
        ('inn', '/home/kris/Work/Data/DA_results/Ablation_Study/PAI/Domain_and_Tissue_labels_as_conditioning/cINN/2023_02_28_19_27_48/testing/training'),
        ('unit', '/home/kris/Work/Data/DA_results/Ablation_Study/PAI/Domain_and_Tissue_labels_as_conditioning/UNIT/2023_02_28_19_27_12/testing/training'),
    ]
    results = ExperimentResults()
    for name, path in sources:
        files = list(Path(path).glob('*.npz'))
        data = calculate_mean_spectrum(files)
        tissue_data = [('artery', data.get('artery_spectra_all')), ('vein', data.get('vein_spectra_all'))]
        for tissue, x in tissue_data:
            results.append(name="pai_signal", value=x.flatten())
            results.append(name="wavelength", value=np.tile(settings.pai_wavelengths, x.shape[0]))
            results.append(name="data", value=[name for _ in x.flatten()])
            results.append(name="tissue", value=[tissue for _ in x.flatten()])
    # %%
    df = results.get_df()
    return df


def plot_spectra():
    df = load_data()
    fig, _ = line(data_frame=df,
                  x="wavelength",
                  y="pai_signal",
                  facet_col="tissue",
                  color="data",
                  template="plotly_white",
                  facet_col_spacing=0.05,
                  color_discrete_map=cmap_qualitative,
                  )
    fig.update_layout(font=dict(size=12, family="Libertinus Serif"))
    fig.update_xaxes(title_font=dict(size=12, family="Libertinus Serif"))
    fig.update_yaxes(title_font=dict(size=12, family="Libertinus Serif"))
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.write_html(settings.figures_dir / 'pai_signal.html')
    fig.write_image(settings.figures_dir / 'pai_signal.pdf')
    fig.write_image(settings.figures_dir / 'pai_signal.png', scale=2)
    df.to_csv(settings.figures_dir / 'pai_signal.csv', index=False)


def plot_diff():
    df = load_data()
    df_real = df[df.data == 'real'].copy().groupby(['tissue', 'wavelength'], as_index=True).pai_signal.median()
    df_simulated = df[df.data == 'simulated'].copy().groupby(['tissue', 'wavelength'], as_index=True).pai_signal.median()
    df_inn = df[df.data == 'inn'].copy().groupby(['tissue', 'wavelength'], as_index=True).pai_signal.median()
    df_unit = df[df.data == 'unit'].copy().groupby(['tissue', 'wavelength'], as_index=True).pai_signal.median()

    diff_simulated = ((df_real - df_simulated).abs()).reset_index().dropna()
    diff_simulated['source'] = 'real - simulated'
    diff_inn = ((df_real - df_inn).abs()).reset_index().dropna()
    diff_inn['source'] = 'real - inn'
    diff_unit = ((df_real - df_unit).abs()).reset_index().dropna()
    diff_unit['source'] = 'real - unit'

    diff_df = pd.concat([diff_simulated, diff_unit, diff_inn], sort=True, ignore_index=True, axis=0)
    diff_df.rename({'pai_signal': 'difference'}, axis=1, inplace=True)
    fig, _ = line(data_frame=diff_df,
                  x="wavelength",
                  y="difference",
                  color="source",
                  facet_col="tissue",
                  color_discrete_map=cmap_qualitative_diff
                  )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(font_size=16, font_family='Libertinus Serif')
    fig.write_html(settings.figures_dir / 'pai_diff.html')
    fig.write_image(settings.figures_dir / 'pai_diff.pdf')
    fig.write_image(settings.figures_dir / 'pai_diff.png')
    diff_df.to_csv(settings.figures_dir / 'pai_diff.csv')


@click.command()
@click.option('--spectra', is_flag=True, help="plot PA signal for different tissue structures and methods")
@click.option('--diff', is_flag=True, help="plot PA signal differences for different methods")
def main(spectra: bool, diff: bool):
    if spectra:
        plot_spectra()
    if diff:
        plot_diff()


if __name__ == '__main__':
    main()
