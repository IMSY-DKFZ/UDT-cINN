import click

from src.data.multi_layer_loader import SimulationDataLoader
from src.util.susi import adapt_to_camera_reflectance
from src import settings


def adapt_simulations_to_cameras():
    loader = SimulationDataLoader()
    df = loader.get_database(simulation='generic_depth', splits=['train', 'test'])
    # adapt to camera filter responses
    df_adapted = adapt_to_camera_reflectance(batch=df,
                                             filter_response=settings.tivita_cam_filters_file,
                                             irradiance=settings.tivita_irradiance)
    results_folder = settings.intermediates_dir / 'simulations' / 'multi_layer' / 'generic_depth_adapted'
    results_folder.mkdir(exist_ok=True)
    df_adapted[df_adapted['split'] == 'train'].to_csv(results_folder / 'train.csv')
    df_adapted[df_adapted['split'] == 'test'].to_csv(results_folder / 'test.csv')


@click.command()
@click.option('--adapt', default=False, type=bool, help="Adapt simulations to cameras")
def main(adapt: bool):
    if adapt:
        adapt_simulations_to_cameras()


if __name__ == '__main__':
    main()
