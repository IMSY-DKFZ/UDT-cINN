import click
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

from src import settings


def plot_importance():
    model_file = settings.results_dir / 'rf' / 'rf_classifier_adapted_inn.joblib'
    model: RandomForestClassifier = joblib.load(model_file)
    importance = model.feature_importances_
    df = pd.DataFrame()
    df['importance'] = importance
    df['wavelength'] = np.arange(500, 1000, 5)

    fig = px.line(data_frame=df,
                  x="wavelength",
                  y="importance")
    fig.write_html(settings.figures_dir / 'rf_feature_importance.html')
    fig.write_image(settings.figures_dir / 'rf_feature_importance.pdf')
    fig.write_image(settings.figures_dir / 'rf_feature_importance.png')


@click.command()
@click.option('--importance', is_flag=True, help="plot feature importance of random forest model")
def main(importance: bool):
    if importance:
        plot_importance()


if __name__ == '__main__':
    main()
