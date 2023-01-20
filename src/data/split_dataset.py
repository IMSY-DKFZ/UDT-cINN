import pandas as pd
import numpy as np
import os


def split_dataset(data_path, save_path, save_keyword: str = "sampled", seed: int = 42):
    np.random.seed(seed)

    df = pd.read_csv(data_path, header=[0, 1], index_col=0)
    df = df.sample(frac=1).reset_index(drop=True)

    n_simulations = df.shape[0]
    split_sizes = np.array((0.7, 0.1, 0.2))*n_simulations
    split_sizes = split_sizes.astype(int)

    train_df = df[0:split_sizes[0]]
    validation_df = df[split_sizes[0]:split_sizes[0] + split_sizes[1]]
    test_df = df[split_sizes[0] + split_sizes[1]:]

    save_path = os.path.join(save_path, save_keyword)
    os.makedirs(save_path, exist_ok=True)

    train_df.to_csv(os.path.join(save_path, "training.csv"))
    validation_df.to_csv(os.path.join(save_path, "validation.csv"))
    test_df.to_csv(os.path.join(save_path, "downstream_test.csv"))


if __name__ == "__main__":
    data_path = "/home/kris/Work/Data/domain_adaptation_simulations/HSI_Data/sampled/training.csv"

    df = pd.read_csv(data_path, header=[0, 1], index_col=0)
    oxy = df.layer0
    oxy = oxy.sao2
    print(df.reflectances[13, :])
    k = "k"
