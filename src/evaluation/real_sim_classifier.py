import numpy as np
from sklearn.linear_model import LogisticRegression
import os
import glob
import seaborn as sns
import joblib
import pandas as pd
import matplotlib.pyplot as plt


real_dataset = "/home/kris/Work/Data/domain_adaptation_simulations/min_max_preprocessed_data_sqrt_ms/real_images"
sim_dataset = "/home/kris/Work/Data/domain_adaptation_simulations/min_max_preprocessed_data_sqrt_ms/good_simulations"

gan_cinn_dataset = "/home/kris/Work/Data/DA_results/gan_cinn/2023_01_23_22_47_44/testing"
unit_dataset = "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/unit/2023_02_01_01_14_47/testing"

model_path = "/home/kris/Work/Data/DA_results/Classifiers/real_sim_classifier/logistic_regressor.sav"


def gather_images(data_path: str, data_split: str = "training"):
    file_list = glob.glob(os.path.join(data_path, data_split, "*.np*"))

    images = [np.load(file)["reconstruction"].flatten() for file in file_list]

    return images


def shuffle_data_and_labels(data: np.ndarray, labels: np.ndarray):
    assert len(data) == len(labels)
    indices = np.random.permutation(len(data))
    return data[indices], labels[indices]


if __name__ == "__main__":
    Train_Classifier = False

    if Train_Classifier:
        print("Preparing training data")

        train_real_images = gather_images(real_dataset)
        train_sim_images = gather_images(sim_dataset)

        training_set_data = np.array(train_sim_images + train_real_images)
        training_set_labels = np.array([0] * len(train_sim_images) + [1] * len(train_real_images))

        training_set_data, training_set_labels = shuffle_data_and_labels(training_set_data, training_set_labels)

        print("Fitting Classifier")
        classifier = LogisticRegression(max_iter=200)
        classifier.fit(training_set_data, training_set_labels)
        joblib.dump(classifier, model_path)
    else:
        print("Loading Classifier")
        classifier = joblib.load(model_path)

    print("Predicting values")

    ts = sns.load_dataset("tips")

    val_real_images = gather_images(real_dataset, "validation")
    val_sim_images = gather_images(sim_dataset, "validation")
    val_gan_cinn_images = gather_images(gan_cinn_dataset, "training")
    val_unit_images = gather_images(unit_dataset, "training")

    sim_predictions = classifier.predict_proba(val_sim_images)
    real_predictions = classifier.predict_proba(val_real_images)
    gan_cinn_predictions = classifier.predict_proba(val_gan_cinn_images)
    unit_predictions = classifier.predict_proba(val_unit_images)

    # sim_predictions_indices = np.stack([np.round(sim_predictions)[:, 0] == 1, np.round(sim_predictions)[:, 1] == 1],
    #                                    axis=1)
    # real_predictions_indices = np.stack([np.round(real_predictions)[:, 0] == 1, np.round(real_predictions)[:, 1] == 1],
    #                                     axis=1)
    # gan_cinn_predictions_indices = np.stack([np.round(gan_cinn_predictions)[:, 0] == 1, np.round(gan_cinn_predictions)[:, 1] == 1],
    #                                        axis=1)
    # unit_predictions_indices = np.stack([np.round(unit_predictions)[:, 0] == 1, np.round(unit_predictions)[:, 1] == 1],
    #                                        axis=1)

    sim_predictions = sim_predictions[:, 1]
    real_predictions = real_predictions[:, 1]
    gan_cinn_predictions = gan_cinn_predictions[:, 1]
    unit_predictions = unit_predictions[:, 1]

    res_array = np.concatenate([sim_predictions, real_predictions, gan_cinn_predictions, unit_predictions], axis=0)

    predictions_df = pd.DataFrame(data=res_array, columns=["class predictions"])

    label_list = ["simulations"] * len(sim_predictions) + ["real"] * len(real_predictions) \
               + ["gan_cinn"] * len(gan_cinn_predictions) + ["unit"] * len(unit_predictions)
    predictions_df["dataset_type"] = label_list

    sns.violinplot(data=predictions_df, x="dataset_type", y="class predictions", inner="points", scale="width")
    plt.show()

