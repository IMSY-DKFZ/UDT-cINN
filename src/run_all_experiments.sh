#!/bin/sh

# This is the bash script that reproduces the full paper submission of the MICCAI23 paper #3159 entitled:
# Unsupervised Domain Transfer with Conditional Invertible Neural Networks

# For full reproduction, please uncomment Experiments 2, 3, 5, 6, 8, 10, 11, 12.

# TODO Adjust the following line to the parent folder of where you have stored the folders 'HSI_Data' and 'PAI_Data'!
export DATA_BASE_PATH="/Path/to/raw/data"

# TODO Adjust the following line depending on where you want to save all the output data!
export SAVE_DATA_PATH="/Path/to/save/folder"
mkdir "$SAVE_DATA_PATH/results"
mkdir "$SAVE_DATA_PATH/results/inn"
mkdir "$SAVE_DATA_PATH/results/unit"

export HSI_DATA_PATH="$DATA_BASE_PATH/HSI_Data/organ_data"
export PAT_DATA_PATH="$DATA_BASE_PATH/PAT_Data"

export RUN_BY_BASH="True"
export PYTHON_PATH="$PWD"

# 1st Experiment: cINN_DY for PAT
export EXPERIMENT_NAME="gan_cinn"
export EXPERIMENT_ID="cINN_DY"
echo "Starting training of $EXPERIMENT_ID!"
python3 train.py                                               # for cINN_DY (PAT)
python3 evaluation/post_process_generated_data.py "--path" "$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID"
cINN_DY_PATH="$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID"

# 2nd Experiment: cINN_D for PAT
#export EXPERIMENT_ID="cINN_D"
#echo "Starting training of $EXPERIMENT_ID!"
#python3 train.py "--condition" "domain"                       # for cINN_D (PAT)
#python3 evaluation/post_process_generated_data.py "--path" "$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID"
#cINN_D_PATH="$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID"

# 3rd Experiment: cINN_without_GAN for PAT
#export EXPERIMENT_ID="cINN_without_GAN"
#echo "Starting training of $EXPERIMENT_ID!"
#python3 train.py "--adversarial_training" "0"                 # for cINN_without_GAN (PAT)
#python3 evaluation/post_process_generated_data.py "--path" "$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID"
#cINN_WG_PATH="$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID"

# 4th Experiment: cINN_DY for HSI
export EXPERIMENT_NAME="gan_cinn_hsi"
export EXPERIMENT_ID="cINN_DY"
echo "Starting training of $EXPERIMENT_ID!"
python3 train.py                                                # for cINN_DY (HSI)
cINN_DY_PATH_HSI="$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID/testing/generated_spectra_data"

# 5th Experiment: cINN_D for HSI
#export EXPERIMENT_ID="cINN_D"
#echo "Starting training of $EXPERIMENT_ID!"
#python3 train.py "--condition" "domain"                       # for cINN_D (HSI)
#cINN_D_PATH_HSI="$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID/testing/generated_spectra_data"

# 6th Experiment: cINN_without_GAN for HSI
#export EXPERIMENT_ID="cINN_without_GAN"
#echo "Starting training of $EXPERIMENT_ID!"
#python3 train.py "--adversarial_training" "0"                 # for cINN_without_GAN (HSI)
#cINN_WG_PATH_HSI="$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID/testing/generated_spectra_data"

# 7th Experiment: UNIT_Y for PAT
export EXPERIMENT_NAME="unit"
export EXPERIMENT_ID="UNIT_Y"
echo "Starting training of $EXPERIMENT_ID!"
python3 train.py                                               # for UNIT_Y (PAT)
python3 evaluation/post_process_generated_data.py "--path" "$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID"
UNIT_Y_PATH="$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID"

# 8th Experiment: UNIT for PAT
#export EXPERIMENT_ID="UNIT"
#echo "Starting training of $EXPERIMENT_ID!"
#python3 train.py "--condition" "domain"                       # for UNIT (PAT)
#python3 evaluation/post_process_generated_data.py "--path" "$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID"
#UNIT_PATH="$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID"

# 9th Experiment: UNIT_Y for HSI
export EXPERIMENT_NAME="unit_hsi"
export EXPERIMENT_ID="UNIT_Y"
echo "Starting training of $EXPERIMENT_ID!"
python3 train.py                                               # for UNIT_Y (HSI)
UNIT_Y_PATH_HSI="$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID/testing/generated_spectra_data"

# 10th Experiment: UNIT for HSI
#export EXPERIMENT_ID="UNIT"
#echo "Starting training of $EXPERIMENT_ID!"
#python3 train.py "--condition" "domain"                       # for UNIT (HSI)
#UNIT_PATH_HSI="$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID/testing/generated_spectra_data"

# 11th Experiment: CycleGAN for PAT
#export EXPERIMENT_ID="CycleGAN"
#export EXPERIMENT_NAME="cycle_gan"
#echo "Starting training of $EXPERIMENT_ID!"
#python3 train.py                                              # for CycleGAN (PAT)
#python3 evaluation/post_process_generated_data.py "--path" "$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID"
#CYCLEGAN_PATH="$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID"

# 12th Experiment: CycleGSN for HSI
#export EXPERIMENT_NAME="cycle_gan_hsi"
#echo "Starting training of $EXPERIMENT_ID!"
#python3 train.py                                              # for CycleGAN (HSI)
#CYCLEGAN_PATH_HSI="$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID/testing/generated_spectra_data"

python3 evaluation/post_process_generated_data.py "--path" "$cINN_DY_PATH"
python3 evaluation/post_process_generated_data.py "--path" "$UNIT_Y_PATH"

# Uncomment the following lines for full reproducibility
#python3 evaluation/post_process_generated_data.py "--path" "$cINN_D_PATH"
#python3 evaluation/post_process_generated_data.py "--path" "$cINN_WG_PATH"
#python3 evaluation/post_process_generated_data.py "--path" "$UNIT_PATH"
#python3 evaluation/post_process_generated_data.py "--path" "$CYCLEGAN_PATH"

# The following line produces the classification results in table 1 of the paper.
python3 evaluation/artery_vein_classifier.py "--target" "test" "--data_base_root" "$PAT_DATA_PATH" "--gan_cinn_root" "$cINN_DY_PATH/testing/training" "--unit_root" "$UNIT_Y_PATH/testing/training"

# Uncomment the following line for full reproducibility and comment the line above. Please note that cINN_D and UNIT will be the values for 'inn' and 'unit', respectively.
#python3 evaluation/artery_vein_classifier.py "--target" "test" "--data_base_root" "$PAT_DATA_PATH" "--gan_cinn_root" "$cINN_D_PATH/testing/training" "--unit_root" "$UNIT_PATH/testing/training"

# Uncomment the following line for full reproducibility and comment the lines above. Please note that cINN_without_GAN and CycleGAN will be the values for 'inn' and 'unit', respectively.
#python3 evaluation/artery_vein_classifier.py "--target" "test" "--data_base_root" "$PAT_DATA_PATH" "--gan_cinn_root" "$cINN_WG_PATH/testing/training" "--unit_root" "$CYCLEGAN_PATH/testing/training"

# The following lines plot the figures of the paper. Since we only showed cINN_DY and UNIT_Y in the paper, we don't include this for the other models.
# This means for full reproducibility of Table 1, these lines can be commented.
python3 evaluation/compute_pai_pca.py "--pca" "--target" "test" "--data_base_root" "$PAT_DATA_PATH" "--gan_cinn_root" "$cINN_DY_PATH/testing/training" "--unit_root" "$UNIT_Y_PATH/testing/training"
python3 visualization/plot_pai_spectra.py "--spectra" "--diff" "--target" "test" "--data_base_root" "$PAT_DATA_PATH" "--gan_cinn_root" "$cINN_DY_PATH/testing/training" "--unit_root" "$UNIT_Y_PATH/testing/training"

# The following lines plot the figures of the paper. Since we only showed cINN_DY and UNIT_Y in the paper, we don't include this for the other models.
# This means for full reproducibility of Table 1, these lines can be commented.
cp -r "$cINN_DY_PATH_HSI" "$SAVE_DATA_PATH/results/inn/generated_spectra_data"
cp -r "$UNIT_Y_PATH_HSI" "$SAVE_DATA_PATH/results/unit/generated_spectra_data"
python3 evaluation/si_classifier.py "--rf" "--target" "test"
python3 visualization/plot_semantic_spectra.py "--spectra" "--diff" "--pca"
python3 visualization/manuscript_plots.py

# Uncomment the following lines for full reproducibility. Please note that cINN_D and UNIT will be the values for 'inn' and 'unit', respectively.
#cp -r "$cINN_D_PATH_HSI" "$SAVE_DATA_PATH/results/inn/generated_spectra_data"
#cp -r "$UNIT_PATH_HSI" "$SAVE_DATA_PATH/results/unit/generated_spectra_data"
#python3 evaluation/si_classifier.py "--rf" "--target" "test"
#
# Uncomment the following lines for full reproducibility. Please note that cINN_without_GAN and CycleGAN will be the values for 'inn' and 'unit', respectively.
#cp -r "$cINN_WG_PATH_HSI" "$SAVE_DATA_PATH/results/inn/generated_spectra_data"
#cp -r "$CYCGLEGAN_PATH_HSI" "$SAVE_DATA_PATH/results/unit/generated_spectra_data"
#python3 evaluation/si_classifier.py "--rf" "--target" "test"
