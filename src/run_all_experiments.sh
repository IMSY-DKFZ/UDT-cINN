#!/bin/sh

export RUN_BY_BASH="True"

export DATA_BASE_PATH="/home/kris/Work/Data/domain_adaptation_simulations"
#export DATA_BASE_PATH="/Path/to/raw/data"

export SAVE_DATA_PATH="/home/kris/Work/Data/Test/"
#export SAVE_DATA_PATH="/Path/to/save/folder"

export HSI_DATA_PATH="$DATA_BASE_PATH/HSI_Data/organ_data"
export PAT_DATA_PATH="$DATA_BASE_PATH/PAT_Data"

export PYTHON_PATH="$PWD"

#export EXPERIMENT_NAME="gan_cinn"
#export EXPERIMENT_ID="cINN_DY"
#echo "Starting training of $EXPERIMENT_ID!"
#python3 train.py "--epochs" "1"                                              # for cINN (PAT)
#python3 evaluation/post_process_generated_data.py "--path" "$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID"
#cINN_DY_PATH="$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID"
#
#export EXPERIMENT_ID="cINN_D"
#echo "Starting training of $EXPERIMENT_ID!"
#python3 train.py "--condition" "domain"                       # for cINN_D (PAT)
#python3 evaluation/post_process_generated_data.py "--path" "$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID"
#cINN_D_PATH="$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID"
#
#export EXPERIMENT_ID="cINN_without_GAN"
#echo "Starting training of $EXPERIMENT_ID!"
#python3 train.py "--adversarial_training" "0"                 # for cINN_without_GAN (PAT)
#python3 evaluation/post_process_generated_data.py "--path" "$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID"
#cINN_WG_PATH="$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID"
#
#export EXPERIMENT_NAME="gan_cinn_hsi"
#export EXPERIMENT_ID="cINN_DY"
#echo "Starting training of $EXPERIMENT_ID!"
#python3 train.py                                              # for cINN (HSI)
#cINN_DY_PATH_HSI="$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID/testing/generated_spectra_data"
#
#export EXPERIMENT_ID="cINN_D"
#echo "Starting training of $EXPERIMENT_ID!"
#python3 train.py "--condition" "domain"                       # for cINN_D (HSI)
#cINN_D_PATH_HSI="$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID/testing/generated_spectra_data"
#
#export EXPERIMENT_ID="cINN_without_GAN"
#echo "Starting training of $EXPERIMENT_ID!"
#python3 train.py "--adversarial_training" "0"                 # for cINN_without_GAN (HSI)
#cINN_WG_PATH_HSI="$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID/testing/generated_spectra_data"
#
#export EXPERIMENT_NAME="unit"
#export EXPERIMENT_ID="UNIT_Y"
#echo "Starting training of $EXPERIMENT_ID!"
#python3 train.py                                              # for UNIT_Y (PAT)
#python3 evaluation/post_process_generated_data.py "--path" "$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID"
#UNIT_Y_PATH="$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID"
#
#export EXPERIMENT_ID="UNIT"
#echo "Starting training of $EXPERIMENT_ID!"
#python3 train.py "--condition" "domain"                       # for UNIT (PAT)
#python3 evaluation/post_process_generated_data.py "--path" "$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID"
#UNIT_PATH="$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID"
#
#export EXPERIMENT_NAME="unit_hsi"
#export EXPERIMENT_ID="UNIT_Y"
#echo "Starting training of $EXPERIMENT_ID!"
#python3 train.py                                              # for UNIT_Y (HSI)
#UNIT_Y_PATH_HSI="$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID/testing/generated_spectra_data"
#
#export EXPERIMENT_ID="UNIT"
#echo "Starting training of $EXPERIMENT_ID!"
#python3 train.py "--condition" "domain"                       # for UNIT (HSI)
#UNIT_PATH_HSI="$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID/testing/generated_spectra_data"
#
#export EXPERIMENT_ID="CycleGAN"
#export EXPERIMENT_NAME="cycle_gan"
#echo "Starting training of $EXPERIMENT_ID!"
#python3 train.py                                              # for CycleGAN (PAT)
#python3 evaluation/post_process_generated_data.py "--path" "$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID"
#CYCLEGAN_PATH="$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID"
#
#export EXPERIMENT_NAME="cycle_gan_hsi"
#echo "Starting training of $EXPERIMENT_ID!"
#python3 train.py                                              # for CycleGAN (HSI)
#CYCLEGAN_PATH_HSI="$SAVE_DATA_PATH/$EXPERIMENT_NAME/$EXPERIMENT_ID/testing/generated_spectra_data"

#python3 evaluation/post_process_generated_data.py "--path" "$cINN_DY_PATH"
#python3 evaluation/post_process_generated_data.py "--path" "$UNIT_Y_PATH"

#python3 evaluation/artery_vein_classifier.py "--target" "test" "--data_base_root" "$PAT_DATA_PATH" "--gan_cinn_root" "$cINN_DY_PATH/testing/training" "--unit_root" "$UNIT_Y_PATH/testing/training"

#cp -r "$cINN_DY_PATH_HSI" "$HSI_DATA_PATH/results/inn"
#cp -r "$UNIT_Y_PATH_HSI" "$HSI_DATA_PATH/results/unit"
#python3 evaluation/si_classifier.py "--rf" "--target" "test"

#jupyter nbconvert --to notebook --inplace --execute visualization/manuscript_plots.ipynb

