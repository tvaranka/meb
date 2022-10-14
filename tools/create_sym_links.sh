#!/bin/bash

datasets=("SMIC" "CASME" "CASME2" "SAMM" "4DMicro" "MMEW" "CASME3")
for dataset in ${datasets[@]}
do
  mkdir -p data/$dataset
done

ln -sfn "/my_path/micro_expressions/SMIC/smic.xlsx" "data/SMIC/smic.xlsx"
ln -sfn "/my_path/micro_expressions/SMIC/SMIC_all_cropped/HS" "data/SMIC/SMIC_all_cropped/HS"
ln -sfn "/my_path/micro_expressions/SMIC/HS" "data/SMIC/HS"
ln -sfn "/my_path/micro_expressions/SMIC/smic_uv_frames_secrets_of_OF.npy" "data/SMIC/smic_uv_frames_secrets_of_OF.npy"

ln -sfn "/my_path/micro_expressions/CASME/casme.xlsx" "data/CASME/casme.xlsx"
ln -sfn "/my_path/micro_expressions/CASME/Cropped2" "data/CASME/Cropped"
ln -sfn "/my_path/micro_expressions/CASME/CASME_raw_selected" "data/CASME/CASME_raw_selected"
ln -sfn "/my_path/micro_expressions/CASME/casme_uv_frames_secrets_of_OF.npy" "data/CASME/casme_uv_frames_secrets_of_OF.npy"

ln -sfn "/my_path/micro_expressions/CASME2/CASME2-coding-updated.xlsx" "data/CASME2/CASME2-coding-updated.xlsx"
ln -sfn "/my_path/micro_expressions/CASME2/Cropped_original" "data/CASME2/Cropped_original"
ln -sfn "/my_path/micro_expressions/CASME2/CASME2_RAW_selected" "data/CASME2/CASME2_RAW_selected"
ln -sfn "/my_path/micro_expressions/CASME2/casme2_uv_frames_secrets_of_OF.npy" "data/CASME2/casme2_uv_frames_secrets_of_OF.npy"

ln -sfn "/my_path/micro_expressions/SAMM/SAMM_Micro_FACS_Codes_v2.xlsx" "data/SAMM/SAMM_Micro_FACS_Codes_v2.xlsx"
ln -sfn "/my_path/micro_expressions/SAMM/SAMM_CROP2" "data/SAMM/SAMM_CROP"
ln -sfn "/my_path/micro_expressions/SAMM/SAMM" "data/SAMM/SAMM"
ln -sfn "/my_path/micro_expressions/SAMM/samm_uv_frames_secrets_of_OF.npy" "data/SAMM/samm_uv_frames_secrets_of_OF.npy"

ln -sfn "/my_path/micro_expressions/4DMicro/4DME_Labelling_Micro_release v1.xlsx" "data/4DMicro/4DME_Labelling_Micro_release v1.xlsx"
ln -sfn "/my_path/micro_expressions/4DMicro/gray_micro_crop" "data/4DMicro/gray_micro_crop"
ln -sfn "/my_path/micro_expressions/4DMicro/gray_micro" "data/4DMicro/gray_micro"
ln -sfn "/my_path/micro_expressions/4DMicro/4d_uv_frames_secrets_of_OF.npy" "data/4DMicro/4d_uv_frames_secrets_of_OF.npy"

ln -sfn "/my_path/micro_expressions/MMEW/MMEW_Micro_Exp.xlsx" "data/MMEW/MMEW_Micro_Exp.xlsx"
ln -sfn "/my_path/micro_expressions/MMEW/MMEW/Micro_Expression" "data/MMEW/Micro_Expression"
ln -sfn "/my_path/micro_expressions/MMEW/MMEW/Micro_Expression" "data/MMEW/Micro_Expression"
ln -sfn "/my_path/micro_expressions/MMEW/mmew_uv_frames_secrets_of_OF.npy" "data/MMEW/mmew_uv_frames_secrets_of_OF.npy"

ln -sfn "/my_path/micro_expressions/CASME3/part_A/annotation/cas(me)3_part_A_ME_label_JpgIndex_v2.xlsx"  "data/CASME3/cas(me)3_part_A_ME_label_JpgIndex_v2.xlsx"
ln -sfn "/my_path/micro_expressions/CASME3/part_A/data/ME_A"  "data/CASME3/ME_A"
ln -sfn "/my_path/micro_expressions/CASME3/part_A/ME_A_cropped" "data/CASME3/ME_A_cropped"
ln -sfn "/my_path/micro_expressions/CASME3/part_A/casme3_uv_frames_secrets_of_OF.npy" "data/CASME3/casme3_uv_frames_secrets_of_OF.npy"

ln -sfn "/my_path/micro_expressions/CASME3/part_C/CAS(ME)3_part_C_ME.xlsx" "data/CASME3/CAS(ME)3_part_C_ME.xlsx"
ln -sfn "/my_path/micro_expressions/CASME3/part_C/ME_C" "data/CASME3/ME_C"
ln -sfn "/my_path/micro_expressions/CASME3/part_C/ME_C_cropped" "data/CASME3/ME_C_cropped"
ln -sfn "/my_path/micro_expressions/CASME3/part_C/casme3c_uv_frames_secrets_of_OF.npy" "data/CASME3/casme3c_uv_frames_secrets_of_OF.npy"
