import os


class config:
    """Used as a config file for storing dataset information"""

    smic_excel_path = "data/SMIC/smic.xlsx"
    smic_cropped_dataset_path = "data/SMIC/SMIC_all_cropped/HS"
    smic_dataset_path = "data/SMIC/HS"
    smic_optical_flow = "data/SMIC/smic_uv_frames_secrets_of_OF.npy"

    casme_excel_path = "data/CASME/casme.xlsx"
    casme_cropped_dataset_path = "data/CASME/Cropped"
    casme_dataset_path = "data/CASME/CASME_raw_selected"
    casme_optical_flow = "data/CASME/casme_uv_frames_secrets_of_OF.npy"

    casme2_excel_path = "data/CASME2/CASME2-coding-updated.xlsx"
    casme2_cropped_dataset_path = "data/CASME2/Cropped_original"
    casme2_dataset_path = "data/CASME2/CASME2_RAW_selected"
    casme2_optical_flow = "data/CASME2/casme2_uv_frames_secrets_of_OF.npy"

    samm_excel_path = "data/SAMM/SAMM_Micro_FACS_Codes_v2.xlsx"
    samm_cropped_dataset_path = "data/SAMM/SAMM_CROP"
    samm_dataset_path = "data/SAMM/SAMM"
    samm_optical_flow = "data/SAMM/samm_uv_frames_secrets_of_OF.npy"

    fourd_excel_path = "data/4DMicro/4DME_Labelling_Micro_release v1.xlsx"
    fourd_cropped_dataset_path = "data/4DMicro/gray_micro_crop"
    fourd_dataset_path = "data/4DMicro/gray_micro"
    fourd_optical_flow = "data/4DMicro/4d_uv_frames_secrets_of_OF.npy"

    mmew_excel_path = "data/MMEW/MMEW_Micro_Exp.xlsx"
    mmew_cropped_dataset_path = "data/MMEW/Micro_Expression"
    mmew_dataset_path = "data/MMEW/Micro_Expression"
    mmew_optical_flow = "data/MMEW/mmew_uv_frames_secrets_of_OF.npy"

    casme3a_excel_path = "data/CASME3/cas(me)3_part_A_ME_label_JpgIndex_v2.xlsx"
    casme3a_dataset_path = "data/CASME3/ME_A"
    casme3a_cropped_dataset_path = "data/CASME3/ME_A_cropped"
    casme3a_optical_flow = "data/CASME3/casme3_uv_frames_secrets_of_OF.npy"

    casme3c_excel_path = "data/CASME3/CAS(ME)3_part_C_ME.xlsx"
    casme3c_dataset_path = "data/CASME3/ME_C"
    casme3c_cropped_dataset_path = "data/CASME3/ME_C_cropped"
    casme3c_optical_flow = "data/CASME3/casme3c_uv_frames_secrets_of_OF.npy"


def check_path(cls: config):
    """
    Checks whether the "data" folder is in the current directory.
    If not tries to find it.
    """
    paths = [path for path in dir(cls) if not path.startswith("__")]
    if "data" in os.listdir("."):
        return cls
    elif "data" in os.listdir("..") and not paths[0].startswith(".."):
        for path in paths:
            modified_path = "../" + getattr(cls, path)
            setattr(cls, path, modified_path)
    elif "data" in os.listdir("../.."):
        for path in paths:
            modified_path = "../../" + getattr(cls, path)
            setattr(cls, path, modified_path)
    else:
        print("Couldn't find data folder")
    return cls


config = check_path(config)

