import pathlib

import regression_models

import pandas as pd

pd.options.display.max_rows = 10
pd.options.display.max_columns = 10

PACKAGE_ROOT = pathlib.Path(regression_models.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "datasets"

# data
TESTING_DATA_FILE = "test.csv"
TRAINING_DATA_FILE = "train_all.csv"
TARGET = "logS"

# variables
FEATURES = "SMILES"

EXTRACTED_RDKIT_DESCRIPTORS = ['MinEStateIndex', 'MinAbsEStateIndex', 'NumRadicalElectrons',
       'FpDensityMorgan1', 'Ipc', 'Kappa3', 'PEOE_VSA10', 'PEOE_VSA11',
       'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA3', 'PEOE_VSA4',
       'PEOE_VSA5', 'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA3', 'SMR_VSA8',
       'SMR_VSA9', 'SlogP_VSA11', 'SlogP_VSA3', 'SlogP_VSA4',
       'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'EState_VSA11',
       'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5',
       'EState_VSA6', 'EState_VSA8', 'VSA_EState4', 'VSA_EState5',
       'VSA_EState6', 'VSA_EState8', 'VSA_EState9',
       'NumAliphaticHeterocycles', 'NumSaturatedRings', 'fr_Al_COO',
       'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_NH', 'fr_COO',
       'fr_COO2', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH1', 'fr_N_O',
       'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_SH', 'fr_aldehyde',
       'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid',
       'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo',
       'fr_barbitur', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo',
       'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether',
       'fr_furan', 'fr_guanido', 'fr_hdrzine', 'fr_hdrzone',
       'fr_imidazole', 'fr_isocyan', 'fr_isothiocyan',
       'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy',
       'fr_morpholine', 'fr_nitrile', 'fr_nitro_arom',
       'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime',
       'fr_para_hydroxylation', 'fr_phenol_noOrthoHbond', 'fr_phos_ester',
       'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd',
       'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd',
       'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole',
       'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']


PIPELINE_NAME = "lasso_regression"
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}_output_v"

# used for differential testing
ACCEPTABLE_MODEL_DIFFERENCE = 0.05
