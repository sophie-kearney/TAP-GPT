import core_functions.data_processing
import sys

random_state = int(sys.argv[1])
k = int(sys.argv[2])
p = int(sys.argv[3])
modality = sys.argv[4]

###
# SPLIT DATA
###

combined_df = core_functions.data_processing.split_data(random_state, k, p, modality)
combined_df.to_csv(f"data/data_splits/{random_state}_{modality}_datasplit_{p}_{k}.csv", index=False)

###
# FEATURE SELECTION
###

if p != 0:
    cols_to_exclude = ["PTID", "split"]
    covariates = ["APOE4", "Gender", "Education", "Age"]
    selected_features = core_functions.data_processing.do_feature_selection(combined_df, p,[*cols_to_exclude, *covariates])
else:
    selected_features = combined_df.columns.tolist()
combined_df = combined_df[selected_features]

###
# CREATE TASK PROMPTS
###

cols_to_exclude = ["PTID", "split"]

instruction_header = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
zero_shot_instruction = "Below is a table of patient records. Each column contains features related to Alzheimer's disease. Based on the information, predict whether the patient has Alzheimer's disease (1) or does not (0). Respond only with 1 or 0."
few_shot_instruction = "Below is a table of patient records. Each column contains features related to Alzheimer's disease. The last row is missing a value in the 'AlzheimersDisease' column. Based on the patterns in the other rows, predict whether the patient in the last row has Alzheimer's disease (1) or does not (0). Respond only with 1 or 0."
test_train_val_df =  combined_df[combined_df['split'].isin(['test', 'train', 'val'])]

# tabular
zero_shot_tab = core_functions.data_processing.create_zero_shot_tabular(instruction_header, zero_shot_instruction,
                                                                    test_train_val_df, cols_to_exclude)
zero_shot_tab.to_csv(f"data/task_prompts/zero_shot_tabular_{random_state}_{modality}_{p}_{k}.csv", index=False)

few_shot_tab = core_functions.data_processing.create_few_shot_tabular(instruction_header, few_shot_instruction,
                                                                      combined_df, random_state, k, cols_to_exclude)
few_shot_tab.to_csv(f"data/task_prompts/few_shot_tabular_{random_state}_{modality}_{p}_{k}.csv", index=False)

# serialized
zero_shot_instruction_ser = f"Below is a patient record with feature-value pairs derived from {modality} imaging-derived data. Based on this information, predict whether the patient has Alzheimer's disease (1) or does not (0). Respond only with 1 or 0."
few_shot_instruction_ser = f"Below is several patient records with feature-value pairs derived from {modality} imaging-derived data. Based on this information, predict whether the LAST patient has Alzheimer's disease (1) or does not (0). Respond only with 1 or 0."

zero_shot_ser = core_functions.data_processing.create_zero_shot_serialized(instruction_header, zero_shot_instruction_ser,
                                                                           test_train_val_df, cols_to_exclude)
zero_shot_ser.to_csv(f"data/task_prompts/zero_shot_serialized_{random_state}_{modality}_{p}_{k}.csv", index=False)

few_shot_ser = core_functions.data_processing.create_few_shot_serialized(instruction_header, few_shot_instruction_ser,
                                                                         combined_df, random_state, k, cols_to_exclude)
few_shot_ser.to_csv(f"data/task_prompts/few_shot_serialized_{random_state}_{modality}_{p}_{k}.csv", index=False)