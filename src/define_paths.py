from pathlib import Path

BASE = Path(r"C:\Users\AalstJE\OneDrive - UMCG\Documents\PhD\Code\AIOS-course\Data\processed")

# Experiment 1: post-surgery case (model not trained on operated anatomy)
SURGERY_EXP = {
    "postop_image": BASE / "surgery_experiment" / "imagesTs" / "HNC-B_782_0000.nii.gz",
    "model_pred_postop": BASE / "surgery_experiment" / "labelsTs" / "HNC-B_782.nii.gz",
    # Optional reference if available later
    "gt_postop": BASE / "surgery_experiment" / "labelsGt" / "HNC-B_782.nii.gz",
}

# Experiment 2: metal robustness (model outputs already in processed folders)
METAL_EXP = {
    "metal_case_image": BASE / "metal_experiment_metal_model" / "imagesTs" / "metal_case.nii.gz",
    "no_metal_case_image": BASE / "metal_experiment_metal_model" / "imagesTs" / "no_metal_case.nii.gz",
    "metal_model_pred_metal": BASE / "metal_experiment_metal_model" / "labelsTs" / "metal_case.nii.gz",
    "metal_model_pred_no_metal": BASE / "metal_experiment_metal_model" / "labelsTs" / "no_metal_case.nii.gz",
    "no_metal_model_pred_metal": BASE / "metal_experiment_no_metal_model" / "labelsTs" / "metal_case.nii.gz",
    "no_metal_model_pred_no_metal": BASE / "metal_experiment_no_metal_model" / "labelsTs" / "no_metal_case.nii.gz",
    # Optional references
    "gt_metal": BASE / "metal_experiment_no_metal_model" / "labelsGt" / "metal_case.nii.gz",
    "gt_no_metal": BASE / "metal_experiment_no_metal_model" / "labelsGt" / "no_metal_case.nii.gz",
}

