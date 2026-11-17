"""Only load models created locally or otherwise trusted."""
def command(run, model):
    return "inference_01_predict_biomass.py", [
        "--run-dir", run, "--model-file", model/"random_forest_biomass.joblib",
        "--feature-file", model/"feature_list.json"]
