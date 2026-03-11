import json
import argparse

# Need a way to get different mHH categories from a single training.

# ------ configurations ----- #

# Example for a single file, 3 cats, doing mjj cut, boosted flag to false
# base_path = "/home/mmcginni/HHbbgg_conditional_classifiers/out_Version_20251203_inc2024_year/optuna_categorization_baseline/"
# dict_inputs = {
#                 "input_file": ["best_cut_params.json"],
#                 "output_file": [], #leave empty to have the same as input, with _finalfits at the end
#                 "n_cats": 3,
#                 "do_mjjcut": True, #True or False
#                 "mHH_bins": [] #bin edges include upper and lowermost bins, -1 for no bound, leave emptry if not doing mHH bins. Should have N input_files + 1
#                 "is_boosted": 0 # -1: no cut on boosted flag, 0: flag to false, 1: flag to true
# }

# Example for a mHH bins, training per mHH bin
# base_path = "/home/mmcginni/HHbbgg_conditional_classifiers/out_Version_20251203_inc2024_kl_year_350_650/"
# dict_inputs = {
#                 "input_file": ["mHH_bin_0_to_350/optuna_categorization/best_cut_params.json", "mHH_bin_350_to_650/optuna_categorization/best_cut_params.json", "mHH_bin_650_to_inf/optuna_categorization/best_cut_params.json"],
#                 "output_file": ["best_cut_params_350_650_finalfits.json"],
#                 "n_cats": 3,
#                 "do_mjjcut": True,
#                 "mHH_bins": [-1, 350, 650, -1]
# }

base_path = "/eos/user/m/mmcginni/Documents/HHtobbgg/HHbbgg_conditional_classifiers_TRAINONGPU/out_Version_20251203_inc2024_year/optuna_categorization_baseline/"
dict_inputs = {
                "input_file": ["best_cut_params.json"],
                "output_file": ["best_cut_params_finalfits_nomjjcut.json"], #leave empty to have the same as input, with _finalfits at the end
                "n_cats": 5,
                "do_mjjcut": False, #True or False
                "mHH_bins": [] #bin edges include upper and lowermost bins, -1 for no bound, leave emptry if not doing mHH bins. Should have N input_files + 1
}

# --------------------------- #


dict_scoremap = {"th_signal": "ggHH_score",
                "th_bg_0": "nonRes_score",
                "th_bg_1": "ttH_score",
                "th_bg_2": "singleH_score"}

if len(dict_inputs["output_file"]) == 0:
    output_file = base_path + dict_inputs["input_file"][0][:-5] + "_finalfits.json"
else:
    output_file = base_path + dict_inputs["output_file"][0]

dict_cats_ff = {}
ifile = 0
while ifile < len(dict_inputs["input_file"]):
    with open(base_path + dict_inputs["input_file"][ifile], 'r') as file:
        dict_cats_in = json.load(file)

    catstr_mHH = ""
    if len(dict_inputs["mHH_bins"]) > 0:
        mHH_low = dict_inputs["mHH_bins"][ifile]
        mHH_high = dict_inputs["mHH_bins"][ifile + 1]
        if mHH_low == -1:
            catstr_mHH = f"(HHbbggCandidate_mass < {mHH_high}) & "
        elif mHH_high == -1:
            catstr_mHH = f"(HHbbggCandidate_mass > {mHH_low}) & "
        else:
            catstr_mHH = f"(HHbbggCandidate_mass > {mHH_low} & HHbbggCandidate_mass < {mHH_high}) & "
    
    catstr_mjj = ""
    if dict_inputs["do_mjjcut"]:
        catstr_mjj = "(dijet_mass > 80 & dijet_mass < 190) & "

    catstr_boosted = ""
    if (dict_inputs["is_boosted"] == 0) | (dict_inputs["is_boosted"] == 1):
        catstr_boosted = f"(is_boosted == {dict_inputs["is_boosted"]}) & "

    catstr_nots = ""
    icat = 0
    while icat < dict_inputs["n_cats"]:
        cuts = dict_cats_in[icat]

        catstr_dnn = f"(ggHH_score > {cuts['th_signal']} & nonRes_score < {cuts['th_bg_0']} & ttH_score < {cuts['th_bg_1']} & singleH_score < {cuts['th_bg_2']})"

        dict_cats_ff[f"cat{dict_inputs['n_cats']*ifile + icat+1}"] = catstr_mHH + catstr_mjj + catstr_boosted + catstr_dnn + catstr_nots

        catstr_nots += " & not" + catstr_dnn
        icat += 1

    ifile += 1


with open(output_file, "w") as file:
    json.dump(dict_cats_ff, file, indent=4)