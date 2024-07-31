import awkward as ak
import numpy as np
import json
import os
from typing import Any, Dict, List, Optional
import pyarrow as pa
import pyarrow.parquet as pq
import glob
import mplhep as hep
import matplotlib.pyplot as plt


class PrepareInputs:
    def __init__(
        self
        ) -> None:
        self.model_type = "mlp"
        self.input_var_json = os.path.join(os.path.dirname(__file__), "input_variables.json")
        self.sample_to_class = {
            "GGJets": "is_non_resonant_bkg",
            "GJetPt20To40": "is_non_resonant_bkg",
            "GJetPt40": "is_non_resonant_bkg",
            "ttHToGG": "is_ttH_bkg",
            "GluGluToHH": "is_GluGluToHH_sig",
            "VBFToHH": "is_VBFToHH_sig"
            }
        self.classes = ["is_non_resonant_bkg", "is_ttH_bkg", "is_GluGluToHH_sig", "is_VBFToHH_sig"]

    def load_vars(self, path):
        print(path)
        with open(path) as f:
            vals = json.load(f)
        return vals

    def load_parquet(self, path, N_files=-1):

        file_list = glob.glob(path + '*.parquet')
        file_list = file_list[:N_files]

        events = ak.from_parquet(file_list)
        print(f"INFO: loaded parquet files from the path {path}")

        return events

    def add_var(self, events: ak.Array) -> ak.Array:

        # add more variables to the events here
        # events["varable"] = variable # the variable has to be calculated from the existing variables in events
        for var in events.fields:
            if 'pt' in var:
                events[var] = np.log(events[var])

        #events["weight"] = ak.ones_like(events.pt)
        
        return events
    
    def plot_pt_variables(self, comb_inputs, vars_for_training):
        for var in vars_for_training:
            if 'pt' in var or 'Pt' in var:
                plt.figure()
                for cls in self.classes:
                    class_events = comb_inputs[comb_inputs[cls] == 1]
                    data_to_plot = class_events[var]
                    binning = np.linspace(0, 10, 41)
                    Hist, Edges = np.histogram(ak.to_numpy(data_to_plot), bins=binning, density=True)
                    hep.histplot((Hist, Edges), histtype='step', label=f'{cls}')
                plt.xlabel(f"{var}")
                plt.ylabel("Events per bin")
                plt.xlim([0, 10])
                plt.ylim(bottom=0)
                plt.legend()
                hep.cms.label("Private work", data=False, year="2022", com=13.6)
                plt.savefig(f'/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/data/{var}_plot.png')
                plt.clf()

    def prep_input(self, samples_path, out_path) -> ak.Array:

        comb_inputs = []

        for samples in self.sample_to_class.keys():

            events = self.load_parquet(f"{samples_path}/{samples}/nominal/", -1)

            # add more variables as required
            events = self.add_var(events)

            # get only valid events, cuts not applied in HiggsDNA
            events = events[(events.mass > 100) | (events.mass < 180)]

            # pad zero for objects not present

            # get only the variables required for training
            vars_for_training = self.load_vars(self.input_var_json)[self.model_type]
            events = events[vars_for_training]

            # add the bools for each class
            for cls in self.classes:  # first intialize everything to zero
                events[cls] = ak.zeros_like(events.pt)

            events[self.sample_to_class[samples]] = ak.ones_like(events.pt) # one-hot encoded
            comb_inputs.append(events)
            print(f"INFO: Number of events in {samples}: {len(events)}")

        comb_inputs = ak.concatenate(comb_inputs, axis=0)

        self.plot_pt_variables(comb_inputs, vars_for_training)

        # Calculate class weights
        class_counts = {}
        for cls in self.classes:
            class_counts[cls] = ak.sum(getattr(comb_inputs, cls))
        print(f"INFO: Class counts: {class_counts}")
    
        class_weights = {cls: 1.0 / class_counts[cls] for cls in self.classes}
        print(f"INFO: Class weights: {class_weights}")

        # Apply class weights
        weight_array = np.zeros(len(events.pt), dtype=np.float64)
        for cls, weight in class_weights.items():
            weight_array += weight * ak.to_numpy(events[cls])

        # Add weight array to events
        events["weight"] = ak.Array(weight_array)

        print("\n", f"INFO: Number of events in is_non_resonant_bkg: {sum(comb_inputs.is_non_resonant_bkg)}")
        print(f" INFO: Number of events in is_ttH_bkg: {sum(comb_inputs.is_ttH_bkg)}")
        print(f" INFO: Number of events in is_GluGluToHH_sig: {sum(comb_inputs.is_GluGluToHH_sig)}")
        print(f" INFO: Number of events in is_VBFToHH_sig: {sum(comb_inputs.is_VBFToHH_sig)}")

        arrow_table = ak.to_arrow_table(comb_inputs)
        pq.write_table(arrow_table, f'{out_path}/dummy_input_for_{self.model_type}.parquet')

        return comb_inputs


out = PrepareInputs()
out.prep_input("/net/scratch_cms3a/kasaraguppe/public/HHbbgg_samples/", "/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models")