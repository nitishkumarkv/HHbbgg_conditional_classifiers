import awkward as ak
import json
from typing import Any, Dict, List, Optional
import pyarrow as pa
import pyarrow.parquet as pq
import glob


class PrepareInputs:
    def __init__(
        self
        ) -> None:
        self.model_type = "mlp"
        self.input_var_json = "input_variables.json"
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
        vals = json.load(open(path))
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

        events["weight"] = ak.ones_like(events.pt)  # now adding dummy weight. This has to be ubdated

        return events

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

        print("\n", f"INFO: Number of events in is_non_resonant_bkg: {sum(comb_inputs.is_non_resonant_bkg)}")
        print(f" INFO: Number of events in is_ttH_bkg: {sum(comb_inputs.is_ttH_bkg)}")
        print(f" INFO: Number of events in is_GluGluToHH_sig: {sum(comb_inputs.is_GluGluToHH_sig)}")
        print(f" INFO: Number of events in is_VBFToHH_sig: {sum(comb_inputs.is_VBFToHH_sig)}")

        arrow_table = ak.to_arrow_table(comb_inputs)
        pq.write_table(arrow_table, f'{out_path}/dummy_input_for_{self.model_type}.parquet')

        return comb_inputs


out = PrepareInputs()
out.prep_input("/net/scratch_cms3a/kasaraguppe/public/HHbbgg_samples/", "/net/scratch_cms3a/kasaraguppe/public/HHbbgg_samples/training_inputs")