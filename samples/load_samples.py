import awkward as ak


def load_parquet(path):

    events = ak.from_parquet(path)
    print(f"INFO: loaded parquet files from the path {path}")

    return events


# all the samples are stored here: /net/scratch_cms3a/kasaraguppe/public/HHbbgg_samples

events = load_parquet("/net/scratch_cms3a/kasaraguppe/public/HHbbgg_samples/GluGluHToGG/nominal/")

fileds = events.fields
with open("output.txt", "w") as file: # you can save the filed in a txt file
    for item in fileds:
        file.write(f"{item}\n")

print(fileds)
print(events.pt)


# example of event selection
#pt_cut = (events.pt > 20)
#num_jets_cut = (events.n_jets > 2)
#events = events[pt_cut & num_jets_cut] # for and operation
#events = events[pt_cut | num_jets_cut] # for or operation

