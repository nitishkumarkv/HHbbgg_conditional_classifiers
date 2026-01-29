import pandas as pd
import pyarrow.parquet as pq

bin_edges = [350]
out_dir = "/eos/user/m/mmcginni/Documents/HHtobbgg/HHbbgg_conditional_classifiers_TRAINONGPU/out_Version_20251203_inc2024_kl_year_350"

df_comb = pd.DataFrame()

i = 0
while i < len(bin_edges) + 1:
    filestr = f"{out_dir}/mHH_bin_"
    if i == 0:
        filestr += f"0_to_{bin_edges[0]}"
    elif i == len(bin_edges):
        filestr += f"{bin_edges[i - 1]}_to_inf"
    else:
        filestr += f"{bin_edges[i - 1]}_to_{bin_edges[i]}"

    filestr += "/merged_samples.parquet"

    df = pq.read_table(filestr).to_pandas()
    df_comb = pd.concat([df_comb, df])

    i += 1

df_comb.to_parquet(f"{out_dir}/merged_samples_mHHcomb.parquet")
