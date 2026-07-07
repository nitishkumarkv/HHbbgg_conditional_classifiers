import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import pyarrow as pa
import pyarrow.parquet as pq

from mlp import MLP


def get_device(cuda_device: str):
  return torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu')


def load_model(model_dict_path, model_path, input_size, device, output_size=None):
  with open(model_dict_path, 'r') as f:
    best_params = json.load(f)

  best_num_layers = best_params['num_layers']
  best_num_nodes = best_params['num_nodes']
  best_act_fn_name = best_params['act_fn_name']
  best_act_fn = getattr(nn, best_act_fn_name)
  best_dropout_prob = best_params['dropout_prob']

  model_state = None
  if output_size is None:
    if 'output_size' in best_params:
      output_size = best_params['output_size']
      print(f"Using output_size from best_params: {output_size}")
    else:
      model_state = torch.load(model_path, map_location=device, weights_only=False)
      for key in reversed(list(model_state['model_state_dict'].keys())):
        if 'weight' in key and 'layers' in key:
          output_size = model_state['model_state_dict'][key].shape[0]
          print(f"Inferred output_size from checkpoint: {output_size}")
          break
      if output_size is None:
        raise ValueError("Could not determine output_size. Please provide output_size.")
  else:
    print(f"Using provided output_size: {output_size}")

  model = MLP(
    input_size,
    best_num_layers,
    best_num_nodes,
    output_size,
    best_act_fn,
    best_dropout_prob
  ).to(device)

  if model_state is None:
    model_state = torch.load(model_path, map_location=device, weights_only=False)

  model.load_state_dict(model_state['model_state_dict'])
  model.eval()
  return model


def predict_numpy(model, X_np, device, batch_size=1024):
  X = torch.tensor(X_np, dtype=torch.float32, device=device)
  y_preds = []

  for i in range(0, len(X), batch_size):
    X_batch = X[i:i + batch_size]
    with torch.no_grad():
      y_batch = model(X_batch)
      y_batch = F.softmax(y_batch, dim=1)
      y_preds.append(y_batch.cpu().numpy())

  y = np.concatenate(y_preds, axis=0)
  print(f"Prediction shape: {y.shape}")
  return y


def read_parquet_table(parquet_path):
  if not os.path.exists(parquet_path):
    raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
  return pq.read_table(parquet_path)


def take_table_rows(table, indices):
  index_array = pa.array(indices.astype(np.int64))
  return table.take(index_array)


def sample_indices(n_total, fraction, seed):
  if not (0 < fraction <= 1.0):
    raise ValueError(f"fraction must be in (0, 1], got {fraction}")
  n_keep = int(n_total * fraction)
  n_keep = max(1, n_keep)
  rng = np.random.default_rng(seed)
  idx = np.sort(rng.choice(n_total, size=n_keep, replace=False))
  return idx


def process_one_sample(
  input_dir,
  output_dir,
  model,
  device,
  fraction,
  seed,
  save_rel_w=True,
  save_events=True,
  overwrite=False,
):
  x_path = os.path.join(input_dir, "X.npy")
  rel_w_path = os.path.join(input_dir, "rel_w.npy")
  events_path = os.path.join(input_dir, "events.parquet")

  if not os.path.exists(x_path):
    print(f"[WARNING] Missing X.npy, skip: {x_path}")
    return

  os.makedirs(output_dir, exist_ok=True)

  out_x_path = os.path.join(output_dir, "X.npy")
  out_y_path = os.path.join(output_dir, "y.npy")
  out_rel_w_path = os.path.join(output_dir, "rel_w.npy")
  out_events_path = os.path.join(output_dir, "events.parquet")
  out_idx_path = os.path.join(output_dir, "sampled_indices.npy")

  if (not overwrite) and os.path.exists(out_x_path) and os.path.exists(out_y_path):
    print(f"[INFO] Output exists, skip: {output_dir}")
    return

  X = np.load(x_path)
  n_total = len(X)
  idx = sample_indices(n_total, fraction, seed)
  X_sub = X[idx]

  print(f"[INFO] {input_dir}")
  print(f"       original events = {n_total}")
  print(f"       kept events     = {len(idx)}")
  print(f"       fraction        = {fraction}")

  np.save(out_x_path, X_sub)
  np.save(out_idx_path, idx)

  if save_rel_w and os.path.exists(rel_w_path):
    rel_w = np.load(rel_w_path)
    if len(rel_w) != n_total:
      raise ValueError(f"Length mismatch: len(rel_w)={len(rel_w)} but len(X)={n_total} in {input_dir}")
    rel_w_sub = rel_w[idx]
    np.save(out_rel_w_path, rel_w_sub)

  if save_events and os.path.exists(events_path):
    table = read_parquet_table(events_path)
    if table.num_rows != n_total:
      raise ValueError(
        f"Length mismatch: events.parquet rows={table.num_rows} but len(X)={n_total} in {input_dir}"
      )
    table_sub = take_table_rows(table, idx)
    pq.write_table(table_sub, out_events_path)

  y_sub = predict_numpy(model, X_sub, device=device)
  np.save(out_y_path, y_sub)

  print(f"[INFO] Saved sampled files to: {output_dir}\n")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Subsample prepared evaluation inputs, regenerate parquet/X/rel_w, and run DNN prediction.")
  parser.add_argument('--model_folder', type=str, required=True, help='Path to the trained model folder')
  parser.add_argument('--samples_path', type=str, required=True, help='Path to prepared samples root')
  parser.add_argument('--config_path', type=str, required=True, help='Path to folder containing training_config.yaml')
  parser.add_argument('--fraction', type=float, default=0.5, help='Sampling fraction, e.g. 0.5')
  parser.add_argument('--seed', type=int, default=None, help='Random seed for subsampling; default uses training_config random_seed')
  parser.add_argument('--suffix', type=str, default='sampled50', help='Suffix for output folders')
  parser.add_argument('--get_pred_nominal', action='store_true', help='Process nominal MC + data samples')
  parser.add_argument('--get_pred_sys', action='store_true', help='Process systematic samples')
  parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output files')
  args = parser.parse_args()

  model_folder = args.model_folder
  model_dict_path = os.path.join(model_folder, "params.json")
  model_path = os.path.join(model_folder, "mlp.pth")

  training_config_path = os.path.join(args.config_path, "training_config.yaml")
  with open(training_config_path, 'r') as f:
    training_config = yaml.safe_load(f)

  seed = args.seed if args.seed is not None else training_config["random_seed"]
  device = get_device(training_config["cuda_device"])
  output_size = len(training_config["classes"])

  print(f"Device: {device}")
  print(f"Sampling fraction: {args.fraction}")
  print(f"Sampling seed: {seed}")
  print(f"Number of output classes: {output_size}")

  # Build model once
  dummy_x_path = None

  if args.get_pred_nominal:
    found = False
    for era in training_config["samples_info"]["eras"]:
      for sample in training_config["samples_info"][era].keys():
        candidate = os.path.join(args.samples_path, "individual_samples", era, sample, "X.npy")
        if os.path.exists(candidate):
          dummy_x_path = candidate
          found = True
          break
      if found:
        break

    if (dummy_x_path is None) and ("data" in training_config["samples_info"]):
      for data_sample in training_config["samples_info"]["data"].keys():
        candidate = os.path.join(args.samples_path, "individual_samples_data", data_sample, "X.npy")
        if os.path.exists(candidate):
          dummy_x_path = candidate
          found = True
          break

  elif args.get_pred_sys:
    found = False
    for era in training_config["samples_info"]["eras"]:
      for sample in training_config["samples_info"][era].keys():
        if sample in ["GGJets", "DDQCDGJET", "TTG_10_100", "TTG_100_200", "TTG_200", "TT", "TTGG"]:
          continue
        for sys in training_config["systematics"]:
          candidate = os.path.join(args.samples_path, "individual_samples", era, sample, sys, "X.npy")
          if os.path.exists(candidate):
            dummy_x_path = candidate
            found = True
            break
        if found:
          break
      if found:
        break

  if dummy_x_path is None:
    raise RuntimeError("Could not find any X.npy to infer input size.")

  dummy_X = np.load(dummy_x_path, mmap_mode='r')
  input_size = dummy_X.shape[1]
  del dummy_X

  model = load_model(
    model_dict_path=model_dict_path,
    model_path=model_path,
    input_size=input_size,
    device=device,
    output_size=output_size,
  )

  if args.get_pred_nominal:
    # nominal MC
    for era in training_config["samples_info"]["eras"]:
      for sample in training_config["samples_info"][era].keys():
        input_dir = os.path.join(args.samples_path, "individual_samples", era, sample)
        output_dir = os.path.join(args.samples_path, f"individual_samples_{args.suffix}", era, sample)

        process_one_sample(
          input_dir=input_dir,
          output_dir=output_dir,
          model=model,
          device=device,
          fraction=args.fraction,
          seed=seed,
          save_rel_w=True,
          save_events=True,
          overwrite=args.overwrite,
        )

    # data
    if "data" in training_config["samples_info"]:
      for data_sample in training_config["samples_info"]["data"].keys():
        input_dir = os.path.join(args.samples_path, "individual_samples_data", data_sample)
        output_dir = os.path.join(args.samples_path, f"individual_samples_data_{args.suffix}", data_sample)

        process_one_sample(
          input_dir=input_dir,
          output_dir=output_dir,
          model=model,
          device=device,
          fraction=args.fraction,
          seed=seed,
          save_rel_w=False,   # data usually has no rel_w.npy
          save_events=True,
          overwrite=args.overwrite,
        )

  elif args.get_pred_sys:
    for era in training_config["samples_info"]["eras"]:
      for sample in training_config["samples_info"][era].keys():
        if sample in ["GGJets", "DDQCDGJET", "TTG_10_100", "TTG_100_200", "TTG_200", "TT", "TTGG"]:
          continue

        for sys in training_config["systematics"]:
          input_dir = os.path.join(args.samples_path, "individual_samples", era, sample, sys)
          output_dir = os.path.join(args.samples_path, f"individual_samples_{args.suffix}", era, sample, sys)

          if not os.path.exists(os.path.join(input_dir, "X.npy")):
            print(f"[WARNING] Missing systematic input, skip: {input_dir}")
            continue

          process_one_sample(
            input_dir=input_dir,
            output_dir=output_dir,
            model=model,
            device=device,
            fraction=args.fraction,
            seed=seed,
            save_rel_w=True,
            save_events=True,
            overwrite=args.overwrite,
          )

  else:
    raise ValueError("Please specify either --get_pred_nominal or --get_pred_sys")