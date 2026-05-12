import json
import os
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn


BINARY_CLASSIFICATION_TYPE = "binary"
MULTICLASS_CLASSIFICATION_TYPE = "multiclass"


def get_classification_type(training_config: Dict[str, Any]) -> str:
    binary_config = training_config.get("binary_classifier", {}) or {}
    if binary_config.get("enabled", False):
        return BINARY_CLASSIFICATION_TYPE
    return training_config.get("classification_type", MULTICLASS_CLASSIFICATION_TYPE)


def is_binary_classification(training_config: Dict[str, Any]) -> bool:
    return get_classification_type(training_config) == BINARY_CLASSIFICATION_TYPE


def load_class_names(input_path: str, training_config: Dict[str, Any]) -> List[str]:
    training_info_path = os.path.join(input_path, "training_info.txt")
    if os.path.exists(training_info_path):
        with open(training_info_path, "r") as f:
            training_info = json.load(f)
        if "classes" in training_info:
            return list(training_info["classes"])

    if "classes" in training_config:
        return list(training_config["classes"])

    raise ValueError(
        "Could not determine class names. Expected classes in "
        f"{training_info_path} or in the training config."
    )


def get_binary_class_groups(
    class_names: Iterable[str],
    training_config: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    binary_config = training_config.get("binary_classifier", {}) or {}
    class_names = list(class_names)

    signal_classes = binary_config.get("signal_classes")
    if signal_classes is None:
        signal_classes = [
            class_name for class_name in class_names
            if class_name in ("is_GluGluToHH_sig", "is_VBFToHH_sig")
        ]

    background_classes = binary_config.get("background_classes")
    if background_classes is None:
        background_classes = ["is_non_resonant_bkg"]

    missing = [
        class_name for class_name in list(signal_classes) + list(background_classes)
        if class_name not in class_names
    ]
    if missing:
        raise ValueError(
            "Binary classifier requested classes that are not present in the "
            f"prepared labels: {missing}. Available classes: {class_names}"
        )

    if len(signal_classes) == 0:
        raise ValueError("Binary classifier needs at least one signal class.")
    if len(background_classes) == 0:
        raise ValueError("Binary classifier needs at least one background class.")

    return list(signal_classes), list(background_classes)


def make_binary_labels_from_one_hot(
    y: np.ndarray,
    class_names: Iterable[str],
    signal_classes: Iterable[str],
    background_classes: Iterable[str],
) -> Tuple[np.ndarray, np.ndarray]:
    class_names = list(class_names)
    signal_indices = [class_names.index(class_name) for class_name in signal_classes]
    background_indices = [class_names.index(class_name) for class_name in background_classes]

    signal_mask = np.any(y[:, signal_indices] == 1, axis=1)
    background_mask = np.any(y[:, background_indices] == 1, axis=1)
    keep_mask = signal_mask | background_mask

    binary_y = signal_mask[keep_mask].astype(np.float32).reshape(-1, 1)
    return binary_y, keep_mask


def prepare_binary_arrays(
    y: np.ndarray,
    class_names: Iterable[str],
    training_config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    signal_classes, background_classes = get_binary_class_groups(class_names, training_config)
    binary_y, keep_mask = make_binary_labels_from_one_hot(
        y,
        class_names,
        signal_classes,
        background_classes,
    )
    metadata = {
        "classification_type": BINARY_CLASSIFICATION_TYPE,
        "output_size": 1,
        "positive_label": "signal",
        "negative_label": "nonres_background",
        "signal_classes": signal_classes,
        "background_classes": background_classes,
    }
    return binary_y, keep_mask, metadata


def weighted_binary_accuracy(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    predictions = (torch.sigmoid(logits).squeeze(dim=1) >= 0.5).float()
    targets = targets.float().squeeze(dim=1)
    return ((predictions == targets).float() * weights).sum() / weights.sum()


def make_loss_function(classification_type: str) -> nn.Module:
    if classification_type == BINARY_CLASSIFICATION_TYPE:
        return nn.BCEWithLogitsLoss(reduction="none")
    return nn.CrossEntropyLoss(reduction="none")
