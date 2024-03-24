import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightning import pytorch as pl
from sklearn.model_selection import train_test_split
import torch

torch.set_float32_matmul_precision("medium")
from chemprop import data, featurizers, models, nn

pl.seed_everything(1)


class MetricTracker(pl.Callback):
    def __init__(self):
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)

    def save_metrics(self, file_path):
        metrics_dict = {}
        for metric_dict in self.metrics:
            for key, value in metric_dict.items():
                if key not in metrics_dict:
                    metrics_dict[key] = []
                metrics_dict[key].append(value.item())

        with open(file_path, "w") as file:
            for key, values in metrics_dict.items():
                file.write(f"{key}: {values}\n")


def get_mol_datapoints(df, smiles_column, target_columns):
    smis = df.loc[:, smiles_column].values
    ys = df.loc[:, target_columns].values
    return [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]


def get_data_loader(df, smiles_column, target_columns, batch_size, num_workers):
    data_points = get_mol_datapoints(df, smiles_column, target_columns)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    dset = data.MoleculeDataset(data_points, featurizer)
    return data.MolGraphDataLoader(dset, num_workers=num_workers)


def create_data_loaders(train_data, val_data, test_data, num_workers=0):
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    train_dset = data.MoleculeDataset(train_data, featurizer)
    val_dset = data.MoleculeDataset(val_data, featurizer)
    test_dset = data.MoleculeDataset(test_data, featurizer)

    train_loader = data.MolGraphDataLoader(train_dset, num_workers=num_workers)
    val_loader = data.MolGraphDataLoader(
        val_dset, num_workers=num_workers, shuffle=False
    )
    test_loader = data.MolGraphDataLoader(
        test_dset, num_workers=num_workers, shuffle=False
    )

    return train_loader, val_loader, test_loader


def calculate_rmse(predicted_spectra, reference_spectra):
    if len(predicted_spectra) != len(reference_spectra):
        print(len(predicted_spectra), len(reference_spectra))
        raise ValueError(
            "Length of predicted and reference spectra lists should be the same."
        )

    rmse_values = []

    for i in range(len(predicted_spectra)):
        predicted = np.array(predicted_spectra[i]).squeeze()
        reference = np.array(reference_spectra[i]).squeeze()

        if len(predicted) != len(reference):
            print(len(predicted), len(reference))
            raise ValueError(
                f"Length of predicted and reference spectra at index {i} should be the same."
            )

        mse = np.mean((predicted - reference) ** 2)
        rmse = np.sqrt(mse)
        rmse_values.append(rmse)

    return rmse_values


def calculate_sis(predicted_spectra, reference_spectra):
    if len(predicted_spectra) != len(reference_spectra):
        print(len(predicted_spectra), len(reference_spectra))
        raise ValueError(
            "Length of predicted and reference spectra lists should be the same."
        )

    sis_values = []

    for i in range(len(predicted_spectra)):
        predicted = np.array(predicted_spectra[i]).squeeze()
        reference = np.array(reference_spectra[i]).squeeze()

        if len(predicted) != len(reference):
            print(len(predicted), len(reference))
            raise ValueError(
                f"Length of predicted and reference spectra at index {i} should be the same."
            )

        # Set any negative values to zero
        predicted[predicted < 0] = 0
        reference[reference < 0] = 0

        # Add a small constant to avoid taking log of zero
        epsilon = 1e-10
        predicted += epsilon
        reference += epsilon

        # Ensure the spectra are normalized
        predicted = predicted / np.sum(predicted)
        reference = reference / np.sum(reference)

        # Check for negative values
        if np.any(predicted < 0) or np.any(reference < 0):
            raise ValueError("Spectra contain negative values")

        # Check for zero values
        if np.any(predicted == 0) or np.any(reference == 0):
            raise ValueError("Spectra contain zero values")

        # Calculate the SID
        sid = np.sum(predicted * np.log(predicted / reference)) + np.sum(
            reference * np.log(reference / predicted)
        )

        # Calculate the SIS
        sis = 1 / (1 + sid)
        sis_values.append(sis)

    return sis_values
