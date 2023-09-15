import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
import numpy as np
from sklearn.metrics import classification_report

from monai.data import decollate_batch, CacheDataset, ThreadDataLoader
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    ScaleIntensityd,
    EnsureTyped,
    Resized,
    CropForegroundd,
    SpatialPadd,
    CastToTyped,
    ConcatItemsd,
    Transform,
)
from monai.utils import set_determinism
from tqdm import tqdm
import pandas as pd
import os 
import argparse
from itertools import combinations
import csv
import torch.nn as nn

def create_oasis_3_multimodal_dataset(csv_path: str, dataset_root: str, transform: Transform, cache_rate: float, modality_names: list, missing_modality: str, resolution: int):
    train_df = pd.read_csv(csv_path, sep=";")
    train_df.fillna('', inplace=True)

    train_data = []
    for index, row in train_df.iterrows():
        data_dict = {}
        has_non_empty = False
        mask = torch.zeros((len(modality_names), resolution, resolution))
        for idx, modality in enumerate(modality_names):
            file_path = row[modality]
            if file_path:
                has_non_empty = True
                data_dict[modality] = os.path.join(dataset_root, file_path)
                mask[idx] = 1
            else:
                if missing_modality == "zeros" or "mask":
                    data_dict[modality] = "empty_volume_2d.nii.gz"
                elif missing_modality == "gauss":
                    data_dict[modality] = "gauss_2d_256.nii.gz"
                else:
                    raise ValueError(f"Invalid missing modality key {missing_modality}")
        if not has_non_empty:
            continue
        data_dict["label"] = row["label"]
        data_dict["mask"] = mask
        train_data.append(data_dict)
    return CacheDataset(data=train_data, transform=transform, cache_rate=cache_rate, num_workers=5, copy_cache=False)

class SafeCropForegroundd:
    def __init__(self, keys, source_key, select_fn, margin=0):
        self.source_key = source_key
        self.crop_foreground = CropForegroundd(keys=keys, source_key=source_key, select_fn=select_fn, margin=margin)

    def __call__(self, data):
        cropped_data = self.crop_foreground(data.copy())
        cropped_image = cropped_data[self.source_key]

        # Check if any dimension (excluding batch and channel dimensions) is zero.
        if np.any(np.asarray(cropped_image.shape[1:]) == 0):
            return data  # Revert to original data if cropped size is zero in any dimension
        
        return cropped_data
    
def save_loss_plot(train_loss_values, val_loss_values, modality_names, plot_out_directory):
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title(",".join(modality_names))
    x = [i + 1 for i in range(len(train_loss_values))]
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Plot the training and validation loss
    plt.plot(x, train_loss_values, label="Training Loss")
    plt.plot(x, val_loss_values, label="Validation Loss", linestyle='--')

    # Set x-axis to display only integer values
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.legend()  # To show the labels on the graph
    plt.grid(True)  # If you want to show a grid on the graph

    # Save the figure to a file
    filename = f"loss_curve_{'_'.join(modality_names)}.png"
    plt.savefig(os.path.join(plot_out_directory, filename), dpi=300, bbox_inches='tight')  # Save with desired resolution and tight bounding box

def run_input_fusion_training(modality_names: list, dataset_root: str, epochs: int, batch_size: int, missing_modality: str, output_table_filename: str, plot_out_dir: str):
    max_epochs = epochs

    resolution = 256
    cache_rate = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    foreground_crop_threshold = 0.1
    transform_list = [
            LoadImaged(keys=modality_names, image_only=True),
            EnsureChannelFirstd(keys=modality_names + ["label"], channel_dim="no_channel"),
            CastToTyped("label", dtype=np.float64),
            ScaleIntensityd(keys=modality_names),
        ]
    for i in range(len(modality_names)):
        transform_list.append(
            SafeCropForegroundd(keys=modality_names[i], source_key=modality_names[i], select_fn=lambda x: x > foreground_crop_threshold, margin=5)
        )
    gpu_keys = ["image"]
    if missing_modality == "mask":
        gpu_keys.append("mask")
    transform_list.extend([
        Resized(keys=modality_names, spatial_size=resolution, size_mode="longest"),
        SpatialPadd(keys=modality_names, spatial_size=(resolution, resolution)),
        ConcatItemsd(keys=modality_names, name="image"),
        EnsureTyped(keys=gpu_keys, device=device),
    ]
    )
    transform = Compose(transform_list)

    train_table_path = "csv/oasis/oasis_3_multimodal_train.csv"
    train_ds = create_oasis_3_multimodal_dataset(csv_path=train_table_path, dataset_root=dataset_root, transform=transform, 
                                                 cache_rate=cache_rate, modality_names=modality_names, missing_modality=missing_modality, resolution=resolution)
    train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=batch_size, shuffle=True)

    val_table_path = "csv/oasis/oasis_3_multimodal_val.csv"
    val_ds = create_oasis_3_multimodal_dataset(csv_path=val_table_path, dataset_root=dataset_root, transform=transform, 
                                               cache_rate=cache_rate, modality_names=modality_names, missing_modality=missing_modality, resolution=resolution)
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=batch_size, shuffle=True)

    class MaskedModel(nn.Module):
        def __init__(self, densenet):
            super(MaskedModel, self).__init__()
            self.densenet = densenet

        def forward(self, x, mask):
            x = x * mask
            x = self.densenet(x)
            return x

    model = DenseNet121(spatial_dims=2, in_channels=len(modality_names), out_channels=1)
    if missing_modality == "mask":
        model = MaskedModel(densenet=model)
    model = model.to(device=device)

    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    scaler = torch.cuda.amp.GradScaler()
    val_interval = 1
    auc_metric = ROCAUCMetric()
    out_model_dir = "./pretrained/"

    modality_name_map = {
        "MR T1w": "T1",
        "MR T2w": "T2",
        "MR T2*": "T2star",
        "MR FLAIR": "FLAIR",
        "MR TOF-MRA": "MRA"
    }

    short_mod_names = [modality_name_map[mod] for mod in modality_names]

    model_file_name = f"DenseNet121_ad_cls_oasis_3_input_fusion{'_'.join(short_mod_names)}_{missing_modality}.pth"

    best_metric = -1
    best_metric_epoch = -1
    metric_values = []
    train_loss_values = []
    val_loss_values = []

    y_pred_trans = Compose([Activations(sigmoid=True)])

    for epoch in range(max_epochs):
        model.train()

        with tqdm(train_loader, unit="batch") as tepoch:
            epoch_loss = 0
            step = 0
            for batch_data in tepoch:
                step += 1
                tepoch.set_description(f"Training Epoch {epoch + 1} / {max_epochs}")

                inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
                if missing_modality == "mask":
                    mask = batch_data["mask"].to(device)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    if missing_modality == "mask":
                        outputs = model(inputs, mask)
                    else:
                        outputs = model(inputs)
                    
                    loss = loss_function(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
            
            epoch_loss /= step
            train_loss_values.append(epoch_loss)

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                epoch_val_loss = 0
                val_step = 0
                for val_data in tqdm(val_loader, "Validation"):
                    val_step += 1
                    val_images, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    model_output = model.densenet(val_images)
                    epoch_val_loss += loss_function(model_output, val_labels).item()
                    y_pred = torch.cat([y_pred, model_output], dim=0)
                    y = torch.cat([y, val_labels], dim=0)
                epoch_val_loss /= val_step
                val_loss_values.append(epoch_val_loss)
                
                y_onehot = torch.cat([i for i in decollate_batch(y, detach=False)], dim=0)
                y_pred_act = torch.cat([y_pred_trans(i) for i in decollate_batch(y_pred)], dim=0)
                auc_metric(y_pred_act, y_onehot)
                result = auc_metric.aggregate()
                auc_metric.reset()
                metric_values.append(result)
                acc_value = torch.eq((y_pred_act > 0.5).long(), y)
                acc_metric = acc_value.float().mean().item()
                del y_pred_act, y_onehot
                if result > best_metric:
                    best_metric = result
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(out_model_dir, model_file_name))
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current AUC: {result:.4f}"
                    f" current accuracy: {acc_metric:.4f}"
                    f" best AUC: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
    print("------------------------------------")
    print(f"Training completed for modalites {','.join(short_mod_names)}")
    print(f"Best AUC = {best_metric:.5f} " f"at epoch {best_metric_epoch}")

    save_loss_plot(train_loss_values=train_loss_values, val_loss_values=val_loss_values, modality_names=short_mod_names, plot_out_directory=plot_out_dir)

    with open(output_table_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([','.join(short_mod_names), best_metric, best_metric_epoch])
    
def main():
    parser = argparse.ArgumentParser(description="Multimoda Alzheimer's Classsification Training")
    parser.add_argument("--dataset", default="/mnt/f/OASIS-3-MR-Sessions-2D/", type=str, help="directory to the OASIS-3 2D dataset")
    parser.add_argument("--epochs", default=20, type=int, help="number of training epochs")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size")
    parser.add_argument("--exclude_modalities", default="", type=str, help="modalities to use")
    parser.add_argument('--ablation', action='store_true', help="Run with all modality combinations")
    parser.add_argument("--missing_modality", type=str, help="Values to use for missing modality in a sample (zeros, gauss or mask)")
    parser.add_argument("--plots", type=str, help="Directory to store loss plots in.")
    args = parser.parse_args()

    # TODO: add csv paths, model out dir, resolution, missing_modality img path, zero gradient flag
    # TODO: update learning rate and weight_decay

    all_modalities = ["MR T1w", "MR T2w", "MR T2*", "MR FLAIR", "MR TOF-MRA"]

    output_table_filename = f"input_level_fusion_{args.missing_modality}_ablation.csv"

    with open(output_table_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Modalities', 'Best AUC', 'BestEpoch'])

    if args.ablation:
        for i in range(2, len(all_modalities)):
            combos = list(combinations(all_modalities, i))
            for combination in combos:
                print(f"Running input level fusion training with modality {combination} and {args.missing_modality}...")
                run_input_fusion_training(modality_names=list(combination), dataset_root=args.dataset, epochs=args.epochs,
                                           batch_size=args.batch_size, missing_modality=args.missing_modality, output_table_filename=output_table_filename, plot_out_dir=args.plots)
    else:
        excluded_modalities = args.exclude_modalities.split(",")
        for mod in excluded_modalities:
            mod = mod.strip()
            while(mod in all_modalities):
                all_modalities.remove(mod)
        run_input_fusion_training(modality_names=all_modalities, dataset_root=args.dataset, epochs=args.epochs,
                                   batch_size=args.batch_size, missing_modality=args.missing_modality, output_table_filename=output_table_filename, plot_out_dir=args.plots)

if __name__ == "__main__":
    main()