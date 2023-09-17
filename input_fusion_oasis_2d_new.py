import os
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
    NormalizeIntensityd
)
from monai.utils import set_determinism
from tqdm import tqdm
import pandas as pd
import os 
import argparse
from itertools import combinations
import csv
import torch.nn as nn
from pconv_channel import PConv2dChannel
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

set_determinism(seed=42069)

def create_oasis_3_multimodal_dataset(csv_path: str, dataset_root: str, transform: Transform, 
                                      cache_rate: float, modality_names: list, missing_modality: str, resolution: int, placeholder_dir: str):
    train_df = pd.read_csv(csv_path, sep=";")
    train_df.fillna('', inplace=True)

    train_data = []
    for index, row in train_df.iterrows():
        data_dict = {}
        has_non_empty = False
        mask = torch.zeros((len(modality_names), resolution, resolution))
        vec = torch.zeros((len(modality_names)))
        for idx, modality in enumerate(modality_names):
            file_path = row[modality]
            if file_path:
                has_non_empty = True
                data_dict[modality] = os.path.join(dataset_root, file_path)
                mask[idx] = 1
                vec[idx] = 1
            else:
                if missing_modality == "gauss":
                    data_dict[modality] = os.path.join(placeholder_dir, "gauss_2d_256.nii.gz")
                else:
                    data_dict[modality] = os.path.join(placeholder_dir, "empty_volume_2d.nii.gz")
        if not has_non_empty:
            continue
        data_dict["label"] = row["label"]
        data_dict["mask"] = mask
        data_dict["vec"] = vec
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
    
def run_input_fusion_training(
                modality_names: list,
                dataset_root: str,
                epochs: int,
                batch_size: int,
                missing_modality: str,
                output_table_filename: str, 
                learning_rate: float,
                reg_weight: float,
                csv_path: str,
                logdir: str,
                use_densnet_paper_params: bool,
                early_stopping: int,
                clean_transform: bool,
                placeholder_dir: str):
    
    max_epochs = epochs
    cache_rate = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    foreground_crop_threshold = 0.1

    resolution = 256
    roi = 196
    
    transform_list = [
            LoadImaged(keys=modality_names, image_only=True),
            EnsureChannelFirstd(keys=modality_names + ["label"], channel_dim="no_channel"),
            CastToTyped("label", dtype=np.float64),
        ]
    if clean_transform:
        gpu_keys = ["image"]
        if missing_modality == "mask":
            gpu_keys.append("mask")
        transform_list.extend([
            CropForegroundd(keys=modality_names,  source_key="T1", k_divisible=[roi, roi]),
            ConcatItemsd(keys=modality_names, name="image"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=gpu_keys, device=device),
        ]
        )
    else:
        for i in range(len(modality_names)):
            transform_list.append(
                SafeCropForegroundd(keys=modality_names[i], source_key=modality_names[i], select_fn=lambda x: x > foreground_crop_threshold, margin=5)
            )
        gpu_keys = ["image"]
        if missing_modality == "mask":
            gpu_keys.append("mask")
        if missing_modality == "gate":
            gpu_keys.append("vec")
        transform_list.extend([
            Resized(keys=modality_names, spatial_size=resolution, size_mode="longest"),
            SpatialPadd(keys=modality_names, spatial_size=(resolution, resolution)),
            ConcatItemsd(keys=modality_names, name="image"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=gpu_keys, device=device),
        ]
        )
    transform = Compose(transform_list)

    if use_densnet_paper_params:
        batch_size = 32

    train_table_path = os.path.join(csv_path, "oasis_3_multimodal_train.csv")
    train_ds = create_oasis_3_multimodal_dataset(csv_path=train_table_path, dataset_root=dataset_root, transform=transform, 
                                                 cache_rate=cache_rate, modality_names=modality_names, missing_modality=missing_modality,
                                                   resolution=resolution, placeholder_dir=placeholder_dir)
    train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=batch_size, shuffle=True)

    val_table_path = os.path.join(csv_path, "oasis_3_multimodal_val.csv")
    val_ds = create_oasis_3_multimodal_dataset(csv_path=val_table_path, dataset_root=dataset_root, transform=transform, 
                                               cache_rate=cache_rate, modality_names=modality_names, missing_modality=missing_modality,
                                                 resolution=resolution, placeholder_dir=placeholder_dir)
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=batch_size, shuffle=True)

    class MaskedModel(nn.Module):
        def __init__(self, densenet):
            super(MaskedModel, self).__init__()
            self.densenet = densenet

        def forward(self, x, mask):
            x = x * mask
            x = self.densenet(x)
            return x
        
    class PConvModel(nn.Module):
        def __init__(self, in_channels: int):
            super(PConvModel, self).__init__()
            pconv_out_channels = 5
            self.pconv = PConv2dChannel(
                in_channels=in_channels,
                out_channels=pconv_out_channels,
                kernel_size=7,
                stride=1,
                padding=2,
                dilation=2,
                bias=True
            )
            self.densenet = DenseNet121(spatial_dims=2, in_channels=pconv_out_channels, out_channels=1)

        def forward(self, x, mask):
            x, _ = self.pconv(x, mask)
            x = self.densenet(x)
            return x
        
    class MaskedSoftmax(nn.Module):
        def __init__(self, dim=1):
            super(MaskedSoftmax, self).__init__()
            self.dim = dim

        def forward(self, logits, mask):
            # Set masked logits to a large negative value
            masked_logits = logits + ((1 - mask) * -1e9)
            masked_logits = masked_logits + (mask * 10)
            return F.softmax(masked_logits, dim=self.dim)
        
    class GatedModel(nn.Module):
        def __init__(self, in_channels: int):
            super(GatedModel, self).__init__()
            
            hidden_dim = 256
            self.in_channels = in_channels

            self.modality_scaler = nn.Sequential(
                nn.Linear(in_features=in_channels, out_features=hidden_dim),
                nn.ReLU(),
                nn.Linear(in_features=hidden_dim, out_features=in_channels),
            )

            self.densenet = DenseNet121(spatial_dims=2, in_channels=in_channels, out_channels=1)

        def forward(self, x, vec):
            logits = self.modality_scaler(vec)
            modality_weights = (MaskedSoftmax()(logits, vec) / torch.sum(vec, 1).unsqueeze(-1)) * self.in_channels
            modality_weights = modality_weights.unsqueeze(-1).unsqueeze(-1)
            x = self.densenet(x * modality_weights)
            return x

    model = DenseNet121(spatial_dims=2, in_channels=len(modality_names), out_channels=1)
    if missing_modality == "mask":
        model = MaskedModel(densenet=model)
    elif missing_modality == "pconv":
        model = PConvModel(in_channels=len(modality_names))
    elif missing_modality == "gate":
        model = GatedModel(in_channels=len(modality_names))

    model = model.to(device=device)

    loss_function = torch.nn.BCEWithLogitsLoss()

    if use_densnet_paper_params:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0125, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg_weight)

    scaler = torch.cuda.amp.GradScaler()
    auc_metric = ROCAUCMetric()
    out_model_dir = "./pretrained/"

    os.makedirs(out_model_dir, exist_ok=True)

    modality_name_map = {
        "T1": "T1",
        "T2": "T2",
        "T2*": "T2star",
        "FLAIR": "FLAIR",
        "MRA": "MRA"
    }

    short_mod_names = [modality_name_map[mod] for mod in modality_names]

    model_file_name = f"DenseNet121_ad_cls_oasis_3_input_fusion_{'_'.join(short_mod_names)}_{missing_modality}.pth"

    best_metric = -1
    best_metric_epoch = -1
    metric_values = []

    y_pred_trans = Compose([Activations(sigmoid=True)])

    writer = SummaryWriter(log_dir=os.path.join(logdir, f"{'_'.join(short_mod_names)}_{missing_modality}"))
    prev_epoch_val_loss = 100000
    epochs_without_val_loss_improvement = 0
    for epoch in range(max_epochs):

        # if epochs_without_val_loss_improvement >= early_stopping:
        #     print("Early stopping...")
        #     break

        if (epoch + 1) % 10 == 0:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * 0.1

        model.train()
        avg_epoch_train_loss = 0
        avg_epoch_val_loss = 0
        with tqdm(train_loader, unit="batch") as tepoch:
            epoch_loss = 0
            step = 0
            for batch_data in tepoch:
                step += 1
                tepoch.set_description(f"Training Epoch {epoch + 1} / {max_epochs}")

                inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
                if missing_modality == "mask" or missing_modality == "pconv":
                    mask = batch_data["mask"].to(device)
                elif missing_modality == "gate":
                    vec = batch_data["vec"].to(device)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    if missing_modality == "mask" or missing_modality == "pconv":
                        outputs = model(inputs, mask)
                    elif missing_modality == "gate":
                        outputs = model(inputs, vec)
                    else:
                        outputs = model(inputs)
                    
                    loss = loss_function(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
            
            epoch_loss /= step
            avg_epoch_train_loss = epoch_loss

        # Validation
        model.eval()
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            epoch_val_loss = 0
            val_step = 0
            for val_data in tqdm(val_loader, "Validation"):
                val_step += 1
                val_images, val_labels, val_masks, val_vecs = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                    val_data["mask"].to(device),
                    val_data["vec"].to(device),
                )
                if missing_modality == "mask":
                    model_output = model.densenet(val_images)
                elif missing_modality == "pconv":
                    model_output = model(val_images, val_masks)
                elif missing_modality == "gate":
                    model_output = model(val_images, val_vecs)
                else:
                    model_output = model(val_images)
                
                epoch_val_loss += loss_function(model_output, val_labels).item()
                y_pred = torch.cat([y_pred, model_output], dim=0)
                y = torch.cat([y, val_labels], dim=0)
            epoch_val_loss /= val_step
            avg_epoch_val_loss = epoch_val_loss

            if epoch_val_loss >= prev_epoch_val_loss:
                epochs_without_val_loss_improvement += 1

            prev_epoch_val_loss = epoch_val_loss
            
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
        writer.add_scalars("Loss", {"Training": avg_epoch_train_loss, "Validation": avg_epoch_val_loss}, epoch)
        if missing_modality == "gate":
            vec = torch.ones(len(modality_names))
            dev_vec = vec.to(device)
            print(f"Weights all:")
            print(MaskedSoftmax(dim=0)(model.modality_scaler(dev_vec), dev_vec))
            
            for idx, name in enumerate(modality_names):
                vec_mod = torch.ones(len(modality_names))
                vec_mod[idx] = 0
                dev_vec = vec_mod.to(device)
                print(f"Weights w/o {name}:")
                print(MaskedSoftmax(dim=0)(model.modality_scaler(dev_vec), dev_vec))

    print("------------------------------------")
    print(f"Training completed for modalites {','.join(short_mod_names)}")
    print(f"Best AUC = {best_metric:.5f} " f"at epoch {best_metric_epoch}")

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
    parser.add_argument("--missing_modality", type=str, help="Values to use for missing modality in a sample (zeros, gauss, mask, pconv, gate)")
    parser.add_argument("--optim_lr", default=1e-5, type=float, help="optimization learning rate")
    parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
    parser.add_argument("--csv_path", type=str, help="directory to the dataset csv files")
    parser.add_argument("--logdir", type=str, help="Directory to store tensorboard logs in")
    parser.add_argument("--use_densnet_paper_params",  action='store_true', help="Run with all modality combinations")
    parser.add_argument("--roi", default=196, type=int, help="Region of interest")
    parser.add_argument("--early_stopping", default=5, type=int, help="Number of to wait for improve in validation loss before stopping.")
    parser.add_argument("--clean_transform",  action='store_true', help="Use transforms for the clean dataset")
    parser.add_argument("--placeholders",  type=str, help="Directory that contains images to use for missing modalities")
    args = parser.parse_args()

    all_modalities = ["T1", "T2", "T2*", "FLAIR", "MRA"]

    output_table_filename = f"input_level_fusion_{args.missing_modality}_ablation.csv"

    with open(output_table_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Modalities', 'Best AUC', 'BestEpoch'])

    if args.ablation:
        for i in range(2, len(all_modalities)):
            combos = list(combinations(all_modalities, i))
            for combination in combos:
                print(f"Running input level fusion training with modality {combination} and {args.missing_modality}...")
                run_input_fusion_training(modality_names=list(combination),
                                            dataset_root=args.dataset,
                                            epochs=args.epochs,
                                            batch_size=args.batch_size,
                                            missing_modality=args.missing_modality,
                                            output_table_filename=output_table_filename,
                                            learning_rate=args.optim_lr,
                                            reg_weight=args.reg_weight,
                                            csv_path=args.csv_path,
                                            logdir=args.logdir,
                                            use_densnet_paper_params=args.use_densnet_paper_params,
                                            early_stopping=args.early_stopping,
                                            clean_transform=args.clean_transform,
                                            placeholder_dir=args.placeholders)
    else:
        excluded_modalities = args.exclude_modalities.split(",")
        for mod in excluded_modalities:
            mod = mod.strip()
            while(mod in all_modalities):
                all_modalities.remove(mod)
        print(f"Running input level fusion training with modality {all_modalities} and {args.missing_modality}...")
        run_input_fusion_training(modality_names=all_modalities,
                                    dataset_root=args.dataset,
                                    epochs=args.epochs,
                                    batch_size=args.batch_size,
                                    missing_modality=args.missing_modality,
                                    output_table_filename=output_table_filename,
                                    learning_rate=args.optim_lr,
                                    reg_weight=args.reg_weight,
                                    csv_path=args.csv_path,
                                    logdir=args.logdir,
                                    use_densnet_paper_params=args.use_densnet_paper_params,
                                    early_stopping=args.early_stopping,
                                    clean_transform=args.clean_transform,
                                    placeholder_dir=args.placeholders)

if __name__ == "__main__":
    main()