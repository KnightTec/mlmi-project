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
    Transform,
)
from monai.utils import set_determinism
from tqdm import tqdm
import pandas as pd
import os
from monai.transforms import Transform
import argparse

modality_names = ["MR T1w", "MR T2w", "MR T2*", "MR FLAIR", "MR TOF-MRA"]

def create_oasis_3_multimodal_dataset(csv_path: str, dataset_root: str, transform: Transform, cache_rate: float):
    train_df = pd.read_csv(csv_path, sep=";")
    train_df.fillna('', inplace=True)

    train_data = []
    for index, row in train_df.iterrows():
        data_dict = {}
        for modality in modality_names:
            file_path = row[modality]
            if file_path:
                data_dict[modality] = os.path.join(dataset_root, file_path)
            else:
                data_dict[modality] = "empty_volume_2d.nii.gz"
            data_dict["has " + modality] = bool(file_path)
    
        data_dict["label"] = row["label"]
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

def main():
    parser = argparse.ArgumentParser(description="Unimodal Alzheimer's Classsification Training")
    parser.add_argument("--dataset", default="/mnt/f/OASIS-3-MR-Sessions-2D/", type=str, help="directory to the OASIS-3 2D dataset")
    parser.add_argument("--exclude_modality", default="", type=str, help="modality to exclude")
    args = parser.parse_args()

    while(args.exclude_modality in modality_names):
        modality_names.remove(args.exclude_modality)

    resolution = 256
    cache_rate = 1.0 # might need to change this based on the amount of memory available
    batch_size = 1

    dataset_root = args.dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    foreground_crop_threshold = 0.1

    transform_list = [
        LoadImaged(keys=modality_names, image_only=True),
        EnsureChannelFirstd(keys=modality_names, channel_dim="no_channel"),  
        ScaleIntensityd(keys=modality_names),
        Resized(keys=modality_names, spatial_size=resolution, size_mode="longest"),
        SpatialPadd(keys=modality_names, spatial_size=(resolution, resolution))
    ]
    for i in range(len(modality_names)):
        transform_list.append(
            SafeCropForegroundd(keys=modality_names[i], source_key=modality_names[i], select_fn=lambda x: x > foreground_crop_threshold, margin=5)
        )
    transform_list.extend([
        Resized(keys=modality_names, spatial_size=resolution, size_mode="longest"),
        SpatialPadd(keys=modality_names, spatial_size=(resolution, resolution)),
        EnsureTyped(keys=modality_names, device=device),
    ]
    )
    transform = Compose(transform_list)

    val_table_path = "csv/oasis/oasis_3_multimodal_val.csv"
    val_ds = create_oasis_3_multimodal_dataset(csv_path=val_table_path, dataset_root=dataset_root, transform=transform, cache_rate=cache_rate)
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=batch_size, shuffle=True)

    model_paths = {
        "MR T1w": "pretrained/DenseNet121_ad_cls_oasis_3_MR_T1.pth",
        "MR T2w": "pretrained/DenseNet121_ad_cls_oasis_3_MR_T2.pth",
        "MR T2*": "pretrained/DenseNet121_ad_cls_oasis_3_MR_T2_STAR.pth",
        "MR FLAIR": "pretrained/DenseNet121_ad_cls_oasis_3_MR_FLAIR.pth",
        "MR TOF-MRA": "pretrained/DenseNet121_ad_cls_oasis_3_MR_TOF_MRA.pth"
    }
    models = {}
    for modality in modality_names:
        model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=1)
        model.load_state_dict(torch.load(model_paths[modality], map_location=device))
        model.to(device=device)
        models[modality] = model

    auc_metric_prod = ROCAUCMetric()
    auc_metric_max = ROCAUCMetric()
    
    y_pred_trans = Compose([Activations(sigmoid=True)])

    for mod, model in models.items():
        model.eval()

    with torch.no_grad():
        y_pred_product = torch.tensor([], dtype=torch.float32, device=device)
        y_pred_max = torch.tensor([], dtype=torch.float32, device=device)
        y = torch.tensor([], dtype=torch.long, device=device)
        i = 0
        for val_data in tqdm(val_loader):

            y_pred_product_sample = 1
            y_pred_max_sample = 0
            for modality in modality_names:
                if not val_data["has " + modality]:
                    continue
                
                model = models[modality]
                val_images = val_data[modality].to(device)
                pred = y_pred_trans(decollate_batch(model(val_images)))

                y_pred_product_sample *= pred[0]
                y_pred_max_sample = max(y_pred_max_sample, pred[0])

            val_labels = val_data["label"].to(device)
            y_pred_product = torch.cat([y_pred_product, y_pred_product_sample], dim=0)
            y_pred_max = torch.cat([y_pred_max, y_pred_max_sample], dim=0)

            y = torch.cat([y, val_labels], dim=0)

        y_onehot = y.cpu()

        auc_metric_prod(y_pred_product.cpu(), y_onehot)
        result_prod = auc_metric_prod.aggregate()

        auc_metric_max(y_pred_max.cpu(), y_onehot)
        result_max = auc_metric_max.aggregate()

        print(f"Product Fusion AUC = {result_prod}")
        print(f"Max Fusion AUC = {result_max}")

if __name__ == "__main__":
    main()