import os
import torch
import numpy as np
from sklearn.metrics import classification_report

from monai.data import decollate_batch, CacheDataset, ThreadDataLoader
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121, swin_unetr
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
    CastToTyped,
)
from monai.utils import set_determinism
from tqdm import tqdm
import pandas as pd
import os
from monai.transforms import Transform
import argparse
import random
import torch.nn as nn
import torch.optim as optim

def set_seed(no):
    torch.manual_seed(no)
    random.seed(no)
    np.random.seed(no)
    os.environ['PYTHONHASHSEED'] = str()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(100)

modality_names = ["MR T1w", "MR T2w", "MR T2*", "MR FLAIR", "MR TOF-MRA"]

# dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')#384
# dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')#768
dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')#1024
# dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')#1536

#bce adam sol

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.2, gamma=0.7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):

        pt = torch.sigmoid(inputs)
        pos_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = -pos_weight * (1 - pt)**self.gamma * pt.log()
        return loss.mean()

class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self, model_name):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False),#unimodal case -> 1 o/w len(modality_names)
            nn.ReLU(),
            nn.BatchNorm2d(3)
        )

        self.lin_in = 0
        self.lin_out = 0

        self.transformer = None

        if model_name == "dinov2_vits14":
            self.transformer = dinov2_vits14
            self.lin_in = 384
            self.lin_out = 256            
        elif model_name == "dinov2_vitb14":
            self.transformer = dinov2_vitb14
            self.lin_in = 768
            self.lin_out = 512     
        elif model_name == "dinov2_vitl14":
            self.transformer = dinov2_vitl14
            self.lin_in = 1024
            self.lin_out = 682      
        elif model_name == "dinov2_vitg14":
            self.transformer = dinov2_vitg14
            self.lin_in = 1536
            self.lin_out = 1024         



        self.classifier = nn.Sequential(
            nn.Linear(self.lin_in, self.lin_out),
            nn.ReLU(),
            nn.Linear(self.lin_out, 1)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x


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
    parser.add_argument("--dataset", default=r"D:\from_ubuntu\OASIS-3-MR-Sessions-2D\\", type=str, help="directory to the OASIS-3 2D dataset")
    parser.add_argument("--exclude_modality", default="", type=str, help="modality to exclude")
    args = parser.parse_args()

    while(args.exclude_modality in modality_names):
        modality_names.remove(args.exclude_modality)

    print(modality_names)

    resolution = 224
    cache_rate = 1.0 # might need to change this based on the amount of memory available
    batch_size = 1

    dataset_root = args.dataset
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
    transform_list.extend([
        Resized(keys=modality_names, spatial_size=resolution, size_mode="longest"),
        SpatialPadd(keys=modality_names, spatial_size=(resolution, resolution)),
        EnsureTyped(keys=modality_names, device=device),
    ]
    )
    transform = Compose(transform_list)

    val_table_path = "oasis_3_multimodal_val.csv"
    val_ds = create_oasis_3_multimodal_dataset(csv_path=val_table_path, dataset_root=dataset_root, transform=transform, cache_rate=cache_rate)
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=batch_size, shuffle=True)

    model_paths = {
        "MR T1w": r"D:\saved_models\\dinov2_vitl14_unimodal_MR T1w_bce_adam_oasis_3.pth",
        "MR T2w": r"D:\saved_models\\dinov2_vitl14_unimodal_MR T2w_bce_adam_oasis_3.pth",
        "MR T2*": r"D:\saved_models\\dinov2_vitl14_unimodal_T2s_bce_adam_oasis_3.pth",
        "MR FLAIR": r"D:\saved_models\\dinov2_vitl14_unimodal_MR FLAIR_bce_adam_oasis_3.pth",
        "MR TOF-MRA": r"D:\saved_models\\dinov2_vitl14_unimodal_TOF_bce_adam_oasis_3.pth"
    }

    models = {}
    for modality in modality_names:
        
        model = DinoVisionTransformerClassifier(model_name="dinov2_vitl14")
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
        for val_data in tqdm(val_loader):

            y_pred_product_sample = 1
            y_pred_max_sample = 0
            i = 0
            for modality in modality_names:
                if not val_data["has " + modality]:
                    continue
                i += 1
                model = models[modality]
                val_images = val_data[modality].to(device)
                pred = y_pred_trans(decollate_batch(model(val_images)))

                y_pred_product_sample *= pred[0]
                y_pred_max_sample = max(y_pred_max_sample, pred[0])

            if i == 0:
                continue

            val_labels = val_data["label"].to(device)
            y_pred_product = torch.cat([y_pred_product, y_pred_product_sample], dim=0)
            
            if y_pred_max_sample == 0:
                y_pred_max_sample = torch.tensor([y_pred_max_sample]).to(device)
                
            y_pred_max = torch.cat([y_pred_max, y_pred_max_sample], dim=0)

            y = torch.cat([y, val_labels], dim=0)

        y_onehot = y.cpu()

        auc_metric_prod(y_pred_product.cpu(), y_onehot)
        result_prod = auc_metric_prod.aggregate()

        auc_metric_max(y_pred_max.cpu(), y_onehot)
        result_max = auc_metric_max.aggregate()

        print('Modality names: ', modality_names)
        print(f"Product Fusion AUC = {result_prod}")
        print(f"Max Fusion AUC = {result_max}")

if __name__ == "__main__":
    main()