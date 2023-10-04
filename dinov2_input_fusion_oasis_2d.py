import os
import matplotlib.pyplot as plt
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
import random
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score

def set_seed(no):
    torch.manual_seed(no)
    random.seed(no)
    np.random.seed(no)
    os.environ['PYTHONHASHSEED'] = str()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(100)

modality_names = ["MR T1w", "MR T2w", "MR T2*", "MR FLAIR", "MR TOF-MRA"]


dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')#384
dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')#768
dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')#1024
# dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')#1536

out_model_dir = r"multimodal\unimodal_pretrained_other_dinos\\"


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
    def __init__(self, model_name, in_dim):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 3, kernel_size=1, stride=1, padding=0, bias=False),#unimodal case -> 1 o/w len(modality_names)
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



def create_oasis_3_multimodal_dataset(csv_path: str, dataset_root: str, transform: Transform, cache_rate: float, modality_names: list, missing_modality: str):
    train_df = pd.read_csv(csv_path, sep=";")
    train_df.fillna('', inplace=True)

    train_data = []
    for index, row in train_df.iterrows():
        data_dict = {}
        has_non_empty = False
        for modality in modality_names:
            file_path = row[modality]
            if file_path:
                has_non_empty = True
                data_dict[modality] = os.path.join(dataset_root, file_path)
            else:
                if missing_modality == 'zeros':
                    data_dict[modality] = r"empty_volume_2d.nii.gz"
                elif missing_modality == 'gauss':
                    data_dict[modality] = r"gauss_2d_256.nii.gz"
        if not has_non_empty:
            continue
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
    parser = argparse.ArgumentParser(description="Multimoda Alzheimer's Classsification Training")
    parser.add_argument("--dataset", default=r"D:\from_ubuntu\OASIS-3-MR-Sessions-2D\\", type=str, help="directory to the OASIS-3 2D dataset")
    parser.add_argument("--epochs", default=8, type=int, help="number of training epochs")
    parser.add_argument("--batch_size", default=4, type=int, help="batch size")
    parser.add_argument("--exclude_modalities", default="", type=str, help="modalities to use")
    args = parser.parse_args()


    models = ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14']# , 'dinov2_vitg14']
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    modality_namess = [["MR T1w", "MR T2w", "MR T2*", "MR FLAIR", "MR TOF-MRA"], ["MR T1w", "MR T2w", "MR T2*", "MR FLAIR"], 
                    ["MR T1w", "MR T2w", "MR T2*"], ["MR T1w", "MR T2w"], ["MR T1w", "MR T2w", "MR FLAIR"]]
    losses = ['bce', 'focal', 'bce_focal']
    optimizers = ['adam', 'sgd']
    missing_modalities = ['zeros', 'gauss']
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


    model_res_to_print = []
    for model_name in models:

        loss_function_1 = FocalLoss()
        loss_function_2 = torch.nn.BCEWithLogitsLoss()
        # optimizer1 = optim.Adam(model.parameters(), lr=1e-5)
        # optimizer2 = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
        val_interval = 1
        auc_metric = ROCAUCMetric()

        for modality in modality_namess:


            dataset_root = args.dataset
            max_epochs = args.epochs
            batch_size = args.batch_size

            resolution = 224
            cache_rate = 1.0
            for loss_f in losses:
                for optmzr in optimizers:
                    if optmzr == "sgd":
                        model = DinoVisionTransformerClassifier(model_name=model_name, in_dim=len(modality))
                        model = model.to(device)
                        scaler = torch.cuda.amp.GradScaler()
                        optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
                    elif optmzr == "adam":
                        model = DinoVisionTransformerClassifier(model_name=model_name, in_dim=len(modality))
                        model = model.to(device)
                        scaler = torch.cuda.amp.GradScaler()
                        optimizer = optim.Adam(model.parameters(), lr=1e-5)

                    for missing_modal in missing_modalities:

                        foreground_crop_threshold = 0.1
                        transform_list = [
                                LoadImaged(keys=modality, image_only=True),
                                EnsureChannelFirstd(keys=modality + ["label"], channel_dim="no_channel"),
                                CastToTyped("label", dtype=np.float64),
                                ScaleIntensityd(keys=modality),
                            ]
                        for i in range(len(modality)):
                            transform_list.append(
                                SafeCropForegroundd(keys=modality[i], source_key=modality[i], select_fn=lambda x: x > foreground_crop_threshold, margin=5)
                            )
                        transform_list.extend([
                            Resized(keys=modality, spatial_size=resolution, size_mode="longest"),
                            SpatialPadd(keys=modality, spatial_size=(resolution, resolution)),
                            ConcatItemsd(keys=modality, name="image"),
                            EnsureTyped(keys=["image"], device=device),
                        ]
                        )
                        transform = Compose(transform_list)

                        train_table_path = "oasis_3_multimodal_train.csv"
                        train_ds = create_oasis_3_multimodal_dataset(csv_path=train_table_path, dataset_root=dataset_root, transform=transform, cache_rate=cache_rate, modality_names=modality, missing_modality= missing_modal)
                        train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=batch_size, shuffle=True)

                        val_table_path = "oasis_3_multimodal_val.csv"
                        val_ds = create_oasis_3_multimodal_dataset(csv_path=val_table_path, dataset_root=dataset_root, transform=transform, cache_rate=cache_rate, modality_names=modality, missing_modality= missing_modal)
                        val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=batch_size, shuffle=True)


                        best_metric = -1
                        best_metric_epoch = -1
                        best_modal_accuracy = -1
                        best_model_precision = -1
                        best_model_recall = -1
                        best_model_f1 = -1
                        metric_values = []

                        y_pred_trans = Compose([Activations(sigmoid=True)])

                        model_file_name = ""


                        if "MR TOF-MRA" in modality:
                            model_file_name = model_name + "_multimodal_input_level_" + "all_" + loss_f + "_" + optmzr + "_oasis_3.pth"
                        elif len(modality) == 4:
                            model_file_name = model_name + "_multimodal_input_level_" + "T1_T2_T2s_FLAIR_" + missing_modal + '_' + loss_f + "_" + optmzr + "_oasis_3.pth"
                        elif len(modality) == 2:
                            model_file_name = model_name + "_multimodal_input_level_" + "T1_T2_" + missing_modal + loss_f + "_" + optmzr + "_oasis_3.pth"
                        elif len(modality) == 3 and "MR FLAIR" in modality:
                            model_file_name = model_name + "_multimodal_input_level_" + "T1_T2_FLAIR" + missing_modal + loss_f + "_" + optmzr + "_oasis_3.pth"         
                        elif len(modality) == 3 and "MR T2*" in modality:
                            model_file_name = model_name + "_multimodal_input_level_" + "T1_T2_T2s" + missing_modal + loss_f + "_" + optmzr + "_oasis_3.pth"                            
                        

                        for epoch in range(max_epochs):
                            model.train()

                            with tqdm(train_loader, unit="batch") as tepoch:
                                for batch_data in tepoch:
                                    tepoch.set_description(f"Training Epoch {epoch + 1} / {max_epochs}")

                                    inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
                                    optimizer.zero_grad()
                                    with torch.cuda.amp.autocast():
                                        outputs = model(inputs)
                                        if loss_f == 'focal':
                                            loss = loss_function_1(outputs, labels) # Focal Loss
                                        elif loss_f == 'bce':
                                            loss = loss_function_2(outputs, labels) # BCE Loss
                                        elif loss_f == 'bce_focal':
                                            loss_1 = loss_function_1(outputs, labels)
                                            loss_2 = loss_function_2(outputs, labels)
                                            loss = 0.8*loss_2 + 0.2*loss_1
                                    
                                    scaler.scale(loss).backward()
                                    scaler.step(optimizer)
                                    scaler.update()

                                    tepoch.set_postfix(loss=loss.item())

                            if (epoch + 1) % val_interval == 0:
                                model.eval()
                                with torch.no_grad():
                                    y_pred = torch.tensor([], dtype=torch.float32, device=device)
                                    y = torch.tensor([], dtype=torch.long, device=device)
                                    for val_data in tqdm(val_loader, "Validation"):
                                        val_images, val_labels = (
                                            val_data["image"].to(device),
                                            val_data["label"].to(device),
                                        )
                                        y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                                        y = torch.cat([y, val_labels], dim=0)
                                    y_onehot = torch.cat([i for i in decollate_batch(y, detach=False)], dim=0)
                                    y_pred_act = torch.cat([y_pred_trans(i) for i in decollate_batch(y_pred)], dim=0)
                                    auc_metric(y_pred_act, y_onehot)
                                    result = auc_metric.aggregate()
                                    auc_metric.reset()
                                    metric_values.append(result)
                                    acc_value = torch.eq((y_pred_act > 0.5).long(), y)
                                    acc_metric = acc_value.float().mean().item()



            
                                    # Convert predictions to binary (1 or 0) based on threshold of 0.5
                                    y_pred_binary = (y_pred_act > 0.5).long()


                                    # Convert tensors to numpy for sklearn metrics
                                    y_numpy = y_onehot.cpu().numpy()
                                    y_pred_binary_numpy = y_pred_binary.cpu().numpy()

                                    # Precision, Recall and F1
                                    precision = precision_score(y_numpy, y_pred_binary_numpy, zero_division=0)
                                    recall = recall_score(y_numpy, y_pred_binary_numpy, zero_division=0)
                                    f1 = f1_score(y_numpy, y_pred_binary_numpy, zero_division=0)

                                    
                                    del y_pred_act, y_onehot
                                    if result > best_metric:
                                        best_metric = result
                                        best_metric_epoch = epoch + 1
                                        best_modal_accuracy = acc_metric
                                        best_model_precision = precision
                                        best_model_recall = recall
                                        best_model_f1 = f1
                                        torch.save(model.state_dict(), os.path.join(out_model_dir, model_file_name))
                                        print("saved new best metric model")
                                    print(
                                        f"current epoch: {epoch + 1} current AUC: {result:.4f}"
                                        f" current accuracy: {acc_metric:.4f}"
                                        f" best AUC: {best_metric:.4f}"
                                        f" Best Precision: {best_model_precision:.4f}"
                                        f" Best Recall: {best_model_recall:.4f}"
                                        f" Best F1: {best_model_f1:.4f}"

                                        f" at epoch: {best_metric_epoch}"
                                    )

                            if (epoch + 1) - best_metric_epoch > 4:
                                break

                        print(f"Training completed - best AUC = {best_metric:.4f} " f"at epoch {best_metric_epoch}")   
                        res_to_print = model_file_name + " " + "best AUC: " + str(best_metric) + " Best Precision: " + str(best_model_precision) + " Best Recall: " + str(best_model_recall) + " Best F1: " + str(best_model_f1) +  " at epoch: " + str(best_metric_epoch)
                        model_res_to_print.append(res_to_print)

    for item in model_res_to_print:
        print(item)

if __name__ == "__main__":
    main() 