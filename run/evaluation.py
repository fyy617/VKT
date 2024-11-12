import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision import transforms, datasets
from sklearn.metrics import roc_auc_score, confusion_matrix
from thop import profile
from vka_model import VKA as vka 


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))

    data_transform = {
        "test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    batch_size = 8
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    test_dataset = datasets.ImageFolder(root="/path/to/dataset/test", transform=data_transform["test"])
    test_num = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=nw)
    print("Using {} images for testing.".format(test_num))

    net = vka(num_classes=num_classes)  
    net.to(device)

    net_weight_path = "/path/to/model/vka.pth"
    torch.cuda.empty_cache()
    
    # Load model with specified map_location
    state_dict = torch.load(net_weight_path, map_location=device)
    net.load_state_dict(state_dict, strict=False)
    print("Using model:", net_weight_path)

    # Calculate the number of parameters and FLOPs
    test_input = torch.randn(3, 3, 224, 224).to(device)  # Example input tensor
    macs, params = profile(net, inputs=(test_input,), verbose=False)
    print(f'Number of parameters: {params / 1e6:.4f} million')
    print(f'FLOPs: {macs / 1e9:.4f} billion')

    net.eval()
    
    # Initialize metrics
    acc = 0.0  # Accumulate accurate number / epoch
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout)
        for test_images, test_labels in test_bar:
            outputs = net(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            
            # Collect predictions and labels for AUC-ROC
            all_preds.extend(predict_y.cpu().numpy())
            all_labels.extend(test_labels.cpu().numpy())

            # Update accuracy
            acc += (predict_y == test_labels.to(device)).sum().item()

    # Convert to numpy arrays for further processing
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate Test Accuracy
    test_accurate = acc / test_num
    print(f'Test Accuracy: {test_accurate:.4f}')

    # Calculate Precision, Sensitivity, Specificity, F1-score
    cm = confusion_matrix(all_labels, all_preds)
    TP = cm[1, 1]  # Assuming class 1 is the positive class
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    recall = sensitivity  # Recall is the same as Sensitivity

    # Calculate AUC-ROC
    try:
        auc_roc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc_roc = float('nan')  # In case of error, e.g., only one class present

    # Calculate MAE, MSE, RMSE (for classification, this may not be meaningful)
    mae = np.mean(np.abs(all_preds - all_labels))
    mse = np.mean((all_preds - all_labels) ** 2)
    rmse = np.sqrt(mse)

    # Print all calculated metrics
    print('Overall Accuracy (OA): {:.4f}'.format(test_accurate))
    print('Precision: {:.4f}'.format(precision))
    print('Sensitivity (Recall): {:.4f}'.format(sensitivity))
    print('Specificity: {:.4f}'.format(specificity))
    print('F1-score: {:.4f}'.format(f1_score))
    print('AUC-ROC: {:.4f}'.format(auc_roc))
    print('Mean Absolute Error (MAE): {:.4f}'.format(mae))
    print('Mean Squared Error (MSE): {:.4f}'.format(mse))
    print('Root Mean Squared Error (RMSE): {:.4f}'.format(rmse))

    print('Finished test')

if __name__ == '__main__':
    main()
