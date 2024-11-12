import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from tqdm import tqdm
from vka_model import VKA as vka 


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

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

    test_dataset = datasets.ImageFolder(root="/path/to/data/test", transform=data_transform["test"])
    test_num = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=nw)
    print("using  {} images for test.".format(test_num))

    net = vka(num_classes=2)  
    net.to(device)

        net_weight_path = "/model/pth/vka.pth"
    net.load_state_dict(torch.load(net_weight_path), strict=False)
    print("Using model:", net_weight_path)

    net.eval()
    acc = 0.0  
    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout)
        for test_images, test_labels in test_bar:
            outputs = net(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]

            for pred, true in zip(predict_y.cpu().numpy(), test_labels.cpu().numpy()):
                print(f"Predicted: {pred}, True: {true}")

            acc += torch.eq(predict_y, test_labels.to(device)).sum().item()

    test_accurate = acc / test_num
    print(acc,test_num)
    print('test_accuracy: %.4f' % (test_accurate))

    print('Finished test')

if __name__ == '__main__':
    main()
