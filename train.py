import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse
from model.convnext import ConvNeXt
from data.isic import ISICDataSet
from data.covid import ChestXrayDataSet
import os
import timm


def download_model():
    model_convnext: ConvNeXt = timm.create_model("convnext_large", pretrained=True)
    # save the model
    torch.save(model_convnext.state_dict(), "convnext_large.pth")


def parse_args():
    parser = argparse.ArgumentParser(description="Train ConvNeXt model")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to model checkpoint to resume training",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training and testing"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of epochs to train"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of classes")
    parser.add_argument(
        "--dataset", type=str, default="isic", help="Dataset to use (isic or covid)"
    )
    parser.add_argument(
        "--train_data_dir", type=str, required=True, help="Path to training data"
    )
    parser.add_argument(
        "--train_list_file", type=str, required=True, help="Path to training list file"
    )
    parser.add_argument(
        "--test_data_dir", type=str, required=True, help="Path to testing data"
    )
    parser.add_argument(
        "--test_list_file", type=str, required=True, help="Path to testing list file"
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of data loading workers"
    )
    parser.add_argument(
        "--save_dir", type=str, default=".", help="Directory to save model checkpoints"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_args = dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])
    model = ConvNeXt(**model_args)
    os.makedirs(args.save_dir, exist_ok=True)
    if args.resume:
        num_features = (
            model.num_features
        )  # Get the number of input features for the classification head
        model.head.fc = nn.Linear(num_features, args.num_classes).to(device)
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        if not os.path.exists("convnext_large.pth"):
            print("Downloading pre-trained ConvNeXt model")
            download_model()
            print("Model downloaded")
        checkpoint = torch.load("convnext_large.pth", map_location=device)
        model.load_state_dict(checkpoint)
        num_features = (
            model.num_features
        )  # Get the number of input features for the classification head
        model.head.fc = nn.Linear(num_features, args.num_classes).to(device)

        # Freeze all layers except the classifier head
        for param in model.parameters():
            param.requires_grad = True
        for param in model.head.fc.parameters():
            param.requires_grad = True
    model.to(device)

    train_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize(
                256, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
            ),
            transforms.RandomRotation(30),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.Resize(
                256, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    if args.dataset == "isic":
        train_dataset = ISICDataSet(
            data_dir=args.train_data_dir,
            image_list_file=args.train_list_file,
            use_melanoma=True,
            transform=train_transforms,
        )
        test_dataset = ISICDataSet(
            data_dir=args.test_data_dir,
            image_list_file=args.test_list_file,
            use_melanoma=True,
            transform=test_transforms,
        )
    elif args.dataset == "covid":
        train_dataset = ChestXrayDataSet(
            data_dir=args.train_data_dir,
            image_list_file=args.train_list_file,
            use_covid=True,
            transform=train_transforms,
        )
        test_dataset = ChestXrayDataSet(
            data_dir=args.test_data_dir,
            image_list_file=args.test_list_file,
            use_covid=True,
            transform=test_transforms,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0.0
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct_preds / total_preds
        print(
            f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )

        model.eval()
        correct_preds = 0
        total_preds = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct_preds += (predicted == labels).sum().item()
                total_preds += labels.size(0)

        test_accuracy = 100 * correct_preds / total_preds
        print(f"Test Accuracy after epoch {epoch+1}: {test_accuracy:.2f}%")

        if test_accuracy > best_accuracy:
            print(
                f"Accuracy improved! Saving model with {test_accuracy:.2f}% accuracy."
            )
            best_accuracy = test_accuracy
            torch.save(
                model.state_dict(), f"{args.save_dir}/best_model_epoch_{epoch+1}.pth"
            )

    print("Training complete.")


if __name__ == "__main__":
    main()
