import os
import faiss
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from torchvision import transforms
from model.convnext import ConvNeXt
from data.isic import ISICDataSet
from data.covid import ChestXrayDataSet
import json
import argparse
import pickle
from gmp import GlobalMaxPooling2d

def parse_args():
    parser = argparse.ArgumentParser(description='Image Retrieval Configuration')
    parser.add_argument('--resume', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, choices=['isic', 'covid'], required=True, help='Dataset to use')
    parser.add_argument('--test_data_dir', type=str, required=True, help='Path to test data directory')
    parser.add_argument('--test_list_file', type=str, required=True, help='Path to test list file')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--pca_components', type=int, default=128, help='Number of PCA components')
    parser.add_argument('--save_index_dir', type=str, default='faiss_index', help='Directory to save index')
    return parser.parse_args()


def main():
    print("Loading model")
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_args = dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])
    model = ConvNeXt(**model_args)
    num_features = model.num_features
    model.head.fc = nn.Linear(num_features, args.num_classes).to(device)
    checkpoint = torch.load(args.resume, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    print("Model loaded")
    gmp = GlobalMaxPooling2d().to(device)

    test_transforms = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if args.dataset == 'isic':
        test_dataset = ISICDataSet(
            data_dir=args.test_data_dir,
            image_list_file=args.test_list_file,
            use_melanoma=True,
            transform=test_transforms
        )
    elif args.dataset == 'covid':
        test_dataset = ChestXrayDataSet(
            data_dir=args.test_data_dir,
            image_list_file=args.test_list_file,
            use_covid=True,
            transform=test_transforms
        )
    print(f"Number of images in test dataset: {len(test_dataset)}")

    print(f"Building index for {args.dataset} dataset")
    image_embeddings = []
    image_ids_labels = {}
    with torch.no_grad():
        for idx, (image, label) in enumerate(tqdm(test_dataset)):
            image = image.to(device)
            image = image.unsqueeze(0)
            features = model.forward_features(image)
            features = gmp(features)
            image_embeddings.append(features.cpu().numpy())
            image_ids_labels[idx] = label.item()

    image_embeddings = np.vstack(image_embeddings)
    image_embeddings = normalize(image_embeddings, norm='l2')

    pca = PCA(n_components=args.pca_components)
    image_embeddings_pca = pca.fit_transform(image_embeddings)

    dimension = image_embeddings_pca.shape[1]
    index = faiss.IndexFlatIP(dimension) # Similarity is calculated using inner product
    index.train(image_embeddings_pca.astype(np.float32))
    index.add(image_embeddings_pca.astype(np.float32))
    print("Index built")
    print("Saving index")
    os.makedirs(args.save_index_dir, exist_ok=True)
    faiss.write_index(index, f"{args.save_index_dir}/index.faiss")

    with open(f'{args.save_index_dir}/image_ids_labels.json', 'w') as f:
        json.dump(image_ids_labels, f)
    
    with open(f'{args.save_index_dir}/test_dataset.pkl', 'wb') as f:
        pickle.dump(test_dataset, f)

    with open(f'{args.save_index_dir}/pca.pkl', 'wb') as f:
        pickle.dump(pca, f)
    print("Index saved")


if __name__ == '__main__':
    main()
