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
import numpy as np
import faiss
import torch
from torch import nn
from tqdm import tqdm
import pandas as pd

class IndexBuilder:
    def __init__(
        self,
        model: ConvNeXt,
        gmp: GlobalMaxPooling2d,
        pca: PCA,
        index: faiss.IndexFlatL2,
        image_ids_labels,
    ):
        self.model = model
        self.gmp = gmp
        self.pca = pca
        self.index = index
        self.image_ids_labels = image_ids_labels


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def retrieve_similar_images(index_builder: IndexBuilder, query_image, k=5):
    query_image = query_image.unsqueeze(0).to(device)  # Add batch dimension
    # query_embedding = model(query_image)  # Get the embedding for the query image
    query_embedding = index_builder.model.forward_features(query_image)  # deit
    # query_embedding = query_embedding.flatten(start_dim=1) # deit
    query_embedding = index_builder.gmp(query_embedding)
    # Convert the query embedding to numpy array
    query_embedding = query_embedding.cpu().detach().numpy()
    query_embedding = normalize(query_embedding, norm="l2")
    query_embedding = index_builder.pca.transform(query_embedding)
    # Search the Faiss index for the top k nearest neighbors
    distances, indices = index_builder.index.search(
        query_embedding.astype(np.float32), k
    )

    return indices[0], distances[0]

def compute_p_at_k(retrieved_indices, ground_truth_labels, k=5):
    """
    Tính Precision tại k cho một truy vấn.
    """
    relevant_count = 0
    for i in range(k):
        if retrieved_indices[i] in ground_truth_labels:
            relevant_count += 1
    return relevant_count / k

def compute_ap_at_k(retrieved_indices, ground_truth_labels, k=5):
    """
    Tính Average Precision tại k cho một truy vấn.
    """
    relevant_count = 0
    ap = 0.0

    for i in range(k):
        if retrieved_indices[i] in ground_truth_labels:
            relevant_count += 1
            ap += relevant_count / (i + 1)

    # Nếu không có mục liên quan, trả về 0
    if relevant_count == 0:
        return 0.0

    return ap / k

def compute_map_at_k(retrieved_indices_list, ground_truth_labels_list, k=5):
    """
    Tính Mean Average Precision tại k.
    """
    ap_list = []

    for retrieved_indices, ground_truth_labels in zip(retrieved_indices_list, ground_truth_labels_list):
        ap = compute_ap_at_k(retrieved_indices, ground_truth_labels, k)
        ap_list.append(ap)

    return np.mean(ap_list)

def compute_precision_at_k(retrieved_indices_list, ground_truth_labels_list, k=5):
    """
    Tính Precision tại k cho tất cả các truy vấn.
    """
    precision_list = []

    for retrieved_indices, ground_truth_labels in zip(retrieved_indices_list, ground_truth_labels_list):
        precision = compute_p_at_k(retrieved_indices, ground_truth_labels, k)
        precision_list.append(precision)

    return np.mean(precision_list)


def parse_args():
    parser = argparse.ArgumentParser(description='Test Image Retrieval')
    parser.add_argument('--resume', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of output classes')
    parser.add_argument('--save_index_dir', type=str, default='faiss_index', help='Directory to save index')
    parser.add_argument('--k', type=int, default=5, help='Number of nearest neighbors to retrieve')
    parser.add_argument('--save_result_path', type=str, default='retrieval_results.csv', help='Path to save retrieval results')
    return parser.parse_args()

def main():
    print("Loading model")
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(f'{args.save_index_dir}/test_dataset.pkl', 'rb') as f:
        test_dataset = pickle.load(f)

    with open(f'{args.save_index_dir}/image_ids_labels.json', 'r') as f:
        image_ids_labels = json.load(f)

    with open(f'{args.save_index_dir}/pca.pkl', 'rb') as f:
        pca = pickle.load(f)

    index = faiss.read_index(f'{args.save_index_dir}/index.faiss')

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

    index_builder = IndexBuilder(
        model=model,
        gmp=gmp,
        pca=pca,
        index=index,
        image_ids_labels=image_ids_labels
    )

    retrieved_indices_list = []
    ground_truth_labels_list = []

    print("Evaluating retrieval performance")
    for i in tqdm(range(len(test_dataset))):
        query_image, ground_truth_labels = test_dataset[i]
        similar_image_ids, distances = retrieve_similar_images(index_builder, query_image, k=args.k + 1)
        sorted_indices = np.argsort(distances)[1:]
        sorted_similar_image_ids = np.array(similar_image_ids)[sorted_indices]
        labels_retrieved = [index_builder.image_ids_labels[str(idx)] for idx in sorted_similar_image_ids]
        retrieved_indices_list.append(labels_retrieved)
        ground_truth_labels_list.append(ground_truth_labels)

    map_at_k = compute_map_at_k(retrieved_indices_list, ground_truth_labels_list, k=args.k)
    print(f"Mean Average Precision at {args.k} (mAP@{args.k}): {map_at_k:.4f}")

    p_at_1 = compute_precision_at_k(retrieved_indices_list, ground_truth_labels_list, k=1)
    p_at_k = compute_precision_at_k(retrieved_indices_list, ground_truth_labels_list, k=args.k)

    print(f"Precision at 1 (P@1): {p_at_1:.4f}")
    print(f"Precision at {args.k} (P@{args.k}): {p_at_k:.4f}")

    df = pd.DataFrame({
        f'mAP@{args.k}': [map_at_k],
        f'P@1': [p_at_1],
        f'P@{args.k}': [p_at_k]
    })

    df.to_csv(args.save_result_path, index=False)
    print(f"Results saved to {args.save_result_path}")

if __name__ == '__main__':
    main()
