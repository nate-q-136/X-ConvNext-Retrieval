# Image Retrieval Pipeline

This project demonstrates how to build an image retrieval system using ConvNeXt and Faiss. The pipeline includes training the model, building the index, and evaluating retrieval performance.

## Setup

### Requirements
- Python 3.8+

Install dependencies:
```bash
pip install -r requirements.txt
```

## Workflow

### 1. Train the Model
Train a ConvNeXt model on the ISIC or COVID datasets.

**Example:**
```bash
python train.py \
  --train_data_dir path/to/train_data \
  --train_list_file path/to/train_list.csv \
  --test_data_dir path/to/test_data \
  --test_list_file path/to/test_list.csv \
  --batch_size 8 \
  --num_epochs 10 \
  --lr 1e-4 \
  --num_classes 3 \
  --save_dir checkpoints
```

### 2. Build Faiss Index
Generate embeddings and build a Faiss index for fast image retrieval.

**Example:**
```bash
python build_index.py \
  --resume checkpoints/best_model_epoch_19.pth \
  --dataset isic \
  --test_data_dir path/to/test_data \
  --test_list_file path/to/test_list.csv \
  --pca_components 128 \
  --save_index_dir faiss_index
```

### 3. Test Image Retrieval
Retrieve similar images and evaluate the performance of the retrieval system.

**Example:**
```bash
python test.py \
  --resume checkpoints/best_model_epoch_19.pth \
  --num_classes 3 \
  --save_index_dir faiss_index \
  --k 5 \
  --save_result_path retrieval_results.csv
```

## Results
- The retrieval performance (mAP@K, P@1, P@K) will be saved to `retrieval_results.csv`.
- The index and PCA model are stored in the `faiss_index` directory.

## File Descriptions
- **train.py** – Train the ConvNeXt model.
- **build_index.py** – Build the image index using embeddings and PCA.
- **test.py** – Retrieve and evaluate similar images from the dataset.
- **model/** – Contains ConvNeXt model definition.
- **data/** – Contains ISIC and COVID dataset loading classes.

## Notes
- Ensure that the dataset paths and checkpoint paths are correctly specified.
- For ISIC and COVID datasets, adjust the number of classes using the `--num_classes` argument.
- Use the `--dataset` argument to switch between ISIC and COVID datasets.

## Acknowledgements
- ConvNeXt model from [timm](https://github.com/rwightman/pytorch-image-models)
- Faiss library from Facebook AI Research

