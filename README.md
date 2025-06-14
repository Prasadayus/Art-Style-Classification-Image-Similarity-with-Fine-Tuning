# Art-Style-Classification-Image-Similarity-with-Fine-Tuning

## Overview
This project is an implementation of the research paper  
**"A Deep Learning Approach for Painting Retrieval based on Genre Similarity"**  
by Tess Masclef, Mihaela Scuturici, Benjamin Bertin, Vincent Barrellon, Vasile-Marian Scuturici, and Serge Miguet.  
You can access the original paper [here](https://link.springer.com/chapter/10.1007/978-3-031-51026-7_24).

With the rapid digitization of art collections, there is a growing need for automated tools to retrieve visually or semantically similar paintings from large databases. Manual search is inefficient and subjective, especially as collections grow. Deep learning enables the extraction of robust, high-level features from images, making it possible to compare and retrieve similar artworks efficiently and objectively.

In this project, we fine-tuned pre-trained deep learning models (**ResNet** and **EfficientNet**) on the large and diverse WikiArt dataset to learn generalizable visual features for artworks. These models were then applied to the National Gallery of Art (NGA) dataset for image similarity retrieval. We developed a full pipeline for preprocessing, feature extraction, similarity search, and evaluation. The NGA dataset was carefully preprocessed (see `nga_dataset_preparation.ipynb`) to ensure data quality, which included removing corrupted, duplicate, or low-quality images and standardizing metadata. This necessary step further reduced the dataset size, reinforcing the need for transfer learning from WikiArt.

---

## Key Features

### Dataset 1: **WikiArt (Feature Learning for Transfer)**

- **Task**: Used for fine-tuning deep learning models (ResNet and EfficientNet) to learn robust and generalizable visual features from a large and diverse collection of artwork images.  
  The primary goal  enable effective feature extraction that can be transferred to other datasets for image similarity tasks.
- **Preprocessing**: The WikiArt dataset was filtered to include only styles that had between **2000** and **5000** images for balanced training.
- **Model**: Fine-tuned **ResNet** and **EfficientNet** models to extract transferable features from artwork images.  
**Source**: The WikiArt dataset is publicly available on Kaggle. You can find the dataset [here](https://www.kaggle.com/datasets/simolopes/wikiart-all-artpieces).

### Dataset 2: **National Gallery of Art (Image Similarity)**

- The **National Gallery of Art dataset** was used to extract features from artwork images.
- **ResNet** and **EfficientNet** were fine-tuned for feature extraction.
- The images were then processed using **Cosine Similarity** or **Euclidean Distance** to find the most similar paintings.

#### Preprocessing

The preprocessing steps for the National Gallery of Art dataset are detailed in the `nga_dataset_preparation.ipynb` notebook.  
Due to preprocessing—such as removing corrupted, duplicate, or low-quality images, and ensuring consistent metadata—the final usable dataset from the National Gallery of Art is significantly smaller. This reduction in data volume is necessary to maintain data quality and integrity for deep learning tasks, but it also means the dataset is not large enough for effective model training or fine-tuning. As a result, models are fine-tuned on the larger WikiArt dataset and then used for feature extraction and similarity retrieval on the NGA dataset.

---

## Rationale for Fine-Tuning on WikiArt and Applying to the NGA Dataset

- The National Gallery of Art (NGA) dataset contains a limited number of images, which is insufficient for training or fine-tuning deep neural networks effectively.
- The WikiArt dataset is significantly larger and more diverse, making it suitable for fine-tuning deep learning models to learn generalizable visual features relevant to artworks.
- By fine-tuning on WikiArt, the model acquires rich and transferable representations of artistic images.
- These learned features are then applied to the NGA dataset for image similarity retrieval, enabling effective performance despite the limited data available in NGA.
- This transfer learning approach is supported by established practices in deep learning and is reflected in the workflow and data analysis in the provided Jupyter notebooks, particularly `nga_dataset_preparation.ipynb`.

---

## Models Used

- **ResNet**: Fine-tuned on the WikiArt dataset for feature extraction.
- **EfficientNet**: Fine-tuned on the WikiArt dataset for feature extraction.

---

## Dataset Creation & Preprocessing

### 1. **WikiArt Dataset (Style Classification)**

#### Filtered Art Styles
- Images from the WikiArt dataset were filtered for **art styles** that had between **2000** and **5000** images.
- The dataset was shuffled and split into **training**, **validation**, and **test** sets.

#### Dataset Structure

- **Training Set**: 70% of the images.
- **Validation Set**: 15% of the images.
- **Test Set**: 15% of the images.

#### Directory Creation

- Directories were created for **each art style** to separate images into the appropriate categories.
- Subdirectories for **train**, **validation**, and **test** sets were created for each style.

#### Image Copying

- Images were copied into their respective directories based on their art style and split ratio.

### 2. **National Gallery of Art Dataset (Image Similarity)**

- The **National Gallery of Art dataset** was used to extract features from artwork images.
- **ResNet** and **EfficientNet** were fine-tuned for feature extraction.
- The images were then processed using **Cosine Similarity** or **Euclidean Distance** to find the most similar paintings.

#### Preprocessing

The preprocessing steps for the National Gallery of Art dataset are detailed in the `nga_dataset_preparation.ipynb` notebook.  
Due to preprocessing—such as removing corrupted, duplicate, or low-quality images, and ensuring consistent metadata—the final usable dataset from the National Gallery of Art is significantly smaller. This reduction in data volume is necessary to maintain data quality and integrity for deep learning tasks, but it also means the dataset is not large enough for effective model training or fine-tuning. As a result, models are fine-tuned on the larger WikiArt dataset and then used for feature extraction and similarity retrieval on the NGA dataset.

---

## Evaluation Methods

### Title-Based Metrics

1. **Precision@K**: 
   - Measures how many retrieved artworks have a similar title to the query artwork.
   - Evaluates the top **K** retrieved artworks based on their title similarity.

2. **Average Precision (AP)**: 
   - Computes relevance-weighted precision for retrievals, ranking artworks based on their relevance to the query.

### Visual Feature-Based Metrics

1. **Cosine Similarity**: 
   - Measures the similarity between image features by calculating the cosine of the angle between feature vectors.
   - A higher cosine similarity indicates visually more similar images.

2. **Mean Similarity**: 
   - Computes the average similarity across all retrieved images.

3. **Feature Precision**: 
   - Percentage of retrieved images exceeding a similarity threshold.

### Combined Metrics

- A **weighted combination** of title-based and visual similarity scores is used to get a more holistic evaluation.

### Evaluation Process

The evaluation process consists of:
1. **Feature Similarity Metrics**: Measures similarity between query and retrieved images based on extracted features.
2. **Visual Similarity Evaluation**: Computes the **mean similarity** and **feature precision** from a sample of images.
3. **Quality Metric Calculation**: The **Quality Metric** is computed to assess the overall retrieval quality.

### Quality Metric Formula

The **Quality Metric** formula is inspired by the paper:

**"A Deep Learning Approach for Painting Retrieval based on Genre Similarity"** by Tess Masclef, Mihaela Scuturici, Benjamin Bertin, Vincent Barrellon, Vasile-Marian Scuturici, and Serge Miguet.

The formula is:

**Quality Metric = k / (n * k_approximate)**

Where:
- **k** = Number of relevant results retrieved.
- **n** = Total number of retrieved images.
- **k_approximate** = Approximate number of relevant results.

This metric helps evaluate the overall retrieval quality by considering the number of relevant images retrieved and their approximate relevance.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
