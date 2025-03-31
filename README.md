# Art-Style-Classification-Image-Similarity-with-Fine-Tuning

# Art Style Classification & Image Similarity with Fine-Tuning

## Overview
This project involves fine-tuning pre-trained models, specifically **ResNet** and **EfficientNet**, for two distinct tasks:

1. **Style Classification** using the WikiArt dataset to classify artwork images into predefined styles.
2. **Image Similarity** using the National Gallery of Art dataset to find similar paintings (e.g., portraits with similar faces or poses).

The goal of this project is to demonstrate the application of **deep learning** for both **image classification** and **similarity tasks** on artwork images, leveraging **fine-tuning** on pre-trained models to improve performance on these tasks.

---

## Key Features

### Dataset 1: **WikiArt (Style Classification)**

- **Task**: Classify artwork images into predefined art styles.
- **Preprocessing**: The WikiArt dataset was filtered to include only styles that had between **2000** and **5000** images for balanced training.
- **Model**: Fine-tuned **ResNet** and **EfficientNet** models to classify images based on style.
**Source**: The WikiArt dataset is publicly available on Kaggle. You can find the dataset [here](https://www.kaggle.com/datasets/simolopes/wikiart-all-artpieces).
    
### Dataset 2: **National Gallery of Art (Image Similarity)**

- **Task**: Find similar artworks based on feature representations extracted using pre-trained models.
- **Preprocessing**: The National Gallery of Art dataset was used to extract high-level features from paintings, and cosine similarity  was used to evaluate image similarity.
**Source**: The National Gallery of Art dataset is available on the official GitHub repository. You can access it [here](https://github.com/NationalGalleryOfArt/opendata).  
---

## Models Used

- **ResNet**: Fine-tuned on the WikiArt dataset for style classification.
- **EfficientNet**: Fine-tuned on the WikiArt dataset for style classification.

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

The preprocessing steps for the National Gallery of Art dataset are provided in the `nga_dataset_preparation.ipynb` notebook.

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
