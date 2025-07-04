# Art-Style-Classification-Image-Similarity-with-Fine-Tuning

## Overview
This project is an implementation of the research paper  
**"A Deep Learning Approach for Painting Retrieval based on Genre Similarity"**  
by Tess Masclef, Mihaela Scuturici, Benjamin Bertin, Vincent Barrellon, Vasile-Marian Scuturici, and Serge Miguet.  
You can access the original paper [here](https://link.springer.com/chapter/10.1007/978-3-031-51026-7_24).

With the rapid digitization of art collections, there is a growing need for automated tools to retrieve visually or semantically similar paintings from large databases. Manual search is inefficient and subjective, especially as collections grow. Deep learning enables the extraction of robust, high-level features from images, making it possible to compare and retrieve similar artworks efficiently and objectively.

---

## Web Application (Flask)

This project includes a **Flask-based web application** for interactive painting similarity search:

- **Upload a painting** using a simple web interface.
- The app extracts features from your uploaded image using both ResNet50 and EfficientNet models (preloaded for fast response).
- It performs a fast similarity search using the Annoy index and retrieves the top visually similar paintings from the NGA dataset.
- The app automatically selects the best model for your query based on similarity metrics and displays the top results with their angular distance and cosine similarity.
- Results are shown in a clean, user-friendly layout with side-by-side images and similarity scores.

Below is a sample screenshot of the web app in action:

![Web App Screenshot](path/to/your/screenshot.png)

You can use this web interface for quick, visual exploration of painting similarities.

---

## Core Methodology

The project follows a robust transfer learning approach for image retrieval in the artistic domain:

- **Feature Learning on WikiArt:**  
  Deep learning models are fine-tuned on the large WikiArt dataset. This crucial step ([wikiart-with-transfer-learning.ipynb](wikiart-with-transfer-learning.ipynb)) enables the models to learn generalized visual features relevant to art, such as styles, compositions, and color palettes, making them effective art feature extractors.

- **Transfer Learning to NGA:**  
  The fine-tuned models are then applied to the NGA dataset. Feature vectors (embeddings) are extracted for each of the **3,039 NGA paintings**. This transfer learning approach is vital because the NGA dataset, while substantial, is not large enough for effective deep model fine-tuning from scratch.

- **High-Speed Similarity Search:**  
  To ensure real-time query performance, the ANNOY (Approximate Nearest Neighbors Oh Yeah) library is employed. All extracted feature vectors from the NGA dataset are precomputed and indexed offline. At query time, ANNOY efficiently finds the most similar vectors in milliseconds, bypassing slow exhaustive searches.

---

## Project Structure

- **wikiart-with-transfer-learning.ipynb:** Fine-tuning ResNet50/EfficientNet on WikiArt for robust art features.
- **nga dataset preparation.ipynb:** Downloading, filtering, and preprocessing NGA images.
- **Painiting_similarity_on_NGA_Dataset.ipynb:** Feature extraction, building the search index, and running similarity queries.

---

## Datasets

### 1. WikiArt (for Fine-Tuning)
- Used to fine-tune models for generalizable art features, not for direct search.
- [Kaggle WikiArt Dataset](https://www.kaggle.com/datasets/simolopes/wikiart-all-artpieces)
- **Preprocessing:** Filtered to include only styles with 2,000–5,000 images for balanced training.

### 2. National Gallery of Art (NGA)
- **Source:** NGA Open Data
- **Final Image Count:** 3,039 paintings (after filtering for "painting" in the classification column and successful downloads)
- **Note:** The NGA metadata does not provide fine-grained genre/style labels (e.g., "portrait", "landscape"). Only broad type ("painting", "sculpture", etc.) is available.

---

## Workflow

1. **Data Preparation**
   - Filter `objects.csv` for "painting" in classification.
   - Merge with image URLs and download images (see `nga dataset preparation.ipynb`).
   - Preprocess images to a standard size (e.g., 224x224).

2. **Feature Extraction & Indexing**
   - Use fine-tuned ResNet50/EfficientNet to extract feature vectors for all NGA images.
   - Build an ANNOY index for fast similarity search.
   - Save feature arrays and index for efficient querying.

3. **Similarity Search**
   - For a query image, extract its feature vector.
   - Retrieve the top-k most similar paintings using the ANNOY index.
   - Visualize the query and results, showing angular distance and cosine similarity.

---

## Models

- **ResNet50:** Fine-tuned on WikiArt, used for NGA feature extraction.
- **EfficientNet:** Fine-tuned on WikiArt, used for NGA feature extraction.

Both models are used as fixed feature extractors for the NGA dataset.

---

## Evaluation Methods

### Visual and Distance-Based Evaluation

Since the NGA dataset does not include genre/style labels, evaluation is based on visual inspection and feature-space distance metrics:

- **Angular Distance:**  
  \( \text{Angular Distance} = \sqrt{2 \times (1 - \text{Cosine Similarity})} \)  
  Lower is better (closer in feature space).

- **Cosine Similarity:**  
  Higher is better (closer in feature space).

For each query:
- Average Angular Distance of the top-k results (lower = better).
- Average Cosine Similarity of the top-k results (higher = better).
- Standard deviation of both metrics (lower = more consistent retrieval).

**Visual Inspection:** Plot the query and top-k results with their metrics for qualitative assessment.

#### Example code:
```python
def print_similarity_stats(angular_distances, cosine_similarities, model_name="Model"):
    print(f"{model_name} Similarity Statistics:")
    print(f"  Average Angular Distance: {np.mean(angular_distances):.4f}")
    print(f"  Std Dev Angular Distance: {np.std(angular_distances):.4f}")
    print(f"  Average Cosine Similarity: {np.mean(cosine_similarities):.4f}")
    print(f"  Std Dev Cosine Similarity: {np.std(cosine_similarities):.4f}")
```

---

## Interpretation

- **Lower angular distance** and **higher cosine similarity** indicate better retrieval (images are more similar in feature space).
- **Lower standard deviation** means results are more consistent.

**Why?**  
Cosine similarity measures alignment of feature vectors (1 = identical), angular distance is a normalized dissimilarity (0 = identical).  
See [Wikipedia: Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity).

---

## Limitations

- **No genre-based quantitative evaluation:** The NGA dataset does not contain detailed genre/style labels. Precision@k by genre, as in the original paper, cannot be computed.
- **Evaluation is "label-free":** Assessment is based on feature-space distances and visual relevance, not genre.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
