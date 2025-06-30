import os
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import Model
from annoy import AnnoyIndex

# ---- CONFIG ----
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH_RESNET = 'best_resnet_finetuned.keras'
MODEL_PATH_EFFICIENTNET = 'best_efficientnet_finetuned.keras'
FEATURES_PATH_RESNET = 'nga_data/resnet_features_2.npy'
FEATURES_PATH_EFFICIENTNET = 'nga_data/efficientnet_features_2.npy'
ANNOY_PATH_RESNET = 'nga_data/resnet_annoy_2.ann'
ANNOY_PATH_EFFICIENTNET = 'nga_data/efficientnet_annoy_2.ann'
PATHS_TXT_RESNET = 'nga_data/resnet_features_2_paths.txt'
PATHS_TXT_EFFICIENTNET = 'nga_data/efficientnet_features_2_paths.txt'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models and indexes once at startup
resnet_model = tf.keras.models.load_model(MODEL_PATH_RESNET, compile=False)
resnet_feature_extractor = Model(inputs=resnet_model.inputs, outputs=resnet_model.get_layer(name="dense").output)
resnet_features = np.load(FEATURES_PATH_RESNET)
with open(PATHS_TXT_RESNET) as f:
    resnet_image_paths = [line.strip() for line in f]
resnet_annoy = AnnoyIndex(resnet_features.shape[1], 'angular')
resnet_annoy.load(ANNOY_PATH_RESNET)

effnet_model = tf.keras.models.load_model(MODEL_PATH_EFFICIENTNET, compile=False)
effnet_feature_extractor = Model(inputs=effnet_model.inputs, outputs=effnet_model.get_layer(name="dense_2").output)
effnet_features = np.load(FEATURES_PATH_EFFICIENTNET)
with open(PATHS_TXT_EFFICIENTNET) as f:
    effnet_image_paths = [line.strip() for line in f]
effnet_annoy = AnnoyIndex(effnet_features.shape[1], 'angular')
effnet_annoy.load(ANNOY_PATH_EFFICIENTNET)

def get_query_feature(feature_extractor, query_img_path, layer_name):
    img = tf.keras.preprocessing.image.load_img(query_img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    if layer_name == "dense":
        img_array = tf.keras.applications.resnet.preprocess_input(img_array)
    else:
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return feature_extractor.predict(img_array, verbose=0)[0]

def angular_distance(vec1, vec2):
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)
    cosine_sim = np.dot(vec1_norm, vec2_norm)
    angular_dist = np.sqrt(2 * (1 - cosine_sim))
    return angular_dist, cosine_sim

def get_topk_similar(query_img_path, feature_extractor, features, annoy_index, image_paths, layer_name, top_k=3):
    query_feature = get_query_feature(feature_extractor, query_img_path, layer_name)
    similar_indices, annoy_dists = annoy_index.get_nns_by_vector(query_feature, top_k+1, include_distances=True)
    results = []
    for idx in similar_indices:
        img_path = image_paths[idx]
        if os.path.basename(img_path) != os.path.basename(query_img_path):
            ang_dist, cos_sim = angular_distance(query_feature, features[idx])
            results.append({
                'img': img_path,
                'ang_dist': ang_dist,
                'cos_sim': cos_sim
            })
        if len(results) == top_k:
            break
    avg_ang = np.mean([r['ang_dist'] for r in results])
    avg_cos = np.mean([r['cos_sim'] for r in results])
    return results, avg_ang, avg_cos

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)

            # ResNet
            resnet_results, resnet_avg_ang, resnet_avg_cos = get_topk_similar(
                filepath, resnet_feature_extractor, resnet_features, resnet_annoy, resnet_image_paths, "dense", top_k=3
            )
            # EfficientNet
            effnet_results, effnet_avg_ang, effnet_avg_cos = get_topk_similar(
                filepath, effnet_feature_extractor, effnet_features, effnet_annoy, effnet_image_paths, "dense_2", top_k=3
            )
            # Choose best model (higher avg cosine similarity is better)
            if resnet_avg_cos >= effnet_avg_cos:
                best_model = "ResNet50"
                best_results = resnet_results
            else:
                best_model = "EfficientNet"
                best_results = effnet_results

            # Convert image paths for HTML display
            for r in best_results:
                if r['img'].startswith('images/'):
                    r['img'] = r['img']
                else:
                    r['img'] = os.path.relpath(r['img'], 'static')

            return render_template('results.html',
                                   query_image=filepath,
                                   best_model=best_model,
                                   results=best_results)
    return render_template('index.html')

@app.route('/api/similarity', methods=['POST'])
def api_similarity():
    file = request.files.get('query_image')
    if not file:
        return jsonify({'error': 'No image uploaded'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)

    # ResNet
    resnet_results, resnet_avg_ang, resnet_avg_cos = get_topk_similar(
        filepath, resnet_feature_extractor, resnet_features, resnet_annoy, resnet_image_paths, "dense", top_k=3
    )
    # EfficientNet
    effnet_results, effnet_avg_ang, effnet_avg_cos = get_topk_similar(
        filepath, effnet_feature_extractor, effnet_features, effnet_annoy, effnet_image_paths, "dense_2", top_k=3
    )
    if resnet_avg_cos >= effnet_avg_cos:
        best_model = "ResNet50"
        best_results = resnet_results
    else:
        best_model = "EfficientNet"
        best_results = effnet_results
    return jsonify({
        'best_model': best_model,
        'results': [
            {
                'img': url_for('static', filename=r['img']),
                'angular_distance': float(r['ang_dist']),
                'cosine_similarity': float(r['cos_sim'])
            } for r in best_results
        ]
    })

if __name__ == '__main__':
    app.run(debug=True)