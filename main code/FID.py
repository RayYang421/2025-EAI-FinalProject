import tensorflow as tf
import numpy as np
import os
import cv2
from scipy import linalg
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration
# ==========================================
BASELINE_MODEL_PATH = './generator_baseline.keras'           # Original (unpruned) generator
PRUNED_FT_MODEL_PATH = './fine_tuned_pruned_generator.keras' # Pruned + fine-tuned generator
DATA_DIR = 'shoes_data/ut-zap50k-images-square'              # Dataset root directory

# Evaluation settings
IMG_HW = 128
EVAL_SAMPLES = 2000   # Number of samples used for FID
BATCH_SIZE = 32       # Batch size for evaluation

# GPU setup
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ==========================================
# 2. InceptionV3 Model Setup
# ==========================================
def get_inception_model():
    inception = tf.keras.applications.InceptionV3(
        include_top=False,
        pooling='avg',
        input_shape=(299, 299, 3)
    )
    return inception

# ==========================================
# 3. Data Loading Utilities
# ==========================================
def list_all_files(directory):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files


def load_data_for_eval(image_file):
    # Load real shoe image
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [IMG_HW, IMG_HW])

    # Load corresponding edge image
    edge_path = tf.strings.regex_replace(image_file, '.jpg', '.edges.jpg')
    edge_image = tf.io.read_file(edge_path)
    edge_image = tf.image.decode_jpeg(edge_image, channels=1)
    edge_image = tf.image.convert_image_dtype(edge_image, tf.float32)
    edge_image = tf.image.resize(edge_image, [IMG_HW, IMG_HW])

    # Normalize to [-1, 1] for generator input
    norm_image = (image * 2.0) - 1.0
    norm_edge = (edge_image * 2.0) - 1.0

    # Return:
    #   edge input (normalized),
    #   real image (normalized),
    #   real image in [0,1] (for visualization / FID)
    return norm_edge, norm_image, image

# ==========================================
# 4. FID Computation
# ==========================================
def calculate_fid_score(real_features, fake_features):
    mu1 = real_features.mean(axis=0)
    sigma1 = np.cov(real_features, rowvar=False)

    mu2 = fake_features.mean(axis=0)
    sigma2 = np.cov(fake_features, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))

    # Numerical stability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# ==========================================
# 5. Feature Extraction
# ==========================================
def get_features(inception_model, generator, dataset, num_samples, is_real=False):
    features = []

    for edges, _, real_imgs_01 in tqdm(
        dataset.take(num_samples // BATCH_SIZE),
        desc="Extracting Features"
    ):
        if is_real:
            # Use real images directly
            imgs_to_process = tf.image.resize(real_imgs_01, [299, 299])
        else:
            # Generate images from edges
            generated = generator(edges, training=False)
            generated_01 = (generated * 0.5) + 0.5
            imgs_to_process = tf.image.resize(generated_01, [299, 299])

        # Scale to Inception input range [-1, 1]
        imgs_to_process = (imgs_to_process * 2.0) - 1.0

        batch_features = inception_model.predict(imgs_to_process, verbose=0)
        features.append(batch_features)

    features = np.concatenate(features, axis=0)
    return features

# ==========================================
# 6. Main Evaluation Pipeline
# ==========================================
def main():
    print("[INFO] Preparing evaluation dataset...")
    shoes_dirs = list_all_files(DATA_DIR)
    shoes_dirs = [f for f in shoes_dirs if 'edges' not in f and '.jpg' in f]

    import random
    random.shuffle(shoes_dirs)
    if len(shoes_dirs) > EVAL_SAMPLES:
        shoes_dirs = shoes_dirs[:EVAL_SAMPLES]

    list_ds = tf.data.Dataset.from_tensor_slices(shoes_dirs)
    ds = list_ds.map(load_data_for_eval).batch(BATCH_SIZE)

    # Pre-fetch a fixed batch for visualization (fair comparison)
    print("[INFO] Pre-fetching visualization batch...")
    vis_edges, _, vis_real = next(iter(ds))
    vis_baseline_imgs = None

    print("[INFO] Loading InceptionV3...")
    inception = get_inception_model()

    # 1. Real image features
    print("\n[Step 1/3] Processing real images...")
    real_features = get_features(inception, None, ds, EVAL_SAMPLES, is_real=True)

    # 2. Baseline model
    print(f"\n[Step 2/3] Evaluating baseline model: {BASELINE_MODEL_PATH}...")
    try:
        baseline_gen = tf.keras.models.load_model(BASELINE_MODEL_PATH)

        # Generate visualization samples before releasing the model
        print("   -> Generating baseline visualization samples...")
        baseline_output = baseline_gen(vis_edges, training=False)
        vis_baseline_imgs = (baseline_output * 0.5) + 0.5

        baseline_features = get_features(
            inception, baseline_gen, ds, EVAL_SAMPLES, is_real=False
        )
        fid_baseline = calculate_fid_score(real_features, baseline_features)
        print(f"   >>> Baseline FID Score: {fid_baseline:.4f}")

        del baseline_gen
        tf.keras.backend.clear_session()

    except Exception as e:
        print(f"   [Error] Failed to load baseline model: {e}")
        fid_baseline = None

    # 3. Pruned + fine-tuned model
    print(f"\n[Step 3/3] Evaluating pruned model: {PRUNED_FT_MODEL_PATH}...")
    try:
        pruned_gen = tf.keras.models.load_model(PRUNED_FT_MODEL_PATH)

        pruned_features = get_features(
            inception, pruned_gen, ds, EVAL_SAMPLES, is_real=False
        )
        fid_pruned = calculate_fid_score(real_features, pruned_features)
        print(f"   >>> Pruned + FT FID Score: {fid_pruned:.4f}")

        # Generate visualization samples
        pruned_output = pruned_gen(vis_edges, training=False)
        vis_pruned_imgs = (pruned_output * 0.5) + 0.5

        # Visual comparison: Edge / Real / Baseline / Pruned
        print("   -> Generating visual comparison (4 rows)...")
        num_cols = min(4, BATCH_SIZE)

        plt.figure(figsize=(12, 10))

        for i in range(num_cols):
            # Input edge
            plt.subplot(4, num_cols, i + 1)
            plt.imshow((vis_edges[i] * 0.5 + 0.5)[:, :, 0], cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title("Input Edge", fontsize=12, fontweight='bold', loc='left')

            # Ground truth
            plt.subplot(4, num_cols, i + 1 + num_cols)
            plt.imshow(vis_real[i])
            plt.axis('off')
            if i == 0:
                plt.title("Real Shoe", fontsize=12, fontweight='bold', loc='left')

            # Baseline output
            plt.subplot(4, num_cols, i + 1 + num_cols * 2)
            if vis_baseline_imgs is not None:
                plt.imshow(vis_baseline_imgs[i])
            else:
                plt.text(0.5, 0.5, "Load Failed", ha='center')
            plt.axis('off')
            if i == 0:
                plt.title("Baseline Model", fontsize=12, fontweight='bold', loc='left')

            # Pruned output
            plt.subplot(4, num_cols, i + 1 + num_cols * 3)
            plt.imshow(vis_pruned_imgs[i])
            plt.axis('off')
            if i == 0:
                plt.title("Pruned Model", fontsize=12, fontweight='bold', loc='left')

        plt.tight_layout()
        plt.savefig('fid_comparison_with_baseline.png')
        print("   -> Visual comparison saved to 'fid_comparison_with_baseline.png'")

    except Exception as e:
        print(f"   [Error] Failed to load pruned model: {e}")
        fid_pruned = None

    # Final summary
    print("\n" + "=" * 40)
    print("FINAL RESULTS")
    print("=" * 40)
    if fid_baseline is not None:
        print(f"Baseline FID    : {fid_baseline:.4f}")
    if fid_pruned is not None:
        print(f"Pruned + FT FID : {fid_pruned:.4f}")
    if fid_baseline is not None and fid_pruned is not None:
        diff = fid_pruned - fid_baseline
        print(f"Difference      : {diff:.4f} ({'Improved' if diff < 0 else 'Degraded'})")
    print("=" * 40)


if __name__ == "__main__":
    main()
