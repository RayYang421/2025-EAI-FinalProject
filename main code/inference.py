import os
# ==============================================================================
# IMPORTANT:
# This environment variable MUST be set before importing TensorFlow.
# It avoids protobuf-related crashes such as "File already exists".
# ==============================================================================
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Use scikit-image instead of OpenCV to avoid ABI / dependency conflicts
from skimage import io, transform, color, feature

# ==========================================
# 1. Configuration
# ==========================================
MODEL_PATH = './fine_tuned_pruned_generator.keras'
INPUT_IMAGE_PATH = 'test_shoes3.jpg'
OUTPUT_DIR = 'inference_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_HW = 128  # Input resolution expected by the generator

# ==========================================
# 2. Image preprocessing
# ==========================================
def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load input image
    original_img = io.imread(image_path)

    # Remove alpha channel if present
    if original_img.shape[-1] == 4:
        original_img = original_img[:, :, :3]

    print(f"[INFO] Resizing image to {IMG_HW}x{IMG_HW}...")

    # Resize to model input resolution
    resized_img = transform.resize(
        original_img,
        (IMG_HW, IMG_HW),
        anti_aliasing=True
    ).astype(np.float32)

    # Convert to grayscale for edge detection
    gray = color.rgb2gray(resized_img)

    # Heuristic to distinguish photo vs. line-art using saturation
    hsv = color.rgb2hsv(resized_img)
    mean_saturation = hsv[:, :, 1].mean()

    # Generate edge map
    if mean_saturation > 0.1:
        # Likely a color photo → apply Canny edge detector
        print("[INFO] Detected color image, applying Canny edge detection...")
        edges = feature.canny(gray, sigma=2.0)
        edge_img = edges.astype(np.float32)[:, :, None]
    else:
        # Likely a sketch / binary drawing
        print("[INFO] Detected grayscale / sketch image...")
        if gray.mean() > 0.5:
            # Invert to match white-on-black edge convention
            print("[INFO] Inverting image to match edge convention...")
            edges = 1.0 - gray
        else:
            edges = gray

        edges = edges > 0.5
        edge_img = edges.astype(np.float32)[:, :, None]

    # Normalize edge input to [-1, 1], consistent with training
    edge_img = (edge_img * 2.0) - 1.0

    # Add batch dimension
    input_batch = tf.expand_dims(edge_img, axis=0)

    return input_batch, resized_img, edge_img

# ==========================================
# 3. Main
# ==========================================
def main():
    print(f"[INFO] Loading model: {MODEL_PATH}")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    print(f"[INFO] Processing input image: {INPUT_IMAGE_PATH}")
    try:
        input_batch, resized_img, edge_img = preprocess_image(INPUT_IMAGE_PATH)
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        return

    # Run inference
    print("[INFO] Running generator inference...")
    generated_img = model(input_batch, training=False)

    # Convert output from [-1, 1] back to [0, 1]
    generated_img = generated_img[0].numpy()
    generated_img = (generated_img * 0.5) + 0.5

    # Visualization
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title(f"Input Image ({IMG_HW}x{IMG_HW})")
    plt.imshow(resized_img)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Edge Map (Generator Input)")
    plt.imshow((edge_img[:, :, 0] + 1.0) / 2.0, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Generated Output")
    plt.imshow(generated_img)
    plt.axis('off')

    # Save result
    filename = os.path.basename(INPUT_IMAGE_PATH)
    save_path = os.path.join(OUTPUT_DIR, f"result_128_{filename}")
    plt.savefig(save_path)
    print(f"[SUCCESS] Output saved to: {save_path}")

if __name__ == "__main__":
    main()
