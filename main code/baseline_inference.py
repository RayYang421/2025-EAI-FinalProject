import os
# ==============================================================================
# Set environment variable to avoid Protobuf-related runtime crashes
# MUST be set before importing TensorFlow
# ==============================================================================
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Use scikit-image for consistent image preprocessing
# (kept identical to inference.py for fair comparison)
from skimage import io, transform, color, feature

# ==========================================
# 1. Configuration
# ==========================================
# Path to the converted baseline generator model (.keras)
MODEL_PATH = './generator_baseline.keras'

# Input image path (same image used in previous Test_data.py)
INPUT_IMAGE_PATH = 'test_shoes5.png'

# Output directory for inference results
OUTPUT_DIR = 'inference_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Expected input resolution for the generator
IMG_HW = 128

# ==========================================
# 2. Image preprocessing (identical to inference.py)
# ==========================================
def preprocess_image(image_path):
    """
    Image preprocessing pipeline.
    This function is intentionally kept IDENTICAL to inference.py to ensure
    that both Baseline and Pruned models receive exactly the same input edge map.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # Load input image
    original_img = io.imread(image_path)

    # Remove alpha channel if present
    if original_img.shape[-1] == 4:
        original_img = original_img[:, :, :3]

    print(f"[INFO] Resizing image to {IMG_HW}x{IMG_HW}...")

    # Resize image to model input size
    resized_img = transform.resize(
        original_img,
        (IMG_HW, IMG_HW),
        anti_aliasing=True
    ).astype(np.float32)

    # Convert to grayscale
    gray = color.rgb2gray(resized_img)

    # Determine whether input is a color photo or a sketch/line drawing
    hsv = color.rgb2hsv(resized_img)
    mean_saturation = hsv[:, :, 1].mean()

    # Generate edge map
    if mean_saturation > 0.1:
        # Case A: Color photo → apply Canny edge detection
        print("[INFO] Color image detected, applying Canny edge detection...")
        edges = feature.canny(gray, sigma=2.0)
        edge_img = edges.astype(np.float32)[:, :, None]
    else:
        # Case B: Sketch / line drawing
        print("[INFO] Grayscale / sketch image detected...")

        # Automatically handle white-background sketches
        if gray.mean() > 0.5:
            print("[INFO] White background detected, inverting colors...")
            edges = 1.0 - gray
        else:
            edges = gray

        # Binarize to enhance edge contrast
        edges = edges > 0.5
        edge_img = edges.astype(np.float32)[:, :, None]

    # Normalize edge map to [-1, 1] (GAN input convention)
    edge_img = (edge_img * 2.0) - 1.0

    # Add batch dimension: (1, 128, 128, 1)
    input_batch = tf.expand_dims(edge_img, axis=0)

    return input_batch, resized_img, edge_img

# ==========================================
# 3. Main inference routine
# ==========================================
def main():
    print(f"[INFO] Loading Baseline generator model: {MODEL_PATH}")
    try:
        # Load exported .keras model directly (no need to redefine U-Net)
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
    print("[INFO] Running inference...")
    generated_img = model(input_batch, training=False)

    # Convert output from [-1, 1] back to [0, 1] for visualization
    generated_img = generated_img[0].numpy()
    generated_img = (generated_img * 0.5) + 0.5

    # Visualization
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title(f"Input Image ({IMG_HW}x{IMG_HW})")
    plt.imshow(resized_img)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Edge Map (Model Input)")
    plt.imshow((edge_img[:, :, 0] + 1.0) / 2.0, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Baseline Generated Output")
    plt.imshow(generated_img)
    plt.axis('off')

    # Save result
    # Append '_baseline' to distinguish from pruned model outputs
    base_name = os.path.basename(INPUT_IMAGE_PATH)
    filename_only = os.path.splitext(base_name)[0]
    save_path = os.path.join(
        OUTPUT_DIR,
        f"result_baseline_{filename_only}.png"
    )

    plt.savefig(save_path, bbox_inches='tight')
    print(f"[SUCCESS] Result saved to: {save_path}")

if __name__ == "__main__":
    main()
