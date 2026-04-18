import tensorflow as tf
import numpy as np
import os


# ==============================================================================
# 1. MACs CALCULATION ENGINE (Keras 3 Compatible)
# ==============================================================================
def count_layer_macs(layer):
    """
    Calculates MACs for a single Keras layer.
    """
    macs = 0

    # Check for Convolutional Layers
    if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Conv2DTranspose)):
        weights = layer.get_weights()
        if not weights: return 0

        # weights[0] is Kernel: (kh, kw, cin, cout)
        kernel_shape = weights[0].shape
        k_h, k_w = kernel_shape[0], kernel_shape[1]
        c_in = kernel_shape[2]
        c_out = kernel_shape[3]

        # --- KERAS 3 FIX ---
        # Get Output Shape safely
        try:
            # Try accessing the output tensor first (Standard for Keras 3)
            out_shape = layer.output.shape
        except AttributeError:
            try:
                # Fallback for older Keras versions
                out_shape = layer.output_shape
            except AttributeError:
                print(f"[WARN] Could not determine output shape for {layer.name}")
                return 0

        # Handle TensorShape object or Tuple
        if isinstance(out_shape, list): out_shape = out_shape[0]

        # out_shape is usually (Batch, H, W, C)
        # We need H and W. Index 1 and 2.
        h_out = out_shape[1]
        w_out = out_shape[2]

        # If dimensions are None (dynamic), try to infer from Input
        if h_out is None or w_out is None:
            # Fallback inference (assuming 128x128 input and standard strides)
            # This is a heuristic if shape inference fails
            h_out, w_out = 128, 128  # Default placeholder to avoid crash

        # MACs = Kernel_Area * Channels * Output_Area
        macs = (k_h * k_w * c_in * c_out * h_out * w_out)

    elif isinstance(layer, tf.keras.layers.Dense):
        weights = layer.get_weights()
        if weights:
            in_units = weights[0].shape[0]
            out_units = weights[0].shape[1]
            macs = in_units * out_units

    return macs


def get_model_macs(model):
    total_macs = 0
    total_params = model.count_params()

    print(f"\n[ANALYSIS] Analyzing {model.name}...")

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            total_macs += get_model_macs(layer)[0]
        else:
            total_macs += count_layer_macs(layer)

    return total_macs, total_params


# ==============================================================================
# 2. ARCHITECTURE BUILDERS (Updated for Keras 3)
# ==============================================================================
def encoder_block(layer_in, filters, batchnorm=True):
    init = tf.keras.initializers.GlorotUniform()
    conv = tf.keras.layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    if batchnorm: conv = tf.keras.layers.BatchNormalization()(conv)
    # Update: alpha -> negative_slope
    conv = tf.keras.layers.LeakyReLU(negative_slope=0.2)(conv)
    return conv


def decoder_block(layer_in, iden_in, filters, dropout=True):
    init = tf.keras.initializers.GlorotUniform()
    upconv = tf.keras.layers.Conv2DTranspose(filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(
        layer_in)
    upconv = tf.keras.layers.BatchNormalization()(upconv)
    if dropout: upconv = tf.keras.layers.Dropout(0.5)(upconv)
    upconv = tf.keras.layers.Concatenate()([upconv, iden_in])
    upconv = tf.keras.layers.Activation('relu')(upconv)
    return upconv


def build_baseline():
    """Reconstructs the original heavy architecture."""
    input_shape = (128, 128, 1)
    net_pattern = [64, 128, 128, 256, 384, 512]

    in_image = tf.keras.layers.Input(input_shape)
    e_x = in_image
    iden_vec = []

    for i, e_units in enumerate(net_pattern):
        e_x = encoder_block(e_x, e_units, batchnorm=(i != 0))
        iden_vec.append(e_x)

    b = tf.keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same')(e_x)
    b = tf.keras.layers.Activation('relu')(b)

    d_x = b
    for i, (d_units, identity) in enumerate(zip(net_pattern[::-1], iden_vec[::-1])):
        d_x = decoder_block(d_x, identity, d_units, dropout=(i < 3))

    out = tf.keras.layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same')(d_x)
    out = tf.keras.layers.Activation('tanh')(out)
    return tf.keras.models.Model(in_image, out, name="Baseline_UNet")


# ==============================================================================
# 3. MAIN EXECUTION
# ==============================================================================
def main():
    print("=" * 60)
    print(" REAL MACs CALCULATION (Keras 3 Compatible)")
    print("=" * 60)

    results = []

    # 1. Baseline
    try:
        baseline = build_baseline()
        macs, params = get_model_macs(baseline)
        results.append(("Baseline", macs, params))
    except Exception as e:
        print(f"Error building Baseline: {e}")
        import traceback
        traceback.print_exc()

    # 2. Pruned / Fine-Tuned Models
    files = [
        'pruned_generator_structural.keras',
        'fine_tuned_30_pruned_generator.keras',
        'fine_tuned_50_pruned_generator.keras'
    ]

    for f in files:
        if os.path.exists(f):
            try:
                # Load model
                model = tf.keras.models.load_model(f)
                macs, params = get_model_macs(model)
                results.append((f, macs, params))
            except Exception as e:
                print(f"[ERROR] Could not load {f}: {e}")
        else:
            print(f"[WARN] File not found: {f}")

    # 3. Final Report
    print("\n" + "=" * 85)
    print(f"{'Model Name':<40} | {'MACs (G)':<12} | {'Params (M)':<12} | {'Speedup':<10}")
    print("-" * 85)

    baseline_macs = results[0][1] if results else 1

    for name, macs, params in results:
        macs_g = macs / 1e9
        params_m = params / 1e6

        if "Baseline" in name:
            speedup = "1.00x"
        else:
            ratio = baseline_macs / macs
            speedup = f"{ratio:.2f}x"

        print(f"{name:<40} | {macs_g:.4f} G    | {params_m:.2f} M      | {speedup}")
    print("=" * 85)


if __name__ == "__main__":
    main()