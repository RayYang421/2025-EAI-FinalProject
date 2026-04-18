import tensorflow as tf
import numpy as np
import os
from GAN_eth_GPT import U_net  # Import your model definition

# =========================
# CONFIG
# =========================

IMG_HW = 128
ORIGINAL_PATTERN = [64, 128, 128, 256, 384, 512]
ORIGINAL_B_FILTERS = 512
CHECKPOINT_DIR = './epoch30_checkpoints'

PRUNING_RATIO = 0.3

# =========================
# Core Logic
# =========================

def get_filter_importance(weights):
    """
    Compute filter importance using L1 norm.

    weights shape: [k_h, k_w, in_ch, out_ch]
    return: importance score per output channel (shape: [out_ch])
    """
    # Sum over spatial and input channel dimensions
    return np.sum(np.abs(weights), axis=(0, 1, 2))


def get_keep_indices(weights, ratio):
    """
    Select filter indices to keep based on importance ranking.
    """
    importance = get_filter_importance(weights)
    n_filters = len(importance)
    n_keep = int(n_filters * (1 - ratio))

    # Keep top-n filters with largest importance
    keep_indices = np.argsort(importance)[::-1][:n_keep]
    keep_indices = np.sort(keep_indices)
    return keep_indices


def prune_and_transfer():
    # 1. Build original (trained) model
    print("[INFO] Building Original Model...")
    original_gen = U_net(
        input_shape=[IMG_HW, IMG_HW, 1],
        net_pattern=ORIGINAL_PATTERN,
        b_filters=ORIGINAL_B_FILTERS,
        d_dropout_until=3
    )

    # Load trained weights
    checkpoint = tf.train.Checkpoint(generator=original_gen)
    checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR)).expect_partial()
    print("[INFO] Weights loaded.")

    # 2. Identify encoder / bottleneck / decoder layers
    # U-Net layout:
    # Encoder blocks * 6 -> Bottleneck -> Decoder blocks * 6 -> Output
    all_layers = [
        l for l in original_gen.layers
        if isinstance(l, (tf.keras.layers.Conv2D, tf.keras.layers.Conv2DTranspose))
    ]

    n_enc = len(ORIGINAL_PATTERN)
    encoder_layers = all_layers[:n_enc]
    bottleneck_layer = all_layers[n_enc]
    decoder_layers = all_layers[n_enc + 1: 2 * n_enc + 1]
    output_layer = all_layers[-1]

    print(f"[INFO] Found {len(encoder_layers)} encoder layers, {len(decoder_layers)} decoder layers.")

    # 3. Compute pruning plan (which filters to keep)
    enc_keep_indices = []

    print("\n[PLANNING] Calculating filter importance...")
    for i, layer in enumerate(encoder_layers):
        w, b = layer.get_weights()
        keep_idx = get_keep_indices(w, PRUNING_RATIO)
        enc_keep_indices.append(keep_idx)
        print(f"  Encoder L{i+1}: {w.shape[-1]} -> {len(keep_idx)} filters")

    # Prune bottleneck as well
    bn_w, bn_b = bottleneck_layer.get_weights()
    bn_keep_idx = get_keep_indices(bn_w, PRUNING_RATIO)
    print(f"  Bottleneck: {bn_w.shape[-1]} -> {len(bn_keep_idx)} filters")

    # Build new network configuration
    new_pattern = [len(idx) for idx in enc_keep_indices]
    new_b_filters = len(bn_keep_idx)

    print(f"\n[INFO] New encoder pattern: {new_pattern}")
    print(f"[INFO] New bottleneck filters: {new_b_filters}")

    # 4. Build pruned (physically smaller) model
    print("\n[INFO] Building Pruned Model...")
    pruned_gen = U_net(
        input_shape=[IMG_HW, IMG_HW, 1],
        net_pattern=new_pattern,
        b_filters=new_b_filters,
        d_dropout_until=3
    )

    # 5. Weight transfer (weight surgery)
    print("\n[INFO] Transferring weights...")

    new_all_layers = [
        l for l in pruned_gen.layers
        if isinstance(l, (tf.keras.layers.Conv2D, tf.keras.layers.Conv2DTranspose))
    ]

    new_enc_layers = new_all_layers[:n_enc]
    new_bn_layer = new_all_layers[n_enc]
    new_dec_layers = new_all_layers[n_enc + 1: 2 * n_enc + 1]
    new_out_layer = new_all_layers[-1]

    # --- Encoder ---
    # Input image channel is fixed (1 channel), so only output filters are pruned
    last_keep_idx = np.arange(1)

    for i in range(n_enc):
        old_w, old_b = encoder_layers[i].get_weights()
        curr_keep_idx = enc_keep_indices[i]

        if i == 0:
            new_w = old_w[:, :, :, curr_keep_idx]
        else:
            new_w = old_w[:, :, last_keep_idx, :][:, :, :, curr_keep_idx]

        new_b = old_b[curr_keep_idx]
        new_enc_layers[i].set_weights([new_w, new_b])

        last_keep_idx = curr_keep_idx

    # --- Bottleneck ---
    old_w, old_b = bottleneck_layer.get_weights()
    new_w = old_w[:, :, last_keep_idx, :][:, :, :, bn_keep_idx]
    new_b = old_b[bn_keep_idx]
    new_bn_layer.set_weights([new_w, new_b])

    last_keep_idx = bn_keep_idx

    # --- Decoder ---
    # Decoder input = Up-sampled output + skip connection
    print("\n[INFO] Transferring decoder weights...")

    skip_keep_indices_list = enc_keep_indices[::-1]

    for i in range(len(decoder_layers)):
        old_w, old_b = decoder_layers[i].get_weights()
        curr_keep_idx = skip_keep_indices_list[i]

        # First decoder layer only receives bottleneck output
        if i == 0:
            w_input_pruned = old_w[:, :, :, last_keep_idx]
            new_w = w_input_pruned[:, :, curr_keep_idx, :]

        # Remaining decoder layers include skip connections
        else:
            skip_layer_idx_in_enc = n_enc - i
            original_skip_channels = ORIGINAL_PATTERN[skip_layer_idx_in_enc]

            total_in_channels = old_w.shape[3]
            original_main_channels = total_in_channels - original_skip_channels

            w_main = old_w[:, :, :, :original_main_channels]
            w_skip = old_w[:, :, :, original_main_channels:]

            w_main_pruned = w_main[:, :, :, last_keep_idx]
            skip_mask = enc_keep_indices[skip_layer_idx_in_enc]
            w_skip_pruned = w_skip[:, :, :, skip_mask]

            w_combined = np.concatenate([w_main_pruned, w_skip_pruned], axis=3)
            new_w = w_combined[:, :, curr_keep_idx, :]

        new_b = old_b[curr_keep_idx]
        new_dec_layers[i].set_weights([new_w, new_b])
        last_keep_idx = curr_keep_idx

    # --- Output Layer ---
    print("\n[INFO] Transferring output layer weights...")
    old_w, old_b = output_layer.get_weights()

    main_path_indices = last_keep_idx
    skip_path_indices = enc_keep_indices[0]

    # Skip connection is located in the second half of input channels
    offset = int(old_w.shape[3] // 2)
    skip_path_indices_with_offset = skip_path_indices + offset

    final_input_indices = np.concatenate(
        [main_path_indices, skip_path_indices_with_offset]
    ).astype(int)

    print(f"  Output layer input channels: {len(final_input_indices)} (original: {old_w.shape[3]})")

    new_w = old_w[:, :, :, final_input_indices]
    new_out_layer.set_weights([new_w, old_b])

    print("[INFO] Pruning and weight transfer completed.")

    # 6. Save pruned model
    save_path = './pruned_generator_structural.keras'
    pruned_gen.save(save_path)
    print(f"[INFO] Saved pruned model to: {save_path}")
    print(f"[INFO] Final encoder pattern: {new_pattern}")

    return pruned_gen


if __name__ == "__main__":
    pruned_model = prune_and_transfer()

    print("\n" + "=" * 30)
    print("      PRUNED MODEL SUMMARY")
    print("=" * 30)
    pruned_model.summary()
