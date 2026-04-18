import tensorflow as tf
import os

# ==========================================
# 1. Paths and configuration
# ==========================================
# Directory containing TensorFlow checkpoints (e.g., ckpt-XX files)
CHECKPOINT_DIR = './epoch30_checkpoints'

# Output path for the exported Keras model
OUTPUT_MODEL_PATH = './generator_baseline.keras'

# Input image resolution expected by the generator
IMG_HW = 128

# ==========================================
# 2. Model definition (U-Net Generator)
# ==========================================

def encoder_block(layer_in, filters, batchnorm=True):
    """
    Encoder block consisting of:
    Conv2D → (optional) BatchNorm → LeakyReLU
    """
    init = tf.keras.initializers.GlorotUniform()
    conv = tf.keras.layers.Conv2D(
        filters,
        (4, 4),
        strides=(2, 2),
        padding='same',
        kernel_initializer=init
    )(layer_in)

    if batchnorm:
        # BatchNorm forced into training mode to match original cGAN implementation
        conv = tf.keras.layers.BatchNormalization()(conv, training=True)

    conv = tf.keras.layers.LeakyReLU(alpha=0.2)(conv)
    return conv


def decoder_block(layer_in, iden_in, filters, dropout=True):
    """
    Decoder block consisting of:
    Conv2DTranspose → BatchNorm → (optional) Dropout → Skip connection → ReLU
    """
    init = tf.keras.initializers.GlorotUniform()
    upconv = tf.keras.layers.Conv2DTranspose(
        filters,
        (4, 4),
        strides=(2, 2),
        padding='same',
        kernel_initializer=init
    )(layer_in)

    # BatchNorm forced into training mode for consistency with training behavior
    upconv = tf.keras.layers.BatchNormalization()(upconv, training=True)

    if dropout:
        # Dropout applied only to early decoder layers
        upconv = tf.keras.layers.Dropout(0.5)(upconv, training=True)

    # Skip connection (U-Net identity mapping)
    upconv = tf.keras.layers.Concatenate()([upconv, iden_in])
    upconv = tf.keras.layers.Activation('relu')(upconv)
    return upconv


def U_net(input_shape, net_pattern, b_filters, d_dropout_until):
    """
    U-Net generator architecture used in the cGAN model.
    """
    init = tf.keras.initializers.GlorotUniform()
    in_image = tf.keras.layers.Input(input_shape)

    e_x = in_image
    iden_vec = []  # Store encoder feature maps for skip connections

    # ----------------------
    # Encoder
    # ----------------------
    layer_counter = 1
    for e_units in net_pattern:
        if layer_counter == 1:
            # First encoder layer does NOT use BatchNorm
            e_x = encoder_block(e_x, e_units, batchnorm=False)
        else:
            e_x = encoder_block(e_x, e_units)

        iden_vec.append(e_x)
        layer_counter += 1

    # ----------------------
    # Bottleneck
    # ----------------------
    b = tf.keras.layers.Conv2D(
        b_filters,
        (4, 4),
        strides=(2, 2),
        padding='same',
        kernel_initializer=init
    )(e_x)
    b = tf.keras.layers.Activation('relu')(b)

    # ----------------------
    # Decoder
    # ----------------------
    d_x = b
    layer_counter = 1
    for d_units, identity in zip(net_pattern[::-1], iden_vec[::-1]):
        if layer_counter <= d_dropout_until:
            d_x = decoder_block(d_x, identity, d_units)
        else:
            d_x = decoder_block(d_x, identity, d_units, dropout=False)
        layer_counter += 1

    # ----------------------
    # Output layer
    # ----------------------
    out = tf.keras.layers.Conv2DTranspose(
        3,
        (4, 4),
        strides=(2, 2),
        padding='same',
        kernel_initializer=init
    )(d_x)
    out_image = tf.keras.layers.Activation('tanh')(out)

    model = tf.keras.models.Model(in_image, out_image)
    return model

# ==========================================
# 3. Checkpoint restoration and export
# ==========================================
def main():
    print("[INFO] Rebuilding U-Net generator architecture...")

    # Reconstruct generator architecture
    generator = U_net(
        input_shape=[IMG_HW, IMG_HW, 1],
        net_pattern=[64, 128, 128, 256, 384, 512],
        b_filters=512,
        d_dropout_until=3
    )

    # Create checkpoint object
    # NOTE: The key name 'generator' MUST match the original checkpoint definition
    checkpoint = tf.train.Checkpoint(generator=generator)

    # Locate the latest checkpoint
    latest_ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)

    if latest_ckpt:
        print(f"[INFO] Found checkpoint: {latest_ckpt}")

        # Restore generator weights only
        checkpoint.restore(latest_ckpt).expect_partial()
        print("[INFO] Weights successfully restored.")

        # Export model in Keras (.keras) format
        generator.save(OUTPUT_MODEL_PATH)
        print(f"[SUCCESS] Model exported to {OUTPUT_MODEL_PATH}")

    else:
        print(f"[ERROR] No checkpoint found in directory: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
