import tensorflow as tf
import os
import time
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

# ==========================================
# 1. Configuration
# ==========================================
PRUNED_MODEL_PATH = './pruned_generator_structural.keras'  # Path to the pruned generator
IMG_HW = 128
BATCH_SIZE = 4
BUFFER_SIZE = 400
EPOCHS = 30            # Number of fine-tuning epochs
LR_GEN = 2e-4
LR_DISC = 2e-4
OUTPUT_DIR = 'fine_tuned_results'
CHECKPOINT_DIR = 'fine_tuned_checkpoints'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"[INFO] Fine-tuning Config: Epochs={EPOCHS}, Batch={BATCH_SIZE}")

# ==========================================
# 2. Dataset Preparation (Standalone Pipeline)
# ==========================================
def norm_to_abs_one(x):
    """Convert image values from [0, 1] to [-1, 1]."""
    return (x * 2.0) - 1.0


def list_all_files(directory):
    """Recursively list all files under a directory."""
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files


# Dataset path
data_dir = 'shoes_data/ut-zap50k-images-square'
if not os.path.exists(data_dir):
    print(f"[ERROR] Data directory not found: {data_dir}")
    exit()

shoes_dirs = list_all_files(data_dir)

# Keep only real shoe images (exclude edge images)
shoes_dirs = [f for f in shoes_dirs if 'edges' not in f and '.jpg' in f]
print(f"[INFO] Found {len(shoes_dirs)} images.")


def load_image_train(image_file):
    # 1. Load real shoe image
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [IMG_HW, IMG_HW])

    # 2. Locate corresponding edge image
    # Rule: shoe1.jpg -> shoe1.edges.jpg
    edge_path = tf.strings.regex_replace(image_file, '.jpg', '.edges.jpg')

    # Load edge image (must exist)
    edge_image = tf.io.read_file(edge_path)
    edge_image = tf.image.decode_jpeg(edge_image, channels=1)
    edge_image = tf.image.convert_image_dtype(edge_image, tf.float32)
    edge_image = tf.image.resize(edge_image, [IMG_HW, IMG_HW])

    # 3. Normalize to [-1, 1]
    image = norm_to_abs_one(image)
    edge_image = norm_to_abs_one(edge_image)

    # 4. Data augmentation
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        edge_image = tf.image.flip_left_right(edge_image)

    return edge_image, image


# Build tf.data pipeline
list_ds = tf.data.Dataset.from_tensor_slices(shoes_dirs)
train_dataset = list_ds.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# ==========================================
# 3. Model Setup
# ==========================================

# A. Load pruned generator
print(f"[INFO] Loading pruned generator from {PRUNED_MODEL_PATH}...")
try:
    generator = tf.keras.models.load_model(PRUNED_MODEL_PATH)
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    print("Please make sure the pruned model file exists and the path is correct.")
    exit()


# B. Build a fresh discriminator (PatchGAN 70x70)
def Downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(
        filters, size, strides=2, padding='same',
        kernel_initializer=initializer, use_bias=False
    ))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result


def build_discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[IMG_HW, IMG_HW, 1], name='input_image')
    tar = tf.keras.layers.Input(shape=[IMG_HW, IMG_HW, 3], name='target_image')

    # Concatenate condition and target
    x = tf.keras.layers.concatenate([inp, tar])

    down1 = Downsample(64, 4, False)(x)
    down2 = Downsample(128, 4)(down1)
    down3 = Downsample(256, 4)(down2)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv = tf.keras.layers.Conv2D(
        512, 4, strides=1,
        kernel_initializer=initializer,
        use_bias=False
    )(zero_pad1)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    last = tf.keras.layers.Conv2D(
        1, 4, strides=1,
        kernel_initializer=initializer
    )(zero_pad2)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


print("[INFO] Building new discriminator...")
discriminator = build_discriminator()

# ==========================================
# 4. Loss Functions and Optimizers
# ==========================================
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
LAMBDA = 100


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


generator_optimizer = tf.keras.optimizers.Adam(LR_GEN, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(LR_DISC, beta_1=0.5)

# ==========================================
# 5. Training Step
# ==========================================
@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
            disc_generated_output, gen_output, target
        )
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    return gen_total_loss, disc_loss

# ==========================================
# 6. Visualization Utilities
# ==========================================
def generate_images(model, test_input, tar, epoch):
    prediction = model(test_input, training=False)
    plt.figure(figsize=(15, 5))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input (Edge)', 'Ground Truth', 'Pruned + Fine-tuned Output']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        if i == 0:
            plt.imshow(display_list[i] * 0.5 + 0.5, cmap='gray')
        else:
            plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')

    save_path = os.path.join(OUTPUT_DIR, f"epoch_{epoch}.png")
    plt.savefig(save_path)
    plt.close()

# ==========================================
# 7. Main Training Loop (with Loss Tracking)
# ==========================================
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator
)

# Keep only the latest 5 checkpoints
manager = tf.train.CheckpointManager(
    checkpoint, directory=CHECKPOINT_DIR, max_to_keep=5
)

gen_loss_history = []
disc_loss_history = []

print(f"\n[INFO] Starting fine-tuning for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    start = time.time()

    epoch_gen_losses = []
    epoch_disc_losses = []

    for n, (input_image, target) in tqdm(
        enumerate(train_dataset),
        desc=f'Epoch {epoch + 1}/{EPOCHS}'
    ):
        gen_loss, disc_loss = train_step(input_image, target, epoch)

        epoch_gen_losses.append(gen_loss.numpy())
        epoch_disc_losses.append(disc_loss.numpy())

        # Save visualization for the first batch
        if n == 0:
            generate_images(generator, input_image, target, epoch + 1)

    avg_gen_loss = np.mean(epoch_gen_losses)
    avg_disc_loss = np.mean(epoch_disc_losses)

    gen_loss_history.append(avg_gen_loss)
    disc_loss_history.append(avg_disc_loss)

    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        save_path = manager.save(checkpoint_number=(epoch + 1))
        print(f"[Saved] Checkpoint saved at epoch {epoch + 1}: {save_path}")

    print(
        f"Time: {time.time() - start:.2f}s | "
        f"Gen Loss: {avg_gen_loss:.4f} | "
        f"Disc Loss: {avg_disc_loss:.4f}\n"
    )

# Save final generator
final_save_path = './fine_tuned_pruned_generator.keras'
generator.save(final_save_path)
print(f"[SUCCESS] Fine-tuning complete. Model saved to {final_save_path}")

# ==========================================
# 8. Plot and Save Loss Curves
# ==========================================
def plot_loss_curve(g_loss, d_loss, save_path='loss_curve.png'):
    plt.figure(figsize=(10, 6))

    plt.plot(g_loss, label='Generator Loss', linewidth=2)
    plt.plot(d_loss, label='Discriminator Loss', linewidth=2)

    plt.title('GAN Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path)
    print(f"[INFO] Loss curve saved to {save_path}")
    plt.close()


plot_loss_curve(
    gen_loss_history,
    disc_loss_history,
    save_path=os.path.join(OUTPUT_DIR, 'loss_curve.png')
)
