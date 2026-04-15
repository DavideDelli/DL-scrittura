# prepara_dati.py — versione con augmentation e inversione colore
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

IMG_HEIGHT = 128
IMG_WIDTH = 128
CHANNELS = 1

def load_and_preprocess_image(image_path, label):
    """Carica, inverte (tratti bianchi su nero) e normalizza."""
    img_raw = tf.io.read_file(image_path)
    img = tf.image.decode_png(img_raw, channels=CHANNELS)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0, 1]
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    # --- INVERSIONE: sfondo nero, tratti bianchi ---
    # Le tue immagini originali hanno sfondo bianco e tratti scuri.
    # Invertiamo così il modello impara a "disegnare" i tratti (valore alto = inchiostro).
    img = 1.0 - img
    return img, label

def augment_image(image, label):
    """
    Augmentation realistica per scrittura a mano.
    Applicata solo al training set.
    """
    # 1. Piccola rotazione casuale (±10 gradi)
    #    tf.keras.layers.RandomRotation vuole radianti come frazione di 2pi
    angle_rad = tf.random.uniform([], -0.055, 0.055)  # ≈ ±10 gradi in frazione (10/180*pi ≈ 0.174, /pi ≈ 0.055)
    # Usiamo tfa se disponibile, altrimenti usiamo un workaround con tf.raw_ops
    # Metodo compatibile con base TF: zoom + traslazione come proxy
    # Rotazione tramite tf.keras preprocessing
    image = _rotate_tensor(image, angle_rad)

    # 2. Piccolo zoom casuale (0.90 – 1.10)
    scale = tf.random.uniform([], 0.90, 1.10)
    new_h = tf.cast(tf.cast(IMG_HEIGHT, tf.float32) * scale, tf.int32)
    new_w = tf.cast(tf.cast(IMG_WIDTH, tf.float32) * scale, tf.int32)
    image = tf.image.resize(image, [new_h, new_w])
    image = tf.image.resize_with_crop_or_pad(image, IMG_HEIGHT, IMG_WIDTH)

    # 3. Piccola traslazione casuale (±8 pixel)
    image = _random_translate(image, max_shift=8)

    # 4. Rumore gaussiano leggero
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.03)
    image = tf.clip_by_value(image + noise, 0.0, 1.0)

    # 5. Variazione di contrasto leggera
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label

def _rotate_tensor(image, angle_rad):
    """Rotazione 2D tramite trasformazione affine (compatibile TF base)."""
    import math
    cos_a = tf.math.cos(angle_rad)
    sin_a = tf.math.sin(angle_rad)
    # Matrice di trasformazione per tf.raw_ops.ImageProjectiveTransformV3
    # [a0, a1, a2, b0, b1, b2, c0, c1] dove la trasf è:
    # x' = (a0*x + a1*y + a2) / k,  k = c0*x + c1*y + 1
    # Per rotazione attorno al centro: shift -> ruota -> shift
    cx = IMG_WIDTH / 2.0
    cy = IMG_HEIGHT / 2.0
    transform = [
        cos_a, -sin_a, cx - cx * cos_a + cy * sin_a,
        sin_a,  cos_a, cy - cx * sin_a - cy * cos_a,
        0.0, 0.0
    ]
    transform = tf.cast(transform, tf.float32)
    transform = tf.reshape(transform, [1, 8])
    image_4d = tf.expand_dims(image, 0)  # [1, H, W, C]
    try:
        rotated = tf.raw_ops.ImageProjectiveTransformV3(
            images=image_4d,
            transforms=transform,
            output_shape=[IMG_HEIGHT, IMG_WIDTH],
            interpolation="BILINEAR",
            fill_mode="CONSTANT",
            fill_value=0.0
        )
    except AttributeError:
        # Fallback se V3 non disponibile (TF < 2.12)
        rotated = tf.raw_ops.ImageProjectiveTransformV2(
            images=image_4d,
            transforms=transform,
            output_shape=[IMG_HEIGHT, IMG_WIDTH],
            interpolation="BILINEAR"
        )
    return tf.squeeze(rotated, 0)

def _random_translate(image, max_shift=8):
    """Traslazione casuale con padding nero."""
    dx = tf.random.uniform([], -max_shift, max_shift, dtype=tf.int32)
    dy = tf.random.uniform([], -max_shift, max_shift, dtype=tf.int32)
    image = tf.roll(image, shift=dy, axis=0)
    image = tf.roll(image, shift=dx, axis=1)
    return image

def prepare_and_load_data(dataset_dir="dataset_normalizzato", batch_size=32,
                           test_split_size=0.2, random_state=42,
                           augment_train=True, augment_multiplier=4):
    """
    Carica il dataset.
    augment_multiplier: quante copie augmentate aggiungere per ogni immagine di training.
    Con 50 immagini/classe e multiplier=4 → 200 immagini/classe effettive.
    """
    image_paths = []
    labels_str = []

    print(f"Scansione: {os.path.abspath(dataset_dir)}")
    for label_name in sorted(os.listdir(dataset_dir)):
        label_dir_path = os.path.join(dataset_dir, label_name)
        if os.path.isdir(label_dir_path):
            for image_filename in os.listdir(label_dir_path):
                if image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(label_dir_path, image_filename))
                    labels_str.append(label_name)

    if not image_paths:
        raise ValueError(f"Nessuna immagine trovata in: {dataset_dir}")

    unique_labels_str = sorted(list(set(labels_str)))
    label_to_int_map = {label: i for i, label in enumerate(unique_labels_str)}
    int_to_label_map = {i: label for i, label in enumerate(unique_labels_str)}

    integer_labels = [label_to_int_map[label] for label in labels_str]
    num_classes_val = len(unique_labels_str)

    image_paths_np = np.array(image_paths)
    integer_labels_np = np.array(integer_labels)

    X_train_paths, X_val_paths, y_train_labels, y_val_labels = train_test_split(
        image_paths_np, integer_labels_np,
        test_size=test_split_size, random_state=random_state,
        stratify=integer_labels_np
    )

    # --- VALIDATION SET: solo preprocessing, niente augmentation ---
    val_ds = tf.data.Dataset.from_tensor_slices((X_val_paths, y_val_labels))
    val_ds = val_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # --- TRAINING SET: preprocessing + augmentation ---
    base_train_ds = tf.data.Dataset.from_tensor_slices((X_train_paths, y_train_labels))
    base_train_ds = base_train_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    if augment_train and augment_multiplier > 0:
        # Crea copie augmentate e concatenale al dataset originale
        augmented_parts = [base_train_ds]
        for _ in range(augment_multiplier):
            aug_ds = base_train_ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
            augmented_parts.append(aug_ds)
        train_ds = augmented_parts[0]
        for part in augmented_parts[1:]:
            train_ds = train_ds.concatenate(part)
        print(f"Augmentation applicata: {augment_multiplier}x → ~{len(X_train_paths) * (augment_multiplier + 1)} immagini di training totali")
    else:
        train_ds = base_train_ds

    train_ds = train_ds.shuffle(buffer_size=len(X_train_paths) * (augment_multiplier + 1), seed=random_state)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    print(f"Totale immagini: {len(image_paths)} | Classi: {num_classes_val}")
    print(f"Train base: {len(X_train_paths)} | Val: {len(X_val_paths)}")

    return train_ds, val_ds, num_classes_val, label_to_int_map, int_to_label_map


if __name__ == "__main__":
    train_dataset, val_dataset, num_classes, lti, itl = prepare_and_load_data(batch_size=4)
    print(f"Classi: {num_classes}")
    for images, labels_batch in train_dataset.take(1):
        print("Batch immagini:", images.shape, "min:", images.numpy().min(), "max:", images.numpy().max())
        print("Batch etichette:", labels_batch.numpy())