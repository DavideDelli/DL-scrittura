# main_training.py — versione corretta
# Fix applicati:
#   1. KL weight bilanciato (beta-VAE con beta piccolo)
#   2. Reconstruction loss con reduce_mean invece di reduce_sum
#   3. Label injection anche a metà decoder (conditional guidance più forte)
#   4. Dropout nell'encoder per ridurre overfitting
#   5. EarlyStopping patience ridotta, LR schedule più aggressivo
#   6. Salvataggio label_info.json nella stessa cartella dei checkpoint
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K
import numpy as np
import os
import json
import prepara_dati

# ---------------------------------------------------------------------------
# 1. CARICAMENTO DATI
# ---------------------------------------------------------------------------
print("Caricamento dati...")
BATCH_SIZE_TRAINING = 32
train_dataset, val_dataset, NUM_CLASSES, label_to_int, int_to_label = \
    prepara_dati.prepare_and_load_data(
        dataset_dir="dataset_normalizzato",
        batch_size=BATCH_SIZE_TRAINING,
        augment_train=True,
        augment_multiplier=5   # 50 img → 300 img/classe nel training
    )

print(f"Classi rilevate: {NUM_CLASSES}")
if NUM_CLASSES == 0:
    raise ValueError("Nessuna classe. Controlla 'dataset_normalizzato'.")

# ---------------------------------------------------------------------------
# 2. PARAMETRI
# ---------------------------------------------------------------------------
IMAGE_HEIGHT = prepara_dati.IMG_HEIGHT   # 128
IMAGE_WIDTH  = prepara_dati.IMG_WIDTH    # 128
IMAGE_CHANNELS = prepara_dati.CHANNELS  # 1
LATENT_DIM   = 64
EMBEDDING_DIM = 32

# --- BETA-VAE ---
# Con reduce_mean sulla reconstruction loss, i valori sono nell'ordine di 0.01–0.3.
# La KL (reduce_mean su LATENT_DIM termini) è nell'ordine di 1–5 nats.
# beta = 0.5 dà un buon bilanciamento iniziale; puoi aumentarlo a 1.0 se lo spazio
# latente risulta ancora disorganizzato dopo training.
KL_WEIGHT = 0.5

# ---------------------------------------------------------------------------
# 3. ARCHITETTURA
# ---------------------------------------------------------------------------

def build_encoder(latent_dim_param, num_classes_param, embedding_dim_param):
    image_input = layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
                               name="encoder_image_input")
    label_input = layers.Input(shape=(1,), name="encoder_label_input")

    # Label embedding
    label_emb = layers.Embedding(num_classes_param, embedding_dim_param,
                                 name="enc_label_emb")(label_input)
    label_emb = layers.Flatten()(label_emb)
    label_feat = layers.Dense(128, activation="relu", name="enc_label_dense")(label_emb)

    # CNN sull'immagine
    x = layers.Conv2D(64,  3, strides=2, padding="same", activation="relu")(image_input)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    img_feat = layers.Flatten()(x)

    # Dropout per ridurre overfitting (nuovo)
    img_feat = layers.Dropout(0.3)(img_feat)

    merged = layers.Concatenate()([img_feat, label_feat])
    merged = layers.Dense(512, activation="relu")(merged)
    merged = layers.Dropout(0.2)(merged)

    z_mean    = layers.Dense(latent_dim_param, name="z_mean")(merged)
    z_log_var = layers.Dense(latent_dim_param, name="z_log_var")(merged)

    return Model([image_input, label_input], [z_mean, z_log_var], name="encoder")


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim   = tf.shape(z_mean)[1]
        eps   = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * eps


def build_decoder(latent_dim_param, num_classes_param, embedding_dim_param):
    latent_input = layers.Input(shape=(latent_dim_param,), name="decoder_latent_input")
    label_input  = layers.Input(shape=(1,), name="decoder_label_input")

    # Label embedding — condiviso tra inizio e metà decoder
    label_emb = layers.Embedding(num_classes_param, embedding_dim_param,
                                 name="dec_label_emb")(label_input)
    label_emb = layers.Flatten()(label_emb)
    label_feat_start = layers.Dense(embedding_dim_param * 2, activation="relu",
                                    name="dec_label_dense_start")(label_emb)
    # Seconda proiezione per injection a metà decoder
    label_feat_mid = layers.Dense(16 * 16 * 32, activation="relu",
                                  name="dec_label_dense_mid")(label_emb)
    label_feat_mid_2d = layers.Reshape((16, 16, 32), name="dec_label_reshape")(label_feat_mid)

    # Ingresso iniziale: z + label
    merged = layers.Concatenate()([latent_input, label_feat_start])
    x = layers.Dense((IMAGE_HEIGHT // 8) * (IMAGE_WIDTH // 8) * 256, activation="relu")(merged)
    x = layers.Reshape((IMAGE_HEIGHT // 8, IMAGE_WIDTH // 8, 256))(x)   # 16x16x256

    # Primo upscaling 16→32
    x = layers.Conv2DTranspose(256, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    # --- RE-INJECTION della label a 32x32 ---
    # Upscale label_feat_mid_2d da 16×16 a 32×32
    label_mid_up = layers.UpSampling2D(size=(2, 2))(label_feat_mid_2d)
    x = layers.Concatenate(axis=-1)([x, label_mid_up])   # 32x32x(256+32)

    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(64,  3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    # Output: usa sigmoid → valori [0,1] con 1=inchiostro, 0=sfondo
    image_output = layers.Conv2DTranspose(IMAGE_CHANNELS, 3, padding="same",
                                          activation="sigmoid")(x)

    return Model([latent_input, label_input], image_output, name="decoder")


class CVAE(Model):
    def __init__(self, encoder_model, decoder_model, kl_loss_weight=KL_WEIGHT, **kwargs):
        super().__init__(**kwargs)
        self.encoder      = encoder_model
        self.decoder      = decoder_model
        self.sampling     = Sampling()
        self.kl_loss_weight = kl_loss_weight

        self.total_loss_tracker        = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker           = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def call(self, inputs):
        image, label = inputs
        z_mean, z_log_var = self.encoder([image, label])
        z = self.sampling([z_mean, z_log_var])
        return self.decoder([z, label])

    def _compute_losses(self, images_batch, labels_batch):
        z_mean, z_log_var = self.encoder([images_batch, labels_batch])
        z = self.sampling([z_mean, z_log_var])
        reconstruction = self.decoder([z, labels_batch])

        # --- FIX CHIAVE: reduce_mean invece di reduce_sum ---
        # reduce_mean normalizza per numero di pixel → valori ~0.01-0.3
        # compatibile con KL_WEIGHT piccolo (0.5)
        recon_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(images_batch, reconstruction)
        )

        # KL loss: reduce_mean su batch e su dimensione latente
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )

        total_loss = recon_loss + self.kl_loss_weight * kl_loss
        return total_loss, recon_loss, kl_loss

    def train_step(self, data):
        images_batch, labels_batch = data
        with tf.GradientTape() as tape:
            total_loss, recon_loss, kl_loss = self._compute_losses(images_batch, labels_batch)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        images_batch, labels_batch = data
        total_loss, recon_loss, kl_loss = self._compute_losses(images_batch, labels_batch)
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

# ---------------------------------------------------------------------------
# 4. BUILD & COMPILE
# ---------------------------------------------------------------------------
print("Costruzione CVAE...")
encoder_instance = build_encoder(LATENT_DIM, NUM_CLASSES, EMBEDDING_DIM)
decoder_instance = build_decoder(LATENT_DIM, NUM_CLASSES, EMBEDDING_DIM)
cvae_instance = CVAE(encoder_instance, decoder_instance, kl_loss_weight=KL_WEIGHT)
cvae_instance.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

encoder_instance.summary()
decoder_instance.summary()

# ---------------------------------------------------------------------------
# 5. SALVA label_info.json (serve a genera_caratteri.py)
# ---------------------------------------------------------------------------
checkpoint_dir = "./cvae_checkpoints_v2"
os.makedirs(checkpoint_dir, exist_ok=True)

label_info = {
    "num_classes": NUM_CLASSES,
    "label_to_int": label_to_int,
    "int_to_label": {str(k): v for k, v in int_to_label.items()}
}
with open(os.path.join(checkpoint_dir, "label_info.json"), "w") as f:
    json.dump(label_info, f, indent=2)
print(f"label_info.json salvato in {checkpoint_dir}")

# ---------------------------------------------------------------------------
# 6. CALLBACKS
# ---------------------------------------------------------------------------
checkpoint_filepath = os.path.join(
    checkpoint_dir,
    "cvae_epoch_{epoch:03d}-val_{val_total_loss:.4f}.weights.h5"
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor="val_total_loss",
        mode="min",
        save_best_only=True
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_total_loss",
        patience=15,          # Ridotto da 25: se stagna 15 epoche, fermati
        verbose=1,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_total_loss",
        factor=0.3,           # Riduci LR del 70%
        patience=5,           # Dopo 5 epoche senza miglioramento
        min_lr=1e-7,
        verbose=1
    ),
]

# ---------------------------------------------------------------------------
# 7. TRAINING
# ---------------------------------------------------------------------------
NUM_EPOCHS = 300
print(f"\nInizio training per max {NUM_EPOCHS} epoche...")
history = cvae_instance.fit(
    train_dataset,
    epochs=NUM_EPOCHS,
    validation_data=val_dataset,
    callbacks=callbacks
)
print("Training completato!")

# Salva pesi finali
final_weights = os.path.join(checkpoint_dir, "cvae_final_weights_v2.weights.h5")
cvae_instance.save_weights(final_weights)
print(f"Pesi finali salvati: {final_weights}")

# ---------------------------------------------------------------------------
# 8. PLOT HISTORY
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, key, title in zip(
    axes,
    ["total_loss", "reconstruction_loss", "kl_loss"],
    ["Total Loss", "Reconstruction Loss", "KL Loss"]
):
    ax.plot(history.history[key], label="Training")
    val_key = f"val_{key}"
    if val_key in history.history:
        ax.plot(history.history[val_key], label="Validation")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

plt.tight_layout()
plot_path = os.path.join(checkpoint_dir, "training_history_v2.png")
plt.savefig(plot_path)
print(f"Grafico salvato: {plot_path}")