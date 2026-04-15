# genera_caratteri.py — versione aggiornata per checkpoint v2
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K
import numpy as np
import os
import json
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. CONFIGURAZIONE — modifica questi percorsi se necessario
# ---------------------------------------------------------------------------
CHECKPOINT_DIR     = "./cvae_checkpoints_v2"
LABEL_INFO_PATH    = os.path.join(CHECKPOINT_DIR, "label_info.json")
MODEL_WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, "cvae_final_weights_v2.weights.h5")

IMAGE_HEIGHT   = 128
IMAGE_WIDTH    = 128
IMAGE_CHANNELS = 1
LATENT_DIM     = 64
EMBEDDING_DIM  = 32

# ---------------------------------------------------------------------------
# 2. CARICA MAPPING ETICHETTE
# ---------------------------------------------------------------------------
if not os.path.exists(LABEL_INFO_PATH):
    raise FileNotFoundError(f"label_info.json non trovato in {LABEL_INFO_PATH}. "
                            "Esegui prima main_training.py.")

with open(LABEL_INFO_PATH, "r") as f:
    label_info = json.load(f)

NUM_CLASSES      = label_info["num_classes"]
label_to_int_map = label_info["label_to_int"]
int_to_label_map = {int(k): v for k, v in label_info["int_to_label"].items()}
print(f"Classi caricate: {NUM_CLASSES}")

# ---------------------------------------------------------------------------
# 3. RICOSTRUISCI L'ARCHITETTURA (identica a main_training.py)
# ---------------------------------------------------------------------------

def build_encoder(latent_dim_param, num_classes_param, embedding_dim_param):
    image_input = layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
                               name="encoder_image_input")
    label_input = layers.Input(shape=(1,), name="encoder_label_input")
    label_emb   = layers.Embedding(num_classes_param, embedding_dim_param, name="enc_label_emb")(label_input)
    label_emb   = layers.Flatten()(label_emb)
    label_feat  = layers.Dense(128, activation="relu", name="enc_label_dense")(label_emb)
    x = layers.Conv2D(64,  3, strides=2, padding="same", activation="relu")(image_input)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    img_feat = layers.Flatten()(x)
    img_feat = layers.Dropout(0.3)(img_feat)
    merged   = layers.Concatenate()([img_feat, label_feat])
    merged   = layers.Dense(512, activation="relu")(merged)
    merged   = layers.Dropout(0.2)(merged)
    z_mean    = layers.Dense(latent_dim_param, name="z_mean")(merged)
    z_log_var = layers.Dense(latent_dim_param, name="z_log_var")(merged)
    return Model([image_input, label_input], [z_mean, z_log_var], name="encoder")

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = K.random_normal(shape=(tf.shape(z_mean)[0], tf.shape(z_mean)[1]))
        return z_mean + K.exp(0.5 * z_log_var) * eps

def build_decoder(latent_dim_param, num_classes_param, embedding_dim_param):
    latent_input = layers.Input(shape=(latent_dim_param,), name="decoder_latent_input")
    label_input  = layers.Input(shape=(1,), name="decoder_label_input")
    label_emb    = layers.Embedding(num_classes_param, embedding_dim_param, name="dec_label_emb")(label_input)
    label_emb    = layers.Flatten()(label_emb)
    label_feat_start = layers.Dense(embedding_dim_param * 2, activation="relu",
                                    name="dec_label_dense_start")(label_emb)
    label_feat_mid   = layers.Dense(16 * 16 * 32, activation="relu",
                                    name="dec_label_dense_mid")(label_emb)
    label_feat_mid_2d = layers.Reshape((16, 16, 32), name="dec_label_reshape")(label_feat_mid)
    merged = layers.Concatenate()([latent_input, label_feat_start])
    x = layers.Dense((IMAGE_HEIGHT // 8) * (IMAGE_WIDTH // 8) * 256, activation="relu")(merged)
    x = layers.Reshape((IMAGE_HEIGHT // 8, IMAGE_WIDTH // 8, 256))(x)
    x = layers.Conv2DTranspose(256, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    label_mid_up = layers.UpSampling2D(size=(2, 2))(label_feat_mid_2d)
    x = layers.Concatenate(axis=-1)([x, label_mid_up])
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(64,  3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    image_output = layers.Conv2DTranspose(IMAGE_CHANNELS, 3, padding="same", activation="sigmoid")(x)
    return Model([latent_input, label_input], image_output, name="decoder")

class CVAE(Model):
    def __init__(self, encoder_model, decoder_model, **kwargs):
        super().__init__(**kwargs)
        self.encoder  = encoder_model
        self.decoder  = decoder_model
        self.sampling = Sampling()
    def call(self, inputs):
        image, label = inputs
        z_mean, z_log_var = self.encoder([image, label])
        z = self.sampling([z_mean, z_log_var])
        return self.decoder([z, label])

# ---------------------------------------------------------------------------
# 4. CARICA PESI
# ---------------------------------------------------------------------------
print("Costruzione modello e caricamento pesi...")
enc = build_encoder(LATENT_DIM, NUM_CLASSES, EMBEDDING_DIM)
dec = build_decoder(LATENT_DIM, NUM_CLASSES, EMBEDDING_DIM)
cvae = CVAE(enc, dec)

# Chiamata fittizia per costruire il grafo prima di load_weights
dummy_img   = np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.float32)
dummy_label = np.zeros((1, 1), dtype=np.int32)
_ = cvae([dummy_img, dummy_label])

cvae.load_weights(MODEL_WEIGHTS_PATH)
print("Pesi caricati.")

# ---------------------------------------------------------------------------
# 5. GENERAZIONE
# ---------------------------------------------------------------------------
def generate_characters(labels_str_list, n_samples_per_label=3, temperature=1.0):
    """
    Genera n_samples_per_label immagini per ogni etichetta in labels_str_list.
    temperature > 1 → più varietà/casualità; < 1 → più concentrato sulla media.
    """
    results = []
    for lbl_str in labels_str_list:
        if lbl_str not in label_to_int_map:
            print(f"Attenzione: '{lbl_str}' non trovata nel mapping. Saltata.")
            continue
        lbl_int = label_to_int_map[lbl_str]
        for _ in range(n_samples_per_label):
            z = tf.random.normal(shape=(1, LATENT_DIM)) * temperature
            lbl_tensor = np.array([[lbl_int]], dtype=np.int32)
            img = cvae.decoder.predict([z, lbl_tensor], verbose=0)
            results.append((lbl_str, img[0, :, :, 0]))
    return results

def display_and_save(results, output_path=None, invert_display=True):
    """
    Mostra i caratteri generati.
    invert_display=True: mostra tratti scuri su sfondo bianco
    (inverte l'inversione fatta in prepara_dati.py per l'output visivo).
    """
    n = len(results)
    if n == 0:
        print("Nessun risultato da mostrare.")
        return
    cols = min(n, 6)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.5))
    axes = np.array(axes).reshape(-1) if n > 1 else [axes]

    for ax, (lbl, img) in zip(axes, results):
        display_img = 1.0 - img if invert_display else img
        ax.imshow(display_img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(lbl, fontsize=10)
        ax.axis("off")

    # Nascondi assi inutilizzati
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Salvato: {output_path}")
    plt.show()


if __name__ == "__main__":
    # Etichette che vuoi generare (usa i nomi esatti del tuo dataset)
    etichette = [
        "A", "B", "C", "D", "E", "F",
        "1", "2", "3",
        "PUNTO", "VIRGOLA", "ESCLAMATIVO", "INTERROGATIVO"
    ]

    print("Generazione caratteri...")
    risultati = generate_characters(
        labels_str_list=etichette,
        n_samples_per_label=2,   # 2 varianti per etichetta
        temperature=0.8          # Leggermente sotto 1.0 per output più nitidi
    )

    output_img = os.path.join(CHECKPOINT_DIR, "generated_v2.png")
    display_and_save(risultati, output_path=output_img)