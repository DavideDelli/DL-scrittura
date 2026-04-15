import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K
import numpy as np
import os
import json
import random
from PIL import Image, ImageOps, ImageEnhance

# ---------------------------------------------------------------------------
# 1. CONFIGURAZIONE PERCORSI
# ---------------------------------------------------------------------------
CHECKPOINT_DIR     = "./cvae_checkpoints_v2"
LABEL_INFO_PATH    = os.path.join(CHECKPOINT_DIR, "label_info.json")
MODEL_WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, "cvae_final_weights_v2.weights.h5")

IMAGE_HEIGHT, IMAGE_WIDTH = 128, 128
LATENT_DIM, EMBEDDING_DIM = 64, 32

with open(LABEL_INFO_PATH, "r") as f:
    label_info = json.load(f)

NUM_CLASSES      = label_info["num_classes"]
label_to_int_map = label_info["label_to_int"]

# ---------------------------------------------------------------------------
# 2. ARCHITETTURA (Copiata esattamente dal tuo main_training.py)
# ---------------------------------------------------------------------------

def build_encoder(latent_dim_param, num_classes_param, embedding_dim_param):
    image_input = layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), name="encoder_image_input")
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
    label_feat_start = layers.Dense(embedding_dim_param * 2, activation="relu", name="dec_label_dense_start")(label_emb)
    label_feat_mid   = layers.Dense(16 * 16 * 32, activation="relu", name="dec_label_dense_mid")(label_emb)
    label_feat_mid_2d = layers.Reshape((16, 16, 32), name="dec_label_reshape")(label_feat_mid)
    merged = layers.Concatenate()([latent_input, label_feat_start])
    x = layers.Dense(16 * 16 * 256, activation="relu")(merged)
    x = layers.Reshape((16, 16, 256))(x)
    x = layers.Conv2DTranspose(256, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    label_mid_up = layers.UpSampling2D(size=(2, 2))(label_feat_mid_2d)
    x = layers.Concatenate(axis=-1)([x, label_mid_up])
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(64,  3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    image_output = layers.Conv2DTranspose(1, 3, padding="same", activation="sigmoid")(x)
    return Model([latent_input, label_input], image_output, name="decoder")

class CVAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampling = Sampling()
    def call(self, inputs):
        img, lbl = inputs
        z_m, z_v = self.encoder([img, lbl])
        z = self.sampling([z_m, z_v])
        return self.decoder([z, lbl])

# --- INIZIALIZZAZIONE ---
enc = build_encoder(LATENT_DIM, NUM_CLASSES, EMBEDDING_DIM)
dec = build_decoder(LATENT_DIM, NUM_CLASSES, EMBEDDING_DIM)
full_cvae = CVAE(enc, dec)
_ = full_cvae([np.zeros((1, 128, 128, 1)), np.zeros((1, 1))])
full_cvae.load_weights(MODEL_WEIGHTS_PATH)
print("Pesi caricati correttamente!")

# ---------------------------------------------------------------------------
# 3. LOGICA DI SCRITTURA
# ---------------------------------------------------------------------------

def pulisci_tratto(raw_img_np):
    img = Image.fromarray(raw_img_np.astype(np.uint8)).convert("L")
    img = ImageEnhance.Contrast(img).enhance(3.0)
    # ELIMINA QUADRATINI: Forza bianco se pixel > 190
    img = img.point(lambda p: 255 if p > 190 else p)
    return img

def genera_char_img(c, temp=0.6):
    mapping = {'?': 'INTERROGATIVO', '!': 'ESCLAMATIVO', '.': 'PUNTO', ',': 'VIRGOLA'}
    c_key = c.upper()
    if c == "'":
        c_key = 'APOSTROFO' if 'APOSTROFO' in label_to_int_map else 'VIRGOLA'
    else:
        c_key = mapping.get(c, c_key)

    if c_key not in label_to_int_map: return None
    
    idx = label_to_int_map[c_key]
    z = tf.random.normal(shape=(1, LATENT_DIM)) * temp
    gen = full_cvae.decoder.predict([z, np.array([[idx]])], verbose=0)[0, :, :, 0]
    return (1.0 - gen) * 255

def scrivi_testo_handwriting(testo, output="risultato_perfetto.png"):
    # Normalizza accenti (à -> A')
    acc = {'à':"A'", 'è':"E'", 'é':"E'", 'ì':"I'", 'ò':"O'", 'ù':"U'"}
    for k, v in acc.items(): testo = testo.replace(k, v)
    
    foglio = Image.new('L', (2400, 1800), 255)
    x, y = 80, 100
    baseline_riga = 150 

    parole = testo.split(' ')
    for parola in parole:
        immagini_parola = []
        larghezza_parola = 0
        
        for c in parola:
            raw = genera_char_img(c)
            if raw is not None:
                img_c = pulisci_tratto(raw)
                inv = ImageOps.invert(img_c)
                bbox = inv.getbbox()
                if bbox:
                    crop = img_c.crop(bbox)
                    # Rimpicciolisci i segni piccoli
                    if c in ".,'":
                        crop = crop.resize((int(crop.width*0.65), int(crop.height*0.65)), Image.LANCZOS)
                    
                    rot = crop.rotate(random.uniform(-1.5, 1.5), resample=Image.BICUBIC, expand=True)
                    immagini_parola.append((rot, c))
                    larghezza_parola += rot.width + 5
        
        if x + larghezza_parola > 2300: # Vai a capo
            x, y = 80, y + 220
        
        for img_l, char_orig in immagini_parola:
            # POSIZIONAMENTO VERTICALE
            if char_orig in ".,": # A terra
                pos_y = y + baseline_riga - img_l.height 
            elif char_orig == "'": # In alto
                pos_y = y + 10 
            else: # Standard
                pos_y = y + (baseline_riga - 125) 
            
            mask = ImageOps.invert(img_l).point(lambda p: 255 if p > 50 else 0)
            foglio.paste(img_l, (x, pos_y + random.randint(-2, 2)), mask=mask)
            x += img_l.width + 5
        x += 80

    foglio.save(output)
    print(f"Salvato in: {output}")

# --- ESECUZIONE ---
scrivi_testo_handwriting("Ciao Davide! L'universita' e' fantastica, specialmente quando i progetti funzionano.")