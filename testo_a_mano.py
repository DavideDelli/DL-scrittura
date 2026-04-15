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
print("Classi caricate:", list(label_to_int_map.keys()))

# ---------------------------------------------------------------------------
# 2. ARCHITETTURA ORIGINALE (Copiata esattamente dal tuo main_training.py)
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

# --- INIZIALIZZAZIONE E CARICAMENTO ---
enc = build_encoder(LATENT_DIM, NUM_CLASSES, EMBEDDING_DIM)
dec = build_decoder(LATENT_DIM, NUM_CLASSES, EMBEDDING_DIM)
cvae_model = CVAE(enc, dec)
_ = cvae_model([np.zeros((1, 128, 128, 1)), np.zeros((1, 1))])
cvae_model.load_weights(MODEL_WEIGHTS_PATH)
print("Pesi caricati con successo!")

# ---------------------------------------------------------------------------
# 3. LOGICA DI RENDERING
# ---------------------------------------------------------------------------

def genera_char_img(char, temp=0.6):
    mapping = {
        '?': 'INTERROGATIVO', '!': 'ESCLAMATIVO', '.': 'PUNTO', 
        ',': 'VIRGOLA', "'": 'VIRGOLA' # Se non hai 'APOSTROFO', usa VIRGOLA
    }
    # Prova a cercare APOSTROFO se presente, altrimenti segui il mapping
    c_key = char.upper()
    if char == "'":
        c_key = 'APOSTROFO' if 'APOSTROFO' in label_to_int_map else 'VIRGOLA'
    else:
        c_key = mapping.get(char, c_key)

    if c_key not in label_to_int_map:
        return None
    
    idx = label_to_int_map[c_key]
    z = tf.random.normal(shape=(1, LATENT_DIM)) * temp
    gen = cvae_model.decoder.predict([z, np.array([[idx]])], verbose=0)[0, :, :, 0]
    return (1.0 - gen) * 255

def scrivi_testo_migliorato(testo, output="risultato_finale.png"):
    # Gestione automatica accenti -> lettera + apostrofo
    acc = {'à':"A'", 'è':"E'", 'é':"E'", 'ì':"I'", 'ò':"O'", 'ù':"U'", 'À':"A'", 'È':"E'", 'É':"E'", 'Ì':"I'", 'Ò':"O'", 'Ù':"U'"}
    for k, v in acc.items(): testo = testo.replace(k, v)
    
    foglio = Image.new('L', (2400, 1800), 255)
    x, y = 80, 120
    riga_h = 220
    baseline_y = 150 # Altezza ideale dove appoggiano le lettere

    parole = testo.split(' ')
    for parola in parole:
        imgs = []
        w_parola = 0
        for c in parola:
            raw = genera_char_img(c)
            if raw is not None:
                img_c = Image.fromarray(raw.astype(np.uint8)).convert("L")
                img_c = ImageEnhance.Contrast(img_c).enhance(2.5)
                # PULIZIA DRASTICA QUADRATINI
                img_c = img_c.point(lambda p: 255 if p > 195 else p)
                
                inv = ImageOps.invert(img_c)
                bbox = inv.getbbox()
                if bbox:
                    crop = img_c.crop(bbox)
                    # Rimpicciolisci segni punteggiatura
                    if c in ".,'":
                        crop = crop.resize((int(crop.width*0.65), int(crop.height*0.65)), Image.LANCZOS)
                    
                    rot = crop.rotate(random.uniform(-1.8, 1.8), resample=Image.BICUBIC, expand=True)
                    imgs.append((rot, c))
                    w_parola += rot.width + 5
        
        if x + w_parola > 2300:
            x, y = 80, y + riga_h
        
        for img_l, char_orig in imgs:
            # POSIZIONAMENTO VERTICALE DINAMICO
            if char_orig in ".,":
                pos_y = y + baseline_y - img_l.height # Attaccato alla base
            elif char_orig == "'":
                pos_y = y + 15 # In alto
            else:
                pos_y = y + (baseline_y - 120) # Lettere normali un po' più in alto della base
            
            mask = ImageOps.invert(img_l).point(lambda p: 255 if p > 60 else 0)
            foglio.paste(img_l, (x, pos_y + random.randint(-2, 2)), mask=mask)
            x += img_l.width + 5
        x += 80

    foglio.save(output)
    print(f"Creato: {output}")

# --- ESECUZIONE ---
scrivi_testo_migliorato("Ciao Davide! L'università è fantastica, specialmente quando i progetti funzionano.")