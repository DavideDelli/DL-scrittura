import os
import numpy as np
from sklearn.model_selection import train_test_split # Per dividere il dataset
import tensorflow as tf # Lo importeremo più avanti per il data loading

DATASET_NORMALIZZATO_DIR = "dataset_normalizzato" # La tua cartella con le immagini 128x128

image_paths = []
labels = [] # Etichette come stringhe (es. "A", "B", "PUNTO")

# Scansiona le sottocartelle (ognuna è un'etichetta)
for label_name in os.listdir(DATASET_NORMALIZZATO_DIR):
    label_dir = os.path.join(DATASET_NORMALIZZATO_DIR, label_name)
    if os.path.isdir(label_dir):
        for image_filename in os.listdir(label_dir):
            if image_filename.lower().endswith('.png'): # Assicurati che siano file png
                image_paths.append(os.path.join(label_dir, image_filename))
                labels.append(label_name)

print(f"Trovate {len(image_paths)} immagini.")
print(f"Numero di etichette uniche (classi): {len(np.unique(labels))}")
# Stampa alcune etichette uniche per verifica
print(f"Esempi di etichette: {np.unique(labels)[:10]}")

# Crea un mapping da etichetta stringa a intero
unique_labels = sorted(list(np.unique(labels))) # Ordina per consistenza
label_to_int = {label: i for i, label in enumerate(unique_labels)}
int_to_label = {i: label for i, label in enumerate(unique_labels)}

# Converte le etichette stringa in interi
integer_labels = [label_to_int[label] for label in labels]

num_classes = len(unique_labels)
print(f"Numero di classi (num_classes): {num_classes}")
print(f"Mapping per 'A': {label_to_int.get('A', 'Non Trovata')}")
print(f"Mapping per 'PUNTO': {label_to_int.get('PUNTO', 'Non Trovata')}")

# Converti le liste in array NumPy per scikit-learn
image_paths_np = np.array(image_paths)
integer_labels_np = np.array(integer_labels)

# Suddivisione (es. 80% training, 20% validazione)
# 'stratify=integer_labels_np' assicura che la proporzione delle classi sia mantenuta in entrambi i set
X_train_paths, X_val_paths, y_train_labels, y_val_labels = train_test_split(
    image_paths_np, 
    integer_labels_np, 
    test_size=0.2,       # 20% per la validazione
    random_state=42,     # Per riproducibilità
    stratify=integer_labels_np # Mantiene la proporzione delle classi
)

print(f"Immagini nel training set: {len(X_train_paths)}")
print(f"Etichette nel training set: {len(y_train_labels)}")
print(f"Immagini nel validation set: {len(X_val_paths)}")
print(f"Etichette nel validation set: {len(y_val_labels)}")

import tensorflow as tf # Ora importiamo TensorFlow

IMG_HEIGHT = 128
IMG_WIDTH = 128
CHANNELS = 1 # Scala di grigi

def load_and_preprocess_image(image_path, label):
    # Leggi il file immagine
    img_raw = tf.io.read_file(image_path)
    # Decodifica l'immagine PNG (o JPG se usi JPG) in scala di grigi
    img = tf.image.decode_png(img_raw, channels=CHANNELS) 
    # Assicurati che sia float32 e normalizza i pixel a [0, 1]
    img = tf.image.convert_image_dtype(img, tf.float32) 
    # Ridimensiona (anche se dovrebbero già essere 128x128, è una buona pratica confermarlo)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    return img, label

# Creiamo i dataset di TensorFlow
BATCH_SIZE = 32 # O 64, dipende dalla memoria GPU

# Training Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_paths, y_train_labels))
train_dataset = train_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=len(X_train_paths)) # Mescola i dati
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE) # Ottimizza le prestazioni

# Validation Dataset
val_dataset = tf.data.Dataset.from_tensor_slices((X_val_paths, y_val_labels))
val_dataset = val_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

print("\nDataset di training e validazione pronti!")
# Puoi ispezionare un batch per vedere la forma dei dati (opzionale)
# for images, labels_batch in train_dataset.take(1):
#     print("Forma del batch di immagini:", images.shape)
#     print("Forma del batch di etichette:", labels_batch.shape)
#     print("Esempio di etichetta nel batch:", labels_batch.numpy()[0])