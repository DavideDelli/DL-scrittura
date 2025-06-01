import cv2
import os
import numpy as np

# --- CONFIGURAZIONE ---
INPUT_DIR_SEGMENTATO = "dataset_segmentato"  # Cartella con i caratteri segmentati
OUTPUT_DIR_NORMALIZZATO = "dataset_normalizzato" # Cartella dove salvare i caratteri normalizzati
TARGET_SIZE = 128  # Dimensione finale dell'immagine (TARGET_SIZE x TARGET_SIZE)
BACKGROUND_COLOR = 255  # Colore di sfondo/padding (255 per bianco)

# --- Assicurati che la cartella base di output esista ---
os.makedirs(OUTPUT_DIR_NORMALIZZATO, exist_ok=True)

print(f"Leggo immagini da: {os.path.abspath(INPUT_DIR_SEGMENTATO)}")
print(f"Immagini normalizzate verranno salvate in: {os.path.abspath(OUTPUT_DIR_NORMALIZZATO)}")
print(f"Dimensione target: {TARGET_SIZE}x{TARGET_SIZE} pixel")

# --- Itera su ogni cartella di etichetta (A, B, PUNTO, ecc.) ---
for label_name in os.listdir(INPUT_DIR_SEGMENTATO):
    label_input_path = os.path.join(INPUT_DIR_SEGMENTATO, label_name)
    label_output_path = os.path.join(OUTPUT_DIR_NORMALIZZATO, label_name)

    if not os.path.isdir(label_input_path):
        continue # Salta file che non sono directory (es. .DS_Store su macOS)

    os.makedirs(label_output_path, exist_ok=True)
    print(f"\nProcesso etichetta: {label_name}")
    
    processed_count = 0
    # --- Itera su ogni immagine di carattere segmentata ---
    for image_filename in os.listdir(label_input_path):
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            char_image_path = os.path.join(label_input_path, image_filename)

            # Carica l'immagine in scala di grigi
            img = cv2.imread(char_image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Errore: Impossibile caricare {char_image_path}. Saltata.")
                continue

            img_h, img_w = img.shape

            # Calcola il rapporto di aspetto e la nuova dimensione
            aspect_ratio = img_w / img_h

            if img_w > img_h:
                # L'immagine è più larga che alta
                new_w = TARGET_SIZE
                new_h = int(TARGET_SIZE / aspect_ratio)
            else:
                # L'immagine è più alta che larga, o quadrata
                new_h = TARGET_SIZE
                new_w = int(TARGET_SIZE * aspect_ratio)
            
            # Assicurati che new_w e new_h non siano zero (se l'immagine originale è degenere)
            if new_w == 0: new_w = 1
            if new_h == 0: new_h = 1

            # Ridimensiona l'immagine mantenendo l'aspect ratio
            # cv2.INTER_LINEAR è una buona scelta generale.
            # cv2.INTER_AREA è spesso buono per rimpicciolire.
            # cv2.INTER_CUBIC o cv2.INTER_LANCZOS4 per ingrandire (più lenti ma qualità migliore).
            # Dato che potremmo sia ingrandire che rimpicciolire per adattare, INTER_LINEAR è un compromesso.
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Crea un'immagine quadrata di sfondo bianco (canvas)
            canvas = np.full((TARGET_SIZE, TARGET_SIZE), BACKGROUND_COLOR, dtype=np.uint8)

            # Calcola la posizione per centrare l'immagine ridimensionata sul canvas
            y_offset = (TARGET_SIZE - new_h) // 2
            x_offset = (TARGET_SIZE - new_w) // 2

            # Incolla l'immagine ridimensionata sul canvas
            canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized_img
            
            # Salva l'immagine normalizzata
            output_image_save_path = os.path.join(label_output_path, image_filename)
            cv2.imwrite(output_image_save_path, canvas)
            processed_count += 1
            
    print(f"Salvate {processed_count} immagini normalizzate per l'etichetta '{label_name}'.")

print("\n--- Processo di normalizzazione completato! ---")