import cv2
import os
import numpy as np

# --- CONFIGURAZIONE ---
# MODIFICA QUESTI VALORI SE NECESSARIO!
INPUT_IMAGE_DIR = "dataset"  # Percorso alla tua cartella 'dataset'
OUTPUT_DIR_BASE = "dataset_segmentato" # Cartella dove salvare i caratteri ritagliati
MIN_CONTOUR_AREA = 50  # Area minima del contorno (potrebbe servire aggiustarla)

# --- Assicurati che la cartella base di output esista ---
os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)

print(f"Cerco immagini in: {os.path.abspath(INPUT_IMAGE_DIR)}")
print(f"I caratteri segmentati verranno salvati in: {os.path.abspath(OUTPUT_DIR_BASE)}")

# --- Itera su tutti i file nella cartella di input ---
for filename in os.listdir(INPUT_IMAGE_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        input_image_path = os.path.join(INPUT_IMAGE_DIR, filename)
        filename_no_ext = os.path.splitext(filename)[0] # Nome file senza estensione

        character_label = None # Inizializza l'etichetta

        # --- Logica per estrarre l'etichetta del carattere ---
        if filename_no_ext.startswith("lettere-maiuscole-"):
            char_candidate = filename_no_ext.split("-")[-1]
            if len(char_candidate) == 1 and char_candidate.isalpha():
                character_label = char_candidate.upper()
        
        elif filename_no_ext.startswith("numeri-"):
            char_candidate = filename_no_ext.split("-")[-1]
            if char_candidate.isdigit():
                character_label = char_candidate
        
        # NUOVA SEZIONE PER I SIMBOLI:
        elif filename_no_ext.startswith("simboli-"):
            # Estrae la parte dopo "simboli-"
            # es. da "simboli-APOSTROFO" prende "APOSTROFO"
            label_candidate = filename_no_ext.split("-", 1)[1] # Dividi solo al primo trattino
            if label_candidate: # Assicurati che non sia vuoto
                character_label = label_candidate # Usa direttamente il nome descrittivo
                                                # Non c'è bisogno di .upper() qui perché sono già descrittivi
        
        if character_label is None:
            # print(f"File {filename} non corrisponde ai pattern noti. Saltato.")
            continue
        
        print(f"\nProcesso il file: {filename} per l'etichetta: '{character_label}'")

        output_char_dir = os.path.join(OUTPUT_DIR_BASE, character_label)
        os.makedirs(output_char_dir, exist_ok=True)

        image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Errore: Impossibile caricare l'immagine da {input_image_path}")
            continue

        inverted_image = cv2.bitwise_not(image)
        contours, hierarchy = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        char_count_for_file = 0
        if contours:
            bounding_boxes = [cv2.boundingRect(c) for c in contours]
            contours_and_boxes = sorted(zip(contours, bounding_boxes), key=lambda b: (b[1][1], b[1][0]))
        else:
            contours_and_boxes = []

        for contour, (x,y,w,h) in contours_and_boxes:
            if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
                char_image_crop = image[y:y+h, x:x+w]
                
                current_files_in_subdir = len(os.listdir(output_char_dir))
                output_filename = os.path.join(output_char_dir, f"{character_label}_{current_files_in_subdir + 1:03d}.png")
                
                cv2.imwrite(output_filename, char_image_crop)
                char_count_for_file += 1

        print(f"Salvati {char_count_for_file} caratteri per il file {filename} nella cartella '{character_label}'.")
    else:
        print(f"File {filename} non è un'immagine supportata. Saltato.")

print("\n--- Processo di segmentazione completato! ---")