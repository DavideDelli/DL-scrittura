import cv2
import os
import numpy as np

# --- CONFIGURAZIONE ---
# MODIFICA QUESTI VALORI SE NECESSARIO!
INPUT_IMAGE_DIR = "dataset"  # Percorso alla tua cartella 'dataset'
OUTPUT_DIR_BASE = "dataset_segmentato" # Cartella dove salvare i caratteri ritagliati
MIN_CONTOUR_AREA = 50  # Area minima del contorno per essere un carattere (potrebbe servire aggiustarla)
# Per i simboli multi-parte, potremmo aver bisogno di una soglia più piccola per i puntini.
# La logica speciale per INTERROGATIVO/ESCLAMATIVO cerca di gestirlo.

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
        
        elif filename_no_ext.startswith("simboli-"): # "simboli-" plurale
            label_candidate = filename_no_ext.split("-", 1)[1] 
            if label_candidate: 
                character_label = label_candidate
        
        if character_label is None:
            # print(f"File {filename} non corrisponde ai pattern noti o etichetta non valida. Saltato.")
            continue
        
        print(f"\nProcesso il file: {filename} per l'etichetta: '{character_label}'")

        output_char_dir = os.path.join(OUTPUT_DIR_BASE, character_label)
        os.makedirs(output_char_dir, exist_ok=True)

        image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Errore: Impossibile caricare l'immagine da {input_image_path}")
            continue

        inverted_image = cv2.bitwise_not(image) # Oggetti bianchi su sfondo nero per findContours
        
        final_bounding_boxes = [] # Lista per i bounding box finali (x,y,w,h) da ritagliare

        # --- Logica di gestione dei contorni ---
        if character_label in ["INTERROGATIVO", "ESCLAMATIVO"]:
            print(f"Applico logica speciale per: {character_label}")
            # Usiamo RETR_LIST per ottenere tutti i contorni per questi simboli
            contours, _ = cv2.findContours(inverted_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            potential_parts = []
            if contours: # Controlla se la lista contorni non è vuota
                for c in contours:
                    # Usiamo una frazione di MIN_CONTOUR_AREA per i puntini, ma non troppo piccola
                    if cv2.contourArea(c) > max(5, MIN_CONTOUR_AREA / 4): 
                        potential_parts.append(c)
            
            main_bodies_bbs = [] # Bounding boxes dei corpi principali
            dot_bbs = []       # Bounding boxes dei puntini

            if potential_parts:
                areas = [cv2.contourArea(p) for p in potential_parts]
                # Una soglia per distinguere corpi e puntini. Potrebbe necessitare di aggiustamenti.
                # Ad esempio, un puntino è significativamente più piccolo del corpo principale.
                # Se ci sono poche parti, questa euristica è delicata.
                # Proviamo a definire un puntino come qualcosa che è < 40% dell'area più grande trovata.
                if areas:
                    max_area_found = np.max(areas)
                    # Definiamo una soglia per i puntini più robusta, es. se area < 30% della max_area_found
                    # e comunque sopra una minima soglia assoluta (es. 5-10 pixel)
                    dot_area_threshold = max_area_found * 0.3 
                    min_absolute_dot_area = 10 # Un puntino deve avere almeno quest'area

                    for c in potential_parts:
                        area = cv2.contourArea(c)
                        if area < dot_area_threshold and area > min_absolute_dot_area :
                            dot_bbs.append(cv2.boundingRect(c))
                        elif area >= dot_area_threshold: # Altrimenti è un corpo principale
                            main_bodies_bbs.append(cv2.boundingRect(c))
                
            processed_dots_indices = [False] * len(dot_bbs)
            for mb_x, mb_y, mb_w, mb_h in main_bodies_bbs:
                best_dot_idx = -1
                
                mb_center_x = mb_x + mb_w / 2
                mb_bottom_y = mb_y + mb_h

                # Criteri per associare un puntino
                # (queste tolleranze potrebbero aver bisogno di aggiustamenti)
                max_y_distance_for_dot = mb_h * 1.0  # Max distanza Y dal fondo del corpo al top del puntino
                max_x_offset_for_dot = mb_w * 0.75 # Max disallineamento X del centro del puntino

                # Cerchiamo il puntino più plausibile
                closest_dot_y_dist = max_y_distance_for_dot 

                for i, (d_x, d_y, d_w, d_h) in enumerate(dot_bbs):
                    if processed_dots_indices[i]:
                        continue
                    
                    d_center_x = d_x + d_w / 2
                    
                    # Il puntino deve essere SOTTO il corpo principale
                    is_below = d_y > mb_y + mb_h * 0.5 # Il puntino inizia sotto la metà del corpo
                    # Allineamento orizzontale
                    is_aligned_x = abs(mb_center_x - d_center_x) < max_x_offset_for_dot
                    # Distanza verticale (dal fondo del corpo al top del puntino)
                    y_distance = d_y - mb_bottom_y
                    is_close_y = 0 <= y_distance < closest_dot_y_dist # Puntino sotto e non troppo lontano

                    if is_below and is_aligned_x and is_close_y:
                        best_dot_idx = i
                        closest_dot_y_dist = y_distance # Aggiorna per trovare il più vicino
                
                if best_dot_idx != -1:
                    d_x, d_y, d_w, d_h = dot_bbs[best_dot_idx]
                    processed_dots_indices[best_dot_idx] = True
                    
                    combined_x = min(mb_x, d_x)
                    combined_y = min(mb_y, d_y)
                    combined_w = max(mb_x + mb_w, d_x + d_w) - combined_x
                    combined_h = max(mb_y + mb_h, d_y + d_h) - combined_y
                    final_bounding_boxes.append((combined_x, combined_y, combined_w, combined_h))
                else:
                    final_bounding_boxes.append((mb_x, mb_y, mb_w, mb_h)) # Salva corpo senza puntino
            
            # Opzionale: salva puntini non associati se sono abbastanza grandi (debug)
            # for i, (d_x, d_y, d_w, d_h) in enumerate(dot_bbs):
            #     if not processed_dots_indices[i]:
            #         print(f"WARN: Puntino non associato ({d_x},{d_y},{d_w},{d_h}) per {character_label} in {filename}")
            #         # final_bounding_boxes.append((d_x, d_y, d_w, d_h)) # Decommenta per salvarli separatamente

        else: # Logica originale per caratteri a singola parte
            contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Filtra e ordina i contorni per caratteri singoli
                valid_contours_with_boxes = []
                for c in contours:
                    if cv2.contourArea(c) > MIN_CONTOUR_AREA:
                        valid_contours_with_boxes.append(cv2.boundingRect(c))
                
                # Ordina per y, poi per x (dall'alto in basso, da sinistra a destra)
                valid_contours_with_boxes.sort(key=lambda b: (b[1], b[0]))
                for x,y,w,h in valid_contours_with_boxes:
                     final_bounding_boxes.append((x,y,w,h))
            else:
                print(f"Nessun contorno esterno valido trovato per {character_label} in {filename}")
                # Non fare 'continue' qui, altrimenti il print finale di "Salvati 0 caratteri" non appare

        # --- Ritaglia e salva usando i final_bounding_boxes ---
        char_count_for_file = 0
        # Ri-ordina i bounding box finali (specialmente se mischiati da logica speciale e normale)
        final_bounding_boxes.sort(key=lambda b: (b[1], b[0])) # Ordina per y (riga), poi per x (colonna)

        for x,y,w,h in final_bounding_boxes:
            # Assicurati che le coordinate siano valide (a volte possono essere leggermente fuori)
            y_start, y_end = max(0, y), min(image.shape[0], y + h)
            x_start, x_end = max(0, x), min(image.shape[1], x + w)
            
            if y_end > y_start and x_end > x_start : # Controlla che l'area di crop sia valida
                char_image_crop = image[y_start:y_end, x_start:x_end]
                
                # Trova un nome univoco progressivo per i file nella cartella di destinazione
                # Questo modo di contare è più robusto se si riesegue o si aggiungono file
                existing_files_count = 0
                for f_name in os.listdir(output_char_dir):
                    if f_name.startswith(f"{character_label}_") and f_name.endswith(".png"):
                        existing_files_count +=1
                
                output_filename = os.path.join(output_char_dir, f"{character_label}_{existing_files_count + 1:03d}.png")
                
                cv2.imwrite(output_filename, char_image_crop)
                char_count_for_file += 1
            else:
                print(f"WARN: Bounding box non valido ({x},{y},{w},{h}) per {character_label} in {filename}. Saltato.")


        print(f"Salvati {char_count_for_file} caratteri per il file {filename} nella cartella '{character_label}'.")
    else:
        # print(f"File {filename} non è un'immagine supportata o non corrisponde ai pattern. Saltato.")
        pass


print("\n--- Processo di segmentazione completato! ---")