# verifica_gpu.py
import tensorflow as tf

print(f"Versione di TensorFlow: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU disponibili: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"GPU #{i}: Nome: {gpu.name}, Tipo: {gpu.device_type}")
        try:
            # Prova a ottenere dettagli specifici per DirectML (potrebbe variare)
            details = tf.config.experimental.get_device_details(gpu)
            if details: # Solo se get_device_details restituisce qualcosa
                print(f"  Dettagli: {details.get('device_name', 'N/A')}") # 'device_name' è un esempio
        except Exception as e:
            print(f"  Impossibile ottenere dettagli specifici per la GPU #{i}: {e}")
else:
    print("Nessuna GPU disponibile per TensorFlow.")

print("\nEseguo un semplice calcolo su GPU (se disponibile) e CPU per confronto (opzionale).")
# Prova a eseguire una semplice operazione per vedere se la GPU viene usata
try:
    # Forza l'operazione sulla GPU se disponibile
    if gpus:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c_gpu = tf.matmul(a, b)
        print("Calcolo su GPU eseguito (se la GPU è stata usata, non ci saranno errori qui). Risultato (GPU):")
        print(c_gpu.numpy())

    # Esegui sulla CPU
    with tf.device('/CPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c_cpu = tf.matmul(a, b)
    print("Calcolo su CPU eseguito. Risultato (CPU):")
    print(c_cpu.numpy())

except RuntimeError as e:
    print(f"Errore durante il test di calcolo: {e}")
    print("Questo potrebbe indicare un problema con la configurazione della GPU o DirectML.")