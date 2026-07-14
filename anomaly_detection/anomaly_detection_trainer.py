# 1. Mount the drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone the repository code
!git clone https://github.com/Fabio-Feruglio/Jet-Image-Tagging.git

# Move to the correct folder
%cd Jet-Image-Tagging/anomaly_detection

# 3. Install dependencies
!pip install optuna wandb

# 4. Authenticate on WandB
from google.colab import userdata
import wandb
wandb.login(key=userdata.get('WANDB_API_KEY'))

# 5. COPIA IL DATASET IN LOCALE (CRITICO PER LE PERFORMANCE)
print("Copiando il dataset da Drive allo storage locale... (potrebbe volerci qualche minuto)")
!cp /content/drive/MyDrive/JetTagging/data/jet_images_299.h5 /content/jet_images_299.h5
print("Copia completata!")

# 6. Train the autoencoder (Puntando al file locale /content/ e salvando su Drive)
!python src/train.py \
    --data_path "/content/jet_images_299.h5" \
    --save_dir "/content/drive/MyDrive/JetTagging/optuna_logs" \
    --epochs 50 \
    --batch_size 64 \
    --lr 0.0005 \
    --encoded_space_dim 128