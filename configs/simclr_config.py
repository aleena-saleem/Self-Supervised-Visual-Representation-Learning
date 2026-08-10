
# ==================================================
# VisionSSL - SimCLR Training Configuration
# ==================================================

# Dataset
DATASET = "STL10"

DATA_ROOT = (
    "/content/drive/MyDrive/"
    "VisionSSL/data"
)

MAX_SAMPLES = 5000


# Image
IMAGE_SIZE = 96


# Training
BATCH_SIZE = 64

EPOCHS = 100

LEARNING_RATE = 0.0003


# SimCLR
TEMPERATURE = 0.5


# Checkpoint
CHECKPOINT_NAME = "simclr_resnet50_stl10.pth"

RESUME = False
