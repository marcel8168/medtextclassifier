###############################################################################
# Global parameters
###############################################################################

NUM_LABELS = 2
LABELS_MAP = {
    "human_medicine": [1, 0],
    "veterinary_medicine": [0, 1]
}
MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
EPOCHS = 2
LEARNING_RATE = 1e-5
RANDOM_SEED = 42
PATH_SAVED_MODELS = "./saved_models/"
