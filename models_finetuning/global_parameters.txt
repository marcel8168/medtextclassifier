###############################################################################
# Global parameters
###############################################################################

NUM_LABELS = 2
LABELS_MAP = {
    0: "human_medicine", 
    1: "veterinary_medicine"
    }
MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
EPOCHS = 2
LEARNING_RATE = 1e-5
RANDOM_SEED = 42