from pathlib import Path

num_samples = 180
# originally 50
num_tokens_at_once = 50
# originally 768
embedding_dim = 768
l2_weight = 1.0e-5
num_entity_classes = 8
num_event_classes = 2
dim_ent = 15
seed_value = 5489
# hidden_dim = embedding_dim + dim_ent
hidden_dim = 24
MASK_VALUE_IGNORE_POSITION = 0
beta = 1.0
learning_rate = 0.001
num_training_epochs = 20
log_dir = Path(".").resolve()

lstm_dim = hidden_dim
event_embedding_dim = 2
