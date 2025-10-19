CONFIG = {
    # Entity counts
    "N_PERSON": 3000,
    "N_COMPANY": 500, 
    "N_PRODUCT": 1000,
    "N_TECHNOLOGY": 200,
    "N_LOCATION": 300,
    "N_PROJECT": 800,
    "N_SKILL": 150,
    "N_DATE": 366,
    
    # Document settings
    "DUP_RATE": 0.25,
    "DOC_FACTS": 8,          # Reduced from 15 to 8
    "MIN_DOC_FACTS": 5,      # Reduced from 8 to 5
    "MAX_DOC_FACTS": 12,     # Reduced from 25 to 12
    
    # Training settings - MORE CONSERVATIVE
    "SEED": 42,
    "BATCH_SIZE": 8,              # Smaller batch size
    "GRADIENT_ACCUMULATION": 4,   # Effective batch size = 16
    "LEARNING_RATE": 2e-6,        # Much lower learning rate
    "NUM_EPOCHS": 10,
    "WARMUP_STEPS": 1000,          # Add warmup
    "WEIGHT_DECAY": 0.01,
    "MAX_GRAD_NORM": 1,         # Lower gradient clipping
    
    # Model architecture - CONSISTENT DIMENSIONS
    "HIDDEN_SIZE": 768,           # Standard GPT2 size
    "NUM_LAYERS": 6,
    "NUM_HEADS": 12,              # Must divide HIDDEN_SIZE
    "MAX_LENGTH": 512,
    "RETRIEVAL_K": 4,
    
    # Compatibility keys for model.py
    "MODEL_DIM": 768,             # Same as HIDDEN_SIZE
    "MODEL_LAYERS": 6,            # Same as NUM_LAYERS
    "TRAIN_STEPS": 50000,         # For evaluation reporting
    
    # Data
    "TOP_K_DOCS": 4,  # Reduced from 6 to 4 to fit within 512 tokens
    "MAX_FACTS_PER_DOC": 10,
    "TRAIN_FROM_SCRATCH": True,  # Set to True to train from scratch
}