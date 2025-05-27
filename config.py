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
    "DOC_FACTS": 15,
    "MIN_DOC_FACTS": 8,
    "MAX_DOC_FACTS": 25,
    
    # Training settings - MUCH MORE CONSERVATIVE
    "SEED": 42,
    "BATCH_SIZE": 32,           # Smaller batch
    "LEARNING_RATE": 1e-4,     # Much smaller LR
    "TRAIN_STEPS": 50000,     # Fewer steps for now
    "MODEL_DIM": 128,          # Smaller model
    "MODEL_LAYERS": 2,         # Fewer layers
    "RETRIEVAL_K": 4,
}