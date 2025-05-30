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
    "BATCH_SIZE": 8,
    "LEARNING_RATE": 1e-5,
    "NUM_EPOCHS": 10,
    "MODEL_DIM": 256,          # Smaller model
    "MODEL_LAYERS": 8,         # Fewer layers
    "RETRIEVAL_K": 4,
    
    # Model architecture
    "HIDDEN_SIZE": 1024,
    "NUM_LAYERS": 6,
    "NUM_HEADS": 8,
    "MAX_LENGTH": 512,
    
    # Data
    "TOP_K_DOCS": 3,
    "MAX_FACTS_PER_DOC": 10,
}