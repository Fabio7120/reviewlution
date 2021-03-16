
### GCP configuration - - - - - - - - - - - - - - - - - - -

# not required here (done locally)

### GCP Project - - - - - - - - - - - - - - - - - - - - - -

# not required here (done in Makefile)

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'reviewlution-01'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location

BUCKET_TRAIN_DATA_PATH = 'data/dataset_1.csv'

##### Training  - - - - - - - - - - - - - - - - - - - - - -

# not required here (done in trainer.py)

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'reviewlution'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'Pipeline'

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

# not required here (done in Makefile)

### - - - - - - - - - - - - - - - - - - - - - - - - - - - -
