import json
import numpy as np
from dataset import Datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassfication,
    TrainingArgumnets,
    Trainer
)
import evaluate

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
TRAIN_PATH = "/content/AI--Powered-Contract-Intelligence-Risk-Scoring/clause_train.json"
TEST_PATH = "/content/AI--Powered-Contract-Intelligence-Risk-Scoring/clause_test.json"
LABEL_MAP_PATH = "/content/AI--Powered-Contract-Intelligence-Risk-Scoring/label_mapping.json"
OUTPUT_DIR = "/content/AI--Powered-Contract-Intelligence-Risk-Scoring/clause_classifier_model"

MAX_LENGTH = 256
NUM_EPOCHS = 3
BATCH_SIZE = 16