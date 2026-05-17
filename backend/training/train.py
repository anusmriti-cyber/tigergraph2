import os
import glob
import torch
import numpy as np

from filelock import FileLock
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from torch.utils.data import Dataset


# =========================================================
# PATH SETUP
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

DATA_DIR = os.path.join(BASE_DIR, 'Dataset', 'papers_txt')

LOCK_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'classification.lock'
)


# =========================================================
# CONDITIONS + KEYWORDS
# =========================================================

CONDITIONS = {
    'ASD': [
        'asd',
        'autism',
        'autistic',
        'asperger'
    ],

    'ADHD': [
        'adhd',
        'attention deficit',
        'hyperactivity'
    ],

    'Dyscalculia': [
        'dyscalculia',
        'math learning disability'
    ],

    'Dyslexia': [
        'dyslexia',
        'reading disability'
    ]
}


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    texts = []
    labels = []

    label_map = {
        'ASD': 0,
        'ADHD': 1,
        'Dyscalculia': 2,
        'Dyslexia': 3,
        'Other': 4
    }

    print(f"Searching for papers in:\n{DATA_DIR}\n")

    txt_files = glob.glob(os.path.join(DATA_DIR, '*.txt'))

    print(f"Found {len(txt_files)} text files.\n")

    for filepath in txt_files:

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()

            text_lower = text.lower()

            assigned_label = 'Other'

            # Keyword-based pseudo labeling
            for condition, keywords in CONDITIONS.items():

                if any(keyword in text_lower for keyword in keywords):
                    assigned_label = condition
                    break

            texts.append(text)
            labels.append(label_map[assigned_label])

        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    return texts, labels, label_map


# =========================================================
# DATASET CLASS
# =========================================================

class PaperDataset(Dataset):

    def __init__(self, encodings, labels):

        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):

        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }

        item['labels'] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):

        return len(self.labels)


# =========================================================
# METRICS
# =========================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)

    f1 = f1_score(
        labels,
        predictions,
        average='weighted',
        zero_division=0
    )

    return {
        'accuracy': accuracy,
        'f1': f1
    }


# =========================================================
# MAIN TRAINING FUNCTION
# =========================================================

def main():

    # -----------------------------
    # Load data
    # -----------------------------

    texts, labels, label_map = load_data()

    print(f"Loaded {len(texts)} papers.\n")

    if len(texts) == 0:
        print("No text files found.")
        print("Check Dataset/papers_txt folder.")
        return

    # -----------------------------
    # Load tokenizer
    # -----------------------------

    print("Loading tokenizer...\n")

    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased'
    )

    # -----------------------------
    # Train-validation split
    # -----------------------------

    print("Creating train-validation split...\n")

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=0.30,
        random_state=42
    )

    # -----------------------------
    # Lock file
    # -----------------------------

    with FileLock(LOCK_FILE, timeout=10):

        print("Lock acquired.")
        print("Starting preprocessing...\n")

        # -----------------------------
        # Tokenization
        # -----------------------------

        train_encodings = tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=512
        )

        val_encodings = tokenizer(
            val_texts,
            truncation=True,
            padding=True,
            max_length=512
        )

        train_dataset = PaperDataset(
            train_encodings,
            train_labels
        )

        val_dataset = PaperDataset(
            val_encodings,
            val_labels
        )

        # -----------------------------
        # Load BERT model
        # -----------------------------

        print("Loading BERT model...\n")

        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=len(label_map)
        )

        # -----------------------------
        # Training arguments
        # -----------------------------

        training_args = TrainingArguments(

            output_dir='./results',

            num_train_epochs=3,

            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,

            warmup_steps=50,

            weight_decay=0.01,

            logging_strategy="steps",
            logging_steps=10,

            eval_strategy="epoch",

            save_strategy="epoch",

            load_best_model_at_end=True,

            metric_for_best_model="f1",

            greater_is_better=True,

            save_total_limit=2,

            report_to="none"
        )

        # -----------------------------
        # Trainer
        # -----------------------------

        trainer = Trainer(

            model=model,

            args=training_args,

            train_dataset=train_dataset,

            eval_dataset=val_dataset,

            compute_metrics=compute_metrics
        )

        # -----------------------------
        # Train
        # -----------------------------

        print("=" * 50)
        print("STARTING TRAINING")
        print("=" * 50)

        trainer.train()

        # -----------------------------
        # Evaluate
        # -----------------------------

        print("\nEvaluating model...\n")

        eval_result = trainer.evaluate()

        # -----------------------------
        # Save model
        # -----------------------------

        print("\nSaving model...\n")

        trainer.save_model("./saved_model")

        tokenizer.save_pretrained("./saved_model")

        # -----------------------------
        # Final Results
        # -----------------------------

        print("\n" + "=" * 50)
        print("FINAL RESULTS")
        print("=" * 50)

        print(
            f"Validation Accuracy: "
            f"{eval_result.get('eval_accuracy', 0.0):.4f}"
        )

        print(
            f"Validation F1 Score: "
            f"{eval_result.get('eval_f1', 0.0):.4f}"
        )

        print("\nTraining completed successfully.")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == '__main__':

    main()