# scripts/fine_tune.py

import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
)
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
import torch
import numpy as np  # Added for class weight computation
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from transformers import EarlyStoppingCallback  # Import the early stopping callback

# Import custom modules
from data_utils import preprocess_data
from metrics import compute_metrics, generate_confusion_matrix

# Suppress FutureWarnings (optional)
warnings.filterwarnings("ignore", category=FutureWarning)

# Define a custom Trainer to incorporate class weights
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

@hydra.main(config_path="../config", config_name="config", version_base="1.1")
def fine_tune(cfg: DictConfig):
    print("Configuration:\n", OmegaConf.to_yaml(cfg))

    # Initialize Label Encoder
    label_encoder = LabelEncoder()

    # Define column names
    label_column = "V12_Frame_Risiko_Fortschritt_Gemischt \n0 = nicht relevant, 1 = Risiko, 2 = Fortschritt, 3 = gemischt, 33 = kein Frame"
    text_column = "Artikeltext_Auto"

    # Load and preprocess data
    train_data = preprocess_data(
        cfg.data.train_file,
        text_column,
        label_column,
        label_encoder,
        fit=True  # Fit encoder on training data
    )
    test_data = preprocess_data(
        cfg.data.test_file,
        text_column,
        label_column,
        label_encoder,
        fit=False  # Use the fitted encoder
    )
    val_data = preprocess_data(
        cfg.data.validation_file,
        text_column,
        label_column,
        label_encoder,
        fit=False  # Use the fitted encoder
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

    def tokenize_function(examples):
        return tokenizer(examples[text_column], padding="max_length", truncation=True)

    # Tokenize datasets
    train_data = train_data.map(tokenize_function, batched=True)
    test_data = test_data.map(tokenize_function, batched=True)
    val_data = val_data.map(tokenize_function, batched=True)

    # Set format for PyTorch
    train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Compute class weights for balanced loss
    labels = train_data['labels']
    class_counts = np.bincount(labels)
    # Handle potential zero counts to avoid division by zero
    class_counts = np.where(class_counts == 0, 1, class_counts)
    class_weights = 1.0 / class_counts
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    # print(class_counts, "\n", class_weights)
    # exit()
    # Initialize results storage
    results = []
    train_losses_per_seed = {}
    eval_losses_per_seed = {}

    # Iterate over each seed
    for seed in cfg.random_seeds:
        print(f"\nTraining with random seed: {seed} using model: {cfg.model.model_name}")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Initialize model
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model.model_name, num_labels=len(label_encoder.classes_)
        )

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(cfg.training.output_dir, f"model_{cfg.model.model_name}_seed_{seed}"),
            eval_strategy="epoch",  # Updated from evaluation_strategy to eval_strategy
            save_strategy="epoch",
            learning_rate=cfg.training.learning_rate,
            per_device_train_batch_size=cfg.training.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
            num_train_epochs=cfg.training.num_train_epochs,
            weight_decay=cfg.training.weight_decay,
            warmup_steps=cfg.training.warmup_steps,
            logging_dir=os.path.join(cfg.training.output_dir, f"logs_seed_{seed}"),
            logging_steps=cfg.training.logging_steps,
            max_grad_norm=cfg.training.max_grad_norm,
            load_best_model_at_end=True,
            metric_for_best_model="uar",
            greater_is_better=True,
            save_total_limit=1,  # Keep only the best model
            # fp16=True,  # Use mixed precision if supported
            # dataloader_num_workers=4,  # Optional: adjust based on your system
        )

        # Define optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.training.warmup_steps,
            num_training_steps=len(train_data) * cfg.training.num_train_epochs,
        )

        # Move model and class_weights to the appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        class_weights_device = class_weights.to(device)

        # Initialize WeightedTrainer instead of the standard Trainer
        trainer = WeightedTrainer(
            class_weights=class_weights_device,
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            tokenizer=tokenizer,
            compute_metrics=lambda eval_pred: compute_metrics(eval_pred, label_encoder),
            optimizers=(optimizer, scheduler),
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Set patience to 2 epochs

        )

        # Train the model
        trainer.train()

        # Evaluate the model
        eval_result = trainer.evaluate(test_data)

        # Extract training and evaluation loss per epoch
        log_history = trainer.state.log_history
        train_losses = []
        eval_losses = []
        epoch_train_losses = []

        for log in log_history:
            if 'loss' in log and 'step' in log:
                epoch_train_losses.append(log['loss'])
            if 'eval_loss' in log:
                if epoch_train_losses:
                    avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
                    train_losses.append(avg_train_loss)
                    epoch_train_losses = []
                eval_losses.append(log['eval_loss'])

        # Handle any remaining train losses
        if epoch_train_losses:
            avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            train_losses.append(avg_train_loss)

        # Store losses
        train_losses_per_seed[seed] = train_losses
        eval_losses_per_seed[seed] = eval_losses

        # Extract UAR
        uar = eval_result.get("eval_uar", None)
        if uar is None:
            print(f"Warning: 'eval_uar' not found in evaluation results for seed {seed}.")

        # Append per-class metrics
        per_class_metrics = {k: v for k, v in eval_result.items() if k.startswith("accuracy_class_")}

        # Predict on test data for confusion matrix
        predictions = trainer.predict(test_data)
        preds = torch.argmax(torch.tensor(predictions.predictions), dim=1).numpy()
        labels = predictions.label_ids
        cm_df = generate_confusion_matrix(labels, preds, label_encoder)

        # Print and log Confusion Matrix
        print(f"Confusion Matrix for seed {seed}:\n{cm_df}\n")

        # Append results
        results.append({
            "seed": seed,
            "uar": uar,
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": cm_df
        })

        # Save model and tokenizer
        trainer.save_model(os.path.join(cfg.training.output_dir, f"model_{cfg.model.model_name}_seed_{seed}"))
        tokenizer.save_pretrained(os.path.join(cfg.training.output_dir, f"model_{cfg.model.model_name}_seed_{seed}"))

    # Save log to a file
    log_file = os.path.join(cfg.training.output_dir, "training_log.txt")
    with open(log_file, "w") as f:
        for entry in results:
            f.write(f"Seed: {entry['seed']}, UAR: {entry['uar']}\n")
            for metric, value in entry['per_class_metrics'].items():
                f.write(f"  {metric}: {value}\n")
            f.write(f"Confusion Matrix:\n{entry['confusion_matrix']}\n\n")

if __name__ == "__main__":
    fine_tune()
