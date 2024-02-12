from datasets import Dataset, DatasetDict
import evaluate
from google.colab import drive
import numpy as np
import os
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer
import torch
from torch.nn import functional as F
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, pipeline,
                          TrainingArguments)


def _validate_dataframes(train_df, test_df):
    """
    Validates that the input DataFrames contain the required columns.
    """
    required_columns = ["text", "label"]
    missing_columns_train = [col for col in required_columns if col not in train_df.columns]
    missing_columns_test = [col for col in required_columns if col not in test_df.columns]

    if missing_columns_train or missing_columns_test:
        missing_info = ""
        if missing_columns_train:
            missing_info += f"Training DataFrame is missing columns: {missing_columns_train}. "
        if missing_columns_test:
            missing_info += f"Testing DataFrame is missing columns: {missing_columns_test}."
        raise ValueError(missing_info)


def _prepare_datasets(train_df, test_df):
    """
    Converts train and test dataframes into a DatasetDict.
    """
    _validate_dataframes(train_df, test_df)
    train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
    test_dataset = Dataset.from_pandas(test_df[["text", "label"]])
    return DatasetDict({'train': train_dataset, 'test': test_dataset})


def _initialize_model_and_tokenizer(model_name, checkpoint_path=None):
    """
    Initializes the tokenizer and model either from save or huggingface.
    """
    load_from_weights = checkpoint_path and os.path.isfile(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if load_from_weights:
        model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                   num_labels=2)
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path if checkpoint_path else model_name,
            num_labels=2
        )

    return tokenizer, model


def _tokenize_datasets(dataset_dict, tokenizer):
    """
    Tokenizes the text in a DatasetDict for model training.
    """
    return dataset_dict.map(
        lambda examples: tokenizer(examples["text"], truncation=True),
        batched=True
    )


def _compute_metrics(eval_pred):
    """
    Computes evaluation metrics for model training.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy_metric = evaluate.load("accuracy")
    return accuracy_metric.compute(predictions=predictions, references=labels)


def _train_model(tokenized_datasets, model, tokenizer, training_args):
    """
    Trains the model using the Trainer.
    """
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_compute_metrics,
    )
    trainer.train()
    return trainer


def train_classifier(df, test_size=0.1, model_name="distilbert-base-uncased",
                     save_model_path=None, training_args=None):
    """
    Main function to train a text classifier.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the complete dataset.
    test_size : float, optional
        The proportion of the dataset to include in the test split.
    model_name : str, optional
        The name of the pre-trained model to use.
    save_model_path : str, optional
        The file path where the model should be saved.
    training_args : transformers.TrainingArguments, optional
        Custom training arguments. If None, default arguments are used.

    Returns
    -------
    transformers.pipeline
        A pipeline with the trained model and tokenizer for sentiment analysis.

    Notes
    -----
    This function prepares the data, initializes the model and tokenizer,
    tokenizes the data, and trains the model. It returns a pipeline
    for sentiment analysis with the trained model.
    """
    train_df, test_df = train_test_split(df, test_size=test_size)
    datasets = _prepare_datasets(train_df, test_df)

    tokenizer, model = _initialize_model_and_tokenizer(model_name)

    tokenized_datasets = _tokenize_datasets(datasets, tokenizer)

    if training_args is None:
        training_args = TrainingArguments(
            output_dir=save_model_path if save_model_path else "./models",
            num_train_epochs=1,
            per_device_train_batch_size=16,
            evaluation_strategy="epoch",
        )

    trainer = _train_model(tokenized_datasets, model, tokenizer, training_args)

    classifier = pipeline("sentiment-analysis",
                          model=model,
                          tokenizer=tokenizer,
                          top_k=1,
                          device=0)
    return classifier


def save_model_weights(classifier, model_path):
    """
    Saves the weights of a classifier model to a specified path.

    Parameters
    ----------
    classifier :
        An instance of the classifier containing the model whose weights are to be saved.
    model_path : str
        The file path where the model weights will be saved.

    Returns
    -------
    None
        Saves the model's state dictionary to the specified path.
    """
    torch.save(classifier.model.state_dict(), model_path)


def load_classifier(model_path, model_name="distilbert-base-uncased"):
    """
    Loads a classifier from a saved model checkpoint.

    Parameters
    ----------
    model_path : str
        The path to the saved model checkpoint.
    model_name : str, optional
        The name of the pre-trained model to use for the tokenizer.

    Returns
    -------
    transformers.pipeline
        A pipeline with the trained model and tokenizer for sentiment analysis.

    Notes
    -----
    This function loads a model from a checkpoint and creates a pipeline
    for sentiment analysis with the loaded model.
    """
    # Load the tokenizer and model from the checkpoint
    tokenizer, model = _initialize_model_and_tokenizer(model_name, checkpoint_path=model_path)

    # Create a sentiment analysis pipeline
    classifier = pipeline("sentiment-analysis",
                          model=model,
                          tokenizer=tokenizer,
                          top_k=1,
                          device=0)
    return classifier


def calculate_loss_and_prediction(df, classifier, col_suffix=None):
    """
    Calculates the cross-entropy loss and predictions for a given dataframe using a sentiment analysis classifier.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with a 'text' column for prediction and a 'label' column for loss calculation.
    classifier
        An instance of a transformers pipeline for sentiment analysis.

    Returns
    -------
    pd.DataFrame
        The DataFrame with 'prob' (probability) and 'loss' (cross-entropy loss) columns added.

    """

    # Check for necessary columns in the DataFrame
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("The DataFrame must contain 'text' and 'label' columns")

    # Predict using the classifier
    preds = classifier.predict(df.text.tolist())

    # Calculate probabilities
    probs = np.array(
        [pred[0]["score"] if pred[0]["label"] == "LABEL_1" else 1 - pred[0]["score"]
         for pred in preds]
    )

    # Calculate cross-entropy loss
    def calc_cross_entropy_loss(probs, labels):
        return - (labels * np.log(probs) + (1 - labels) * np.log(1 - probs))

    ce_losses = calc_cross_entropy_loss(probs, df.label)

    # Add probabilities and losses to the DataFrame
    df = df.copy()
    if col_suffix:
        df[f"prob_{col_suffix}"] = probs
        df[f"loss_{col_suffix}"] = ce_losses
    else:
        df[f"prob"] = probs
        df[f"loss"] = ce_losses

    return df


def calculate_classification_metrics(data, prob_col="prob"):
    """
    Calculates classification metrics for binary classification tasks.

    Parameters
    ----------
    data : pd.DataFrame or dict
        If a DataFrame, it should contain 'label' (true labels) and 'prob' 
        (predicted probabilities) columns.
        If a dictionary, keys represent dataset names, and values are DataFrames 
        with 'label' and 'prob' columns.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing classification metrics for each dataset or a 
        single dataset.
    """
    def calculate_metrics_for_df(df, prob_col):
        df = df.copy()
        # Ensure the 'prob' column values are between 0 and 1
        probs = df[[prob_col]].clip(0, 1)

        # Binarize the probabilities to get binary predictions
        binarizer = Binarizer(threshold=0.5)
        y_pred = binarizer.fit_transform(probs).flatten()

        # Extract true labels
        y_true = df['label'].values

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, probs)

        return [accuracy, f1, precision, recall, roc_auc]

    metrics_list = []
    indices = []

    if isinstance(data, dict):
        # Calculate metrics for each DataFrame in the dictionary
        for key, df in data.items():
            metrics = calculate_metrics_for_df(df, prob_col)
            metrics_list.append(metrics)
            indices.append(key)
    else:
        # Single DataFrame, calculate metrics
        metrics = calculate_metrics_for_df(data, prob_col)
        metrics_list.append(metrics)
        indices = [0]  # Default index for a single DataFrame

    metrics_df = pd.DataFrame(metrics_list, columns=['accuracy', 'F1', 'precision', 'recall', 'roc-auc'], index=indices)

    return metrics_df
