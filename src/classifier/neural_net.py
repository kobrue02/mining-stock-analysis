from enum import Enum
from typing import List, Optional, Tuple, Union
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from src.classifier import BaseClassifier


class CNNConfig(Enum):
    """Configuration settings for the CNN classifier."""

    SMALL = {
        "hidden_dims": [128, 64],
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 20,
        "validation_split": 0.1,
        "device": None,
    }
    MEDIUM = {
        "hidden_dims": [256, 128, 64],
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 30,
        "validation_split": 0.1,
        "device": None,
    }
    LARGE = {
        "hidden_dims": [512, 256, 128, 64],
        "learning_rate": 0.001,
        "batch_size": 128,
        "epochs": 40,
        "validation_split": 0.1,
        "device": None,
    }
    CUSTOM = {
        "hidden_dims": [256, 128],
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 20,
        "validation_split": 0.1,
        "device": None,
    }

    @classmethod
    def load_conf(cls, config_name: str) -> "CNNConfig":
        match config_name:
            case "small":
                return cls.SMALL
            case "medium":
                return cls.MEDIUM
            case "large":
                return cls.LARGE
            case "custom":
                return cls.CUSTOM
            case _:
                raise ValueError(f"Invalid configuration name: {config_name}")

    @classmethod
    def from_dict(cls, config: dict) -> "CNNConfig":
        """Load configuration settings from a dictionary."""
        for key, value in config.items():
            if key == "DEVICE":
                setattr(cls, key, value)
            else:
                setattr(cls, key, eval(value))
        return cls

    @classmethod
    def as_dict(cls) -> dict:
        """Return configuration settings as a dictionary."""
        return {
            key: value for key, value in cls.__members__.items() if key != "from_dict"
        }


class CNNLayers(Enum):
    SMALL = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )

    MEDIUM = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )

    LARGE = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(64),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(128),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(256),
    )

    @property
    def layers(self) -> nn.Sequential:
        return self.value


class CNNModel(nn.Module):
    """
    Convolutional Neural Network model for classification.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = [128, 64],
        layers: nn.Sequential | CNNLayers = None,
    ):
        super(CNNModel, self).__init__()

        # Reshape layer dimensions
        self.input_dim = input_dim

        # Assuming embeddings are 1D, reshape to 2D for CNN
        # We'll reshape the input as a square matrix if possible, or closest rectangular shape
        self.side_length = int(np.ceil(np.sqrt(input_dim)))
        self.padded_length = self.side_length**2

        # Convolutional layers
        if layers:
            self.conv_layers = layers.layers or layers
        else:
            self.conv_layers = CNNLayers.SMALL.layers

        # Calculate dimensions after convolution
        conv_output_dim = 32 * (self.side_length // 4) * (self.side_length // 4)

        # Fully connected layers
        fc_layers = []
        prev_dim = conv_output_dim

        for dim in hidden_dims:
            fc_layers.append(nn.Linear(prev_dim, dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(0.3))
            prev_dim = dim

        fc_layers.append(nn.Linear(prev_dim, num_classes))

        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        # Reshape input to 2D for CNN
        batch_size = x.shape[0]
        x = self._reshape_input(x)

        # Apply convolutional layers
        x = self.conv_layers(x)

        # Flatten for fully connected layers
        x = x.view(batch_size, -1)

        # Apply fully connected layers
        x = self.fc_layers(x)
        return x

    def _reshape_input(self, x):
        """Reshape the input embeddings to 2D format for CNN processing."""
        batch_size = x.shape[0]

        # Pad if necessary
        if x.shape[1] < self.padded_length:
            padding = torch.zeros(
                batch_size, self.padded_length - x.shape[1], device=x.device
            )
            x = torch.cat([x, padding], dim=1)

        # Reshape to square matrix and add channel dimension
        return x.view(batch_size, 1, self.side_length, self.side_length)


class CNNClassifier(BaseClassifier):
    """
    A classifier that uses a Convolutional Neural Network for classification tasks.
    """

    def __init__(
        self,
        config: CNNConfig = None,
        hidden_dims: List[int] = [128, 64],
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 20,
        validation_split: float = 0.1,
        device: Optional[str] = None,
        layers: nn.Sequential | CNNLayers = None,
    ):
        """
        Initialize the CNN classifier.

        Args:
            config (Optional[CNNConfig]): Configuration settings for the CNN.
            hidden_dims (List[int]): Dimensions of hidden layers in the CNN.
            learning_rate (float): Learning rate for optimizer.
            batch_size (int): Batch size for training.
            epochs (int): Number of training epochs.
            validation_split (float): Fraction of data to use for validation.
            device (Optional[str]): Device to use for training ('cuda', 'mps', or 'cpu').
                                    If None, will automatically detect available device.
        """
        super(CNNClassifier, self).__init__()
        if config is not None:
            if isinstance(config, str):
                config = CNNConfig.load_conf(config)
            hidden_dims = config.value["hidden_dims"]
            learning_rate = config.value["learning_rate"]
            batch_size = config.value["batch_size"]
            epochs = config.value["epochs"]
            validation_split = config.value["validation_split"]
            device = config.value["device"]
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split

        # Determine device
        if device is None:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        else:
            self.device = device

        self.layers = layers
        self.model = None
        self.num_classes = None
        self.class_to_idx = None
        self.idx_to_class = None

        # Set up logging
        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s"
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using device: {self.device}")

    def _convert_labels_to_integers(
        self, labels: Union[List, np.ndarray, Tensor]
    ) -> np.ndarray:
        # Create mapping from class labels to indices
        unique_classes = list(set(labels))
        self.class_to_idx = {
            cls.item() if torch.is_tensor(cls) else cls: idx
            for idx, cls in enumerate(unique_classes)
        }
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        # Convert string/float labels to integer indices
        if torch.is_tensor(labels):
            if labels.dtype == torch.float32 or labels.dtype == torch.float64:
                return torch.tensor(
                    [self.class_to_idx[lbl.item()] for lbl in labels], dtype=torch.long
                )
            else:
                return torch.tensor(
                    [
                        self.class_to_idx[lbl.item() if torch.is_tensor(lbl) else lbl]
                        for lbl in labels
                    ],
                    dtype=torch.long,
                )
        else:
            return torch.tensor(
                [self.class_to_idx[lbl] for lbl in labels], dtype=torch.long
            )

    def _is_integral_tensor(self, labels) -> bool:
        return (
            labels.dtype == torch.int64
            or labels.dtype == torch.int32
            or labels.dtype == torch.int16
            or labels.dtype == torch.int8
            or labels.dtype == torch.uint8
        )

    def _prepare_data(
        self,
        embeddings: Union[List, np.ndarray, Tensor],
        labels: Union[List, np.ndarray, Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """
        Convert input data to PyTorch tensors.

        Args:
            embeddings: Input embeddings
            labels: Input labels

        Returns:
            Tuple of (embeddings_tensor, labels_tensor)
        """
        # Convert embeddings to tensor if needed
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)

        # Convert labels to integers if they're not already
        if not self._is_integral_tensor(labels):
            numeric_labels = self._convert_labels_to_integers(labels)
        else:
            numeric_labels = labels.long()
            # Create identity mapping
            unique_classes = torch.unique(labels)
            self.class_to_idx = {cls.item(): cls.item() for cls in unique_classes}
            self.idx_to_class = {idx: idx for idx in self.class_to_idx.values()}

        self.num_classes = len(self.class_to_idx)
        return embeddings, numeric_labels

    def _prepare_training(self, embeddings, labels):
        """
        Prepare data for training by converting to PyTorch tensors and splitting into train/validation sets.

        Args:
            embeddings: The embeddings to train the classifier on
            labels: The labels for the embeddings

        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Prepare data
        self.X, self.y = self._prepare_data(embeddings, labels)
        input_dim = self.X.shape[1]

        # Split into train/validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            self.X,
            self.y,
            test_size=self.validation_split,
            stratify=self.y,
            random_state=42,
        )

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        return train_loader, val_loader, input_dim

    def _run_training_loop(self, train_loader, val_loader, criterion, optimizer):
        # Training loop
        best_val_acc = 0.0
        best_model_state = None

        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Track statistics
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()

            train_loss = train_loss / train_total
            train_acc = train_correct / train_total

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).sum().item()

            val_loss = val_loss / val_total
            val_acc = val_correct / val_total

            # Print statistics
            self.logger.info(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.logger.info(f"Best validation accuracy: {best_val_acc:.4f}")

        # Final evaluation on full training set
        self.model.eval()
        with torch.no_grad():
            X, y = self.X.to(self.device), self.y.to(self.device)
            outputs = self.model(X)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y).sum().item() / y.size(0)

        self.logger.info(f"Final training accuracy: {accuracy:.4f}")

    def fit(self, embeddings, labels) -> None:
        """
        Fit the CNN to the provided embeddings and labels.

        Args:
            embeddings: The embeddings to train the classifier on
            labels: The labels for the embeddings
        """
        self.model = CNNModel(
            input_dim=embeddings.shape[1],
            num_classes=len(set(labels)),
            hidden_dims=self.hidden_dims,
            layers=self.layers,
        ).to(self.device)
        # Prepare data
        train_loader, val_loader, input_dim = self._prepare_training(embeddings, labels)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Train the model
        self._run_training_loop(train_loader, val_loader, criterion, optimizer)

        return self

    def predict(self, embeddings) -> np.ndarray:
        """
        Predict class labels for the given embeddings.

        Args:
            embeddings: The embeddings to classify

        Returns:
            Predicted class labels
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")

        # Convert embeddings to tensor if needed
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)

        # Set model to evaluation mode
        self.model.eval()

        # Make predictions
        with torch.no_grad():
            embeddings = embeddings.to(self.device)
            outputs = self.model(embeddings)
            _, predicted_indices = torch.max(outputs.data, 1)

            # Convert indices back to original class labels
            predicted_labels = np.array(
                [self.idx_to_class[idx.item()] for idx in predicted_indices]
            )

        return predicted_labels

    def predict_proba(self, embeddings: Union[List, np.ndarray, Tensor]) -> np.ndarray:
        """
        Predict class probabilities for the given embeddings.

        Args:
            embeddings: The embeddings to classify

        Returns:
            Predicted class probabilities
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")

        # Convert embeddings to tensor if needed
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)

        # Set model to evaluation mode
        self.model.eval()

        # Make predictions
        with torch.no_grad():
            embeddings = embeddings.to(self.device)
            outputs = self.model(embeddings)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

        return probabilities

    def evaluate(
        self,
        embeddings: Union[List, np.ndarray, Tensor],
        labels: Union[List, np.ndarray, Tensor],
    ) -> dict:
        """
        Evaluate the model on the given embeddings and labels.

        Args:
            embeddings: The embeddings to evaluate on
            labels: The true labels

        Returns:
            Dictionary with evaluation metrics
        """
        # Prepare data
        X, y_true = self._prepare_data(embeddings, labels)
        y_true_np = y_true.cpu().numpy()

        # Get predictions
        y_pred = self.predict(X)

        # Map predictions to indices if necessary
        if not np.issubdtype(y_pred.dtype, np.integer):
            # Ensure we have mapping for all classes
            for label in np.unique(y_pred):
                if label not in self.class_to_idx:
                    # This shouldn't happen if predict() is using the correct mapping
                    raise ValueError(f"Unknown class label: {label}")

            y_pred_indices = np.array([self.class_to_idx[lbl] for lbl in y_pred])
        else:
            y_pred_indices = y_pred

        # Calculate metrics
        accuracy = np.mean(y_pred_indices == y_true_np)
        report = classification_report(
            y_true_np,
            y_pred_indices,
            target_names=[str(self.idx_to_class[i]) for i in range(self.num_classes)],
            output_dict=True,
        )

        # Return metrics
        metrics = {"accuracy": accuracy, "classification_report": report}

        return metrics

    def save(self, filepath: str) -> None:
        """Save model to a file."""
        if self.model is None:
            raise RuntimeError("No model to save. Train the model first.")

        # Prepare state dict with all necessary information
        state_dict = {
            "model_state_dict": self.model.state_dict(),
            "class_to_idx": self.class_to_idx,
            "idx_to_class": self.idx_to_class,
            "num_classes": self.num_classes,
            "hidden_dims": self.hidden_dims,
            "input_dim": self.model.input_dim,
        }

        torch.save(state_dict, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load model from a file."""
        state_dict = torch.load(filepath, map_location=self.device)

        # Extract model parameters
        self.class_to_idx = state_dict["class_to_idx"]
        self.idx_to_class = state_dict["idx_to_class"]
        self.num_classes = state_dict["num_classes"]
        self.hidden_dims = state_dict["hidden_dims"]
        input_dim = state_dict["input_dim"]

        # Initialize model
        self.model = CNNModel(
            input_dim=input_dim,
            num_classes=self.num_classes,
            hidden_dims=self.hidden_dims,
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(state_dict["model_state_dict"])
        self.logger.info(f"Model loaded from {filepath}")
