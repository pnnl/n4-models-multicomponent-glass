import tensorflow as tf
from tensorflow import keras    
from sklearn.preprocessing import StandardScaler
import os
import random
import numpy as np
import joblib
import json

# Define loss function OUTSIDE the class
def create_heteroscedastic_loss(output_dim):
    """Factory function to create heteroscedastic loss"""
    def loss(y_true, y_pred):
        mean = y_pred[:, :output_dim]
        log_var = y_pred[:, output_dim:]
        return 0.5 * tf.reduce_mean(log_var + tf.exp(-log_var) * tf.square(y_true - mean))
    return loss

class HeteroscedasticDNN:
    def __init__(self, input_dim, output_dim, hidden_layers=[32, 200, 200, 256], seed=100):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.model = None
        self.seed = seed
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self._model_built = False
        self.set_seed(self.seed)
    
    @staticmethod
    def set_seed(seed_value=42):
        """Set seed for reproducibility"""
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)
        
    def build_model(self):
        """Build the heteroscedastic model"""
        inputs = keras.Input(shape=(self.input_dim,))
        x = inputs
        
        for units in self.hidden_layers:
            x = keras.layers.Dense(units, activation='sigmoid')(x)
        
        outputs = keras.layers.Dense(self.output_dim * 2)(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self._model_built = True
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=5000, batch_size=None):
        """Train the model"""
        X_train_norm = self.scaler_X.fit_transform(X_train)
        y_train_norm = self.scaler_y.fit_transform(y_train)
        
        validation_data = None
        if X_val is not None:
            X_val_norm = self.scaler_X.transform(X_val)
            y_val_norm = self.scaler_y.transform(y_val)
            validation_data = (X_val_norm, y_val_norm)
            
        if not self._model_built:
            self.build_model()
            
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=create_heteroscedastic_loss(self.output_dim)
        )
        
        callbacks = [
            keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=50, factor=0.5)
        ]
        
        history = self.model.fit(
            X_train_norm, y_train_norm,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
        
    def predict_with_uncertainty(self, X):
        """Make predictions with uncertainty estimates"""
        X_norm = self.scaler_X.transform(X)
        pred_norm = self.model.predict(X_norm, verbose=0)
        
        mean_norm = pred_norm[:, :self.output_dim]
        log_var_norm = pred_norm[:, self.output_dim:]
        
        mean = self.scaler_y.inverse_transform(mean_norm)
        std = np.exp(0.5 * log_var_norm) * self.scaler_y.scale_
        
        return mean, std
    
    def save(self, path):
        """Save the model to a directory - ONLY WEIGHTS, NOT COMPILED MODEL"""
        os.makedirs(path, exist_ok=True)
        
        if self.model is not None:
            # IMPORTANT: Save only weights, NOT the full model
            weights_path = os.path.join(path, "model.weights.h5")
            self.model.save_weights(weights_path)
            
            # Save architecture separately as JSON
            config_path = os.path.join(path, "model_architecture.json")
            with open(config_path, 'w') as f:
                # Get architecture config WITHOUT compilation info
                model_config = self.model.get_config()
                json.dump(model_config, f)
        
        # Save scalers
        joblib.dump(self.scaler_X, os.path.join(path, "scaler_X.pkl"))
        joblib.dump(self.scaler_y, os.path.join(path, "scaler_y.pkl"))
        
        # Save hyperparameters
        params = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_layers': self.hidden_layers,
            'seed': self.seed,
            'model_built': self._model_built
        }
        joblib.dump(params, os.path.join(path, "parameters.pkl"))
        
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load the model from a directory"""
        # Load hyperparameters

        params = joblib.load(os.path.join(path, "parameters.pkl"))
        
        # Create new instance
        instance = cls(
            input_dim=params['input_dim'],
            output_dim=params['output_dim'],
            hidden_layers=params['hidden_layers'],
            seed=params['seed']
        )
        instance._model_built = params['model_built']
        
        # Load scalers
        instance.scaler_X = joblib.load(os.path.join(path, "scaler_X.pkl"))
        instance.scaler_y = joblib.load(os.path.join(path, "scaler_y.pkl"))
        
        # Load model architecture and weights
        weights_path = os.path.join(path, "model.weights.h5")
        config_path = os.path.join(path, "model_architecture.json")
        # Check if files exist
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Reconstruct model from architecture
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        
        # Build model from config
        instance.model = keras.Model.from_config(model_config)
        
        # Load weights BEFORE compiling
        instance.model.load_weights(weights_path)
        
        # NOW compile with the loss function
        instance.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=create_heteroscedastic_loss(instance.output_dim)
        )
        
        print(f"Model is None: {instance.model is None}")  # Debug check
        
        return instance