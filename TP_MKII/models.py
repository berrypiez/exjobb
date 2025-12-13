import numpy as np
from typing import List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Input, TimeDistributed,
    Masking, Conv1D, GlobalAveragePooling1D,
    InputLayer, Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping



def ade_fde(preds, targets):
    """Compute Average Displacement Error (ADE) and Final Displacement Error (FDE)."""
    ade_list, fde_list = [], []
    for p, t in zip(preds, targets):
        min_len = min(len(p), len(t))
        p, t = p[:min_len], t[:min_len]
        dist = np.linalg.norm(p[:, :2] - t[:, :2], axis=1)  # only cx, cy
        ade_list.append(np.mean(dist))
        fde_list.append(dist[-1])
    return np.mean(ade_list), np.mean(fde_list)

# Extract only centroid from predicted and true sequences for ADE/FDE
def compute_centroid_ade_fde(preds, targets):
    """
    preds, targets: arrays/lists of shape (N, T, 2) (unscaled coordinates).
    Returns mean ADE and mean FDE.
    """
    preds_arr = np.array(preds)
    targets_arr = np.array(targets)
    N = len(preds_arr)
    ade_list, fde_list = [], []
    for i in range(N):
        p = preds_arr[i]
        t = targets_arr[i]
        min_len = min(len(p), len(t))
        if min_len == 0:
            continue
        dist = np.linalg.norm(p[:min_len] - t[:min_len], axis=1)
        ade_list.append(np.mean(dist))
        fde_list.append(dist[-1])
    if len(ade_list) == 0:
        return 0.0, 0.0
    return float(np.mean(ade_list)), float(np.mean(fde_list))

class Models:
    def __init__(self, feature_size: Optional[int] = None):
        self.feature_size = feature_size
        self.reset_models()
        
    def reset_models(self):
        self.lstm_model = None
        self.cnn_model = None
        self.lin_model = None
        self.knn_model = None
        
        self.lin_Tf = None
        self.lin_F = None
        self.knn_Tf = None
        self.knn_F = None
        print("All models reset.")

    # ----------------------------
    # LSTM Sequence-to-Sequence Model
    # ----------------------------
    def train_lstm(self, X_train, y_train, X_val=None, y_val=None,
                   latent_dim=128, epochs=40, batch_size=1, learning_rate=0.001, verbose=1):
        
        N, T_x, F = X_train.shape
        _, T_y, _ = y_train.shape

        model = Sequential([
            InputLayer(input_shape=(T_x, F)),
            LSTM(latent_dim),
            Dense(T_y * 2)
        ])
        model.compile(optimizer="adam", loss="mse")
        es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

        model.fit(X_train, y_train.reshape(N, -1),
                  epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=verbose)
        
        self.lstm_model = model
        return model, None

    def predict_lstm(self, X, y_len):
        if self.lstm_model is None:
            raise RuntimeError("LSTM model has not been trained yet.")
        preds_flat = self.lstm_model.predict(X)
        N = preds_flat.shape[0]
        return preds_flat.reshape(N, y_len, 2)

    # ----------------------------
    # Temporal CNN Model
    # ----------------------------
    def train_cnn(self, X_train, y_train,
                  filters=64, kernel_size=3, epochs=40, batch_size=1, learning_rate=0.0001, verbose=1):
        
        N, T_x, F = X_train.shape
        _, T_y, _ = y_train.shape
        self.feature_size = F

        model = Sequential([
            InputLayer(shape=(T_x, F)),
            Conv1D(filters=filters, kernel_size=kernel_size, activation="relu", padding="same"),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(T_y * 2)
        ])
        model.compile(optimizer="adam", loss="mse")
        es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)

        model.fit(X_train, y_train.reshape(N, -1),
                  epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=verbose)
        self.cnn_model = model
        return model, None
    
    def predict_cnn(self, X, y_len):
        if self.cnn_model is None:
            raise RuntimeError("CNN model has not been trained yet.")
        preds_flat = self.cnn_model.predict(X)
        N = preds_flat.shape[0]
        return preds_flat.reshape(N, y_len, 2)

    
    # -------------------
    # KNN for Trajectory Prediction
    # -------------------
    def train_knn(self, X_train, y_train, n_neighbors=5):
        N, T_p, Fx = X_train.shape
        _, T_f, _ = y_train.shape
        self.knn_Tf = T_f
        self.knn_F = 2

        X_flat = X_train.reshape(N, T_p * Fx)
        y_flat = y_train.reshape(N, T_f * 2)

        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
        model.fit(X_flat, y_flat)
        self.knn_model = model
        return model
    
    def predict_knn(self, X):
        if self.knn_model is None:
            raise RuntimeError("KNN model has not been trained yet.")
        N, T_p, Fx = X.shape
        X_flat = X.reshape(N, T_p * Fx)
        preds_flat = self.knn_model.predict(X_flat)
        return preds_flat.reshape(N, self.knn_Tf, 2)
    
    # -------------------
    # Linear Regression for Trajectories
    # -------------------
    def train_linear(self, X_train, y_train):
        N, T_p, Fx = X_train.shape
        _, T_f, _ = y_train.shape
        self.lin_Tf = T_f
        self.lin_F = 2

        X_flat = X_train.reshape(N, T_p * Fx)
        y_flat = y_train.reshape(N, T_f * 2)

        model = LinearRegression()
        model.fit(X_flat, y_flat)
        self.lin_model = model
        return model
    
    def predict_linear(self, X):
        if self.lin_model is None:
            raise RuntimeError("Linear model not trained.")
        N, T_p, Fx = X.shape
        X_flat = X.reshape(N, T_p * Fx)
        preds_flat = self.lin_model.predict(X_flat)
        return preds_flat.reshape(N, self.lin_Tf, 2)


'''    def train_lstm(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        latent_dim=128,
        epochs=40,
        batch_size=8,
        learning_rate=0.001,
        verbose=1
    ):
        N, T_x, F = X_train.shape
        _, T_y, _ = y_train.shape
        self.feature_size = F

        # Encoder
        encoder_inputs = Input(shape=(T_x, F), name="encoder_inputs")
        mask = Masking(mask_value=0.0)(encoder_inputs)
        _, state_h, state_c = LSTM(latent_dim, return_state=True, name="encoder_lstm")(mask)
        encoder_states = [state_h, state_c]

        # Decoder
        decoder_inputs = Input(shape=(T_y, F), name="decoder_inputs")
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = TimeDistributed(Dense(F), name="decoder_dense")
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")

        es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

        val_data = None
        if X_val is not None and y_val is not None:
            val_data = ([X_val, np.zeros_like(y_val)], y_val)

        history = model.fit(
            [X_train, np.zeros_like(y_train)],
            y_train,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es],
            verbose=verbose
        )

        self.lstm_model = model
        self.lstm_latent = latent_dim
        return model, history
        
    def predict_lstm(self, X, y_len):
        model: Model = getattr(self, "lstm_model", None)
        if model is None:
            raise RuntimeError("LSTM model has not been trained yet.")
        N, _, F = X.shape
        decoder_input = np.zeros((N, y_len, F), dtype=X.dtype)
        preds = model.predict([X, decoder_input])
        return preds
        
        
        def train_cnn(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        filters: int = 64,
        kernel_size: int = 3,
        epochs=40,
        batch_size=8,
        learning_rate=0.0001,
        verbose=1
    ):
        N, T_x, F = X_train.shape
        _, T_y, _ = y_train.shape
        self.feature_size = F

        model = Sequential()
        model.add(Input(shape=(T_x, F)))
        model.add(Masking(mask_value=0.0))
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation="relu", padding="same"))
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation="relu", padding="same"))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(T_y * F))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")

        es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

        val_data = None
        if X_val is not None and y_val is not None:
            val_data = (X_val, y_val.reshape(X_val.shape[0], -1))

        history = model.fit(
            X_train,
            y_train.reshape(N, -1),
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es],
            verbose=verbose
        )

        self.cnn_model = model
        self.cnn_Ty = T_y
        self.cnn_F = F
        return model, history

    def predict_cnn(self, X):
        model = getattr(self, "cnn_model", None)
        if model is None:
            raise RuntimeError("CNN model has not been trained yet.")
        preds_flat = model.predict(X)
        N = preds_flat.shape[0]
        preds = preds_flat.reshape(N, self.cnn_Ty, self.cnn_F)
        return preds
        
        def train_knn(self, X_train, y_train, n_neighbors=5):
        # Flatten sequences
        N, T_p, F = X_train.shape
        _, T_f, _ = y_train.shape
        self.knn_Tf = T_f
        self.knn_F = F

        X_flat = X_train.reshape(N, T_p * F)
        y_flat = y_train.reshape(N, T_f * F)

        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
        model.fit(X_flat, y_flat)
        self.knn_model = model
        return model

    def predict_knn(self, X):
        if not hasattr(self, "knn_model"):
            raise RuntimeError("KNN model not trained yet.")
        N, T_p, F = X.shape
        X_flat = X.reshape(N, T_p * F)
        preds_flat = self.knn_model.predict(X_flat)
        preds = preds_flat.reshape(N, self.knn_Tf, self.knn_F)
        return preds
        
    def train_linear(self, X_train, y_train):
        N, T_p, F = X_train.shape
        _, T_f, _ = y_train.shape
        self.lin_Tf = T_f
        self.lin_F = F

        X_flat = X_train.reshape(N, T_p * F)
        y_flat = y_train.reshape(N, T_f * F)

        model = LinearRegression()
        model.fit(X_flat, y_flat)
        self.lin_model = model
        return model

    def predict_linear(self, X):
        if not hasattr(self, "lin_model"):
            raise RuntimeError("Linear model not trained yet.")
        N, T_p, F = X.shape
        X_flat = X.reshape(N, T_p * F)
        preds_flat = self.lin_model.predict(X_flat)
        preds = preds_flat.reshape(N, self.lin_Tf, self.lin_F)
        return preds
'''