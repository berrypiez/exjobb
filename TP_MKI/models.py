import numpy as np
from typing import List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Input, TimeDistributed,
    Masking, Conv1D, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from hmmlearn.hmm import GaussianHMM


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


class Models:
    def __init__(self, feature_size: Optional[int] = None):
        self.feature_size = feature_size
        
    def _init_models(self):
        self.lstm_model = None
        self.cnn_model = None
        self.lin_model = None
        self.knn_model = None
        
        self.lin_Tf = None
        self.lin_F = None
        self.knn_Tf = None
        self.knn_F = None
        
    def reset_models(self):
        self._init_models()
        print("All models reset.")

    # ----------------------------
    # LSTM Sequence-to-Sequence Model
    # ----------------------------
    def train_lstm(
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

    # ----------------------------
    # Temporal CNN Model
    # ----------------------------
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

    # ----------------------------
    # Hidden Markov Model (HMM)
    # ----------------------------
    def train_hmm(
        self,
        seqs: List[np.ndarray],
        n_states: int = 4,
        use_pca: int = 4,
        max_iter: int = 100
    ):
        """Train a GaussianHMM using PCA-reduced features."""
        all_frames = np.vstack(seqs)
        pca = PCA(n_components=min(use_pca, all_frames.shape[1]))
        reduced = pca.fit_transform(all_frames)
        lengths = [s.shape[0] for s in seqs]
        model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=max_iter, min_covar=1e-6)
        model.fit(reduced, lengths)
        self.hmm_model = model
        self.hmm_pca = pca
        return model

    def predict_hmm(self, seq: np.ndarray, n_steps: int = 10) -> np.ndarray:
        """Predict the next n_steps frames using a trained HMM."""
        if not hasattr(self, "hmm_model"):
            raise RuntimeError("HMM model has not been trained yet.")
        model = self.hmm_model
        pca = self.hmm_pca

        seq_red = pca.transform(seq)
        logprob, hidden_states = model.decode(seq_red)
        last_state = hidden_states[-1]

        preds = []
        for _ in range(n_steps):
            mean = model.means_[last_state]
            sample = np.random.multivariate_normal(mean, model.covars_[last_state])
            preds.append(sample)
            # move to next likely state
            last_state = np.argmax(model.transmat_[last_state])
        preds = np.array(preds)
        preds_full = pca.inverse_transform(preds)
        return preds_full
    
    # -------------------
    # KNN for Trajectory Prediction
    # -------------------
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
    
    # -------------------
    # Linear Regression for Trajectories
    # -------------------
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
