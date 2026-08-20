import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dialysisml.config import LSTMConfig


class NumDaysLSTM(nn.Module):
    def __init__(self, config: LSTMConfig, input_size, output_size=1):
        super(NumDaysLSTM, self).__init__()

        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers

        # main LSTM layer
        self.lstm = nn.LSTM(
            input_size,
            config.hidden_size,
            config.num_layers,
            dropout=config.dropout_rate if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout_rate)

        # final linear layer to squeeze dimensions
        self.fc = nn.Linear(config.hidden_size, output_size)

    def forward(self, x):
        """
        Defines how inputs pass through the network.
        Input x has shape (batch_size, n_steps, n_features)
        """
        # Passiamo i dati nella LSTM.
        # out contiene gli output di TUTTI i 15 step temporali.
        # ignoriamo (hn, cn) che sono la memoria interna finale a cui non ci serve accedere direttamente.
        out, (hn, cn) = self.lstm(x)

        # out ha forma: (batch_size, 15, hidden_size)

        # A noi interessa solo cosa ha capito la rete alla FINE della finestra di 15 giorni.
        # Quindi estraiamo l'ultimo step temporale (indice -1 sulla dimensione 1)
        last_step = out[:, -1, :]

        # last_step ha forma: (batch_size, hidden_size)

        # Passiamo l'output finale al layer lineare per ottenere la predizione dei giorni
        prediction = self.fc(self.dropout(last_step))

        # prediction ha forma: (batch_size, 1)
        return prediction


class RiskLSTM(nn.Module):
    def __init__(self, config: LSTMConfig, input_size, output_size=4):
        super(RiskLSTM, self).__init__()

        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers

        self.lstm = nn.LSTM(
            input_size,
            config.hidden_size,
            config.num_layers,
            dropout=config.dropout_rate if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout_rate)

        # final linear layer to squeeze dimensions (output_size = 4)
        self.fc = nn.Linear(config.hidden_size, output_size)

    def forward(self, x):
        """
        Input x has shape (batch_size, n_steps, n_features)
        """
        out, (hn, cn) = self.lstm(x)  # shape (batch_size, sequence_length, hidden_size)
        last_step = out[:, -1, :]

        # Le predizioni qui sono "logits" (numeri crudi), la conversione in probabilità
        # la farà direttamente la BCEWithLogitsLoss durante l'addestramento.
        prediction = self.fc(self.dropout(last_step))

        # prediction ha forma: (batch_size, 4)
        return prediction

    def fit(
        self,
        epochs,
        loss_function,
        optimizer,
        train_loader,
        test_loader,
        arch_name: str,
        window_size: int,
        verbose: bool = True,
    ) -> pd.DataFrame:

        records = []

        for epoch in range(1, epochs + 1):
            # start training
            self.train()
            train_losses = []

            # get batches from the training loader
            for X_batch, y_batch_log in train_loader:
                optimizer.zero_grad()  # reset gradients
                predictions_log = self(X_batch)  # make predictions
                loss = loss_function(predictions_log, y_batch_log)  # compute loss
                loss.backward()  # compute gradients
                optimizer.step()  # update weights
                train_losses.append(loss.item())

            mean_train_loss = np.mean(train_losses)

            # start validation
            self.eval()
            test_losses = []
            all_preds_log = []
            all_targets_log = []

            with torch.no_grad():
                # get batches from the test loader
                for X_val, y_test_log in test_loader:
                    pred_test_log = self(X_val)
                    test_loss = loss_function(pred_test_log, y_test_log)
                    test_losses.append(test_loss.item())

                    # Accumuliamo le predizioni e i target
                    all_preds_log.append(pred_test_log.cpu().numpy())
                    all_targets_log.append(y_test_log.cpu().numpy())

            mean_test_loss = np.mean(test_losses)

            # Appiattiamo gli array per il calcolo delle metriche
            preds_log = np.concatenate(all_preds_log).ravel()
            targets_log = np.concatenate(all_targets_log).ravel()

            # --- CALCOLO METRICHE ---

            # 1. Metriche nello spazio logaritmico (log TTE)
            # Epsilon evita divisioni per zero se targets_log contiene uno 0
            eps = 1e-8
            mape_log_tte = (
                np.mean(np.abs((targets_log - preds_log) / (targets_log + eps))) * 100
            )
            mae_log_tte = np.mean(np.abs(targets_log - preds_log))

            # 2. Metriche nello spazio reale (TTE in giorni)
            # Usiamo np.expm1 se in preprocessing hai usato log1p, altrimenti np.exp
            preds_tte = np.expm1(preds_log)
            targets_tte = np.expm1(targets_log)

            # Evitiamo valori negativi nelle predizioni in giorni
            preds_tte = np.clip(preds_tte, a_min=0, a_max=None)

            mape_tte = (
                np.mean(np.abs((targets_tte - preds_tte) / (targets_tte + eps))) * 100
            )
            mae_tte = np.mean(np.abs(targets_tte - preds_tte))

            # --- SALVATAGGIO RIGA ---
            records.append(
                {
                    "epoch": epoch,
                    "train_loss": mean_train_loss,
                    "val_loss": mean_test_loss,
                    "MAPE_log_TTE": mape_log_tte,
                    "MAE_log_TTE": mae_log_tte,
                    "MAPE_TTE": mape_tte,
                    "MAE_TTE": mae_tte,
                }
            )

            if verbose:
                print(
                    f"Epoch [{epoch}/{epochs}] | "
                    f"Loss Train: {mean_train_loss:.4f} | Loss Val: {mean_test_loss:.4f} | "
                    f"MAE TTE: {mae_tte:.2f}gg | MAPE TTE: {mape_tte:.2f}%"
                )

        # Creazione del DataFrame finale con MultiIndex
        df_results = pd.DataFrame(records)

        # Assegniamo i valori di indice per identificare l'esperimento
        df_results["architecture"] = arch_name
        df_results["window_size"] = window_size

        # Impostiamo il MultiIndex richiesto
        df_results.set_index(["architecture", "window_size"], inplace=True)

        return df_results
