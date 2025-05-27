# Autoencoder Model Implementation
# This file will contain the code for the Autoencoder model.

import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV # For the predictive model on top of latent features
from sklearn.metrics import r2_score

# TensorFlow / Keras for Autoencoder
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

from utils import next_window
from portfolio_utils import form_portfolio, calculate_performance_metrics

# Autoencoder Definition
def create_autoencoder(input_dim, encoding_dim=32, ae_activations='relu', ae_optimizer='adam'):
    # Define Encoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation=ae_activations)(input_layer)
    encoded = Dense(64, activation=ae_activations)(encoded)
    encoder_output = Dense(encoding_dim, activation=ae_activations)(encoded) # Bottleneck
    encoder_model = Model(input_layer, encoder_output, name="encoder")

    # Define Decoder
    decoder_input = Input(shape=(encoding_dim,))
    decoded = Dense(64, activation=ae_activations)(decoder_input)
    decoded = Dense(128, activation=ae_activations)(decoded)
    decoder_output = Dense(input_dim, activation='sigmoid')(decoded) # Sigmoid for reconstruction of scaled data (0-1 if MinMaxScaler, or handle negative if StandardScaler)
                                                                # Or linear if not strictly bounded after scaling.
    decoder_model = Model(decoder_input, decoder_output, name="decoder")

    # Define Autoencoder (Encoder + Decoder)
    autoencoder_input = Input(shape=(input_dim,))
    encoded_img = encoder_model(autoencoder_input)
    reconstructed_img = decoder_model(encoded_img)
    autoencoder_model = Model(autoencoder_input, reconstructed_img, name="autoencoder")
    
    autoencoder_model.compile(optimizer=ae_optimizer, loss='mse')
    return autoencoder_model, encoder_model

def run_autoencoder_model():
    df = pd.read_csv('Data/homework_sample_big.csv')
    df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
    df = df.sort_values(['date', 'permno'])
    predictors = pd.read_csv('Data/factors_char_list.csv')['variable'].tolist()
    target = 'stock_exret'

    os.makedirs('outputs', exist_ok=True)

    all_oos_predictions = []
    all_oos_r2 = []
    
    # AE and predictive model parameters (could be tuned)
    encoding_dim = 32 # Number of latent features from AE
    ae_epochs = 50 # Max epochs for AE training
    ae_batch_size = 256
    ridge_alphas = [0.1, 1.0, 10.0] # For predictive model

    print("Starting Autoencoder model processing with expanding window...")

    for i, (train_df, val_df, test_df) in enumerate(next_window(df)):
        current_oos_year = test_df['date'].dt.year.iloc[0]
        print(f"Processing OOS Year: {current_oos_year} for Autoencoder")

        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()

        # Preprocessing: Imputation and Scaling
        medians = train_df[predictors].median()
        train_df[predictors] = train_df[predictors].fillna(medians)
        val_df[predictors] = val_df[predictors].fillna(medians)
        test_df[predictors] = test_df[predictors].fillna(medians)

        scaler = StandardScaler() # StandardScaler is common for NNs
        train_df[predictors] = scaler.fit_transform(train_df[predictors])
        val_df[predictors] = scaler.transform(val_df[predictors])
        test_df[predictors] = scaler.transform(test_df[predictors])
        
        train_df[target] = train_df[target].fillna(0)
        val_df[target] = val_df[target].fillna(0)
        test_df[target] = test_df[target].fillna(0)
        
        input_dim = len(predictors)

        # Create and Train Autoencoder for the current window
        autoencoder, encoder = create_autoencoder(input_dim, encoding_dim)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        # Check for sufficient data
        if train_df[predictors].shape[0] < ae_batch_size or val_df[predictors].shape[0] == 0:
            print(f"Skipping AE training for OOS year {current_oos_year} due to insufficient train/val data.")
            # Set predictions to NaN or 0 and continue
            test_df.loc[:, 'autoencoder_pred'] = np.nan 
        else:
            autoencoder.fit(train_df[predictors], train_df[predictors], 
                            epochs=ae_epochs, batch_size=ae_batch_size, 
                            validation_data=(val_df[predictors], val_df[predictors]), 
                            callbacks=[early_stopping], verbose=0)

            # Extract latent features using the trained encoder
            train_latent_features = encoder.predict(train_df[predictors], verbose=0)
            val_latent_features = encoder.predict(val_df[predictors], verbose=0)
            test_latent_features = encoder.predict(test_df[predictors], verbose=0)

            # Train a predictive model (Ridge) on latent features
            # Using val_latent_features for RidgeCV internal validation
            combined_train_val_latent_features = np.vstack((train_latent_features, val_latent_features))
            combined_train_val_target = pd.concat([train_df[target], val_df[target]])
            
            if combined_train_val_latent_features.shape[0] < 2:
                 print(f"Skipping Ridge training for OOS year {current_oos_year} due to insufficient combined latent features.")
                 test_df.loc[:, 'autoencoder_pred'] = np.nan
            else:
                predictive_model = RidgeCV(alphas=ridge_alphas, store_cv_values=False)
                predictive_model.fit(combined_train_val_latent_features, combined_train_val_target)
                test_df.loc[:, 'autoencoder_pred'] = predictive_model.predict(test_latent_features)

        test_df['autoencoder_pred'] = test_df['autoencoder_pred'].fillna(0)

        oos_r2_year = r2_score(test_df[target], test_df['autoencoder_pred'])
        all_oos_r2.append(oos_r2_year)
        print(f"Year {current_oos_year} - Autoencoder OOS R2: {oos_r2_year:.4f}")

        all_oos_predictions.append(test_df[['date', 'permno', 'autoencoder_pred', target]])
        
        # Optionally save last AE model or predictive model
        if i == len(list(next_window(df))) - 1: 
            if 'encoder' in locals(): # Check if encoder was created
                encoder.save('outputs/autoencoder_encoder_last_window.keras')
                print("Saved final Autoencoder (encoder part) for last window.")
            if 'predictive_model' in locals():
                 joblib.dump(predictive_model, 'outputs/autoencoder_predictor_last_window.pkl')
                 print("Saved final Autoencoder (predictor part) for last window.")

    print("Finished processing all windows for Autoencoder model.")

    if not all_oos_predictions:
        print("No Autoencoder predictions were generated. Exiting.")
        return

    all_predictions_df = pd.concat(all_oos_predictions)
    overall_oos_r2 = np.mean(all_oos_r2) if all_oos_r2 else np.nan
    print(f'Overall Average OOS R² score for Autoencoder: {overall_oos_r2:.4f}')

    metrics_df = pd.DataFrame({'model': ['autoencoder'], 'oos_r2': [overall_oos_r2]})
    metrics_file = 'outputs/metrics.csv'
    if os.path.exists(metrics_file):
        existing_metrics = pd.read_csv(metrics_file)
        existing_metrics = existing_metrics[existing_metrics['model'] != 'autoencoder']
        metrics_df = pd.concat([existing_metrics, metrics_df])
    metrics_df.to_csv(metrics_file, index=False)
    print(f'Overall Autoencoder metrics saved to {metrics_file}')

    all_predictions_df = all_predictions_df.rename(columns={target: 'stock_exret'})
    ae_overall_port_ret = form_portfolio(all_predictions_df, 'autoencoder_pred', n_stocks=50)
    ae_overall_port_ret.columns = ['autoencoder_port_ret']
    ae_overall_port_ret.to_csv('outputs/autoencoder_overall_port_ret.csv')
    print('Overall Autoencoder monthly portfolio returns saved to outputs/autoencoder_overall_port_ret.csv')

    spy_returns = pd.read_excel('Data/SPY returns.xlsx', index_col=0)
    spy_returns.index = pd.to_datetime(spy_returns.index)
    perf_summary_df = calculate_performance_metrics(ae_overall_port_ret, spy_returns, 'autoencoder')
    summary_file = 'outputs/perf_summary.csv'
    if os.path.exists(summary_file):
        existing_summary = pd.read_csv(summary_file)
        existing_summary = existing_summary[existing_summary['model'] != 'autoencoder']
        perf_summary_df = pd.concat([existing_summary, perf_summary_df])
    perf_summary_df.to_csv(summary_file, index=False)
    print(f'Overall Autoencoder performance summary saved to {summary_file}')

if __name__ == "__main__":
    # Set random seeds for reproducibility with TensorFlow
    tf.random.set_seed(42)
    np.random.seed(42)
    run_autoencoder_model() 