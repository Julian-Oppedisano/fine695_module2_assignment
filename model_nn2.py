import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib # For saving scaler if needed, or final model if not Keras native save
import json
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2 # For L2 regularization

from utils import next_window
from portfolio_utils import form_portfolio, calculate_performance_metrics

# Define NN2 Model
def create_nn2_model(input_dim, nn_units1=64, nn_units2=32, dropout_rate=0.2, l2_reg=0.01, optimizer='adam'):
    model = Sequential([
        Dense(nn_units1, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),
        Dense(nn_units2, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),
        Dense(1) # Output layer for regression
    ])
    model.compile(optimizer=optimizer, loss='mse')
    return model

def run_nn2_model():
    df = pd.read_csv('Data/homework_sample_big.csv')
    df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
    df = df.sort_values(['date', 'permno'])
    predictors = pd.read_csv('Data/factors_char_list.csv')['variable'].tolist()
    target = 'stock_exret'

    os.makedirs('outputs', exist_ok=True)

    all_oos_predictions = []
    all_oos_r2 = []

    # NN parameters (could be tuned with a more sophisticated hyperparameter search)
    nn_epochs = 50 # Max epochs for NN training
    nn_batch_size = 256
    # Fixed hyperparameters for this example; a full GridSearchCV equivalent for Keras is more complex (e.g. KerasTuner)
    nn_units1=64
    nn_units2=32
    dropout_rate=0.2
    l2_reg_strength=0.01 

    print("Starting NN2 model processing with expanding window...")

    for i, (train_df, val_df, test_df) in enumerate(next_window(df)):
        current_oos_year = test_df['date'].dt.year.iloc[0]
        print(f"Processing OOS Year: {current_oos_year} for NN2")

        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()

        medians = train_df[predictors].median()
        train_df[predictors] = train_df[predictors].fillna(medians)
        val_df[predictors] = val_df[predictors].fillna(medians)
        test_df[predictors] = test_df[predictors].fillna(medians)

        scaler = StandardScaler()
        train_df[predictors] = scaler.fit_transform(train_df[predictors])
        val_df[predictors] = scaler.transform(val_df[predictors])
        test_df[predictors] = scaler.transform(test_df[predictors])
        
        train_df[target] = train_df[target].fillna(0)
        val_df[target] = val_df[target].fillna(0)
        test_df[target] = test_df[target].fillna(0)

        input_dim = len(predictors)
        
        # Create and Train NN2 Model for the current window
        # Phase 1: Train with validation_data for early stopping to get an idea of good epochs / check convergence
        model_val_phase = create_nn2_model(input_dim, nn_units1, nn_units2, dropout_rate, l2_reg_strength)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        if train_df[predictors].shape[0] < nn_batch_size or val_df[predictors].shape[0] == 0:
            print(f"Skipping NN training for OOS year {current_oos_year} due to insufficient train/val data.")
            test_df.loc[:, 'nn2_pred'] = np.nan
            model_for_prediction = None
        else:
            history = model_val_phase.fit(train_df[predictors], train_df[target],
                                epochs=nn_epochs, batch_size=nn_batch_size,
                                validation_data=(val_df[predictors], val_df[target]),
                                callbacks=[early_stopping], verbose=0)
            # Use the model state restored by EarlyStopping (best weights based on val_loss)
            # Now, re-train on combined train+val data. 
            # Option 1: Use epochs from early stopping (len(history.epoch))
            # Option 2: Re-train for a fixed number of epochs or until loss plateaus on combined set (no val_split here)
            final_epochs = len(history.epoch) # Use epochs determined by early stopping on validation set

            model_for_prediction = create_nn2_model(input_dim, nn_units1, nn_units2, dropout_rate, l2_reg_strength)
            combined_train_val_predictors = np.vstack((train_df[predictors], val_df[predictors]))
            combined_train_val_target = pd.concat([train_df[target], val_df[target]])
            
            if combined_train_val_predictors.shape[0] < nn_batch_size:
                print(f"Skipping final NN training for OOS year {current_oos_year} due to insufficient combined data.")
                test_df.loc[:, 'nn2_pred'] = np.nan
                model_for_prediction = None # Ensure it's not used if not trained
            else:
                 model_for_prediction.fit(combined_train_val_predictors, combined_train_val_target,
                                   epochs=final_epochs, batch_size=nn_batch_size, verbose=0)
        
        if model_for_prediction is not None and test_df[predictors].shape[0] > 0:
            test_df.loc[:, 'nn2_pred'] = model_for_prediction.predict(test_df[predictors], verbose=0).flatten()
        else:
            test_df.loc[:, 'nn2_pred'] = np.nan # if no model trained or no test data

        test_df['nn2_pred'] = test_df['nn2_pred'].fillna(0)

        oos_r2_year = r2_score(test_df[target], test_df['nn2_pred'])
        all_oos_r2.append(oos_r2_year)
        print(f"Year {current_oos_year} - NN2 OOS R2: {oos_r2_year:.4f}")

        all_oos_predictions.append(test_df[['date', 'permno', 'nn2_pred', target]])

        if i == len(list(next_window(df))) - 1 and model_for_prediction is not None:
            model_for_prediction.save('outputs/nn2_model_last_window.keras')
            print("Saved final NN2 model for last window to outputs/nn2_model_last_window.keras")
            # Params are fixed in this example, but could save them if tuned
            params_to_save = {'nn_units1': nn_units1, 'nn_units2': nn_units2, 'dropout': dropout_rate, 'l2': l2_reg_strength, 'oos_r2_last_window': oos_r2_year}
            with open('outputs/nn2_params_last_window.json', 'w') as f:
                 json.dump(params_to_save, f)
            print("Saved params for NN2 (last window) to outputs/nn2_params_last_window.json")

    print("Finished processing all windows for NN2 model.")

    if not all_oos_predictions:
        print("No NN2 predictions were generated. Exiting.")
        return

    all_predictions_df = pd.concat(all_oos_predictions)
    overall_oos_r2 = np.mean(all_oos_r2) if all_oos_r2 else np.nan
    print(f'Overall Average OOS R² score for NN2: {overall_oos_r2:.4f}')

    metrics_df = pd.DataFrame({'model': ['nn2'], 'oos_r2': [overall_oos_r2]})
    metrics_file = 'outputs/metrics.csv'
    if os.path.exists(metrics_file):
        existing_metrics = pd.read_csv(metrics_file)
        existing_metrics = existing_metrics[existing_metrics['model'] != 'nn2']
        metrics_df = pd.concat([existing_metrics, metrics_df])
    metrics_df.to_csv(metrics_file, index=False)
    print(f'Overall NN2 metrics saved to {metrics_file}')

    all_predictions_df = all_predictions_df.rename(columns={target: 'stock_exret'})
    nn2_overall_port_ret = form_portfolio(all_predictions_df, 'nn2_pred', n_stocks=50)
    nn2_overall_port_ret.columns = ['nn2_port_ret']
    nn2_overall_port_ret.to_csv('outputs/nn2_overall_port_ret.csv')
    print('Overall NN2 monthly portfolio returns saved to outputs/nn2_overall_port_ret.csv')

    spy_returns = pd.read_excel('Data/SPY returns.xlsx', index_col=0)
    spy_returns.index = pd.to_datetime(spy_returns.index)
    perf_summary_df = calculate_performance_metrics(nn2_overall_port_ret, spy_returns, 'nn2')
    summary_file = 'outputs/perf_summary.csv'
    if os.path.exists(summary_file):
        existing_summary = pd.read_csv(summary_file)
        existing_summary = existing_summary[existing_summary['model'] != 'nn2']
        perf_summary_df = pd.concat([existing_summary, perf_summary_df])
    perf_summary_df.to_csv(summary_file, index=False)
    print(f'Overall NN2 performance summary saved to {summary_file}')

if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    run_nn2_model() 