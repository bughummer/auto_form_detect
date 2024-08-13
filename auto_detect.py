import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy.signal import savgol_filter
from scipy.stats import zscore

all_data = pd.read_csv('DigitalPP_TechnicalAssignment_Dataset_v1_Aug2024.csv')

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
import plotly.graph_objs as go

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        # Initialize hidden state and cell state
        batch_size = input_seq.size(0)
        self.hidden_cell = (torch.zeros(1, batch_size, self.hidden_layer_size),
                            torch.zeros(1, batch_size, self.hidden_layer_size))
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

# Function to smooth data using Savitzky-Golay filter
def smooth_data(data, window_length=51, polyorder=2):
    smoothed_data = savgol_filter(data['gr_n'], window_length=window_length, polyorder=polyorder)
    data['gr_n_smoothed'] = smoothed_data
    return data

# Function to preprocess data for prediction
def preprocess_data_for_prediction(data, scaler, look_back):
    data['gr_n_scaled'] = scaler.transform(data['gr_n_smoothed'].values.reshape(-1, 1))
    
    def create_dataset(dataset, look_back=look_back):
        X = []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
        return np.array(X)

    dataset = data['gr_n_scaled'].values.reshape(-1, 1)
    X = create_dataset(dataset, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return torch.tensor(X, dtype=torch.float32)

# Function to predict with LSTM and inverse scale the predictions
def predict_lstm(model, X, scaler):
    model.eval()
    with torch.no_grad():
        predictions = model(X).numpy()
    # Inverse transform the predictions to the original scale
    predictions_inverse = scaler.inverse_transform(predictions)
    return predictions_inverse

# Main function to load the model, make predictions, and visualize results
def main(df, well_name):
    # Load the data
    well_data = df[df['wellname'] == well_name].copy()

    # Load the trained LSTM model and scaler
    lstm_units = 50  # Must match the value used in training
    model_path = 'lstm_model.pth'
    model = LSTMModel(input_size=1, hidden_layer_size=lstm_units, output_size=1)
    model.load_state_dict(torch.load(model_path))
    scaler = torch.load('scaler.pth')

    # Smooth the data
    well_data = smooth_data(well_data)

    # Set parameters
    look_back = 50

    # Preprocess the data for prediction
    X = preprocess_data_for_prediction(well_data, scaler, look_back)

    # Make LSTM predictions and inverse transform them to the original scale
    lstm_predictions = predict_lstm(model, X, scaler)
    print("LSTM Predictions (Inverse Scaled):")
    print(lstm_predictions.flatten())

    # Visualization
    fig = go.Figure()

    # Original GR curve
    fig.add_trace(go.Scatter(x=well_data.index, y=well_data['gr_n'], mode='lines', name='Original GR', line=dict(color='gray'), opacity=0.5))

    # Savgol Smoothed GR curve
    fig.add_trace(go.Scatter(x=well_data.index, y=well_data['gr_n_smoothed'], mode='lines', name='Savgol Smoothed GR', line=dict(color='blue')))

    # LSTM Predictions
    fig.add_trace(go.Scatter(x=well_data.index[look_back:], y=lstm_predictions.flatten(), mode='lines', name='LSTM Predictions', line=dict(color='orange')))

    # Highlight zones of interest
    zones_below_combined = []
    in_zone = False
    for i in range(len(lstm_predictions)):
        if well_data['gr_n_smoothed'].iloc[i + look_back] < lstm_predictions[i]:
            if not in_zone:
                start_depth = well_data.index[i + look_back]
                in_zone = True
        else:
            if in_zone:
                end_depth = well_data.index[i + look_back - 1]
                zones_below_combined.append((start_depth, end_depth))
                in_zone = False

    # Plot zones of interest
    for start, end in zones_below_combined:
        fig.add_vrect(x0=start, x1=end, fillcolor="yellow", opacity=0.3, line_width=0)

    # Final layout adjustments
    fig.update_layout(title=f'Gamma Ray Log Predictions for {well_name}',
                      xaxis_title='Depth',
                      yaxis_title='GR Value',
                      template='plotly_white')

    fig.show()

for well_name in all_data.wellname.unique():
    print(well_name)
    main(all_data, well_name)

