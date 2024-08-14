import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objs as go
from scipy.signal import savgol_filter

# Define the LSTM model class
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = torch.nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input_seq):
        batch_size = input_seq.size(0)
        self.hidden_cell = (torch.zeros(1, batch_size, self.hidden_layer_size),
                            torch.zeros(1, batch_size, self.hidden_layer_size))
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

# Function to remove outliers based on z-score
def remove_outliers(data, z_thresh=3):
    mean = np.mean(data)
    std = np.std(data)
    z_scores = np.abs((data - mean) / std)
    return data[z_scores < z_thresh]

# Function to smooth data using Savitzky-Golay filter
def smooth_data_savgol(data, window_length=51, polyorder=2):
    smoothed_data = savgol_filter(data, window_length=window_length, polyorder=polyorder)
    return smoothed_data

# Function to preprocess data for prediction
def preprocess_data_for_prediction(data, scaler, look_back):
    data_scaled = scaler.transform(data.reshape(-1, 1))
    
    def create_dataset(dataset, look_back):
        X = []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
        return np.array(X)

    X = create_dataset(data_scaled, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return torch.tensor(X, dtype=torch.float32)

# Function to predict with LSTM and inverse scale the predictions
def predict_lstm(model, X, scaler):
    model.eval()
    with torch.no_grad():
        predictions = model(X).numpy()
    predictions_inverse = scaler.inverse_transform(predictions)
    return predictions_inverse

# Main function to load model, make predictions, and identify zones of interest
def main(df, selected_wells, look_back, mean_multiplier, merge_threshold, thickness_threshold):
    # Initialize empty lists to hold combined data from all selected wells
    all_well_data_smoothed = []
    all_lstm_predictions = []
    all_combined_predictions = []
    all_depths = []
    all_well_names = []
    
    # Loop over selected wells
    for well_name in selected_wells:
        # Load the data for the current well
        well_data = df[df['wellname'] == well_name].copy()

        # Load the trained LSTM model and scaler
        lstm_units = 50
        model = LSTMModel(input_size=1, hidden_layer_size=lstm_units, output_size=1)
        model.load_state_dict(torch.load('lstm_model.pth'))
        scaler = torch.load('scaler.pth')

        # Remove outliers and smooth the data
        well_data_cleaned = remove_outliers(well_data['gr_n'].values)
        well_data_smoothed = smooth_data_savgol(well_data_cleaned)

        # Preprocess data for prediction
        X = preprocess_data_for_prediction(well_data_smoothed, scaler, look_back)

        # Make LSTM predictions
        lstm_predictions = predict_lstm(model, X, scaler)
        
        # Calculate mean-based cutoff
        mean_cutoff = np.mean(well_data_smoothed) * mean_multiplier

        # Combine LSTM predictions and mean cutoff
        combined_predictions = np.minimum(lstm_predictions.flatten(), mean_cutoff)

        # Store the data for plotting
        all_well_data_smoothed.append(well_data_smoothed)
        all_lstm_predictions.append(lstm_predictions.flatten())
        all_combined_predictions.append(combined_predictions)
        all_depths.append(well_data['tvd_scs'].values[look_back:])
        all_well_names.append(well_name)

    # Visualization
    fig = go.Figure()

    # Plot smoothed data and predictions for each well
    for i, well_name in enumerate(all_well_names):
        fig.add_trace(go.Scatter(x=all_depths[i], y=all_well_data_smoothed[i], mode='lines', name=f'{well_name} Smoothed Data', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=all_depths[i], y=all_lstm_predictions[i], mode='lines', name=f'{well_name} LSTM Predictions', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=all_depths[i], y=all_combined_predictions[i], mode='lines', name=f'{well_name} Combined Cutoff', line=dict(color='red', dash='dash')))

    # Final layout
    fig.update_layout(title=f'Gamma Ray Log Predictions for Selected Wells: {", ".join(all_well_names)}',
                      xaxis_title='Depth',
                      yaxis_title='Gamma Ray (gr_n)',
                      template='plotly_white')

    st.plotly_chart(fig)

# Streamlit app
st.title("LSTM Prediction for Gamma Ray Logs")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Select wells
    wells = st.multiselect("Select wells", options=df['wellname'].unique())

    # Set parameters with sliders
    look_back = st.slider("Look Back Period", min_value=1, max_value=100, value=50, step=1)
    mean_multiplier = st.slider("Mean Multiplier", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
    merge_threshold = st.slider("Merge Threshold", min_value=1, max_value=50, value=10, step=1)
    thickness_threshold = st.slider("Thickness Threshold", min_value=1, max_value=10, value=3, step=1)

    # Show selected parameters
    st.write(f"Selected Wells: {', '.join(wells)}")
    st.write(f"Look Back Period: {look_back}")
    st.write(f"Mean Multiplier: {mean_multiplier}")
    st.write(f"Merge Threshold: {merge_threshold}")
    st.write(f"Thickness Threshold: {thickness_threshold}")

    # Run main function
    if st.button("Run LSTM Prediction"):
        main(df, wells, look_back, mean_multiplier, merge_threshold, thickness_threshold)
