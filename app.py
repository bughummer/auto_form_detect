import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import plotly.graph_objs as go

# Define the LSTM model class
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

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

# Main function to load model, make predictions, identify zones of interest, and visualize the results
def main(df, selected_wells, look_back=50, mean_multiplier=0.5, merge_threshold=10, thickness_threshold=3):
    fig = go.Figure()

    for index, well_name in enumerate(selected_wells):
        # Load the data
        well_data = df[df['wellname'] == well_name].copy()

        # Load the trained LSTM model and scaler
        model = LSTMModel(input_size=1, hidden_layer_size=50, output_size=1)
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

        # Identify zones of interest
        zones_of_interest = []
        in_zone = False
        for i in range(len(combined_predictions)):
            if well_data_smoothed[i + look_back] < combined_predictions[i]:
                if not in_zone:
                    start_depth = well_data['tvd_scs'].iloc[i + look_back]
                    in_zone = True
            else:
                if in_zone:
                    end_depth = well_data['tvd_scs'].iloc[i + look_back - 1]
                    thickness = end_depth - start_depth
                    difference = np.abs(combined_predictions[i - 1] - well_data_smoothed[i + look_back - 1])
                    if thickness >= thickness_threshold:  # Only consider zones with sufficient thickness
                        zones_of_interest.append((start_depth, end_depth, difference, thickness))
                    in_zone = False

        # Merge close zones
        merged_zones = []
        if zones_of_interest:
            current_start, current_end, current_diff, _ = zones_of_interest[0]

            for start_depth, end_depth, diff, thickness in zones_of_interest[1:]:
                if start_depth - current_end <= merge_threshold:
                    current_end = end_depth
                    current_diff = max(current_diff, diff)  # Max difference in the zone
                else:
                    merged_zones.append((current_start, current_end, current_diff))
                    current_start, current_end, current_diff = start_depth, end_depth, diff

            merged_zones.append((current_start, current_end, current_diff))

        # Calculate x-offset for each well
        x_offset = index * 100  # Adjust this value based on your data's depth range for better spacing

        # Plot smoothed data
        fig.add_trace(go.Scatter(x=well_data['tvd_scs'] + x_offset, y=well_data_smoothed, mode='lines', name=f'{well_name} - Smoothed Data', line=dict(color='blue')))

        # Plot LSTM predictions
        fig.add_trace(go.Scatter(x=well_data['tvd_scs'][look_back:] + x_offset, y=lstm_predictions.flatten(), mode='lines', name=f'{well_name} - LSTM Predictions', line=dict(color='orange')))

        # Plot combined cutoff
        fig.add_trace(go.Scatter(x=well_data['tvd_scs'][look_back:] + x_offset, y=combined_predictions, mode='lines', name=f'{well_name} - Combined Cutoff', line=dict(color='red', dash='dash')))

        # Highlight zones of interest with varying colors based on the difference
        for start, end, diff in merged_zones:
            color_intensity = min(max(diff / max([d[2] for d in merged_zones]), 0.1), 1)  # Scale between 0.1 and 1 for better visibility
            color = 'yellow'
            fig.add_vrect(x0=start + x_offset, x1=end + x_offset, fillcolor=color, opacity=color_intensity, line_width=0)

    # Final layout
    fig.update_layout(
        title=f'Gamma Ray Log Predictions for Selected Wells',
        xaxis_title='Depth (with offset for each well)',
        yaxis_title='Gamma Ray (gr_n)',
        template='plotly_white',
        showlegend=True
    )

    st.plotly_chart(fig)

# Streamlit app interface
def streamlit_app():
    st.title("LSTM Prediction for Gamma Ray Logs")

    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Select well names (multiple)
        wells = df['wellname'].unique()
        selected_wells = st.multiselect("Select wells", wells)

        # Set parameters with sliders
        look_back = st.slider("Look Back Period", min_value=1, max_value=100, value=50, step=1)
        mean_multiplier = st.slider("Mean Multiplier", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
        merge_threshold = st.slider("Merge Threshold", min_value=1, max_value=50, value=10, step=1)
        thickness_threshold = st.slider("Thickness Threshold", min_value=1, max_value=10, value=3, step=1)

        # Run prediction and visualization
        if st.button("Run LSTM Prediction"):
            main(df, selected_wells, look_back, mean_multiplier, merge_threshold, thickness_threshold)

# Run the app
if __name__ == "__main__":
    streamlit_app()
