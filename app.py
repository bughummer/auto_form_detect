import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import plotly.graph_objs as go
from plotly.subplots import make_subplots

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
    if not selected_wells:
        st.warning("Please select at least one well.")
        return

    num_wells = len(selected_wells)

    # Determine column widths based on the number of wells
    if num_wells == 1:
        # For a single well, add an invisible subplot to control the width
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.25, 0.75])
    elif num_wells < 4:
        # For 2 or 3 wells, assign a fixed width to each subplot
        fig = make_subplots(rows=1, cols=num_wells, shared_yaxes=True, column_widths=[0.25] * num_wells)
    else:
        # For 4 or more wells, distribute the widths evenly
        column_widths = [1.0 / num_wells] * num_wells
        fig = make_subplots(rows=1, cols=num_wells, shared_yaxes=True, column_widths=column_widths)

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

        # Plot smoothed data
        fig.add_trace(go.Scatter(
            x=well_data_smoothed, y=well_data['tvd_scs'], mode='lines',
            name='Smoothed Data', line=dict(color='blue'), showlegend=(index == 0)
        ), row=1, col=index+1)

        # Plot LSTM predictions
        fig.add_trace(go.Scatter(
            x=lstm_predictions.flatten(), y=well_data['tvd_scs'][look_back:], mode='lines',
            name='LSTM Predictions', line=dict(color='orange'), showlegend=(index == 0)
        ), row=1, col=index+1)

        # Plot combined cutoff
        fig.add_trace(go.Scatter(
            x=combined_predictions, y=well_data['tvd_scs'][look_back:], mode='lines',
            name='Combined Cutoff', line=dict(color='red', dash='dash'), showlegend=(index == 0)
        ), row=1, col=index+1)

        # Highlight zones of interest on each subplot correctly
        for start, end, diff in merged_zones:
            color_intensity = 0.5
            color = 'yellow'
            fig.add_shape(type="rect",
                          x0=0, x1=150,   # Use the range of the GR log
                          y0=start, y1=end,
                          fillcolor=color, opacity=color_intensity, line_width=0,
                          row=1, col=index+1)

    # Final layout
    fig.update_layout(
        title=f'Formation tops detection based on GR log',
        height=1600,  # Make the plot longer
        xaxis_title='Gamma Ray (gr_n)',
        yaxis_title='Depth',
        template='plotly_white',
        showlegend=True,
        yaxis_autorange='reversed'  # Depth increases downwards
    )

    st.plotly_chart(fig)

# Streamlit app interface
def streamlit_app():
    st.set_page_config(layout="wide")  # Set the page to wide mode
    st.title("Detect Formation Tops based on Gamma Ray Logs")

    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Select well names (multiple)
        wells = df['wellname'].unique()
        selected_wells = st.multiselect("Select wells", wells)
        # Set parameters with sliders
        look_back = 50
        mean_multiplier = 1 #st.slider("Mean Multiplier", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
        merge_threshold = st.number_input("Merge zones that have distance between them less than:", min_value=0, max_value=50, value=1, step=1)
        thickness_threshold = st.number_input("Ignore formations that have thickness less than:", min_value=0, max_value=50, value=1, step=1)

        # Run prediction and visualization
        if st.button("Detect"):
            main(df, selected_wells, look_back, mean_multiplier, merge_threshold, thickness_threshold)

# Run the app
if __name__ == "__main__":
    streamlit_app()
