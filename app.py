import streamlit as st
import pandas as pd
import torch
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
import joblib

# Define the LSTM model class
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.linear(lstm_out[:, -1])

# Function to smooth data using Savitzky-Golay filter
def smooth_data(data):
    from scipy.signal import savgol_filter
    data['gr_n_smoothed'] = savgol_filter(data['gr_n'], 51, 2)
    return data

# Preprocess the data for prediction
def preprocess_data_for_prediction(data, scaler, look_back):
    data = smooth_data(data)
    scaled_data = scaler.transform(data['gr_n_smoothed'].values.reshape(-1, 1))
    X = [scaled_data[i:i+look_back] for i in range(len(scaled_data) - look_back)]
    return torch.from_numpy(np.array(X)).float()

# Predict using the LSTM model
def predict_lstm(model, X, scaler):
    model.eval()  # Set model to evaluation mode
    y_pred = model(X).detach().numpy()
    return scaler.inverse_transform(y_pred)

# Streamlit app code
st.title("LSTM Prediction for Gamma Ray Logs")

# Step 1: File upload
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    well_data = pd.read_csv(uploaded_file)
    
    # Step 2: Select well from dropdown
    well_options = well_data['wellname'].unique()
    selected_well = st.selectbox("Select a well", well_options)
    
    if selected_well:
        st.write(f"You selected: {selected_well}")
        well_data = well_data[well_data['wellname'] == selected_well]
        
        # Load the pre-trained LSTM model
        model = LSTMModel()
        model.load_state_dict(torch.load('lstm_model.pth'))

        # Load the pre-fitted scaler
        scaler = joblib.load('scaler.pth')
        
        # Preprocess the data
        look_back = 10
        X = preprocess_data_for_prediction(well_data, scaler, look_back)
        
        # Predict using the LSTM model
        lstm_predictions = predict_lstm(model, X, scaler)
        
        # Visualization
        st.write("LSTM Predictions (Inverse Scaled):")
        st.write(lstm_predictions.flatten())

        fig = go.Figure()

        # Plot original GR curve
        fig.add_trace(go.Scatter(x=well_data.index, y=well_data['gr_n'], name='Original GR', line=dict(color='gray')))

        # Plot smoothed GR curve
        fig.add_trace(go.Scatter(x=well_data.index, y=well_data['gr_n_smoothed'], name='Smoothed GR', line=dict(color='blue')))

        # Plot LSTM Predictions
        fig.add_trace(go.Scatter(x=well_data.index[look_back:], y=lstm_predictions.flatten(), name='LSTM Predictions', line=dict(color='orange')))

        # Show plot in the Streamlit app
        st.plotly_chart(fig)
