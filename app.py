def main(df, selected_wells, look_back=50, mean_multiplier=0.5, merge_threshold=10, thickness_threshold=3):
    # Create subplots with one column per well
    fig = make_subplots(rows=1, cols=len(selected_wells), shared_yaxes=True, subplot_titles=selected_wells)

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
        fig.add_trace(go.Scatter(x=well_data_smoothed, y=well_data['tvd_scs'], mode='lines', name=f'{well_name} - Smoothed Data', line=dict(color='blue')),
                      row=1, col=index+1)

        # Plot LSTM predictions
        fig.add_trace(go.Scatter(x=lstm_predictions.flatten(), y=well_data['tvd_scs'][look_back:], mode='lines', name=f'{well_name} - LSTM Predictions', line=dict(color='orange')),
                      row=1, col=index+1)

        # Plot combined cutoff
        fig.add_trace(go.Scatter(x=combined_predictions, y=well_data['tvd_scs'][look_back:], mode='lines', name=f'{well_name} - Combined Cutoff', line=dict(color='red', dash='dash')),
                      row=1, col=index+1)

        # Highlight zones of interest using add_shape
        for start, end, diff in merged_zones:
            color_intensity = min(max(diff / max([d[2] for d in merged_zones]), 0.1), 1)  # Scale between 0.1 and 1 for better visibility
            color = 'yellow'
            fig.add_shape(type="rect",
                          x0=0, x1=1,  # Use the full width of the subplot
                          y0=start, y1=end,
                          fillcolor=color, opacity=color_intensity, line_width=0,
                          row=1, col=index+1)  # Specify the row and column directly

    # Final layout
    fig.update_layout(
        title=f'Gamma Ray Log Predictions for Selected Wells',
        xaxis_title='Gamma Ray (gr_n)',
        yaxis_title='Depth',
        template='plotly_white',
        showlegend=True,
        yaxis_autorange='reversed'  # Depth increases downwards
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
