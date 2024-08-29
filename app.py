import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

import os
import google.generativeai as genai
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Set up Gemini Pro
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

try:
    model = genai.GenerativeModel("gemini-1.5-pro")
except Exception as e:
    st.error(f"Failed to initialize Gemini Pro: {str(e)}")
    model = None

def get_gemini_response(prompt):
    if model is None:
        return "Gemini Pro API is not available. Using fallback content."
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error getting response from Gemini Pro: {str(e)}")
        return "Unable to generate response. Using fallback content."

def get_fallback_content(prompt):
    fallback_responses = {
        "Summarize the ChaosCNN project in 2-3 sentences based on the provided research paper.": 
            "ChaosCNN is a novel method using convolutional neural networks to predict outcomes in chaotic systems. It transforms multivariate time series into CNN-compatible formats and uses specialized architecture optimized via genetic algorithms. The method shows improved accuracy in predicting road accidents and modeling the Lorenz system compared to traditional approaches.",
        "Explain the Data Preparation step of ChaosCNN methodology in simple terms.":
            "Data preparation in ChaosCNN involves transforming multivariate time series data into a format suitable for CNN input. This typically includes normalizing the data, creating sliding windows of fixed size, and possibly applying data augmentation techniques.",
        "Explain the Network Architecture step of ChaosCNN methodology in simple terms.":
            "The ChaosCNN architecture consists of multiple convolutional layers to extract features from the input data, followed by pooling layers for downsampling. The final layers are fully connected and output the predicted values for the chaotic system.",
        "Explain the Training Process step of ChaosCNN methodology in simple terms.":
            "The ChaosCNN training process involves feeding the prepared data through the network, calculating the loss between predicted and actual values, and using backpropagation to adjust the network weights. It incorporates multi-step training and chaos-aware regularization to enhance long-term prediction accuracy.",
        "Explain the Genetic Hyperparameter Optimization step of ChaosCNN methodology in simple terms.":
            "Genetic hyperparameter optimization in ChaosCNN uses evolutionary algorithms to find the best combination of hyperparameters for the model. This includes parameters like the number of layers, filter sizes, and learning rates, optimizing them to improve the model's performance on chaotic systems.",
        "Explain the Prediction step of ChaosCNN methodology in simple terms.":
            "In the prediction step, ChaosCNN takes new input data and passes it through the trained network to generate predictions. For multi-step predictions, the model may use its own outputs as inputs for subsequent time steps, carefully managing the accumulation of errors in chaotic systems."
    }
    return fallback_responses.get(prompt, "No fallback content available for this prompt.")

def prepare_data(data, window_size=30):
    if data.empty:
        raise ValueError("The input data is empty.")
    
    X = data[['weather_condition', 'time_of_day', 'traffic_intensity', 'road_conditions']].values
    y = data['fatalities'].values

    if len(X) < window_size:
        raise ValueError(f"Not enough data points. Expected at least {window_size}, but got {len(X)}.")

    X_windows = []
    y_windows = []

    for i in range(len(X) - window_size):
        X_windows.append(X[i:i+window_size])
        y_windows.append(y[i+window_size])

    return np.array(X_windows), np.array(y_windows)

def create_model(input_shape):
    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    st.title("ChaosCNN Project Explorer")

    # Project Overview
    st.header("Project Overview")
    overview = get_gemini_response("Summarize the ChaosCNN project in 2-3 sentences based on the provided research paper.")
    if "Using fallback content" in overview:
        overview = get_fallback_content("Summarize the ChaosCNN project in 2-3 sentences based on the provided research paper.")
    st.write(overview)

    # Abstract Display
    st.header("Abstract")
    with st.expander("View Abstract"):
        st.write("""
        This paper introduces ChaosCNN, a novel method employing convolutional neural networks (CNNs) to predict outcomes in chaotic systems. We present an innovative approach to transform multivariate time series into CNN-compatible formats, coupled with a specialized architecture optimized via genetic algorithms. ChaosCNN incorporates multi-step training and chaos-aware regularization to enhance long-term prediction accuracy. We demonstrate its effectiveness on road accident prediction and the Lorenz system, achieving superior results compared to DIGGER and LSTM methods. For accident prediction, ChaosCNN attains a Mean Absolute Error of 0.42, outperforming DIGGER (0.58) and LSTM (0.63). In the Lorenz system, it maintains accuracy for 2-3 Lyapunov times. This research advances chaotic system forecasting, offering potential applications in diverse fields from climate modeling to financial analysis.
        """)

    # Methodology Explanation
    st.header("Methodology")
    methodology_options = ["Data Preparation", "Network Architecture", "Training Process", "Genetic Hyperparameter Optimization", "Prediction"]
    selected_method = st.selectbox("Select a methodology component:", methodology_options)
    
    method_explanation = get_gemini_response(f"Explain the {selected_method} step of ChaosCNN methodology in simple terms.")
    if "Using fallback content" in method_explanation:
        method_explanation = get_fallback_content(f"Explain the {selected_method} step of ChaosCNN methodology in simple terms.")
    st.write(method_explanation)

    # Data Visualization and Model Training
    st.header("Data Visualization and Model Training")
    
    # Load and prepare data
    try:
        data = pd.read_csv('data/accident_data.csv')
        
        if data.empty:
            st.error("The loaded data is empty. Please check your 'accident_data.csv' file.")
            return

        st.write("Data sample:")
        st.write(data.head())
        
        X, y = prepare_data(data)
        
        # Normalize the data
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
        # Create and train model
        model = create_model(input_shape=(X.shape[1], X.shape[2]))
        
        if st.button("Train Model"):
            with st.spinner("Training ChaosCNN model..."):
                history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
            st.success("Model trained successfully!")
            
            # Plot training history
            fig, ax = plt.subplots()
            ax.plot(history.history['loss'], label='Training Loss')
            ax.plot(history.history['val_loss'], label='Validation Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            st.pyplot(fig)
    
        # Make predictions
        if st.button("Make Predictions"):
            y_pred = model.predict(X_test).flatten()
            
            # Plot results
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            st.pyplot(fig)
        
            mae = np.mean(np.abs(y_test - y_pred))
            st.write(f"Mean Absolute Error: {mae:.2f}")

        # ChaosCNN Demonstration
        st.header("ChaosCNN Demonstration")
        st.write("Input parameters for accident prediction:")

        # Input parameters
        weather = st.slider("Weather Condition", 0, 10, 5)
        time_of_day = st.slider("Time of Day", 0, 24, 12)
        traffic_intensity = st.slider("Traffic Intensity", 0, 10, 5)
        road_conditions = st.slider("Road Conditions", 0, 10, 5)

        # Generate prediction
        if st.button("Predict"):
            # Create a sequence of 30 time steps (assuming window_size=30)
            input_sequence = np.tile([weather, time_of_day, traffic_intensity, road_conditions], (30, 1))
            input_sequence = scaler.transform(input_sequence).reshape(1, 30, 4)
            
            prediction = model.predict(input_sequence)[0][0]
            st.subheader("Prediction")
            st.write(f"Predicted number of fatalities: {prediction:.2f}")

    except FileNotFoundError:
        st.error("The file 'data/accident_data.csv' was not found. Please make sure it exists in the correct location.")
    except pd.errors.EmptyDataError:
        st.error("The file 'accident_data.csv' is empty. Please check the file contents.")
    except ValueError as e:
        st.error(f"Error in data preparation: {str(e)}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.write("Please make sure the 'data/accident_data.csv' file exists, is properly formatted, and contains sufficient data.")

if __name__ == "__main__":
    main()