import matplotlib.pyplot as plt
import numpy as np
from google.generativeai import GenerativeModel

def plot_results(y_true, y_pred):
    fig, ax = plt.subplots()
    ax.plot(y_true, label='Actual')
    ax.plot(y_pred, label='Predicted')
    ax.set_xlabel('Time')
    ax.set_ylabel('Fatalities')
    ax.legend()
    return fig

def explain_prediction(model, input_data, prediction):
    model = GenerativeModel('gemini-pro')
    prompt = f"""
    Explain why a ChaosCNN model might predict {prediction:.2f} fatalities given the following input parameters:
    Weather Condition: {input_data[0][0]}
    Time of Day: {input_data[0][1]}
    Traffic Intensity: {input_data[0][2]}
    Road Conditions: {input_data[0][3]}

    Consider the complex interactions between these factors in a chaotic system.
    """
    response = model.generate_content(prompt)
    return response.text