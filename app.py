import pandas as pd
import gradio as gr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def accident_severity_predictor(sex, num_vehicles_involved, num_casualties):
    # Load the dataset
    # Adjust the file name accordingly
    df = pd.read_csv('RTA Dataset modified.csv')

    # Selecting relevant features
    X = df[['Sex_of_driver', 'Number_of_vehicles_involved', 'Number_of_casualties']]

    # Label encode the 'Sex_of_driver' column
    encoder = LabelEncoder()
    X['Sex_of_driver_encoded'] = encoder.fit_transform(X['Sex_of_driver'])
    X_encoded = X[['Sex_of_driver_encoded', 'Number_of_vehicles_involved', 'Number_of_casualties']]

    # Replace 'YourTargetVariable' with the actual target variable name in your dataset
    y = df['Accident_severity']

    # Split the dataset
    X_train, _, y_train, _ = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Make predictions using the user input
    user_input_df = pd.DataFrame({'Sex_of_driver': [sex], 'Number_of_vehicles_involved': [
                                 num_vehicles_involved], 'Number_of_casualties': [num_casualties]})

    # Label encode the user input
    user_input_df['Sex_of_driver_encoded'] = encoder.transform(user_input_df['Sex_of_driver'])
    user_input_encoded = user_input_df[['Sex_of_driver_encoded', 'Number_of_vehicles_involved', 'Number_of_casualties']]

    prediction = model.predict(user_input_encoded)

    return prediction[0]

# Define the input components for Gradio
sex_input = gr.Textbox(label="Enter the Sex (Male, Female, Unknown)")
num_vehicles_input = gr.Number(label="Enter the number of vehicles involved")
num_casualties_input = gr.Number(
    label="Enter the number of casualties involved")

# Define the output component for Gradio
output = gr.Textbox(label="Predicted Severity")

# Create Gradio interface
iface = gr.Interface(fn=accident_severity_predictor,
                     inputs=[sex_input, num_vehicles_input,
                             num_casualties_input],
                     outputs=output,
                     title="ROAD TRAFFIC ACCIDENT SEVERITY PREDICTION",
                     description="This is an END to END Machine Learning project done to predict the accident severity of a person by giving the inputs in the prompt. \n\n"
                                 "This Accident severity project has been deployed to showcase the main objective of the severity happened to the person who met with an accident. \n\n"
                                 "MODEL DEVELOPMENT AND MODEL DEPLOYMENT: Shanth Ruban T. \n\n"
                                 "DEPLOYMENT TOOL: GRADIO \n\n"
                                 "GITHUB: https://github.com/Ruban008 \n\n"
                                 "LINKED IN: https://in.linkedin.com/in/shanthruban-thirunavukarasu-a88b18203 \n\n"
                                 
                                  )

# Launch the Gradio interface
iface.launch(share=True)
