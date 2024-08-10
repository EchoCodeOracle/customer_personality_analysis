import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load the trained model
model3 = pickle.load(open('model3.pkl', 'rb'))

def predict(features):
    # Ensure the input is a numpy array
    features = np.array(features).reshape(1, -1)
    # Predict using the loaded model
    prediction = model3.predict(features)
    return prediction

# Streamlit app
st.title('Customer Personality Analysis')

# App description
st.markdown("""
""")

st.sidebar.header('Customer Information')

# Numeric Inputs
Age = st.sidebar.number_input('Age', min_value=0, max_value=100)
Education = st.sidebar.selectbox('Education', ('Graduation', 'PhD', 'Master', '2n Cycle', 'Basic'))
Education_dict = {'Graduation': 0, 'PhD': 1, 'Master': 2, '2n Cycle': 3, 'Basic': 4}
Education = Education_dict[Education]

Marital_Status = st.sidebar.selectbox('Marital Status', ('Single', 'Married', 'Together', 'Divorced', 'Widow'))
Marital_Status_dict = {'Single': 0, 'Married': 1, 'Together': 2, 'Divorced': 3, 'Widow': 4}
Marital_Status = Marital_Status_dict[Marital_Status]

Income = st.sidebar.number_input('Income', min_value=0)
Total_Children = st.sidebar.number_input('Total Children', min_value=0)
Recency = st.sidebar.number_input('Recency', min_value=0)
Total_Spending = st.sidebar.number_input('Total Spending', min_value=0)
Total_Accepted_Campaigns = st.sidebar.number_input('Total Accepted Campaigns', min_value=0)
NumDealsPurchases = st.sidebar.number_input('Number of Deals Purchases', min_value=0)
Total_Purchases = st.sidebar.number_input('Total Purchases', min_value=0)
Web_Interaction = st.sidebar.number_input('Web Interaction', min_value=0)

# Prepare the input data
input_data = [Age, Education, Marital_Status, Income, Total_Children, Recency, Total_Spending, Total_Accepted_Campaigns, NumDealsPurchases, Total_Purchases, Web_Interaction]

# Predict and display results
if st.sidebar.button('Predict'):
    try:
        prediction = predict(input_data)
        st.write(f'Prediction: Cluster {prediction[0]}')
        
        # Display the input data
        st.subheader('Input Data')
        input_dict = {
            'Age': Age,
            'Education': Education,
            'Marital Status': Marital_Status,
            'Income': Income,
            'Total Children': Total_Children,
            'Recency': Recency,
            'Total Spending': Total_Spending,
            'Total Accepted Campaigns': Total_Accepted_Campaigns,
            'Number of Deals Purchases': NumDealsPurchases,
            'Total Purchases': Total_Purchases,
            'Web Interaction': Web_Interaction
        }
        st.write(pd.DataFrame([input_dict]))

        # Plot each input feature
        st.subheader('Feature Visualizations')

        def plot_feature(feature_name, value, color):
            fig, ax = plt.subplots()
            ax.bar([feature_name], [value], color=color)
            ax.set_ylabel(feature_name)
            ax.legend([feature_name], loc='upper right')
            st.pyplot(fig)

        plot_feature('Income', Income, 'blue')
        plot_feature('Total Spending', Total_Spending, 'green')
        plot_feature('Recency', Recency, 'purple')
        plot_feature('Total Children', Total_Children, 'gold')
        plot_feature('Total Accepted Campaigns', Total_Accepted_Campaigns, 'red')
        plot_feature('Number of Deals Purchases', NumDealsPurchases, 'orange')
        plot_feature('Total Purchases', Total_Purchases, 'brown')
        plot_feature('Web Interaction', Web_Interaction, 'cyan')

    except Exception as e:
        st.error(f"An error occurred: {e}")
