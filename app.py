import streamlit as st
import pandas as pd
import joblib
from streamlit_option_menu import option_menu
#from Premium_pipeline import SmartPremiumPipeline  # your pipeline file

# Load trained pipeline (if already saved), or initialize and fit manually
#pipeline = SmartPremiumPipeline()

train_df = pd.read_csv("/Users/muralidharanv/Documents/GUVI /PROJECTS/Smart Premium/DATA/playground-series-s4e12 (1)/train.csv",index_col=0)
#XGB_model = joblib.load('/Users/muralidharanv/Documents/GUVI /PROJECTS/Smart Premium/saved_models/XGBoost_model.pkl') 
Randomforest_model = joblib.load('/Users/muralidharanv/Documents/GUVI /PROJECTS/Smart Premium/saved_models/RandomForest_model.pkl') 

#pipeline.fit(train_df, target_column = "Premium Amount", model_name="RandomForest", model_params={"max_depth": 20, "max_features": 'sqrt', "min_samples_leaf": 2, "min_samples_split": 2, "n_estimators": 200})


# Streamlit UI
st.set_page_config(layout="wide")
#st.title("Smart Premium Estimator")
 
with st.sidebar:
    selected = option_menu(
            "Menu",
            ["Home", "Premium Calculator"],
            icons=["house", "calculator"],
            #menu_icon="cast",
            #menu_icon="menu-hamburger",
            default_index=0,
        )


if selected == "Home":
    st.title("Smart Premium Estimator")
    st.write("Welcome to Smart Premium Estimator!")
    #st.image("", width=900)
    st.write("This application is designed to help Customers to estimate their insurance premium based on various factors.")
    st.write("Please navigate to the 'Premium Calculator' section to get started.")

  
    with st.form("login_form"):
        username = st.text_input("Username" )
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button(label="Login")
        if submit_button:
            if username == "User" and password == "premium":
                st.success("Login Successful!")
            else:
                st.error("Invalid Credentials!")


elif selected == "Premium Calculator":
    st.title("Premium Estimation Criteria")
    st.header("Select the Applicable Criteria for premium estimation", divider =True)
    # User input form
    with st.form("premium_form"):
        Age = st.number_input("Age", min_value=18, max_value=100, step=1)
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Annual_Income = st.number_input("Annual Income", min_value=0,max_value=100000 ,step=5000)
        Marital_Status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        Number_of_Dependents = st.number_input("Number of Dependents", min_value=0, max_value=6, step=1)
        Education_Level = st.selectbox("Education Level", ["High School", "Bachelors", "Masters", "PhD"])
        Occupation = st.selectbox("Occupation", ["Employed", "Self-Employed", "Unemployed"])
        Health_Score = st.number_input("Health Score", min_value=0, max_value=100, step=1)
        Location = st.radio("Location", ["Urban", "Suburban", "Rural"])
        Policy_Type = st.selectbox("Policy Type",["Comprehnesive","Basic","Premium"])
        Previous_Claims = st.number_input("Previous Claims", min_value=0, max_value=5, step=1)
        Vehicle_Age = st.number_input("Vehicle Age", min_value = 0 , max_value = 20, step = 1)
        Credit_score = st.number_input("Credit Score", min_value= 250, max_value= 1000, step =50)
        Insurance_Duration = st.number_input("Insurance Duration (in years)", min_value= 1, max_value=10, step =1)
        Policy_start_date = st.date_input("Policy Start Date",format="DD/MM/YYYY")
        Customer_feedback = st.radio("Customer Feedback", ["Poor", "Average", "Good"])
        smoking_status = st.radio("Smoking Status", ["Yes", "No"])
        Exercise_frequency = st.radio("Exercise Frequency", ["Daily", "Weekly", "Monthly", "Rarely"])
        property_type = st.selectbox("Property Type", ["House", "Apartment", "Condo"])
        
        # Two columns: left for submit, right for cancel
        col1, col2 = st.columns([1, 1])
        with col1:
            Submit = st.form_submit_button("Predict Premium")
        with col2:
            Cancel = st.form_submit_button("Cancel", type="secondary")

    if Submit:
        # Prepare input DataFrame
        input_data = pd.DataFrame([{
            "Age": Age,
            "Gender": Gender,
            "Credit Score": Credit_score,
            "Annual Income": Annual_Income,
            "Marital Status": Marital_Status,
            "Number of Dependents": Number_of_Dependents,
            "Education Level": Education_Level,
            "Occupation": Occupation,
            "Health Score": Health_Score,
            "Location": Location,
            "Policy Type": Policy_Type,
            "Previous Claims": Previous_Claims,
            "Vehicle Age": Vehicle_Age,
            "Insurance Duration": Insurance_Duration,
            "Policy Start Date": Policy_start_date,
            "Customer Feedback": Customer_feedback,
            "Smoking Status": smoking_status,
            "Exercise Frequency": Exercise_frequency,
            "Property Type": property_type
        }])
        # Convert date to datetime format
        input_data['Policy Start Date'] = pd.to_datetime(input_data['Policy Start Date'], format="%d/%m/%Y", errors='coerce')
        if 'Age' in input_data.columns:
            input_data['Age Group'] = pd.cut(input_data['Age'], bins=[18, 25, 35, 45, 60, 100],
                                     labels=['18-25', '26-35', '36-45', '46-60', '60+'])

        if 'Credit Score' in input_data.columns:
            input_data['CreditScoreGroup'] = pd.cut(input_data['Credit Score'], bins=[300, 500, 650, 750, 850],
                                            labels=['Poor', 'Average', 'Good', 'Excellent'])



        # Predict
        prediction = Randomforest_model.predict(input_data)
        st.success(f"Estimated Premium Amount: ₹{round(float(prediction[0]), 2)}")
    
    elif Cancel:
        st.warning("Premium estimation process cancelled..!")
        st.stop()