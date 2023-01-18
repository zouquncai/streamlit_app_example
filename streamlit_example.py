#!/usr/bin/env python
# coding: utf-8

# ### Design an app using streamlit
# - Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science.
# - Tornado is a Python web framework and an asynchronous networking library that relies on non-blocking network I/O to serve web applications.

# In[2]:


# !pip install streamlit
# !pip install tornado==5.1
# !pip install plotly


# In[2]:


import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder #ordinal encoding categorical features

import pickle  #to load a saved model
import base64  #to open .gif files in streamlit app

import shap #for prediction explanation
import streamlit.components.v1 as components


# In[3]:


@st.cache(suppress_st_warning=True)
def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value
@st.cache(suppress_st_warning=True)
def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value
@st.cache(suppress_st_warning=True)   
def cramers_V(var1,var2):
    crosstab = np.array(pd.crosstab(var1,var2, rownames=None, colnames=None)) # Cross table building
    stat = chi2_contingency(crosstab)[0] # Keeping of the test statistic of the Chi2 test
    obs = np.sum(crosstab) # Number of observations
    mini = min(crosstab.shape)-1 # Take the minimum value between the columns and the rows of the cross table
    return (stat/(obs*mini))

#get SHAP plots
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

df = pd.read_excel("database(well-being of students in Nice).xls")
cols = df.columns
num_cols = df._get_numeric_data().columns
cat_cols = list(set(cols) - set(num_cols))

# rows= []
# df_cat = df[cat_cols]
# for var1 in df_cat:
#     col = []
#     for var2 in df_cat:
#         cramers =cramers_V(df_cat[var1], df_cat[var2]) # Cramer's V test
#         col.append(round(cramers,2)) # Keeping of the rounded value of the Cramer's V  
#     rows.append(col)
  
# cramers_results = np.array(rows)
# df_cramerV = pd.DataFrame(cramers_results, columns = df_cat.columns, index =df_cat.columns)

app_mode = st.sidebar.selectbox('Select Page',['Home',"Exploratory Data Analysis", 'Prediction']) #two pages


# In[ ]:


if app_mode=='Home':
    st.sidebar.subheader('Early Detection of Major Depressive Disorder in Adolescents Using Demographics and Electronic Health Records')  
    st.image('image.jpeg')
    st.markdown("Mental health is an important part of overall health of a human being. Major Depressive Disorder has long been identified as one of the most common mental disorders in the United States. For some individuals, especially adolescents, major depression can result in severe impairments that interfere with or limit their ability to carry out major life activities, such as learning at school. High school students around the world are particularly vulnerable to mental health issues while facing unprecedented challenges imposed by fast changing technology and social and political environment. The covid-19 pandemic has only made it worse. According to a report released in March 2022 by US researchers, more than a third of high school students surveyed in the United States experienced stress, anxiety or depression, and nearly a fifth said they seriously considered suicide during the COVID-19 pandemic.")
    st.markdown("Therefore early detection of major depression among high school students is of paramount importance. .... This study aims to apply various machine learning algorithms on demographics and electronic health records with the goal of detecting major depression at early stage. The results of the analysis can potentially provide insights into how schools can proactively involve with students with potential depressive disorders.")
    
elif app_mode == "Exploratory Data Analysis":
    st.title('Exploratory Data Analysis')  
    st.subheader('Dataset Preview')
    
#     df = pd.read_excel("database(well-being of students in Nice).xls")
    st.write(df.head())
    
    st.subheader('Distribution of each feature in the data')
    
    fig=plt.figure(figsize = (3, 2))
    var_selected = st.selectbox("Please select a feature:", cols)
    if var_selected in num_cols:
        sns.histplot(data = df, x = var_selected, bins = 15)
        st.pyplot(fig)
    elif var_selected in cat_cols:
        sns.countplot(data = df, x = var_selected)
        st.pyplot(fig)
    
#     st.subheader("Check the correlation among numeric features")
#     fig=plt.figure(figsize = (12, 9))
#     plt.figure(figsize = (10,8))
#     sns.heatmap(df.corr(), cmap = "plasma")
#     st.pyplot(fig)

    
#     st.subheader("Check the correlation among categorical features")
#     fig=plt.figure(figsize = (12, 9))
#     plt.figure(figsize=(12,9)) 
#     sns.heatmap(df_cramerV, annot=False, fmt='.1g', vmin=0, vmax=1, cmap='plasma')
#     st.pyplot(fig)
        
    
elif app_mode == 'Prediction':
    st.image('prediction.jpeg')

    st.subheader('Please answer the questions on the left and then click the Predict button below')
    st.sidebar.header("Informations about the student:")
    
    dict_Difficulty_memorizing_lessons = {'no':0, 'yes':1}
    dict_Anxiety_symptoms = {'no':0, 'yes':1}
#     Height Imputed 
#     Physical activity(3 levels) 
    dict_Physical_activity = {'no':0, 'occasionally':1, 'regularly':2}
    dict_Satisfied_with_living_conditions = {'no':0, 'unknown':1, 'yes':2}
#     BMI_eng_imputed
#     HeartRate_imputed
    dict_Financial_difficulties = {'no':0, 'yes':1}
    dict_Learning_disabilities = {'no':0, 'yes':1}
    dict_Having_only_one_parent = {'no':0, 'unknown':1, 'yes':2}
    dict_Unbalanced_meals = {'no':0, 'yes':1}
    dict_Eating_junk_food = {'no':0, 'yes':1}
#     Cigarette smoker (5 levels)
    dict_Cigarette_smoker = {'frequently':0, 'heavily':1, 'no':2, 'occasionally':3, 'regularly':4, 'unknown':5}
    
    Difficulty_memorizing_lessons = st.sidebar.radio("Difficulty memorizing lessons?", tuple(dict_Difficulty_memorizing_lessons.keys()))
    Anxiety_symptoms = st.sidebar.radio("Anxiety symptoms?", tuple(dict_Anxiety_symptoms.keys()))
    Height_Imputed = st.sidebar.number_input("Height (cm)")
    Physical_activity = st.sidebar.radio("Physical activity", tuple(dict_Physical_activity.keys()))
    Satisfied_with_living_conditions = st.sidebar.radio("Satisfied with living_conditions?", tuple(dict_Satisfied_with_living_conditions.keys()))
    BMI_eng_imputed = st.sidebar.number_input("Body Mass Index (BMI)")
    HeartRate_imputed = st.sidebar.number_input("Heart Rate")
    Financial_difficulties = st.sidebar.radio("Financial difficulties?", tuple(dict_Financial_difficulties.keys()))
    Learning_disabilities = st.sidebar.radio("Learning_disabilities?", tuple(dict_Learning_disabilities.keys()))
    Having_only_one_parent = st.sidebar.radio("Having only one parent?", tuple(dict_Having_only_one_parent.keys()))
    Unbalanced_meals = st.sidebar.radio("Unbalanced meals?", tuple(dict_Unbalanced_meals.keys()))
    Eating_junk_food = st.sidebar.radio("Eating junk food?", tuple(dict_Eating_junk_food.keys()))
    Cigarette_smoker = st.sidebar.radio("Cigarette smoker?", tuple(dict_Cigarette_smoker.keys()))
    


# In[ ]:


    data1={
       'Difficulty memorizing lessons' : Difficulty_memorizing_lessons,
       'Anxiety symptoms' : Anxiety_symptoms,
       'Height Imputed' : Height_Imputed,
       'Physical activity(3 levels)' : Physical_activity,
       'Satisfied with living conditions' : Satisfied_with_living_conditions,
       'BMI_eng imputed' : BMI_eng_imputed,
       'HeartRate_imputed' : HeartRate_imputed,
       'Financial difficulties' : Financial_difficulties,
       'Learning disabilities':Learning_disabilities,
       'Having only one parent':Having_only_one_parent,
       'Unbalanced meals':Unbalanced_meals,
       'Eating junk food':Eating_junk_food,
       'Cigarette smoker (5 levels)' : Cigarette_smoker
       }

    feature_list=[Height_Imputed,
             HeartRate_imputed,
             BMI_eng_imputed,
             get_value(Financial_difficulties, dict_Financial_difficulties),
             get_value(Eating_junk_food, dict_Eating_junk_food),
             get_value(Physical_activity, dict_Physical_activity),
             get_value(Cigarette_smoker, dict_Cigarette_smoker),
             get_value(Having_only_one_parent, dict_Having_only_one_parent),
             get_value(Anxiety_symptoms, dict_Anxiety_symptoms),
             get_value(Learning_disabilities, dict_Learning_disabilities),
             get_value(Satisfied_with_living_conditions, dict_Satisfied_with_living_conditions),
             get_value(Unbalanced_meals, dict_Unbalanced_meals),
             get_value(Difficulty_memorizing_lessons, dict_Difficulty_memorizing_lessons)
            ]        

    single_sample = np.array(feature_list).reshape(1,-1)


# In[ ]:


    if st.button("Predict"):
#     file_ = open("images_pass.jpeg", "rb")
#     contents = file_.read()
#     data_url = base64.b64encode(contents).decode("utf-8")
#     file_.close()

#     file = open("images_warning.jpeg", "rb")
#     contents = file.read()
#     data_url_no = base64.b64encode(contents).decode("utf-8")
#     file.close()

        pickled_model = pickle.load(open('model_rf.pkl', 'rb'))
#         st.write(single_sample)
        prediction = pickled_model.predict_proba(single_sample)
        
        col1, col2 = st.columns(2)
        if prediction[:,1] < 0.2:
            col1.metric(label = "Model predicted probability of depression", value = np.round(prediction[:,1], 2))
            col2.metric(label = 'Depression Risk', value = "Low")
    #         st.image('images_pass.jpeg')
        else:
            col1.metric(label = "Model predicted probability of depression", value = np.round(prediction[:,1], 2))
            col2.metric(label = 'Prediction', value = "High")
    #         st.image('images_warning.jpeg')


# In[ ]:


        explainer = shap.TreeExplainer(pickled_model)
        shap.initjs()
        X_nonlinear_train = pd.read_pickle("df_nonlinear_train.pkl")
        shap_values = explainer.shap_values(single_sample)
#         st.write(explainer.expected_value)
#         st.write(shap_values)
        # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
        st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], single_sample, X_nonlinear_train.columns, link="logit"))

        # visualize the training set predictions
#         st_shap(shap.force_plot(explainer.expected_value, shap_values, X_nonlinear_train), 400)

