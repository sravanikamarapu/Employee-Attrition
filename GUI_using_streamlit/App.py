import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import base64

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp
     {
    background-image: url("data:image/png;base64,%s");

    background-size: cover;
    
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('EAttr2.png')
css = """
    input[type=text] {
        border: 2px solid #000000;
        border-radius: 5px;
        padding: 5px;
        outline: none;
        box-shadow: none;
    }
"""

st.write(f"<style>{css}</style>", unsafe_allow_html=True)

df = pd.read_csv(r'C:\Users\Vani\OneDrive\Desktop\ML Project\Copy of train.csv')

df = df.drop('id', axis = 1) 
df = df.drop('StandardHours', axis = 1) 
df = df.drop('EmployeeCount', axis = 1) 
df = df.drop('Over18', axis = 1)
df = df.drop('Attrition', axis = 1)

df["BusinessTravel"] = df["BusinessTravel"].replace({"Travel_Frequently" : 0, "Travel_Rarely" : 1, "Non-Travel" : 2})
df["Department"] = df["Department"].replace({"Research & Development" : 0, "Sales" : 1, "Human Resources" : 2})
df["EducationField"] = df["EducationField"].replace({"Medical" : 0, "Other" : 1, "Marketing" : 2, "Life Sciences" : 3, "Technical Degree" : 4, "Human Resources" : 5})
df["Gender"] = df["Gender"].replace({"Male" : 0, "Female" : 1})
df["JobRole"] = df["JobRole"].replace({"Laboratory Technician" : 0, "Sales Representative" : 1, "Sales Executive" : 2, "Healthcare Representative" : 3, "Manager" : 4, "Manufacturing Director" : 5, "Research Scientist" : 6, "Human Resources" : 7, "Research Director" : 8})
df["MaritalStatus"] = df["MaritalStatus"].replace({"Married" : 0, "Single" : 1, "Divorced" : 2})
df["OverTime"] = df["OverTime"].replace({"No" : 0, "Yes" : 1})

scaler = StandardScaler()
scaler.fit(df)
pickle_in = open('model_pickel.pkl' , 'rb')

regressor = pickle.load(pickle_in)

def predict_attrition(x):
    res = regressor.predict_proba(x)[:,1]
    return np.round(res[0], 6)

st.markdown("<h1 style = 'color : black; font_size: 3rem'border-radius: 10px>𝑬𝒎𝒑𝒍𝒐𝒚𝒆𝒆 𝑨𝒕𝒕𝒓𝒊𝒕𝒊𝒐𝒏</h1>", unsafe_allow_html=True)
#st.write("<h1 style='color: white;'>This is some text.</h1>", unsafe_allow_html=True)
# st.markdown(styles, unsafe_allow_html=True)
# st.markdown(
#     """
#     <style>
#     div.stTextInput > div:first-child > input[type= "text"] {
#         background-color: #f2f2f2;
#     }
    
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

Age = st.slider("𝑨𝒈𝒆", min_value = 19, max_value = 60, step = 1, value = 55)

st.markdown('''<style>
select{
    border: 1px solid black;
}
</style>''', unsafe_allow_html=True)
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        Gender = st.radio("𝑮𝒆𝒏𝒅𝒆𝒓", ["Male", "Female"], index = 0)
        BusinessTravel = st.selectbox("𝑩𝒖𝒔𝒊𝒏𝒆𝒔𝒔𝑻𝒓𝒂𝒗𝒆𝒍", ['Travel_Frequently' ,'Travel_Rarely', 'Non-Travel'], index = 0)
        DailyRate = st.number_input("𝑫𝒂𝒊𝒍𝒚𝑹𝒂𝒕𝒆", min_value = 100, max_value = 1500)
        Department = st.selectbox("𝑫𝒆𝒑𝒂𝒓𝒕𝒎𝒆𝒏𝒕",  ['Research & Development', 'Sales', 'Human Resources'], index = 0)
        DistanceFromHome = st.number_input("𝑫𝒊𝒔𝒕𝒂𝒏𝒄𝒆𝑭𝒓𝒐𝒎𝑯𝒐𝒎𝒆", min_value = 1, max_value = 29)
        Education = st.selectbox("𝑬𝒅𝒖𝒄𝒂𝒕𝒊𝒐𝒏", [1, 2, 3, 4, 5, 15])
        EducationField = st.selectbox("𝑬𝒅𝒖𝒄𝒂𝒕𝒊𝒐𝒏𝑭𝒊𝒆𝒍𝒅",  ['Medical', 'Other', 'Marketing', 'Life Sciences', 'Technical Degree',
            'Human Resources'], index = 0)
        EnvironmentSatisfaction = st.selectbox("𝑬𝒏𝒗𝒊𝒓𝒐𝒏𝒎𝒆𝒏𝒕𝑺𝒂𝒕𝒊𝒔𝒇𝒂𝒄𝒕𝒊𝒐𝒏", [1, 2, 3, 4])
        HourlyRate = st.number_input("𝑯𝒐𝒖𝒓𝒍𝒚𝑹𝒂𝒕𝒆", min_value = 30, max_value = 100)
        JobInvolvement = st.selectbox("𝑱𝒐𝒃𝑰𝒏𝒗𝒐𝒍𝒗𝒆𝒎𝒆𝒏𝒕", [1, 2, 3, 4])
        JobLevel = st.selectbox("𝑱𝒐𝒃𝑳𝒆𝒗𝒆𝒍", [1, 2, 3, 4, 5, 7])
        JobRole = st.selectbox("𝑱𝒐𝒃𝑹𝒐𝒍𝒆",  ['Laboratory Technician' ,'Sales Representative', 'Sales Executive',
            'Healthcare Representative', 'Manager', 'Manufacturing Director',
            'Research Scientist', 'Human Resources', 'Research Director'], index = 0)
        JobSatisfaction = st.selectbox("𝑱𝒐𝒃𝑺𝒂𝒕𝒊𝒔𝒇𝒂𝒄𝒕𝒊𝒐𝒏", [1, 2, 3, 4])
        MaritalStatus = st.selectbox("𝑴𝒂𝒓𝒊𝒕𝒂𝒍𝑺𝒕𝒂𝒕𝒖𝒔",   ['Married' ,'Divorced' ,'Single'], index = 0)
    with col2:
        OverTime = st.radio("𝑶𝒗𝒆𝒓𝑻𝒊𝒎𝒆", ["Yes", "No"], index = 0)
        MonthlyIncome = st.number_input("𝑴𝒐𝒏𝒕𝒉𝒍𝒚𝑰𝒏𝒄𝒐𝒎𝒆", min_value = 1000, max_value = 20000)
        MonthlyRate = st.number_input("𝑴𝒐𝒏𝒕𝒉𝒍𝒚𝑹𝒂𝒕𝒆", min_value = 500, max_value = 27000)
        NumCompaniesWorked = st.number_input("𝑵𝒖𝒎𝑪𝒐𝒎𝒑𝒂𝒏𝒊𝒆𝒔𝑾𝒐𝒓𝒌𝒆𝒅", min_value = 0, max_value = 9)
        PercentSalaryHike = st.number_input("𝑷𝒆𝒓𝒄𝒆𝒏𝒕𝑺𝒂𝒍𝒂𝒓𝒚𝑯𝒊𝒌𝒆", min_value = 11, max_value = 25)
        PerformanceRating = st.selectbox("𝑷𝒆𝒓𝒇𝒐𝒓𝒎𝒂𝒏𝒄𝒆𝑹𝒂𝒕𝒊𝒏𝒈", [3, 4])
        RelationshipSatisfaction = st.selectbox("𝑹𝒆𝒍𝒂𝒕𝒊𝒐𝒏𝒔𝒉𝒊𝒑𝑺𝒂𝒕𝒊𝒔𝒇𝒂𝒄𝒕𝒊𝒐𝒏", [1, 2, 3, 4])
        StockOptionLevel = st.selectbox("𝑺𝒕𝒐𝒄𝒌𝑶𝒑𝒕𝒊𝒐𝒏𝑳𝒆𝒗𝒆𝒍", [0, 1, 2, 3])
        TotalWorkingYears = st.number_input("𝑻𝒐𝒕𝒂𝒍𝑾𝒐𝒓𝒌𝒊𝒏𝒈𝒀𝒆𝒂𝒓𝒔", min_value = 0, max_value = 41)
        TrainingTimesLastYear = st.selectbox("𝑻𝒓𝒂𝒊𝒏𝒊𝒏𝒈𝑻𝒊𝒎𝒆𝒔𝑳𝒂𝒔𝒕𝒀𝒆𝒂𝒓", [0, 1, 2, 3, 4, 5, 6])
        WorkLifeBalance = st.selectbox("𝑾𝒐𝒓𝒌𝑳𝒊𝒇𝒆𝑩𝒂𝒍𝒂𝒏𝒄𝒆", [1, 2, 3, 4])
        YearsAtCompany = st.number_input("𝒀𝒆𝒂𝒓𝒔𝑨𝒕𝑪𝒐𝒎𝒑𝒂𝒏𝒚", min_value = 0, max_value = 41)
        YearsInCurrentRole = st.number_input("𝒀𝒆𝒂𝒓𝒔𝑰𝒏𝑪𝒖𝒓𝒓𝒆𝒏𝒕𝑹𝒐𝒍𝒆", min_value = 0, max_value = 18)
        YearsSinceLastPromotion = st.number_input("𝒀𝒆𝒂𝒓𝒔𝑺𝒊𝒏𝒄𝒆𝑳𝒂𝒔𝒕𝑷𝒓𝒐𝒎𝒐𝒕𝒊𝒐𝒏", min_value = 0, max_value = 15)
        
YearsWithCurrManager = st.number_input("𝒀𝒆𝒂𝒓𝒔𝑾𝒊𝒕𝒉𝑪𝒖𝒓𝒓𝑴𝒂𝒏𝒂𝒈𝒆𝒓", min_value = 0, max_value = 17)


business_travel_map = {"Travel_Frequently" : 0, "Travel_Rarely" : 1, "Non-Travel" : 2}
department_map = {"Research & Development" : 0, "Sales" : 1, "Human Resources" : 2}
education_field_map = {"Medical" : 0, "Other" : 1, "Marketing" : 2, "Life Sciences" : 3, "Technical Degree" : 4, "Human Resources" : 5}
gender_map = {"Male" : 0, "Female" : 1}
job_role_map = {"Laboratory Technician" : 0, "Sales Representative" : 1, "Sales Executive" : 2, "Healthcare Representative" : 3, "Manager" : 4, "Manufacturing Director" : 5, "Research Scientist" : 6, "Human Resources" : 7, "Research Director" : 8}
marital_status_map = {"Married" : 0, "Single" : 1, "Divorced" : 2}
Over_Time_map = {"No" : 0, "Yes" : 1}

business_travel_num = business_travel_map[BusinessTravel]
department_num = department_map[Department]
education_field_num = education_field_map[EducationField]
gender_num = gender_map[Gender]
job_role_num = job_role_map[JobRole]
marital_status_num = marital_status_map[MaritalStatus]
Over_Time_num = Over_Time_map[OverTime]


x = [Age, business_travel_num, DailyRate, department_num, DistanceFromHome, Education, education_field_num, EnvironmentSatisfaction, gender_num, 
       HourlyRate, JobInvolvement, JobLevel, job_role_num, JobSatisfaction, marital_status_num, MonthlyIncome, MonthlyRate, NumCompaniesWorked,
       Over_Time_num, PercentSalaryHike, PerformanceRating,
       RelationshipSatisfaction, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance,
       YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager]
x_scaled = scaler.transform([x])
if st.button("Predict"):
        result = predict_attrition(x_scaled)
        # print(result)
        # st.success("The Attrition probability is \: {}".format(result))
        st.markdown(f'<p style="background-color:#26282A;color:white;padding:10px; border-radius: 7px">The Attrition prediction is {result}</p>',
    unsafe_allow_html=True
)
print(x)



