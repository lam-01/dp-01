import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import pickle
# import spacy


st.header("Phân tích và dự đoán kết quả học tập 🏨 ")



#with st.expander('Data'):
#     st.write('**Raw data**')
df = pd.read_csv('https://raw.githubusercontent.com/lam-01/Data/main/Student_performance_data_2.csv')
   #st.write(df)

# Hiển thị dữ liệu ban đầu
# st.subheader("Dữ liệu ban đầu")
# st.write(df.head())
# with st.expander('Data visualization'):
#   st.scatter_chart(data=df, x='Abesence', y='GPA', color='GradeClass')
#   st.scatter_chart(data=df, x='StudyTimeWeekly', y='GPA', color='GradeClass')

# Thực hiện One-Hot Encoding cho các biến phân loại
# st.subheader("Áp dụng One-Hot Encoding cho các biến phân loại")
cat_cols = ['Sports', 'Volunteering', 'ParentalSupport', 'Music', 'Extracurricular', 'ParentalEducation', 'Tutoring', 'Ethnicity']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
# Khởi tạo scaler


num_cols = ['StudyTimeWeekly', 'Absences','Age']
scaler = StandardScaler()
df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])

# st.write("Dữ liệu sau khi áp dụng One-Hot Encoding:")
# st.write(df_encoded.head())

# Phân tách dữ liệu
# st.subheader("Phân tách dữ liệu")
X = df_encoded.drop(['GradeClass', 'StudentID'], axis=1)
y = df_encoded['GradeClass']

# st.write("Biến đầu vào (X):")
# st.write(X.head())

# st.write("Biến đầu ra (y):")
# st.write(y.head())

# Áp dụng SMOTE để cân bằng dữ liệu
smote = SMOTE(sampling_strategy='auto')
X_res, y_res = smote.fit_resample(X, y)

# st.write(f"Train set size after SMOTE: {X_res.shape[0]} samples")

# chia tập dữ liệu 
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# st.write(f"Train set size: {X_train.shape[0]} samples")
# st.write(f"Test set size: {X_test.shape[0]} samples")

 
# Xây dựng
with st.sidebar:
    st.header('Input features')

    gender_map = {"Male": 0, "Female": 1}
    gender_selected = st.selectbox('Gender', ('Male', 'Female'))
    gender_encoded = gender_map[gender_selected]

    age = st.slider('Age', 15, 18, 16)
   
    ethnicity_map = {"Caucasian": 0, "African American": 1, "Asian": 2, "Other": 3}
    ethnicity_selected = st.selectbox('Ethnicity', ('Caucasian', 'African American', 'Asian', 'Other'))
    ethnicity_encoded = ethnicity_map[ethnicity_selected]

    parental_education_map = {"None": 0, "High School": 1, "Some College": 2, "Bachelor": 3, "Higher": 4}
    parental_education_selected = st.selectbox('ParentalEducation', ('None', 'High School', 'Some College', 'Bachelor', 'Higher'))
    parental_education_encoded = parental_education_map[parental_education_selected]

    tutoring_map = {"Yes": 1, "No": 0}
    tutoring_selected = st.selectbox('Tutoring', ('Yes', 'No'))
    tutoring_encoded = tutoring_map[tutoring_selected]

    parental_support_map = {"None": 0, "Low": 1, "Moderate": 2, "High": 3, "Very High": 4}
    parental_support_selected = st.selectbox('ParentalSupport', ('None', 'Low', 'Moderate', 'High', 'Very High'))
    parental_support_encoded = parental_support_map[parental_support_selected]

    extracurricular_map = {"Yes": 1, "No": 0}
    extracurricular_selected = st.selectbox('Extracurricular', ('Yes', 'No'))
    extracurricular_encoded = extracurricular_map[extracurricular_selected]

   
    # music_map = {"Yes": 1, "No": 0}
    # music_selected = st.selectbox('Music', ('Yes', 'No'))
    # music_encoded = music_map[music_selected]

   
    sport_map = {"Yes": 1, "No": 0}
    sport_selected = st.selectbox('Sport', ('Yes', 'No'))
    sport_encoded = sport_map[sport_selected]
   
    volunteering_map = {"Yes": 1, "No": 0}
    volunteering_selected = st.selectbox('Volunteering', ('Yes', 'No'))
    volunteering_encoded = volunteering_map[volunteering_selected]

    #study_time_weekly = st.number_input('Study Time Weekly (hours)', min_value=0, max_value=20)
    study_time_weekly = st.slider('Study Time Weekly (hours)', 0.00,20.00,10.00)
    absences = st.number_input('Absences', min_value=0, max_value=30)

    # Create a DataFrame for the input features
    data = {
        'Gender': gender_encoded,
         'Age':age,
        'Ethnicity': ethnicity_encoded,
        'ParentalEducation': parental_education_encoded,
        'Tutoring': tutoring_encoded,
        'ParentalSupport': parental_support_encoded,
        'Extracurricular': extracurricular_encoded,
      # 'Music':music_encoded,
        'Sport':sport_encoded,
        'Volunteering': volunteering_encoded,
        'StudyTimeWeekly': study_time_weekly,
        'Absences': absences,
       
    }


input_df = pd.DataFrame(data, index=[0])
input_penguins = pd.concat([input_df, X], axis=0)


# Mô hình 
clf = RandomForestRegressor(random_state=42)
clf.fit(X_train, y_train)
# Dự đoán giá trị cho tập dữ liệu kiểm tra
y_pred = clf.predict(X_test)

# Tính toán Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
st.write(mse)

# Tính toán R² Score
r2 = r2_score(y_test, y_pred)
st.write(r2)

# Hàm dự đoán
# def predict_gpa(mode, X_test):
#     prediction = mode.predict(X_test)
#     return prediction[0]
# Hàm dự đoán lớp điểm
def predict_grade_class(model, X_test):
    prediction = model.predict(X_test)
    return int(prediction[0])

#Hàm chuyển đổi GPA sang GradeClass
# def gpa_to_grade_class(gpa):
#     if gpa >= 3.5:
#         return 'A'
#     elif gpa > 3.0:
#         return 'B'
#     elif gpa >= 2.5:
#         return 'C'
#     elif gpa >= 2.0:
#         return 'D'
#     else:
#         return 'F'

# Dự đoán GPA khi nhấn nút Predict
# if st.button('Dự đoán'):
#     gpa_prediction = predict_gpa(clf, X_test)
#     grade_class = gpa_to_grade_class(gpa_prediction)
#     st.success(f'Predicted GPA: {gpa_prediction:.2f}')
#     st.success(f'Grade Class: {grade_class}')
# Dự đoán GradeClass khi nhấn nút Predict
if st.button('Dự đoán'):
    grade_class_prediction = predict_grade_class(clf, X_test)
    # Mapping từ số nguyên (0, 1, 2, 3, 4) sang các lớp điểm 'A', 'B', 'C', 'D', 'F'
    grade_class_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}
    predicted_grade_class = grade_class_map[grade_class_prediction]
    
    st.success(f'Predicted Grade Class: {predicted_grade_class}')




