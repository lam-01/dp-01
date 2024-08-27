import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

st.header("Phân tích và dự đoán kết quả học tập 🏨")

# Đọc dữ liệu từ file CSV
df = pd.read_csv('https://raw.githubusercontent.com/lam-01/Data/main/Student_performance_data_2.csv')

# Thực hiện One-Hot Encoding cho các biến phân loại
cat_cols = ['Sports', 'Volunteering', 'ParentalSupport', 'Music', 'Extracurricular', 'ParentalEducation', 'Gender', 'Tutoring', 'Ethnicity']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Phân tách dữ liệu
X = df_encoded.drop(['GradeClass', 'StudentID'], axis=1)
y = df_encoded['GradeClass']

# Áp dụng SMOTE để cân bằng dữ liệu
smote = SMOTE(sampling_strategy='auto')
X_res, y_res = smote.fit_resample(X, y)

# Chia tập dữ liệu thành train và test
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Sidebar input features
with st.sidebar:
    st.header('Input features')

    gender_map = {"Male": 0, "Female": 1}
    gender_selected = st.selectbox('Gender', ('Male', 'Female'))
    gender_encoded = gender_map[gender_selected]

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

    volunteering_map = {"Yes": 1, "No": 0}
    volunteering_selected = st.selectbox('Volunteering', ('Yes', 'No'))
    volunteering_encoded = volunteering_map[volunteering_selected]

    study_time_weekly = st.slider('Study Time Weekly (hours)', 0.00, 20.00, 10.00)
    absences = st.number_input('Absences', min_value=0, max_value=30)

# Tạo DataFrame cho các đặc trưng đầu vào
input_data = {
    'Gender': gender_encoded,
    'Ethnicity': ethnicity_encoded,
    'ParentalEducation': parental_education_encoded,
    'Tutoring': tutoring_encoded,
    'ParentalSupport': parental_support_encoded,
    'Extracurricular': extracurricular_encoded,
    'Volunteering': volunteering_encoded,
    'StudyTimeWeekly': study_time_weekly,
    'Absences': absences,
}

input_df = pd.DataFrame(input_data, index=[0])

# Tối ưu hóa mô hình Random Forest với GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='r2')
grid_search.fit(X_train, y_train)

# In ra các tham số tốt nhất
st.write(f"Best parameters: {grid_search.best_params_}")

# Khởi tạo mô hình với các tham số tốt nhất
best_rf = RandomForestRegressor(
    n_estimators=grid_search.best_params_['n_estimators'],
    max_features=grid_search.best_params_['max_features'],
    max_depth=grid_search.best_params_['max_depth'],
    min_samples_split=grid_search.best_params_['min_samples_split'],
    min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
    bootstrap=grid_search.best_params_['bootstrap'],
    random_state=42
)

# Huấn luyện mô hình tốt nhất
best_rf.fit(X_train, y_train)

# Hàm dự đoán
def predict_grade_class(model, input_features):
    prediction = model.predict(input_features)
    return prediction[0]

# Hàm chuyển đổi GPA sang GradeClass
def gpa_to_grade_class(gpa):
    if gpa >= 3.5:
        return 'A'
    elif gpa >= 3.0:
        return 'B'
    elif gpa >= 2.5:
        return 'C'
    elif gpa >= 2.0:
        return 'D'
    else:
        return 'F'

# Dự đoán Grade Class khi nhấn nút Predict
if st.button('Dự đoán'):
    grade_class_prediction = predict_grade_class(best_rf, input_df)
    grade_class = gpa_to_grade_class(grade_class_prediction)
    st.success(f'Predicted Grade Class: {grade_class}')
