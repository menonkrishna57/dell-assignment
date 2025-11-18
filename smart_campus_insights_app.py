import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


data = {'StudentID': [1, 1, 2, 2, 3, 3, 1, 2, 3],
        'Date': ['2023-10-01', '2023-10-02', '2023-10-01', '2023-10-02', '2023-10-01', '2023-10-02', '2023-10-03', '2023-10-03', '2023-10-03'],
        'Status': ['Present', 'Absent', 'Present', 'Present', 'Absent', 'Present', 'Present', 'Present', 'Present']}
attendance_df = pd.DataFrame(data)
data = {'StudentID': [1, 2, 3, 1, 2], 'EventName': ['Hackathon', 'Seminar', 'Hackathon', 'Workshop', 'Seminar']}
events_df = pd.DataFrame(data)
data = {'StudentID': [1, 2, 3, 1, 2, 3], 'SessionDuration': [45, 60, 20, 50, 70, 30], 'PagesViewed': [15, 20, 5, 18, 22, 8]}
lms_df = pd.DataFrame(data)

st.title("📊 Smart Campus Insights")
st.sidebar.header("🔍 Filters")

students = attendance_df['StudentID'].unique()
student_options = ['All'] + sorted(list(students))
selected_student = st.sidebar.selectbox("Select Student", student_options)

if selected_student == 'All':
    filtered_attendance = attendance_df
    filtered_events = events_df
    filtered_lms = lms_df
else:
    filtered_attendance = attendance_df[attendance_df['StudentID'] == selected_student]
    filtered_events = events_df[events_df['StudentID'] == selected_student]
    filtered_lms = lms_df[lms_df['StudentID'] == selected_student]

st.subheader("📋 Attendance Trends")
if not filtered_attendance.empty:
    if selected_student == 'All':
        attendance_summary = filtered_attendance.groupby(['Date', 'Status']).size().unstack(fill_value=0)
        st.line_chart(attendance_summary)
    else:
        attendance_summary = filtered_attendance['Status'].value_counts().rename('Count')
        st.bar_chart(attendance_summary)
else:
    st.write("No attendance data available for the selection.")

st.subheader("🎓 Event Participation")
if not filtered_events.empty:
    event_counts = filtered_events['EventName'].value_counts()
    st.bar_chart(event_counts)
else:
    st.write("No event participation data available for the selection.")

st.subheader("💻 LMS Usage Patterns")
if not filtered_lms.empty:
    lms_summary = filtered_lms.groupby('StudentID')[['SessionDuration', 'PagesViewed']].mean()
    st.dataframe(lms_summary)
else:
    st.write("No LMS usage data available for the selection.")

st.subheader("🤖 Predict Student Engagement Risk")

ml_data = pd.merge(attendance_df.groupby('StudentID')['Status'].apply(lambda x: (x == 'Absent').mean()).reset_index(name='AbsenceRate'),
                   lms_df.groupby('StudentID')[['SessionDuration', 'PagesViewed']].mean().reset_index(),
                   on='StudentID')

ml_data['Engagement'] = (ml_data['AbsenceRate'] < 0.2).astype(int)

if len(ml_data) > 1:
    X = ml_data[['AbsenceRate', 'SessionDuration', 'PagesViewed']]
    y = ml_data['Engagement']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.text("Model Performance:")
    st.code(classification_report(y_test, y_pred))

    st.subheader("📈 Predict Engagement for New Student")
    absence_rate = st.number_input("Absence Rate (0 to 1)", min_value=0.0, max_value=1.0, value=0.1)
    session_duration = st.number_input("Average Session Duration (minutes)", min_value=0.0, value=30.0)
    pages_viewed = st.number_input("Average Pages Viewed", min_value=0.0, value=10.0)

    if st.button("Predict Engagement"):
        prediction = model.predict([[absence_rate, session_duration, pages_viewed]])
        result = "Engaged" if prediction[0] == 1 else "At Risk"
        st.success(f"Predicted Engagement Status: **{result}**")
else:
    st.warning("Not enough data to train the prediction model.")
