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
selected_students = st.sidebar.multiselect("Select Students", students)

# If no students are selected, show all students
if not selected_students:
    selected_students = students
    st.sidebar.info("Showing all students")
else:
    st.sidebar.success(f"Showing {len(selected_students)} selected student(s)")

filtered_attendance = attendance_df[attendance_df['StudentID'].isin(selected_students)]
filtered_events = events_df[events_df['StudentID'].isin(selected_students)]
filtered_lms = lms_df[lms_df['StudentID'].isin(selected_students)]

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

st.subheader("🔴 Data Poisoning Attack Simulation")
st.markdown("""
This section demonstrates how **data poisoning** can compromise model integrity by injecting 
malicious training data that flips labels or introduces bias.
""")

poison_enabled = st.checkbox("Enable Data Poisoning Attack", value=False)

if poison_enabled:
    st.warning("⚠️ Data Poisoning Active - Model will be trained on corrupted data")
    
    # Create poisoned dataset
    ml_data_poisoned = ml_data.copy()
    
    # Attack strategy: Flip labels for students with low absence rates
    # This will make the model learn incorrect patterns
    poison_percentage = st.slider("Poisoning Percentage (%)", 0, 100, 30)
    num_poison = int(len(ml_data_poisoned) * poison_percentage / 100)
    
    # Target engaged students (low absence rate) and flip their labels
    engaged_mask = ml_data_poisoned['AbsenceRate'] < 0.2
    engaged_indices = ml_data_poisoned[engaged_mask].index
    
    if len(engaged_indices) > 0:
        poison_indices = engaged_indices[:min(num_poison, len(engaged_indices))]
        ml_data_poisoned.loc[poison_indices, 'Engagement'] = 1 - ml_data_poisoned.loc[poison_indices, 'Engagement']
    
    st.info(f"🎯 Poisoned {len(poison_indices)} records out of {len(ml_data_poisoned)}")
    
    # Train poisoned model
    X_poisoned = ml_data_poisoned[['AbsenceRate', 'SessionDuration', 'PagesViewed']]
    y_poisoned = ml_data_poisoned['Engagement']
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_poisoned, y_poisoned, random_state=42)
    
    model_poisoned = DecisionTreeClassifier()
    model_poisoned.fit(X_train_p, y_train_p)
    y_pred_poisoned = model_poisoned.predict(X_test_p)
    
    # Compare models side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ Clean Model")
        st.text(classification_report(y_test, y_pred))
    
    with col2:
        st.markdown("### 🔴 Poisoned Model")
        st.text(classification_report(y_test_p, y_pred_poisoned))
    
    # Visual comparison
    st.subheader("📊 Model Accuracy Comparison")
    from sklearn.metrics import accuracy_score
    clean_accuracy = accuracy_score(y_test, y_pred)
    poisoned_accuracy = accuracy_score(y_test_p, y_pred_poisoned)
    
    comparison_df = pd.DataFrame({
        'Model': ['Clean Model', 'Poisoned Model'],
        'Accuracy': [clean_accuracy, poisoned_accuracy]
    })
    
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(comparison_df['Model'], comparison_df['Accuracy'], color=['green', 'red'])
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0, 1])
    ax.set_title('Model Performance: Clean vs Poisoned')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    st.pyplot(fig)
    
    st.markdown("""
    ### 🔍 Impact Analysis:
    - **Accuracy Degradation**: The poisoned model shows reduced accuracy
    - **False Predictions**: Students who should be flagged as "At Risk" may be misclassified as "Engaged"
    - **Security Risk**: Attackers can manipulate model behavior by corrupting training data
    
    ### 🛡️ Mitigation Strategies:
    1. **Data Validation**: Implement anomaly detection on training data
    2. **Robust Training**: Use robust loss functions less sensitive to outliers
    3. **Data Provenance**: Track data sources and verify integrity
    4. **Regular Audits**: Monitor model performance for unexpected changes
    """)
    
    # Use poisoned model for predictions
    current_model = model_poisoned
    st.subheader("📈 Predict with Poisoned Model")
else:
    st.success("✅ Using Clean Model - No data poisoning")
    current_model = model
    st.subheader("📈 Predict Engagement for New Student")

absence_rate = st.number_input("Absence Rate (0 to 1)", min_value=0.0, max_value=1.0, value=0.1)
session_duration = st.number_input("Average Session Duration (minutes)", min_value=0.0, value=30.0)
pages_viewed = st.number_input("Average Pages Viewed", min_value=0.0, value=10.0)

if st.button("Predict Engagement"):
    prediction = current_model.predict([[absence_rate, session_duration, pages_viewed]])
    result = "Engaged" if prediction[0] == 1 else "At Risk"
    
    if poison_enabled:
        st.error(f"⚠️ Poisoned Model Prediction: {result}")
        st.caption("This prediction may be unreliable due to data poisoning")
    else:
        st.success(f"Predicted Engagement Status: {result}")
