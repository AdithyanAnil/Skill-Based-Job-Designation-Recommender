import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.title("Skill-Based Job Designation Recommender")

df = pd.read_csv('dataset/skill_based_data.csv')

df.drop(['Candidate_ID','Name','Age','Gender','City','Degree','Experience_Years','Projects_Count','Certifications','Resume_Summary'], axis=1, inplace=True)

skills = df.Skills
skills = skills.apply(lambda x: ' '.join(i.strip().lower() for i in x.split(',')))

df['Target_Designation'] = df['Target_Designation'].map({
    'Front End Developer': 1,
    'Backend Developer': 2,
    'Data Analyst': 3,
    'Cloud Engineer': 4,
    'ML Engineer': 5,
    'HR Manager': 6,
    'Designer': 7,
    'Digital Marketing': 8
})

tfidf = TfidfVectorizer()
x = tfidf.fit_transform(skills)

y = df['Target_Designation']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rnd = RandomForestClassifier(n_estimators=100, random_state=42)
rnd.fit(x_train, y_train)
y_predict = rnd.predict(x_test)

accuracy = accuracy_score(y_test, y_predict)
precision = precision_score(y_test, y_predict, average='weighted')
recall = recall_score(y_test, y_predict, average='weighted')
f1 = f1_score(y_test, y_predict, average='weighted')

with st.expander("Model info"):
    st.write(f"Accuracy: {accuracy*100:.2f}%")
    st.write(f"Precision (weighted): {precision*100:.2f}%")
    st.write(f"Recall (weighted): {recall*100:.2f}%")
    st.write(f"F1-Score (weighted): {f1*100:.2f}%")

designation_skills = {
    1: [
        "Front End Developer",
        ["HTML5","CSS3","JavaScript","TypeScript","React.js","Angular","Vue.js","Bootstrap","Tailwind CSS","jQuery","Redux","Responsive Design","Webpack","REST API","Git","Figma"]
    ],
    2: [
        "Backend Developer",
        ["Python","Java","C#","Node.js","Express.js","Django","Flask","Spring Boot","PHP","Laravel","ASP.NET Core","REST API","GraphQL","MySQL","PostgreSQL","MongoDB","Redis","Docker","Git","Linux"]
    ],
    3: [
        "Data Analyst",
        ["Python","SQL","Excel","Power BI","Tableau","Pandas","NumPy","Matplotlib","Seaborn","Statistics","Data Cleaning","Data Visualization","Business Intelligence","ETL","Google Sheets","R Programming"]
    ],
    4: [
        "Cloud Engineer",
        ["AWS","Azure","Google Cloud Platform","Docker","Kubernetes","Terraform","Linux","Bash","CI/CD","Jenkins","GitHub Actions","Ansible","CloudFormation","Networking","Python","Monitoring"]
    ],
    5: [
        "ML Engineer",
        ["Python","TensorFlow","PyTorch","Scikit-learn","Pandas","NumPy","OpenCV","NLP","Deep Learning","Machine Learning","Feature Engineering","Model Deployment","Docker","FastAPI","MLflow","SQL"]
    ],
    6: [
        "HR Manager",
        ["Recruitment","Talent Acquisition","Employee Relations","Payroll","HRMS","Performance Management","Onboarding","Training & Development","Compliance","Labor Laws","Communication","Conflict Resolution","MS Excel","MS Office","Leadership","Interviewing"]
    ],
    7: [
        "Designer",
        ["Adobe Photoshop","Adobe Illustrator","Figma","Adobe XD","Sketch","Canva","UI Design","UX Design","Wireframing","Prototyping","Typography","Color Theory","Responsive Design","Branding","After Effects","InDesign"]
    ],
    8: [
        "Digital Marketing",
        ["SEO","SEM","Google Ads","Meta Ads","Content Marketing","Email Marketing","Social Media Marketing","Google Analytics","Keyword Research","WordPress","Canva","Copywriting","Marketing Automation","Lead Generation","CRM","Campaign Management"]
    ]
}

user_inp = st.text_input("Your skills (comma-separated)", placeholder="e.g. Python, SQL, Power BI, Excel")

if st.button("Recommend Designation", type="primary"):
    if not user_inp.strip():
        st.warning("Please enter at least one skill.")
    else:
        inp_list = [i.strip() for i in user_inp.split(',')]
        len_of_sp = len(inp_list)

        sample = [' '.join(i.strip().lower() for i in user_inp.split(','))]
        sample = tfidf.transform(sample)
        prediction = rnd.predict(sample)

        designation = designation_skills[prediction[0]][0]
        required_skills = set(designation_skills[prediction[0]][1])
        user_skills = set(inp_list)

        missing_skills = sorted(required_skills - user_skills)
        threshold = len(required_skills) / 2.5

        # st.subheader(f"Predicted Designation: {designation}")
        # st.write("User Skills :", sorted(user_skills))
        # st.write("Required Skills :", sorted(required_skills))
        # st.write("Missing Skills :", missing_skills)
        # st.write("Threshold :", threshold)

        if len(user_skills) >= threshold:
            st.success(f"Your predicted designation is: {designation}")
            st.write("Better if you have these skills too:")

            for skill in missing_skills:
                st.write(f"- {skill}")
        elif len(missing_skills) == len(required_skills):
            st.error("You are not eligible for any of our designations.")
        else:
            st.info(f"You need to learn at least {int(threshold)-len(user_skills)} more skills for {designation}.")