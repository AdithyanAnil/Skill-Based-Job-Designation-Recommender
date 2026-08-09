import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.header("Skill-Based Job Designation Recommender")

df = pd.read_csv("dataset/skill_based_data_noisy.csv")

df.drop([
    "Candidate_ID",
    "Name",
    "Age",
    "Gender",
    "City",
    "Degree",
    "Experience_Years",
    "Projects_Count",
    "Certifications",
    "Resume_Summary"
], axis=1, inplace=True)

skills = df["Skills"]
skills = skills.apply(
    lambda x: " ".join(i.strip().lower() for i in x.split(","))
)

df["Target_Designation"] = df["Target_Designation"].map({
    "Front End Developer": 1,
    "Backend Developer": 2,
    "Data Analyst": 3,
    "Cloud Engineer": 4,
    "ML Engineer": 5,
    "HR Manager": 6,
    "Designer": 7,
    "Digital Marketing": 8
})

tfidf = TfidfVectorizer()
x = tfidf.fit_transform(skills)
y = df["Target_Designation"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

rnd = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rnd.fit(x_train, y_train)

y_predict = rnd.predict(x_test)

accuracy = accuracy_score(y_test, y_predict)
precision = precision_score(y_test, y_predict, average="weighted")
recall = recall_score(y_test, y_predict, average="weighted")
f1 = f1_score(y_test, y_predict, average="weighted")

with st.sidebar.expander("Model Info"):
    st.write(f"Accuracy: {accuracy * 100:.2f}%")
    st.write(f"Precision (weighted): {precision * 100:.2f}%")
    st.write(f"Recall (weighted): {recall * 100:.2f}%")
    st.write(f"F1-Score (weighted): {f1 * 100:.2f}%")

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

with st.sidebar.expander("Available Designations"):
    for designation, skills in designation_skills.values():
        st.write(designation)
with st.sidebar:
    st.subheader("Skills by Designation")

    for designation, skills in designation_skills.values():
        with st.expander(designation):
            for skill in skills:
                st.write(f"- {skill}")
name = st.text_input("Enter your name")

user_inp = st.text_area(
    "Enter your skills (comma-separated)",
    placeholder="e.g. Python, SQL, Power BI, Excel"
)

if st.button("Find My Best Role", type="secondary"):

    if not name.strip():
        st.warning("Please enter your name.")

    elif not user_inp.strip():
        st.warning("Please enter at least one skill.")

    else:
        user_skills = set(
            i.strip().lower()
            for i in user_inp.split(",")
            if i.strip()
        )

        sample = [
            " ".join(
                i.strip().lower()
                for i in user_inp.split(",")
            )
        ]

        sample = tfidf.transform(sample)

        prediction = rnd.predict(sample)

        designation = designation_skills[prediction[0]][0]

        required_skills = set(
            skill.lower()
            for skill in designation_skills[prediction[0]][1]
        )

        matched_skills = required_skills.intersection(user_skills)

        missing_skills = sorted(
            required_skills - user_skills
        )

        threshold = len(required_skills) / 2.5

        if len(matched_skills) >= threshold:

            st.success(
                f"Hi {name}! Your predicted designation is: {designation}"
            )

            st.write(
                "You may also benefit from having these skills:"
            )

            if missing_skills:
                for skill in missing_skills:
                    st.write(f"- {skill.title()}")
            else:
                st.write(
                    "You already have all the required skills!"
                )

        elif len(matched_skills) == 0:

            st.error(
                f"Sorry {name}, your current skills don't match "
                f"the requirements for any of our designations."
            )

        else:

            needed = int(threshold) - len(matched_skills)

            if needed < 1:
                needed = 1

            st.info(
                f"{name}, you need to develop at least "
                f"{needed} more relevant skills to qualify "
                f"for the {designation} role."
            )
st.markdown("""
<style>
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    text-align: center;
    padding: 8px;
    
    color: gray;
    font-size: 16px;
    border-top: 1px solid #ddd;
    z-index: 999;
}
</style>

<div class="footer">
    © 2026 Adithyan S Anil · Skill-Based Job Designation Recommender
</div>
""", unsafe_allow_html=True)