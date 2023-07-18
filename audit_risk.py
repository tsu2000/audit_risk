import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import plotly.graph_objects as go

import io
import requests

from PIL import Image
from streamlit_extras.badges import badge

# Import sklearn methods/tools
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Import all sklearn algorithms used
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def report_table(data):
    data = data.rename(index = {'0': '0 (Non-fraud)', '1': '1 (Fraud)', 'accuracy': 'Accuracy', 'macro avg': 'Macro Average', 'weighted avg': 'Weighted Average'}).round(4)
    fig = go.Figure(data = [go.Table(columnwidth = [2, 1.75],
                            header = dict(values = ['Metric', 'Precision', 'Recall', 'F1 Score', 'Support'],
                                            fill_color = 'navy',
                                            line_color = 'black',
                                            font = dict(color = 'white', size = 14),
                                            height = 27.5),
                                cells = dict(values = [data.index, data['precision'], data['recall'], data['f1-score'], data['support']], 
                                            fill_color = [['gainsboro']*2 + ['palegreen'] + ['gainsboro']*2, ['lightblue']*2 + ['palegreen'] + ['lightblue']*2],
                                            line_color = 'black',
                                            align = ['right', 'center'],
                                            font = dict(color = [['black'], ['navy']*2 + ['darkgreen'] + ['navy']*2], 
                                                        size = [14, 14]),
                                            height = 27.5))])
    fig.update_layout(height = 180, width = 800, margin = dict(l = 5, r = 5, t = 5, b = 5))
    return st.plotly_chart(fig, use_container_width = True)


def main():
    col1, col2, col3 = st.columns([0.05, 0.265, 0.035])
    
    with col1:
        url = 'https://github.com/tsu2000/audit_risk/raw/main/images/audit.png'
        response = requests.get(url)
        img = Image.open(io.BytesIO(response.content))
        st.image(img, output_format = 'png')

    with col2:
        st.title('&nbsp; Audit Risk ML Application')

    with col3:
        badge(type = 'github', name = 'tsu2000/audit_risk', url = 'https://github.com/tsu2000/audit_risk')

    st.markdown('### 🕵️ &nbsp; Audit Risk Machine Learning Web App')
    st.markdown('This web application aims to explore various classification models for classifying fradulent firms based on present and historical risk factors, using an audit risk dataset with information to predict whether a firm is fraudulent or not. The original source of the data can be found [**here**](<https://www.kaggle.com/datasets/sid321axn/audit-data>).')

    # Initialise dataframe
    url = 'https://raw.githubusercontent.com/tsu2000/audit_risk/main/audit_risk.csv'
    df = pd.read_csv(url)

    # Cleaning data:
    # df[df.isin([np.NaN, -np.Inf, np.Inf]).any(axis=1)] # Checks if any data has missing/infinite values
    df['Money_Value'].fillna(value=df['Money_Value'].mean(), inplace=True)

    # Standard scaling
    df2 = df.drop(['Risk', 'LOCATION_ID', 'Detection_Risk'], axis = 1)
    num_col = df2.columns.tolist()
    Scaler = StandardScaler()
    df2[num_col] = Scaler.fit_transform(df2[num_col])

    # Set cleaned data and target:
    X = df2
    y = df['Risk']

    st.write('')

    options = st.selectbox('Select a feature/machine learning model:', ['Exploratory Data Analysis',
                                                                        'K-Nearest Neighbours',
                                                                        'Naïve Bayes',
                                                                        'Logistic Regression',
                                                                        'Support Vector Machine',
                                                                        'Random Forest Classifier'])

    st.markdown('---')

    if options == 'Exploratory Data Analysis':
        eda(initial_data = df, cleaned_data = df2)
    elif options == 'K-Nearest Neighbours':
        knn_model(data = X, target = y)
    elif options == 'Naïve Bayes':
        nb_model(data = X, target = y)
    elif options == 'Logistic Regression':
        lr_model(data = X, target = y)
    elif options == 'Support Vector Machine':
        svm_model(data = X, target = y)
    elif options == 'Random Forest Classifier':
        rf_model(data = X, target = y)
    

def eda(initial_data, cleaned_data):
    st.markdown('## 🔎 &nbsp; Exploratory Data Analysis (EDA)')

    st.write('')

    with st.sidebar:
        st.markdown('# 📈 &nbsp; DataFrame Information')
        st.markdown('### Brief Overview of Certain Features')
        st.markdown('- `Sector_score`: Historical risk score value of the firm using analytical procedures.')
        st.markdown('- `PARA_A`: Discrepancy found in planned expenditures.')
        st.markdown('- `PARA_B`: Discrepancy found in unplanned expenditures.')
        st.markdown('- `TOTAL`: Total amount of discrepancy found in other reports.')
        st.markdown('- `numbers`: Historical discrepancy score.')
        st.markdown('- `Money_Value`: Amount of money involved in misstatements in the past audits.')
        st.markdown('- `Risk`: Whether the firm is fraudulent or not. (1 being fraudulent and 0 being not)')

    st.markdown('### Initial DataFrame:')
    st.dataframe(initial_data)
    st.write(f'Shape of data:', initial_data.shape)

    st.markdown('### Summary Statistics:')
    st.dataframe(initial_data.describe())

    st.markdown('### EDA Heatmap:')
    df = cleaned_data.corr().reset_index().rename(columns = {'index': 'Variable 1'})
    df = df.melt('Variable 1', var_name = 'Variable 2', value_name = 'Correlation')

    base_chart = alt.Chart(df).encode(
        x = 'Variable 1',
        y = 'Variable 2'
    ).properties(
        title = 'Correlation Matrix between Different Features',
        width = 850,
        height = 850
    )

    heatmap = base_chart.mark_rect().encode(
        color = alt.Color('Correlation',
                          scale = alt.Scale(scheme = 'viridis', reverse = True)
        )
    )

    text = base_chart.mark_text().encode(
        text = alt.Text('Correlation', format = ',.2f'),
        color = alt.condition(
            alt.datum['Correlation'] > 0.5, 
            alt.value('white'),
            alt.value('black')
        )
    )

    final = (heatmap + text).configure_title(
        fontSize = 25,
        offset = 10,
        anchor = 'middle'
    )

    st.altair_chart(final, use_container_width = True, theme = 'streamlit')

    st.markdown('### EDA Donut Chart - Proportion of Fraudulent Firms in Dataset:')

    risk_count = initial_data['Risk'].value_counts()
    risk_count = risk_count.rename(index = {0: 'Non-fraudulent', 1: 'Fraudulent'})    
    risk_count = risk_count.reset_index().rename(columns = {'Risk': 'Type', 'count': 'Count'})

    donut = alt.Chart(risk_count).mark_arc(innerRadius = 80).encode(
        theta = alt.Theta(field = 'Count', type = 'quantitative'),
        color = alt.Color(field = 'Type', type = 'nominal'),
    )

    st.altair_chart(donut, use_container_width = True, theme = 'streamlit')

    st.markdown('---')

    
def knn_model(data, target):
    st.markdown('## 🏘️ &nbsp; K-Nearest Neighbours Algorithm')

    st.write('')

    with st.sidebar:
        st.markdown('# 🔢 &nbsp; User Inputs')
        selected_size = st.slider('Select test size:', min_value = 0.15, max_value = 0.35, value = 0.25)
        selected_neighbours = st.slider('Select number of neighbours:', min_value = 1, max_value = 20, value = 3)
        selected_cv = st.slider('Select number of K-Fold cross-validations:', min_value = 2, max_value = 30, value = 5)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = selected_size)

    # Initialise machine learning model
    knn = KNeighborsClassifier(n_neighbors = selected_neighbours)
    knn.fit(X_train, y_train)

    # Show results
    st.markdown('### 📊 &nbsp; Results')
    st.markdown(f'- Model Accuracy Score: &emsp; **:red[{knn.score(X_test, y_test)}]**')
    st.markdown(f'- Cross-Validation Score (k = {selected_cv}): &nbsp; **:blue[{np.mean(cross_val_score(knn, data, target, cv = selected_cv))}]**')

    st.write('')

    # Create classification report
    y_pred = knn.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict = True)
    df = pd.DataFrame(report).transpose()
    
    st.markdown('##### Classification Report:')
    report_table(df)

    st.markdown('---')
    

def nb_model(data, target):
    st.markdown('## 🤷🏻‍♂️ &nbsp; Naïve Bayes Algorithm (Gaussian NB)')

    st.write('')

    with st.sidebar:
        st.markdown('### 🔢 &nbsp; User Inputs')
        selected_size = st.slider('Select test size:', min_value = 0.15, max_value = 0.35, value = 0.25)
        selected_cv = st.slider('Select number of K-Fold cross-validations:', min_value = 2, max_value = 30, value = 5)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = selected_size)

    # Initialise machine learning model
    nb = GaussianNB()
    nb.fit(X_train, y_train)

    # Show results
    st.markdown('### 📊 &nbsp; Results')
    st.markdown(f'- Model Accuracy Score: &emsp; **:red[{nb.score(X_test, y_test)}]**')
    st.markdown(f'- Cross-Validation Score (k = {selected_cv}): &nbsp; **:blue[{np.mean(cross_val_score(nb, data, target, cv = selected_cv))}]**')

    st.write('')

    # Create classification report
    y_pred = nb.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict = True)
    df = pd.DataFrame(report).transpose()

    st.markdown('##### Classification Report:')
    report_table(df)

    st.markdown('---')


def lr_model(data, target):
    st.markdown('## 🪵 &nbsp; Logistic Regression Algorithm')

    st.write('')

    with st.sidebar:
        st.markdown('# 🔢 &nbsp; User Inputs')
        selected_size = st.slider('Select test size:', min_value = 0.15, max_value = 0.35, value = 0.25)
        selected_cv = st.slider('Select number of K-Fold cross-validations:', min_value = 2, max_value = 30, value = 5)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = selected_size)

    # Initialise machine learning model
    lr = LogisticRegression(solver = 'liblinear', multi_class = 'ovr', max_iter = 1000)
    lr.fit(X_train, y_train)

    # Show results
    st.markdown('### 📊 &nbsp; Results')
    st.markdown(f'- Model Accuracy Score: &emsp; **:red[{lr.score(X_test, y_test)}]**')
    st.markdown(f'- Cross-Validation Score (k = {selected_cv}): &nbsp; **:blue[{np.mean(cross_val_score(lr, data, target, cv = selected_cv))}]**')

    st.write('')

    # Create classification report
    y_pred = lr.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict = True)
    df = pd.DataFrame(report).transpose()

    st.markdown('##### Classification Report:')
    report_table(df)

    st.markdown('---')


def svm_model(data, target):
    st.markdown('## ⚙️ &nbsp; Support Vector Machine (SVM) Algorithm')

    st.write('')

    with st.sidebar:
        st.markdown('# 🔢 &nbsp; User Inputs')
        selected_size = st.slider('Select test size:', min_value = 0.15, max_value = 0.35, value = 0.25)
        selected_cv = st.slider('Select number of K-Fold cross-validations:', min_value = 2, max_value = 30, value = 5)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = selected_size)

    # Initialise machine learning model
    svm = SVC()
    svm.fit(X_train, y_train)

    # Show results
    st.markdown('### 📊 &nbsp; Results')
    st.markdown(f'- Model Accuracy Score: &emsp; **:red[{svm.score(X_test, y_test)}]**')
    st.markdown(f'- Cross-Validation Score (k = {selected_cv}): &nbsp; **:blue[{np.mean(cross_val_score(svm, data, target, cv = selected_cv))}]**')

    st.write('')

    # Create classification report
    y_pred = svm.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict = True)
    df = pd.DataFrame(report).transpose()

    st.markdown('##### Classification Report:')
    report_table(df)

    st.markdown('---')


def rf_model(data, target):
    st.markdown('## 🌲 &nbsp; Random Forest Classifier Algorithm')

    st.write('')

    with st.sidebar:
        st.markdown('# 🔢 &nbsp; User Inputs')
        selected_size = st.slider('Select test size:', min_value = 0.15, max_value = 0.35, value = 0.25)
        selected_estimators = st.slider('Select number of trees in Random Forest model:', min_value = 1, max_value = 50, value = 20)
        selected_cv = st.slider('Select number of K-Fold cross-validations:', min_value = 2, max_value = 30, value = 5)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = selected_size)

    # Initialise machine learning model
    rf = RandomForestClassifier(n_estimators = selected_estimators)
    rf.fit(X_train, y_train)

    # Show results
    st.markdown('### 📊 &nbsp; Results')
    st.markdown(f'- Model Accuracy Score: &emsp; **:red[{rf.score(X_test, y_test)}]**')
    st.markdown(f'- Cross-Validation Score (k = {selected_cv}): &nbsp; **:blue[{np.mean(cross_val_score(rf, data, target, cv = selected_cv))}]**')

    st.write('')

    # Create classification report
    y_pred = rf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict = True)
    df = pd.DataFrame(report).transpose()

    st.markdown('##### Classification Report:')
    report_table(df)

    st.markdown('---')


if __name__ == "__main__":
    st.set_page_config(page_title = 'Audit Risk ML App', page_icon = '🕵️')
    main()
