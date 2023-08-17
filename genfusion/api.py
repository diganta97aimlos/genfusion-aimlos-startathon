import streamlit as st
import logging
from zipfile import ZipFile
import shutil, os
from airflow.utils.dot_renderer import render_dag
from airflow.models.dagbag import DagBag
import pandas as pd
import json
from airflow.api.client.local_client import Client
from airflow.models.dagrun import DagRun
import time
import torch
import pickle, io
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from PIL import Image
# from packages.fedgen.fedgen.predict import Predict

batch_size = 32
epochs = 2

c = Client(None, None)

with open('credentials.yml') as file:
    cred = yaml.load(file, Loader=SafeLoader)
file.close()

authentication_status = None

st.subheader('Welcome to FedGen: Collaborative Private Learning')

authenticator = stauth.Authenticate(
    cred['credentials'],
    cred['cookie']['name'],
    cred['cookie']['key'],
    cred['cookie']['expiry_days']
)

name, authentication_status, username = authenticator.login('Login', 'main')
worker = username
st.warning('Please enter your username and password')

def extract_zip_file(zipFile, worker):
    for filename in zipFile:
        with ZipFile(f'../uploads/{filename.name}', 'r') as zipObj:
            zipObj.extractall('../uploads/')
            if f'{worker}' in os.listdir('../workers'):
                for category in os.listdir(f'../uploads'):
                    if category in os.listdir(f'../workers/{worker}'):
                        for fileData in os.listdir(f'../uploads/{category}'):
                            src = f'../uploads/{category}/{fileData}'
                            dest = f'../workers/{worker}/{category}/{fileData}'
                            shutil.copyfile(src, dest)
        os.remove(f'../uploads/{filename.name}')

if authentication_status is None:
    try:
        if authenticator.register_user('Register user', preauthorization=False):
            st.success('User registered successfully')
    except Exception as e:
        st.error(e)

if authentication_status:
    worker = username
    
    sidebarObj = st.sidebar

    option = sidebarObj.selectbox('Choose Your Service',
    (None, 'Model Collaboration', 'Data Generation', 'FedGen AI'))

    if option is not None and option == 'FedGen AI':
        st.subheader('Choose Your Vertical')
        vertical = st.selectbox('Vertical', (None, 'Health & Medical'))
        if vertical == 'Health & Medical':
            st.subheader('Choose Your Dataset')
            dataset = st.selectbox('Dataset', (None, 'Covid Prediction & Analysis'))
            if dataset == 'Covid Prediction & Analysis':
                zipFile = st.file_uploader("Upload Your Data", type=(["zip"]), accept_multiple_files=True)
                if zipFile is not None:
                    extract_zip_file(zipFile, worker)
            trainStatus = sidebarObj.selectbox('Perform Collaborative Training', ('No', 'Yes'))
            if trainStatus == 'Yes':
                epochs = sidebarObj.slider('Training Rounds', 2, 50, 100)
                batch_size = sidebarObj.slider('Training Batch Size', 16, 32, 48)
                st.subheader("Apache Airflow - DAG Viewer")
                dag_id = 'trainer'
                dag_bag = DagBag(dag_folder='airflow/dags', include_examples=False)
                dag = dag_bag.dags[dag_id]
                doc_md = getattr(dag, "doc_md", None)
                if doc_md:
                    st.markdown(doc_md)
                with st.expander('FedGen: Collaborator & Generator'):
                    st.graphviz_chart(render_dag(dag), use_container_width=False)
                start_process = sidebarObj.button('Start Process')
                if start_process:
                    st.info('Federation Process Started')
                    state = 'failed'
                    with st.spinner('Please wait for the process to complete...'):
                        with open('run_config.json', 'w') as file:
                            config_dict = {'run_id': 1, 'mode': 'train', 'epochs': epochs, 'batch_size': batch_size,
                                'selected_model': None, 'client': worker, 'prediction_mode': None}
                            json.dump(config_dict, file)
                        file.close()
                        c.trigger_dag('trainer')
                        while state != 'success':
                            dag_runs = DagRun.find(dag_id='trainer')
                            dag_runs = dag_runs[-1]
                            state = dag_runs.state
                            time.sleep(60)
                        if state == 'success':
                            st.success('Process Successfully Completed!')
            dag_runs = DagRun.find(dag_id='trainer')
            dag_runs = dag_runs[-1]
            state = dag_runs.state
            if state == 'success':
                # results = pd.read_csv('/home/ubuntu/GenAI-Rush/metrics.csv', index_col=False)
                # st.subheader('Model Performance')
                # workerResults = {}
                # for col in results.columns:
                #     workerResults[col] = list()
                # if worker is not None:
                #     for index in range(len(results)):
                #         if worker.lower() in results['Client'][index].lower():
                #             workerResults['Client'].append(results['Client'][index])
                #             workerResults['Federated'].append(results['Federated'][index])
                #             workerResults['Test Loss'].append(results['Test Loss'][index])
                #             workerResults['Test Accuracy'].append(results['Test Accuracy'][index])
                #             workerResults['Precision'].append(results['Precision'][index])
                #             workerResults['Recall'].append(results['Recall'][index])
                #             workerResults['F1 Score'].append(results['F1 Score'][index])
                #     workerResults = pd.DataFrame(workerResults)
                #     st.dataframe(workerResults)
                st.subheader('Download Your Trained Model Weights')
                col1, col2, col3 = st.columns(3)
                weights = torch.load(f'/home/ubuntu/GenAI-Rush/models/encrypted_{worker}.pth')
                pklData = io.BytesIO()
                pickle.dump(weights, pklData)
                col1.download_button(
                    label="Client Model - Encrypted",
                    data=pklData,
                    file_name=f'encrypted_{worker}.pkl',
                )
                weights = torch.load(f'/home/ubuntu/GenAI-Rush/models/encrypted_federated_{worker}.pth')
                pklData = io.BytesIO()
                pickle.dump(weights, pklData)
                col2.download_button(
                    label="Client Model - Encrypted & Federated",
                    data=pklData,
                    file_name=f'encrypted_{worker}.pkl',
                )
                weights = torch.load(f'/home/ubuntu/GenAI-Rush/models/encrypted_global_federated.pth')
                pklData = io.BytesIO()
                pickle.dump(weights, pklData)
                col3.download_button(
                    label="Global Model - Encrypted & Federated",
                    data=pklData,
                    file_name=f'encrypted_{worker}.pkl',
                )
                st.subheader('Use Your Models')
                modelUsage = sidebarObj.selectbox('Use Your Model', (None, 'Single Files (Max 5)', 'Set of Files'))
                # workerResults['Model Score'] = (workerResults['Test Accuracy'] * workerResults['Precision'] * workerResults['Recall'] \
                #     * workerResults['F1 Score'])/workerResults['Test Loss']
                # workerResults = workerResults.sort_values(by=['Model Score'], ascending=False)
                # selectedModel = sidebarObj.selectbox('Select Your Model', 
                #     (None, 
                #     f"{workerResults['Client'][0]}: Federated: {workerResults['Federated'][0]} (Recommended)", 
                #     f"{workerResults['Client'][1]}: Federated: {workerResults['Federated'][1]}", 
                #     f"{workerResults['Client'][2]}: Federated: {workerResults['Federated'][2]} (Low Performance)"))
                # predictionMode = False
                # files = None
                # directory = None
                # if modelUsage is not None:
                #     if modelUsage == 'Single Files (Max 5)':
                #         files = st.file_uploader("Upload your files", type=(["jpg", "jpeg", "png"]), accept_multiple_files=True)
                #         if files is not None:
                #             files = [filename.name for filename in files]
                #             if len(files) > 5:
                #                 st.error('Max limit of 5 files allowed')
                #             else:
                #                 predictionMode = True
                #     elif modelUsage == 'Set of Files':
                #         zipFile = st.file_uploader("Upload your files", type=(["zip"]), accept_multiple_files=True)
                #         if zipFile is not None:
                #             for filename in zipFile:
                #                 with ZipFile(f'/home/ubuntu/kreedaAI/fedgen/to_predict/{filename.name}', 'r') as zipObj:
                #                     zipObj.extractall('/home/ubuntu/kreedaAI/fedgen/to_predict')
                #                 directory = filename.name.split('.')[0]
                #             predictionMode = True
                #     if predictionMode is True and selectedModel is not None:
                #         if 'Global' in selectedModel:
                #             selectedModel = 'Global Model: Encrypted & Federated'
                #         elif 'Federated: False' in selectedModel:
                #             selectedModel = 'Client Model - Encrypted'
                #         elif 'Federated: True' in selectedModel:
                #             selectedModel = 'Client Model - Encrypted & Federated'
                #         predictButton = sidebarObj.button('Get Results')
                #         if predictButton:
                #             state = 'failed'
                #             with st.spinner('Please wait for the process to complete...'):
                #                 with open('/home/ubuntu/airflow/dags/run_config.json', 'w') as file:
                #                     config_dict = {'run_id': 1, 'mode': 'predict', 'epochs': epochs, 'batch_size': batch_size,
                #                                     'selected_model': selectedModel, 'client': worker, 'prediction_mode': modelUsage}
                #                     json.dump(config_dict, file)
                #                 file.close()
                #                 c.trigger_dag('fedgen-trigger', conf={})
                #                 while state != 'success':
                #                     dag_runs = DagRun.find(dag_id='fedgen-trigger')
                #                     dag_runs = dag_runs[-1]
                #                     state = dag_runs.state
                #                     time.sleep(10)
                #                 if state == 'success':
                #                     st.subheader('Predictions')
                #                     results = pd.read_csv('/home/ubuntu/airflow/results/predictions.csv', index_col=False)
                #                     grid = st.columns(5)
                #                     for index in range(0, len(results), 5):
                #                         for idx in range(5):
                #                             try:
                #                                 grid[idx].image(results['File Path'][idx+index], caption=results['Prediction'][idx+index].capitalize())
                #                             except:
                #                                 pass
    if authenticator.logout('Logout', 'main', key='unique_key'):
        authentication_status = None
elif authentication_status is False:
    st.error('Username/password is incorrect')
