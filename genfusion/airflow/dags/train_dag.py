import os
import airflow
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import timedelta
import datetime
import json
import sys
sys.path.insert(0,os.path.abspath(os.path.dirname('/home/ubuntu/GenAI-Rush/GenAI-Rush-FedGen/fedgen')))

from fedgen.fetchData import PreProcess
from fedgen.utils import CreateLoaders
from fedgen.trainer import FedTrainer
from fedgen.federator import SimpleFederatedAveraging
from fedgen.test import TestFederated
from fedgen.model import EncryptData
from fedgen.generator import GeneratorFramework
import shutil

workers = os.listdir('/home/ubuntu/GenAI-Rush/workers')

def check_data_balance(**kwargs):
	worker = kwargs['client']
	data_dir = f'/home/ubuntu/GenAI-Rush/workers/{worker}'
	samples = dict()
	for label in os.listdir(data_dir):
		samples[label] = len(os.listdir(os.path.join(data_dir, label)))
	samples = sorted(samples.items(), key=lambda x:x[1])
	with open(f'/home/ubuntu/GenAI-Rush/generated/imbalance_{worker}.json', 'w') as file:
		json.dump(samples, file)
	file.close()

def generate_module(**kwargs):
	worker = kwargs['client']
	if f'imbalance_{worker}.json' in os.listdir('/home/ubuntu/GenAI-Rush/generated'):
		with open(f'/home/ubuntu/GenAI-Rush/generated/imbalance_{worker}.json') as file:
			samples = json.load(file)
		file.close()
		max_value = samples[samples[-1]]
		for key, value in samples.items():
			diff = max_value - value
			if diff < 0.6*max_value:
				try:
					os.mkdir(f'/home/ubuntu/GenAI-Rush/generated/{worker}/{key}')
				except:
					pass
				os.mkdir(f'/home/ubuntu/GenAI-Rush/generated/temp_{worker}')
				shutil.move(f'/home/ubuntu/GenAI-Rush/workers/{worker}/{key}', f'/home/ubuntu/GenAI-Rush/generated/temp_{worker}')

				gen = GeneratorFramework(dataroot=f'/home/ubuntu/GenAI-Rush/generated/temp_{worker}', difference=diff)
				gen.process()

				shutil.move(f'/home/ubuntu/GenAI-Rush/generated/temp_{worker}/{key}', f'/home/ubuntu/GenAI-Rush/workers/{worker}/')
				os.rmdir(f'/home/ubuntu/GenAI-Rush/generated/temp_{worker}')
		os.remove(f'/home/ubuntu/GenAI-Rush/generated/imbalance_{worker}.json')

def encrypt_module(**kwargs):
	with open('/home/ubuntu/GenAI-Rush/GenAI-Rush-FedGen/run_config.json') as file:
		run_config = json.load(file)
	file.close()
	with open('/home/ubuntu/GenAI-Rush/GenAI-Rush-FedGen/config.json') as file:
		config = json.load(file)['workers_info'][0]
	file.close()
	worker = kwargs['client']
	categories = len(os.listdir(f'/home/ubuntu/GenAI-Rush/workers/{worker}'))
	processData = PreProcess(data_dir='/home/ubuntu/GenAI-Rush/workers', worker=worker, save_path='/home/ubuntu/GenAI-Rush/processed')
	processData.splitData()
	encrypt_obj = EncryptData(categories=categories, worker=worker)
	loader = CreateLoaders(save_path='/home/ubuntu/GenAI-Rush/processed', batch_size=run_config['batch_size'], 
		max_physical_batch_size=config['MAX_PHYSICAL_BATCH_SIZE'], worker=worker)
	train_loader, _, _ = loader.fetchLoaders()
	encrypt_obj.encrypt(trainDataLoader=train_loader, epochs=run_config['epochs'],
		MAX_GRAD_NORM=config['MAX_GRAD_NORM'], EPSILON=config['EPSILON'], DELTA=config['DELTA'])

def training_module(**kwargs):
	with open('/home/ubuntu/GenAI-Rush/GenAI-Rush-FedGen/run_config.json') as file:
		run_config = json.load(file)
	file.close()
	with open('/home/ubuntu/GenAI-Rush/GenAI-Rush-FedGen/config.json') as file:
		config = json.load(file)['workers_info'][0]
	file.close()
	worker = kwargs['client']
	categories = len(os.listdir(f'/home/ubuntu/GenAI-Rush/workers/{worker}'))
	loader = CreateLoaders(save_path='/home/ubuntu/GenAI-Rush/processed', batch_size=run_config['batch_size'], 
		max_physical_batch_size=config['MAX_PHYSICAL_BATCH_SIZE'], worker=worker)
	train_loader, val_loader, test_loader = loader.fetchLoaders()
	trainer_obj = FedTrainer(worker=worker, categories=categories,
		batch_size=run_config['batch_size'], epochs=run_config['epochs'], MAX_PHYSICAL_BATCH_SIZE=config['MAX_PHYSICAL_BATCH_SIZE'], 
		MAX_GRAD_NORM=config['MAX_GRAD_NORM'], EPSILON=config['EPSILON'], DELTA=config['DELTA'], 
		trainDataLoader=train_loader, valDataLoader=val_loader)
	trainer_obj.runTrainingSession()

def federate_module(**kwargs):
	with open('/home/ubuntu/GenAI-Rush/GenAI-Rush-FedGen/run_config.json') as file:
		run_config = json.load(file)
	file.close()
	with open('/home/ubuntu/GenAI-Rush/GenAI-Rush-FedGen/config.json') as file:
		config = json.load(file)['workers_info'][0]
	file.close()
	worker = kwargs['client']
	categories = len(os.listdir(f'/home/ubuntu/GenAI-Rush/workers/{worker}'))
	federator = SimpleFederatedAveraging(workers=os.listdir('/home/ubuntu/GenAI-Rush/workers'), categories=categories,
		MAX_GRAD_NORM=config['MAX_GRAD_NORM'], EPSILON=config['EPSILON'], DELTA=config['DELTA'], EPOCHS=run_config['epochs'])
	loaders = list()
	for worker in os.listdir('/home/ubuntu/GenAI-Rush/workers'):
		loader = CreateLoaders(save_path='/home/ubuntu/GenAI-Rush/processed', batch_size=run_config['batch_size'], 
			max_physical_batch_size=config['MAX_PHYSICAL_BATCH_SIZE'], worker=worker)
		train_loader, _, _ = loader.fetchLoaders()
		loaders.append(train_loader)
	federator.federateGlobalModel(loaders=loaders)
	federator.federateClientModels()

def test_module(**kwargs):
	with open('/home/ubuntu/GenAI-Rush/GenAI-Rush-FedGen/run_config.json') as file:
		run_config = json.load(file)
	file.close()
	with open('/home/ubuntu/GenAI-Rush/GenAI-Rush-FedGen/config.json') as file:
		config = json.load(file)['workers_info'][0]
	file.close()
	worker = kwargs['client']
	categories = len(os.listdir(f'/home/ubuntu/GenAI-Rush/workers/{worker}'))
	testObj = TestFederated(workers=os.listdir(f'/home/ubuntu/GenAI-Rush/workers/'), categories=categories,
		MAX_GRAD_NORM=config['MAX_GRAD_NORM'], EPSILON=config['EPSILON'], DELTA=config['DELTA'],
		max_physical_batch_size=config['MAX_PHYSICAL_BATCH_SIZE'], save_path='/home/ubuntu/GenAI-Rush/processed', batch_size=run_config['batch_size'])
	testObj.testClientModels()
	testObj.testFederatedClientModels()
	performance = testObj.testFederatedGlobalModel()
	performances.to_csv('/home/ubuntu/GenAI-Rush/metrics.csv', index=None)

dag = DAG(
	dag_id = "trainer",
	schedule_interval=None,	
	dagrun_timeout=timedelta(minutes=60),
    start_date=datetime.datetime(2023, 1, 1)
)

start_task = DummyOperator(
	task_id='start',
	dag=dag
)

data_check_tasks = list()
for idx in range(len(workers)):
	task = PythonOperator(
		task_id=f'check_data_{workers[idx]}',
		python_callable=check_data_balance,
		op_kwargs={'client': workers[idx]},
		dag=dag
	)
	data_check_tasks.append(task)


delay1 =  DummyOperator(
	task_id='delay1',
	dag=dag
)

generate_tasks = list()
for idx in range(len(workers)):
	task = PythonOperator(
		task_id=f'generate_data_{workers[idx]}',
		python_callable=generate_module,
		op_kwargs={'client': workers[idx]},
		dag=dag
	)
	generate_tasks.append(task)

delay2 =  DummyOperator(
	task_id='delay2',
	dag=dag
)

encrypt_tasks = list()
for idx in range(len(workers)):
	task = PythonOperator(
		task_id=f'encrypt_data_{workers[idx]}',
		python_callable=encrypt_module,
		op_kwargs={'client': workers[idx]},
		dag=dag
	)
	encrypt_tasks.append(task)

delay3 =  DummyOperator(
	task_id='delay3',
	dag=dag
)

training_tasks = list()
for idx in range(len(workers)):
	task = PythonOperator(
		task_id=f'train_{workers[idx]}',
		python_callable=training_module,
		op_kwargs={'client': workers[idx]},
		dag=dag
	)
	training_tasks.append(task)


federate_task = PythonOperator(
	task_id='federate',
	python_callable=federate_module,
	op_kwargs={'client': workers[idx]},
	dag=dag
)

test_task = PythonOperator(
	task_id='test',
	python_callable=test_module,
	op_kwargs={'client': workers[idx]},
	dag=dag
)

end_task = DummyOperator(
	task_id='end',
	dag=dag
)

start_task >> data_check_tasks >> delay1 >> generate_tasks >> delay2 >> encrypt_tasks >> delay3 >> training_tasks >> federate_task >> test_task >> end_task