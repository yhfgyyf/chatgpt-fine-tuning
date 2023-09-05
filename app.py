import openai
import os
import  time
from dotenv import load_dotenv, find_dotenv
_ =load_dotenv(find_dotenv())

openai.api_key = os.environ.get("OPENAI_API_KEY")
openai.api_base = os.environ.get("OPENAI_API_BASE")
file_path =os.environ.get("FILE_PATH")

#upload file
file=open(file_path,"r",encoding="utf-8") 
training_file=openai.File.create(file=file, purpose="fine-tune")
status = openai.File.retrieve(training_file.id).status

start_time=time.time()
while status != "processed":
    print(f"Uploading file,Status:{status}... {time.time() - start_time:.2f}s", end="\r", flush=True)
    time.sleep(5)
    status = openai.File.retrieve(training_file.id).status 
print(f"File {training_file.id} uploaded after {time.time() - start_time:.2f} seconds.")

job=openai.FineTuningJob.create(
    training_file=training_file.id,
    model="gpt-3.5-turbo-0613",
    n_epochs=2
)

#training
job_status = openai.FineTuningJob.retrieve(job.id).status
start_time = time.time()
while job_status != "succeeded":
   print(f"fine-tuning,Status:{job_status}... {time.time() - start_time:.2f}s", end="\r", flush=True)
   time.sleep(5)
   job = openai.FineTuningJob.retrieve(job.id)
   job_status = job.status
print(f"Job {job.id} finished after {time.time() - start_time:.2f} seconds.")

model_name = job.fine_tuned_model
print(f"Model name: {model_name}")
