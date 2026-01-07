import kfp
from kfp import dsl
import logging
import os

#Fetch the AWS keys from environment variables

aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

# Check if the environment variables are set

if aws_access_key_id and aws_secret_access_key:
    logging.info("AWS Access Key and Secret Key have been retrieved successfully.")
    logging.info("AWS Access Key ID: %s", aws_access_key_id)
    logging.info("AWS Secret Access Key: %s",
        aws_secret_access_key[:4] + "*" * 16 + aws_secret_access_key[-4:])
else:
    raise EnvironmentError("AWS Access Key or Secret Key not set properly.")

@dsl.component(packages_to_install=['boto3', 'pandas', 'scikit-learn', 'numpy'])
def build_model(aws_access_key_id: str, aws_secret_access_key: str):
    #import libraries inside the component
    import boto3
    import pandas as pd
    import pickle
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    # from sklearn.metrics import mean_squared_error

    bucket_name = 'loan-default-ml-bucket'
    data_s3_path = 'data/balanced_logistic_dataset.csv'
    model_s3_path = 'model/loan_default_model.pkl'
    local_data_path = '/balanced_logistic_dataset.csv'
    local_model_path = 'loan_default_model.pkl'
 
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    # Download dataset from S3
    s3_client.download_file(bucket_name, data_s3_path, local_data_path)

    # Load dataset
    targetDF = pd.read_csv(local_data_path)

    #Prepare the data for training
    X = targetDF[['Age', 'Salary', 'CreditScore', 'LoanAmount', 'Tenure']].values
    y = targetDF["LoanDefault"].values

    # Split the data into training and testing sets
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.20)

    # Train the Logistic Regression model
    lr = LogisticRegression()
    model = lr.fit(X_train, y_train)

    # Save the trained model to a file
    with open(local_model_path, 'wb') as f:
        pickle.dump(model, f)

    # Upload the model file to S3
    s3_client.upload_file(local_model_path, bucket_name, model_s3_path)

@dsl.pipeline
def loan_default_pipeline():
    build_model(aws_access_key_id=aws_access_key_id,aws_secret_access_key=aws_secret_access_key)
        
#Main execution
if __name__ == "__main__":
    #kfp.compiler.Compiler().compile(pipeline_func=loan_default_pipeline,package_path="loan_default_pipeline.yaml" )
    
    kfp_endpoint = None
    client = kfp.Client(host=kfp_endpoint)

    # Experiment name
    experiment_name = "Loan_Default_Prediction_Experiment"

    #List all Experiments
    experiments = client.list_experiments()

    #Search for the experiment by name
    experiment = next((exp for exp in experiments.experiments if exp.display_name == experiment_name), None)
       
    if experiment:
        #if experiment exists, get the experiment ID
        experiment_id = experiment.experiment_id
        logging.info(f"Found Experiment {experiment_name}, Experiment ID: {experiment_id}")
    else:
        # If the experiment does not exist, create it
        logging.info(f"Experiment '{experiment_name}' not found. Creating a new experiment.")
        experiment = client.create_experiment(experiment_name)
        experiment_id = experiment.experiment_id
        logging.info(f"Created new Experiment, Experiment ID: {experiment_id}")

    # List and delete existing runs if any exist
    list_runs = client.list_runs(experiment_id=experiment_id)
    if list_runs.runs:
        previous_run_id = list_runs.runs[0].run_id
        logging.info(f"Deleting previous run with ID: {previous_run_id}")
        #Delete the previous run
        client.delete_run(previous_run_id)
    else:
        logging.info("No previous runs found to delete.")

    try:
        client.create_run_from_pipeline_func(
            loan_default_pipeline,
            experiment_name=experiment_name,
            enable_caching=False
        )
        logging.info(f"Pipeline run initiated")
    except Exception as e:
        logging.error(f"Failed to start pipeline run: {e}")