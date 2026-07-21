import mlflow
import os
import sys

def main():
    workspace = os.environ.get('GITHUB_WORKSPACE', os.getcwd())
    db_path = os.path.join(workspace, 'mlflow.db')
    tracking_uri = f"sqlite:///{db_path}"
    
    print(f"Connecting to MLflow Tracking URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    
    all_experiments = client.search_experiments()
    exp_ids = [e.experiment_id for e in all_experiments]
    print(f"Found experiment IDs: {exp_ids}")
    
    runs = client.search_runs(experiment_ids=exp_ids, order_by=['attribute.start_time DESC']) if exp_ids else []
    
    if runs:
        run_id = runs[0].info.run_id
        print(f"Latest Run ID: {run_id}")
        
        github_env = os.environ.get('GITHUB_ENV')
        if github_env:
            with open(github_env, 'a') as f:
                f.write(f"LATEST_RUN_ID={run_id}\n")
    else:
        print("Error: No MLflow runs found!")
        sys.exit(1)

if __name__ == "__main__":
    main()
