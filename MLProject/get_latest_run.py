import mlflow
import os

def main():
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment_ids=['0'], order_by=['attribute.start_time DESC'])
    
    if runs:
        run_id = runs[0].info.run_id
        print(f"Latest Run ID: {run_id}")
        
        github_env = os.environ.get('GITHUB_ENV')
        if github_env:
            with open(github_env, 'a') as f:
                f.write(f"LATEST_RUN_ID={run_id}\n")
    else:
        print("No MLflow runs found.")

if __name__ == "__main__":
    main()
