## Running the SHAP notebook with Docker

```bash
docker build -t dml:1.0 .
docker run --rm -p 8888:8888 -p 5000:5000 dml:1.0
```

Once the container is running, open in your browser:

- MLflow UI (browse the logged runs): http://localhost:5000. Look for the runs inside `first_event_cap365`. Each run in that experiment corresponds to a trained neural network.
By clicking on a run you can copy it's run id and paste it into the shap notebook and run it, in order to run the SHAP analysis for that neural network.
- JupyterLab (run `notebooks/shap.ipynb` from here): http://localhost:8888

If a port is already in use, change the left-hand number, for example `-p 8889:8888`. Stop the container with Ctrl+C.
