# Federated IoT Lab – UCI HAR with Flower & PyTorch

Federated learning playground that emulates five non-IID Human Activity Recognition (HAR) clients collaborating through a Flower server. Each client trains a lightweight PyTorch MLP on a per-user shard of the (pre-processed) UCI HAR dataset, packaged for quick experimentation on resource-constrained IoT setups.

## Project Layout
- `server/fl_server.py` – Flower FedAvg server with centralized evaluation.
- `client/fl_client.py` – Flower NumPyClient wrapper executing local PyTorch training.
- `shared/model.py` – Compact MLP classifier.
- `shared/utils.py` – Data loading, normalization, metric helpers, and model parameter utilities.
- `shared/data/client*.csv` – Non-IID partitions (one synthetic HAR-style user per client).
- `client/Dockerfile` – Base image for both server and clients.
- `docker-compose.yml` – Spins up the server plus five clients, sharing the `/app/shared` volume.
- `requirements.txt` – Python dependencies.

## Quick Start (Docker Compose)
1. Build and launch all containers:
   ```bash
   docker-compose up --build
   ```
2. Monitor the Flower server logs (`fl-server`) for round progression and aggregated metrics.
-> 실행하고 나면 /server/fl_training_log.csv에 round,loss,accuracy가 생김

3. Shut down when finished:
   ```bash
   docker-compose down
   ```


All containers mount the `shared` directory, so updated models or data instantly propagate across the federation.

## Local Execution (no Docker)
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Start the Flower server:
   ```bash
   python server/fl_server.py --address 127.0.0.1:8080 --rounds 5
   ```
3. In separate terminals, launch each client (adjust `--client-id`):
   ```bash
   python client/fl_client.py --client-id 1 --server-address 127.0.0.1:8080
   ```

## Dataset Notes
- The `shared/data` CSV files contain a compact HAR-like feature set (10 representative signals + label) carved into five non-IID shards—each shard is skewed toward one activity to simulate user-specific data.
- Labels follow the original UCI HAR mapping (1–6) but are zero-indexed internally for PyTorch.
- Normalization happens per-client at load time; centralized evaluation re-normalizes the combined test split for fairness.

### Regenerating from the Full UCI HAR Dataset
To swap in the complete dataset:
1. Download the *Human Activity Recognition Using Smartphones Data Set* from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones).
2. Extract the archive into `shared/raw/`.
3. Adapt a preprocessing script (e.g., extend `shared/utils.py`) to:
   - Merge the `train`/`test` splits,
   - Filter the desired sensor features,
   - Assign one subject per client shard, and
   - Export CSV files alongside the provided samples.
4. Replace the existing `client*.csv` files before starting the federation.

## Customisation Tips
- Adjust local training hyperparameters in `server/fl_server.py` (`fit_config_fn`) or override via Flower config.
- Swap `ActivityMLP` for more complex architectures while keeping the lightweight footprint in mind.
- Extend `docker-compose.yml` with GPU-enabled images or additional clients—Flower automatically scales federation rounds once availability thresholds are met.

Enjoy experimenting with federated HAR on modest hardware! Pull requests and improvements are welcome.
