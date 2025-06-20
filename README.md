# two-towers-gigglers

See [presentation.pdf](./resources/presentation.pdf) for architecture etc 

![egg](./resources/egg.gif)

TODO:

- Add loss logging to wandb
- Design evaluations
- Create and test vector DB application

## Dev set up

- Python version 3.10.13
- Install uv python package manager (https://github.com/astral-sh/uv)
- Wandb profile, and added to the right project (see `wandb_init.py` for details)
  - Login by running `wandb login` in the console
  - To look at logging to console run: `wandb.watch(model, log="all", log_freq=100)`

### To run files

1. Run `uv sync` to install all dependencies

- To set the venv in VSCode (on Mac, you can do Shift Command P - you can select the interpreter to be virtual env uv has set up, in .venv)

2. If `export PYTHONPATH=./src` (bash) and create `__init__.py` files for each folder (should already be there)

### Scripts to run

1. `process_bing_dataset.py` will process the MS/Marcho dataset to create two datasets:
- Input dataset info:
   - Hugging face datasets: outputs validation, train and test datasets
   - Has 82326 rows - for query_id
   - Column names: 'answers', 'passages', 'query', 'query_id', 'query_type', 'wellFormedAnswers'
- docs dataset that has all the unique docs per pandas row saved to data/docs_processed.parquet (that is the tokenized docs) and also data/docs.parquet (which is untokenized i.e. words) & uploaded to wandb
- data dataset that includes query (string), doc (string), and whether it was clicked (0/1), for all queries (we are going to assume these are all positive docs). Saved data/query.parquet & uploaded to wandb

2. (Optional - since the outputs are saved to wandb and those are used later) `create_embeddings.py` that will download the fasttext model and create vocab embeddings. This uploads the output (embeddings) to wandb, which can be accessed instead of running this file. Also uploads the vocab.json to wandb
3. `train_two_towers.py` - this initialises two models (the query model and the doc model). These then get trained jointly - query model gets trained on queries, and doc model gets trained on the positive sample (an entry from query dataset), and a negative sample (what we deem to be a random document)
4. `evaluate.py` - this will evaluate a specified model on the validation dataset and will return the [mean average precision](<https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision>) score. Example usage: `uv run src/evaluate.py --query_model_artifact query_model:latest --doc_model_artifact doc_model:latest`. This requires the validation set to be downloaded (`uv run src/process_bing_dataset.py --split validation`)
5. `inference.py` - this will run the models (based on latest), and make an inference based on a query (string) being passed through (either via running the test file, or through the frontend).

### Training

To train our two tower model run:

```bash
uv run src/train_two_towers.py
```

This will do the run with default hyper params and no evaluation.

#### Evaluate in the Training Run after each run

```bash
uv run src/train_two_towers.py --evaluate
```

This will run an evaluation of the model to discover how accurate it is being with the "mean average precision" calculation done above.

#### Run training across multiple hyperparams with Wandb sweeps

```bash
uv run src/train_two_towers.py --sweep
```

This will run a sweep of various hyper params without the slightly costly evaluation step but it is usually best to combine sweeps and evaluation such that you can review which models performed the best.

```bash
uv run src/train_two_towers.py --sweep --evaluate
```

You should see logged in Wandb data for trainings that can give you a view into model performance and what improved the model over time.

### Running the API & front end (without docker)
1. Run API with `uvicorn src.api:app --reload` (ensure `export PYTHONPATH=./src` has been run)
   - To check it is working, go to http://127.0.0.1:8000/health-check which should return a message
   - And to test the API with a query you can pass, run `./src/tests/test_api.sh` (you can alter the query here)
2. Whilst ensuring the API is running, in another terminal run: `streamlit run src/streamlit_app.py`, where you can access the interface at: http://localhost:8501/

## Running with docker compose
1. Ensure .env is setup, with WANDB_API_KEY that is copied from wandb website
2. Start up Docker (e.g. `colima start`).
3. Run `docker-compose up --build` to build all the docker containers etc
   - If you want to cd into a specific docker container, go `docker container ls` to find the container id, then ` docker exec -it <container id> /bin/bash`

## Running redis database etc on little laptops (aka Helen's laptop)
1. Change the Dockerfile.api - comment out api and frontend (not needed)
2. Do `docker-compose up --build` to spin up all the redis database stuff & keep that running in the terminal
3. In another terminal - ensure `export REDIS_HOST=localhost` & `export PYTHONPATH=./src` and run `python3 src/index_docs_to_redis.py` and then `uvicorn src.api:app`
   - If memory (error code 137) runs out, then consider altering `python3 src/index_docs_to_redis.py` to only do `.head()` on the doc and doc_processed dataframes
   - If you don't need to test the frontend, then run the test_inference.py file
4. In another terminal - ensure `export API_URL=http://localhost:8000` & `export PYTHONPATH=./src` and run `streamlit run src/streamlit_app.py`

## Brainstorming

Steps/architecture decided in brainstorming session

![Image of architecture](https://github.com/user-attachments/assets/9713a5dc-d4bc-445e-a6fb-79a0049e9265)
