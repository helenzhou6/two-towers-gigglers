# two-towers-gigglers
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
- docs dataset that has all the unique docs per pandas row saved to data/docs.parquet 
- data dataset that includes query (string), doc (string), and whether it was clicked (0/1), for all queries (we are going to assume these are all positive docs). Saved data/query.parquet
2. (Optional - since the outputs are saved to wandb and those are used later) `create_embeddings.py` that will download the fasttext model and create vocab embeddings. This uploads the output (embeddings) to wandb, which can be accessed instead of running this file. Also uploads the vocab.json to wandb
3. `run_two_towers.py` - this initialises two models (the query model and the doc model). These then get trained jointly - query model gets trained on queries, and doc model gets trained on the positive sample (an entry from query dataset), and a negative sample (what we deem to be a random document)

## Input dataset
- Hugging face datasets: outputs validation, train and test datasets
- Has 82326 rows - for query_id
- Column names: 'answers', 'passages', 'query', 'query_id', 'query_type', 'wellFormedAnswers'


## Brainstorming
Steps/architecture decided in brainstorming session

![Image of architecture](https://github.com/user-attachments/assets/9713a5dc-d4bc-445e-a6fb-79a0049e9265)