# two-towers-gigglers

## Dev set up
- Python version 3.9.3
- Install uv python package manager (https://github.com/astral-sh/uv)

### To run files
1. Run `uv sync` to install all dependencies
2. To run a file: `uv run main.py`  - this should print a statement
- To set the venv in VSCode (on Mac, you can do Shift Command P - you can select the interpreter to be virtual env uv has set up, in .venv) 
3. If `export PYTHONPATH=./src` (bash) and create `__init__.py` files for each folder (should already be there)


### Scripts to run
- `process_bing_dataset.py` will create data/docs.parquet and data/query.parquet 

## Input dataset
- Hugging face datasets: outputs validation, train and test datasets
- Has 82326 rows - for query_id
- Column names: 'answers', 'passages', 'query', 'query_id', 'query_type', 'wellFormedAnswers'


## Brainstorming
Steps/architecture decided in brainstorming session

![Image of architecture](https://github.com/user-attachments/assets/9713a5dc-d4bc-445e-a6fb-79a0049e9265)