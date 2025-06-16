# two-towers-gigglers

## Dev set up
- Python version 3.9.3
- Install uv python package manager (https://github.com/astral-sh/uv)

### To run files
1. Run `uv sync` to install all dependencies
2. To run a file: `uv run main.py`  - this should print a statement
- To set the venv in VSCode (on Mac, you can do Shift Command P - you can select the interpreter to be virtual env uv has set up, in .venv) 
3. If `export PYTHONPATH=./src` (bash) and create `__init__.py` files for each folder (should already be there).