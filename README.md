# billybank

# Setting upo the environment (do this first)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

# To Run the dataset generator (while .venv is activated)
```bash
python3 generator.py
```

# To Run ipynb (while .venv is activated) 
```bash
python -m ipykernel install --user --name=billybank --display-name="Python (billybank)"
jupyter notebook

#Select Analysis.ipynb and select kernel - Python (billybank)
```