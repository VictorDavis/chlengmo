# Character-Level N-Gram Model

[![Unit Tests](https://github.com/VictorDavis/chlengmo/actions/workflows/main.yml/badge.svg)](https://github.com/VictorDavis/chlengmo/actions/workflows/main.yml)

**Chlengmo**: ("Kling-mo") A lean, simple, _fast_ character-level n-gram model. Faster than [NLTK](https://www.nltk.org/api/nltk.lm.html). Zero dependencies.

## Sample Usage

```python

# retrieve corpus from NLTK
# NOTE: NLTK is not a dependency, only using it to fetch a training corpus.
filename = "melville-moby_dick.txt"
text = gutenberg.raw(filename)
start = "Call me Ishmael"
start_idx = text.index(start)
text = text[start_idx:]

# 15-gram model
model = Chlengmo(n=15).fit(text)
model.generate(length=981, prompt="Call me ", seed=42)
```

> Call me Ishmael . Some years ago -- never mind how long precisely -- who knows ? Certain I am , however , the sperm whale \' s food ; and , also , calling to mind the regular , ascertained seasons for hunting him in particular that , in the internal parts of the vessel ; the becharmed crew maintaining the profoundest homage ; yea , an all - abounding adoration ! for almost all the tapers , lamps , and candles that burn round the globe , by girdling it with guineas , one to every three parts of an inch ; stabbing him in the ventricles of his heart . He was in Radney the chief mate , said ,--" Take the rope , sir -- I give it into thy hands , Starbuck watched the Pequod \' s jaw - bone tiller had several times descended from heaven by the way of this fire - ship on the sea . A short space elapsed , and up into this noiselessness came Ahab alone from his cabin . He was a small , short , youngish man , sprinkled all over his face with freckles , and wearing redundant yellow hair .

## Unit Tests

```bash
# from inside the tests/ subfolder
cd tests/

# create virtual environment
virtualenv .venv
. .venv/bin/activate

# install unit test dependencies
python3 -m pip install -r requirements.txt

# run unit tests
pytest

# run coverage report
coverage run -m pytest
coverage report
coverage html
open htmlcov/index.html

# cleanup
deactivate
rm -rf .venv
```

## Notebooks

REF: https://janakiev.com/blog/jupyter-virtual-envs/

```bash
# from inside the notebooks/ subfolder
cd notebooks/

# create virtual environment
virtualenv .venv
. .venv/bin/activate

# install notebook dependencies
python3 -m pip install -r requirements.txt

# create jupyter kernel
python3 -m pip install ipykernel
python3 -m ipykernel install --user --name=.venv

# exit virtual environment & run jupyter
deactivate
PYTHONPATH=../ jupyter notebook

# cleanup
rm -rf .venv
```
