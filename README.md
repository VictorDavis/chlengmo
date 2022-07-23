# chlengmo

## Character-Level N-Gram Model

Chlengmo: ("Kling-mo") A lean, simple, _fast_ character-level n-gram model. Faster than [NLTK](https://www.nltk.org/api/nltk.lm.html). Zero dependencies.

## Sample Usage

```python
# 15-gram model trained on Moby Dick
text = "Call me Ishmael..."
Chlengmo(n=15).fit(text).generate(length=981, prompt="Call me ", seed=42)
```

> Call me Ishmael . Some years ago -- never mind how long precisely -- who knows ? Certain I am , however , the sperm whale \' s food ; and , also , calling to mind the regular , ascertained seasons for hunting him in particular that , in the internal parts of the vessel ; the becharmed crew maintaining the profoundest homage ; yea , an all - abounding adoration ! for almost all the tapers , lamps , and candles that burn round the globe , by girdling it with guineas , one to every three parts of an inch ; stabbing him in the ventricles of his heart . He was in Radney the chief mate , said ,--" Take the rope , sir -- I give it into thy hands , Starbuck watched the Pequod \' s jaw - bone tiller had several times descended from heaven by the way of this fire - ship on the sea . A short space elapsed , and up into this noiselessness came Ahab alone from his cabin . He was a small , short , youngish man , sprinkled all over his face with freckles , and wearing redundant yellow hair .

## Unit Tests

```bash
# create virtual environment
virtualenv .venv
. .venv/bin/activate

# install unit test dependencies
python3 -m pip install -r tests/requirements.txt

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
