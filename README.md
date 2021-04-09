[![Coverage Status](https://coveralls.io/repos/github/j2kun/pmfp-code/badge.svg?branch=main)](https://coveralls.io/github/j2kun/pmfp-code?branch=main)


```
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

To run the test suite:

```
pytest

# with code coverage
pytest --cov-report html:cov_html  --cov-report annotate:cov_annotate --cov
```
