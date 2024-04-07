# Practical Math for Programmers

[![Coverage Status](https://coveralls.io/repos/github/j2kun/pmfp-code/badge.svg?branch=main)](https://coveralls.io/github/j2kun/pmfp-code?branch=main)

This is the source code respository for the book, Practical Math for Programmers.

## Running tests

Requires at least Python 3.10.

```
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

To run the test suite:

```
pytest

# with code coverage
pytest --cov-report html:cov_html  --cov-report term-missing --cov
```
