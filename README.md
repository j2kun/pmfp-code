# Practical Math for Programmers

[![CircleCI](https://circleci.com/gh/j2kun/pmfp-code.svg?style=shield)](https://circleci.com/gh/j2kun/pmfp-code)
[![Coverage Status](https://coveralls.io/repos/github/j2kun/pmfp-code/badge.svg?branch=main)](https://coveralls.io/github/j2kun/pmfp-code?branch=main)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/j2kun/pmfp-code.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/j2kun/pmfp-code/context:python)

This is the source code respository for the book, Practical Math for Programmers.

## Running tests

Requires at least Python 3.9.

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
