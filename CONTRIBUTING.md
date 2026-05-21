# Contributing

## Development

### Set up your dev environment

The following tools are required:

- [git](https://git-scm.com)
- [python](https://www.python.org) (v3.14+)
- [pip](https://pypi.org/project/pip/) (v23.0+)

You can setup your dev environment using [tox](https://tox.wiki/en/latest/), an environment orchestrator which allows for setting up environments for and invoking builds, unit tests, formatting, linting, etc. Install tox with:

```sh
pip install -r setup_requirements.txt
```

If you want to manage your own virtual environment instead of using `tox`, you can install `vllm_detector_adapter` and all dependencies with:

```sh
pip install .
```

### Unit tests

Unit tests are enforced by the CI system. When making changes, run the tests before pushing the changes to avoid CI issues.

Running unit tests against all supported Python versions is as simple as:

```sh
tox
```

Running tests against a single Python version can be done with:

```sh
tox -e py
```

### Coding style

vllm-detector-adapter follows the python [pep8](https://peps.python.org/pep-0008/) coding style. [FUTURE] The coding style is enforced by the CI system, and your PR will fail until the style has been applied correctly.

We use [pre-commit](https://pre-commit.com/) to enforce coding style using [black](https://github.com/psf/black), [prettier](https://github.com/prettier/prettier) and [isort](https://pycqa.github.io/isort/).

You can invoke formatting with:

```sh
tox -e fmt
```

In addition, we use [pylint](https://www.pylint.org) to perform static code analysis of the code.

You can invoke the linting with the following command

```sh
tox -e lint
```
