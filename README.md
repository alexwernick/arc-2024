# arc_2024
This project is a solver for puzzles from the [ARC](https://github.com/fchollet/ARC-AGI). The code was written to solve the tasks in the [2014 Kaggle competition](https://www.kaggle.com/competitions/arc-prize-2024). This particular solution uses Induction Logic Programming to construct defintions in First Order Logic using the an implementation of the [FOIL algorithmn](https://link.springer.com/content/pdf/10.1007/BF00117105.pdf). For more on these topics, check out Artificial Intelegence - A Modern Approach by Stuart Russell & Peter Norvig, 4th Edition with particular attention to Chapter 20 - Knowledge in Learning

## Overview
The code is best understood starting at [arc_2024/runner.py](arc_2024/runner.py). The function `run` is called from [arc_2024/main.py](arc_2024/main.py) if running localy or from the Juypter Notebook [arc_2024/arcprize_2024.ipynb](arc_2024/arcprize_2024.ipynb) when running on Kaggle. The runner has three main parts or interest.The `DataManager` will not be explained here but is simply an abstraction layer to deal with reading and writing the Competetion json data files:

1. Using the `Interpreter` to generate `Shape` objects that are contained in the Inputs of the ARC tasks. These shapes attempt the allow the solver to 'see' the problems as a human would, in shapes. Later these shapes can be used to build backgound knowledge realting to a set of Predicates used in FOIL. More on that later...
2. Using the `GridSizeSolver` to solve the output grid sizes for a certain task. This class constructs `Predicates` and background knowledge that is then fed to the FOIL algorithmn.
3. Using the result of the `GridSizeSolver`, the `Solver` similarly constructs `Predicates` and background knowledge that is then fed to the FOIL algorithmn. This time the range of Predicates is much larger in scope to handle the huge amount of variance seen in the ARC tasks. In fact the range of Predicates here is too limited to solve the majority of ARC tasks but serves a good examples as how this kind of approach can be used on a subset of tasks.


---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

To get started with this project, make sure you have [Poetry](https://python-poetry.org/docs/#installation) installed.

### Clone the Repository

```bash
git clone https://github.com/your-username/your-project-name.git
cd your-project-name
```

### Install Dependencies
```bash
poetry install
```

This command will create a virtual environment, if one doesn’t already exist, and install all dependencies specified in pyproject.toml.

## Usage
Run the application with:

```bash
poetry run python your_script.py
```

Alternatively, activate the virtual environment using:
```bash
poetry shell
```
Then, you can run the application or use other commands directly.

## Development
To add or update dependencies, use Poetry’s dependency management commands:

- Add a Dependency: `poetry add <package-name>`
- Add a Development Dependency: `poetry add --dev <package-name>`

### Updating Dependencies
To update dependencies, you can run:

```bash
poetry update
```
This will update all dependencies to the latest versions within the constraints specified in `pyproject.toml`.

## Testing
This project uses `pytest` for testing. To run tests, execute:

```bash
poetry run pytest
```

Or, if you have activated the Poetry shell, just run:

```bash
pytest
```

## Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix (git checkout -b feature-name).
3. Make your changes.
4. Commit your changes (git commit -am 'Add new feature').
5. Push to the branch (git push origin feature-name).
6. Create a pull request.

Please ensure that your code adheres to the coding standards (use the pre commit checks) and is well-documented.