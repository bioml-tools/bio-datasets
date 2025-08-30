# Contributing to Bio Datasets

First off: **Thank you** for considering a contribution to **datasets-bio**!

## Getting Started & Development Setup

1. Fork the repository on GitHub by clicking the "Fork" button at the top right of the page.
2. Clone your forked repository to your local machine.
3. Create a new branch for your feature or bug fix.
4. Install the required dependencies (including development dependencies) using:
    - If you don't have Poetry installed, follow one of the installation options in:
      https://python-poetry.org/docs/
    - Then, run:
    ```bash
    poetry install --with dev
    ```
5. Build and cache local chemistry reference data from the PDB Chemical Component Dictionary (CCD):
    ```bash
    python setup_ccd.py
    ```
    It will take a few minutes to download and process the data.
6. Verify your installation by running the test suite:
   ```bash
   pytest
   ```
   All tests should pass without errors.
   ```
7. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

You are now ready to create a branch and start making commits! When ready, push your branch to your
fork and open a pull request against the main repository.
