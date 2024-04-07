

=== USAGE INSTRUCTIONS #2 ===

If you don't think you have the dependencies installed then Pipenv can be used to create a fresh environment with them installed. Install Pipenv by following the instructions here: https://pipenv.pypa.io/en/latest/

Having done so, open a terminal and navigate to the directory containing the Pipfile and Pipfile.lock and run:

> pipenv install

If you don't have a recent Python 3 available you can execute the following instead:

> pipenv install --three

It may take a little while to populate your fresh environment with depencies. Once this has completed, you can run the following:

> pipenv run jupyter notebook

A browser window should open. Navigate to the file 'challenge.ipynb' for further instructions.
