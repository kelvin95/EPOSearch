#!/usr/bin/env sh
poetry init
poetry run pip install -r pipenv_requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
# install jupyter kernel for project
poetry run python -m ipykernel install --user --name gradmtl
