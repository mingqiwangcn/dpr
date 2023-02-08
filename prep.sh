python3.7 -m venv ../pyenv/dpr
source ../pyenv/dpr/bin/activate
pip install .
pip uninstall -y nvidia-cublas-cu11
python -m spacy download en_core_web_sm

