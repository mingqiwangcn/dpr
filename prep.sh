conda create -y --name dpr
conda activate dpr

conda install -y -c pytorch faiss-gpu
pip install -r requirements.txt
pip list --format=freeze | grep 'nvidia-cublas-cu*' | xargs pip uninstall -y
python -m spacy download en_core_web_sm

sudo apt update
sudo apt install -y wget gcc-8 unzip libssl1.0.0 software-properties-common
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt-get install -y --only-upgrade libstdc++6

