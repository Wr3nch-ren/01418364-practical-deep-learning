# 01418364-practical-deep-learning
my private repository to contains jupyter notebook files for machine learning

# Installing Python
Install python 3.0 using Anaconda
https://www.anaconda.com/download

# Setup
Open Anaconda Prompt and run these following commands

```
conda create --name dl_env python=3.10

conda activate dl_env

conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0



# Anything above 2.10 is not supported on the GPU on Windows Native

python -m pip install "tensorflow<2.11"



# Verify Tensorflow GPU installation:

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"



pip install Pillow scikit-learn numpy pandas matplotlib pandas seaborn missingno graphviz imutils notebook tensorflow_datasets==4.9.2

pip install --upgrade plotly

pip install --upgrade ipykernel ipyflow



ipython kernel install --user --name=dl_env

python -m ipykernel install --user --name dl_env --display-name dl_env



conda install -y -c conda-forge jupyter_contrib_nbextensions jupyter_nbextensions_configurator

jupyter contrib nbextension install --user

jupyter nbextensions_configurator enable --user



pip install notebook==6.1.5

jupyter nbextension enable scroll_down/main

jupyter nbextension enable autoscroll/main

jupyter nbextension enable execute_time/ExecuteTimes
```

### To run jupyter notebook, use this following command
```
jupyter notebook
```
### or
```
python -m notebook
```