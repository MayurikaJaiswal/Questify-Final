
![1](https://github.com/MayurikaJaiswal/Questify-Final/assets/114226673/f6549fd2-502e-422c-867f-363753bb6a3b)

### install these after installing miniconda

1. create environment
```shell
conda create -n py39 python=3.9
```

2. install llama-cpp
```shell
conda install -c conda-forge llama-cpp-python=0.2.24 -y
```

3. install remaining requirements
```shell
pip install -r requirements.txt
```

4. run streamlit
```
streamlit run app.py
```
