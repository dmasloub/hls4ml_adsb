# hls4ml_adsb

## system req
- ubuntu 20.04 
- vivado v2020.1

## how-to

- make sure system requirements are met 
- create virtualenv, install libraries from requirements.txt
- import data into data folder, seperate into test, train, validation data
- run create create_dirs.py to create dirs for models
- run train_model.py to train keras model 
- run convert_hls.py to convert keras model to hls
