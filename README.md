# hls4ml_adsb

## system req
- ubuntu 20.04 
- vivado v2020.1

## how-to

- Run "docker build -t adsb_pynqz2 -f docker/Dockerfile ." in root dir to build docker

- Run "docker run -p 8888:8888 -v /path/to/your/local/directory:/home/jovyan/work -it adsb_pynqz2" to run docker, replace "/path/to/your/local/directory" with your own path


