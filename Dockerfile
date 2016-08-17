FROM jupyter/datascience-notebook
MAINTAINER Ben Neely <nigelneely@gmail.com>

RUN conda install -c https://conda.binstar.org/menpo opencv3
