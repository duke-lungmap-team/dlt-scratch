# lungmap-scratch
*anything anyone wants to work on*

![mail](images/lungmap.png)


## Environments

#### Native - On my machine
>To complete all work, I'm unsing Anaconda python with the following descritpors:

platform : osx-64
conda version : 4.1.0
conda-build version : 1.19.0
python version : 3.5.1.final.0
requests version : 2.10.0

With this native Anaconda install, I create a Conda environment just for this project,
called `opencv3`. To recreate this environment, follow these steps:

```
#Python 3 and openCV3
conda create -n opencv3 numpy scipy scikit-learn matplotlib python=3
source activate opencv3
conda install -c https://conda.binstar.org/menpo opencv3
conda install spyder
```
**Note**: I don't use the usual conda install library due to this [issue](https://github.com/conda/conda/issues/2448).

#### Jupyter
>Using the `Dockerfile` in this repository, you can recreate our Jupyter environment that we are using, which is being hosted here (TBD).
