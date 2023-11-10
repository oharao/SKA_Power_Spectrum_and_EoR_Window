FROM fdulwich/oskar-python3:2.8.3

RUN apt-get update
RUN apt-get install -y git make g++ gcc python wget python3-pip

RUN apt-get -y install build-essential libgsl-dev libfftw3-dev

RUN pip3 install Datetime==4.5 h5py==3.7.0 logger==1.4 noise==1.2.2 pandas==1.3.5 scipy matplotlib==3.4.3
RUN pip install python-casacore healpy git+https://github.com/telegraphic/pygdsm 21cmFAST scikit-rf joblib
RUN pip install -U tools21cm