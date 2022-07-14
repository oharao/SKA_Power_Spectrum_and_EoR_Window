FROM fdulwich/oskar-python3:2.8.3

RUN apt-get update && \
    apt-get install -y python3-pip

RUN pip3 install Datetime==4.5 h5py==3.7.0 logger==1.4 noise==1.2.2 pandas==1.3.5 scipy==1.7.3
RUN pip install python-casacore