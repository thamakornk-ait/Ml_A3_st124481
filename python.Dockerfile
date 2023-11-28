FROM akraradets/ait-ml-base:2023

RUN pip3 install --upgrade pip
RUN pip3 install ipykernel
RUN pip3 install scikit-learn
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install mlflow
RUN pip install seaborn
RUN pip install ppscore
RUN pip install dash

COPY ./code /root/code
CMD tail -f /dev/null