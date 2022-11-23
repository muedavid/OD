#!/bin/bash

IP=10.5.220.249
HOST=david@${IP}
BASE_PATH=/home/david/SemesterProject/OD
DIR1=data_processing
DIR2=losses
DIR3=metrics
DIR4=models
DIR5=plots
DIR6=utils
FILE1=edge.ipynb
pw=IhD999777

sshpass -p $pw scp -r ${BASE_PATH}/${DIR1} ${HOST}:${BASE_PATH}/
sshpass -p $pw scp -r ${BASE_PATH}/${DIR2} ${HOST}:${BASE_PATH}/
sshpass -p $pw scp -r ${BASE_PATH}/${DIR3} ${HOST}:${BASE_PATH}/
sshpass -p $pw scp -r ${BASE_PATH}/${DIR4} ${HOST}:${BASE_PATH}/
sshpass -p $pw scp -r ${BASE_PATH}/${DIR5} ${HOST}:${BASE_PATH}/
sshpass -p $pw scp -r ${BASE_PATH}/${DIR6} ${HOST}:${BASE_PATH}/
sshpass -p $pw scp -r ${BASE_PATH}/${FILE1} ${HOST}:${BASE_PATH}/
