#! /bin/bash

#Script Configuration
source .env

docker build -t $IMAGE .
