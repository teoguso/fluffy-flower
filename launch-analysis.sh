#!/usr/bin/env bash
set -xe
docker build . -t returning
#docker run -v $PWD/data:/home/returning/data -v $PWD/models:/home/returning/models returning
docker run -it returning
