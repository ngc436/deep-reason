#!/bin/bash

docker volume create -d local  --name es-data-vol\
    --opt device="/data/storage/kg-exps/es-data-vol" \
    --opt type="none" \
    --opt o="bind"

