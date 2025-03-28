#!/bin/bash

set -ex

docker pull node2.bdcl:5000/deep-reason:py3.12

docker run -it \
        -v /data/storage/kg_graphrag:/data \
        node2.bdcl:5000/deep-reason:py3.12 \
        /bin/bash