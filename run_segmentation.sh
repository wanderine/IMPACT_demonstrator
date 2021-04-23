#!/bin/bash

#docker run --rm --gpus '"device=0"' -v /raid/andek67/IMPACT_demonstrator/testsubject:/in -v /raid/andek67/IMPACT_demonstrator/testoutput:/out kerasdicom

docker run --rm --gpus '"device=0"' -v /raid/andek67/IMPACT_demonstrator/testsubject_T1GD_qMRIGD:/in -v /raid/andek67/IMPACT_demonstrator/testoutput:/out kerasdicom

