#! /bin/bash

python ../algorithms/4_algorithms_AG.py 1 0 '../datasets/colon_cancer(train).csv' '../datasets/colon_cancer(test).csv'
python ../algorithms/4_algorithms_AG.py 3 0 '../datasets/colon_cancer(train).csv' '../datasets/colon_cancer(test).csv'

python ../algorithms/4_algorithms_AG.py 1 0 '../datasets/madelon(train).csv' '../datasets/madelon(test).csv'
python ../algorithms/4_algorithms_AG.py 3 0 '../datasets/madelon(train).csv' '../datasets/madelon(test).csv'

python ../algorithms/4_algorithms_AG.py 1 0 '../datasets/PCMAC(train).csv' '../datasets/PCMAC(test).csv'
python ../algorithms/4_algorithms_AG.py 3 0 '../datasets/PCMAC(train).csv' '../datasets/PCMAC(test).csv'