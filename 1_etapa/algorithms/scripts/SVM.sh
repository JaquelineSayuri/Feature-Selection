#! /bin/bash

python ../algorithms/4_algorithms_AG.py 1 2 '../datasets/colon_cancer(train)stand.csv' '../datasets/colon_cancer(test)stand.csv'
python ../algorithms/4_algorithms_AG.py 3 2 '../datasets/colon_cancer(train)stand.csv' '../datasets/colon_cancer(test)stand.csv'

python ../algorithms/4_algorithms_AG.py 1 2 '../datasets/madelon(train)stand.csv' '../datasets/madelon(test)stand.csv'
python ../algorithms/4_algorithms_AG.py 3 2 '../datasets/madelon(train)stand.csv' '../datasets/madelon(test)stand.csv'

python ../algorithms/4_algorithms_AG.py 1 2 '../datasets/PCMAC(train)stand.csv' '../datasets/PCMAC(test)stand.csv'
python ../algorithms/4_algorithms_AG.py 3 2 '../datasets/PCMAC(train)stand.csv' '../datasets/PCMAC(test)stand.csv'