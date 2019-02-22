#! /bin/bash
# Install deepai_nlp
cd deepai_nlp
pip install -e .
cd ..
# Install elmo
cd ELMoForManyLangs
python setup.py install
cd ..
