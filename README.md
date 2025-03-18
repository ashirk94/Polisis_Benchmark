# Polisis_Benchmark

This repository is our effort to reproduce Polisis results for privacy policy classification based on their paper: https://arxiv.org/abs/1802.02561 

Modified by Alan Shirk for Introduction to Privacy-Aware Computing final project.

# Setup instructions
1. Setup a virtual environment using any tool (e.g., pip) and activate it:
```
pip install pipenv
pipenv shell
```
2. Install dependecies from the requirement file: ```pip install -r requirement.txt```
3. install NLTK tokenizer: ```python -m nltk.downloader punkt```
4. Download glove.6B.zip from https://nlp.stanford.edu/projects/glove/ and insert the extracted glove.6B.300d.txt file into the data/glove.6B directory.

# Usage instructions
To run the experiment: ```python -u cnn_multi_label_classifier.py```
