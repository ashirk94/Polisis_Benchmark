#!/usr/bin/env python
# coding: utf-8

# # Example of a Convolutional Neural Network (Multilabel) for Text Classification
import nltk
nltk.data.path.append("C:/Users/alan/AppData/Roaming/nltk_data")

# In[1]:

#Imports needed from pytorch
import torch
from torch.utils.data import Dataset
from collections import OrderedDict


#Some built-in imports
import matplotlib.pyplot as plt
import numpy as np
import pickle
from os.path import join, isfile
from os import listdir

#Imports from the repository
from data_processing import get_weight_matrix, get_tokens
import data_processing as dp
from privacy_policies_dataset import PrivacyPoliciesDataset as PPD
from cnn import CNN


# If this is not the first time you run the code and you are using the same embeddings and the same dataset then you can skip several cells and just run cell 5 to load the weights matrix, cell 7 to load labels and cell 10 to load datasets. Once you have run these cells you can jump straight into section 3 and run the CNN.

# In[ ]:


def confusion_matrices(model, dataset, threshold):
    
    metrics = {}
    x = PPD.collate_data(dataset)[0]
    y_hat  = model(x)
    """ge = greater equal than threshold"""
    y_hat = torch.ge(y_hat, threshold).double() #predicted
    y = dataset.labels_tensor.double()  #actual
    tp = (y * y_hat).sum(0).numpy()
    tn = ((1 - y) * (1 - y_hat)).sum(0).numpy()
    fp = (y_hat * (1 - y)).sum(0).numpy()
    fn = ((1 - y_hat) * y).sum(0).numpy()
    metrics['TP'] = tp
    metrics['TN'] = tn
    metrics['FP'] = fp
    metrics['FN'] = fn
    
    return metrics

def confusion_matrices_change(metrics_05, metrics_best_t):
    
    labels = range(12)
    fig = plt.figure(figsize=(15,10))
    fig.suptitle("metrics differences")
    tp_ax = fig.add_subplot(221)
    tn_ax = fig.add_subplot(222)
    fp_ax = fig.add_subplot(223)
    fn_ax = fig.add_subplot(224)
    
    tp_ax.plot(labels, metrics_05['TP'], label = 't = 0.5')
    tn_ax.plot(labels, metrics_05['TN'], label = 't = 0.5')
    fp_ax.plot(labels, metrics_05['FP'], label = 't = 0.5')
    fn_ax.plot(labels, metrics_05['FN'], label = 't = 0.5')
    
    tp_ax.set_ylabel('TP')
    tn_ax.set_ylabel('TN')
    fp_ax.set_ylabel('FP')
    fn_ax.set_ylabel('FN')
    
    tp_ax.plot(labels, metrics_best_t['TP'], label = 'best t')
    tn_ax.plot(labels, metrics_best_t['TN'], label = 'best t')
    fp_ax.plot(labels, metrics_best_t['FP'], label = 'best t')
    fn_ax.plot(labels, metrics_best_t['FN'], label = 'best t')
    
    tp_ax.legend()
    tn_ax.legend()
    fp_ax.legend()
    fn_ax.legend()
    
    #plt.show()


# ## 1. Generating word embeddings matrix

# We read from raw_data all the files and get all the different words we can find within all the files. If we already have a file named dictionary.pkl and read set to True, it will read the dictionary from this file. 

# In[ ]:


dictionary = get_tokens("raw_data", "embeddings_data", read = True)


# The next step is to load the pretrained embeddings. We will get two python dictionaries. Both have the words as the keys of the python dictionaries and one has the vectors as the keys whilst the other one has the position on the dictionary.

# In[ ]:


fast_text_model = 'data/fastText-0.1.0/corpus_vectors_default_300d'

word2vector_fast_text = dp.get_fast_text_dicts(fast_text_model, "embeddings_data", 300, missing = False, read = True)


# In[ ]:


word2vector_glove = dp.get_glove_dicts( "data",  "glove.6B", "embeddings_data", 300, read = True)

print("Number of words in the pretrained embeddings: {}".format(len(word2vector_glove)))


# Now we compute the matrix containing all the word embeddings that we will need for the embedding layer of the CNN and 
# we obtain a word2idx of just all the words inside dictionary and not all the words present in the word embeddings. Usually the pretrained embeddings that we will use have more words than what we need, that is the reason why we need to obtain a new word2idx of just all the words that we found in the files inside train and test folders.

# `get_weigth_matrix()` variables:
# 1. dimensions
# 2. folder
# 3. read
# 4. oov_random
# 5. kwargs: dictionary, word2vector

# In[ ]:


#weights_matrix = get_weight_matrix(300, "embeddings_data", read = True)
#weights_matrix = get_weight_matrix(300, "embeddings_data", oov_random = False, dictionary = dictionary, word2vector = word2vector_glove)
weights_matrix = get_weight_matrix(300, "embeddings_data", oov_random = False, dictionary = dictionary, word2vector = word2vector_fast_text)


# ### 2. Creation of Datasets

# The first step before obtaining the prefectly cleaned data that will be used in the CNN is to aggregate the labels. The raw_data folder provides a series of files in csv format with repeated sentences. The reason behind this is that some sentences have several labels assigned to them. The last step is to aggregate segments and obtain a list of labels per sentence. The following function gets all the data from raw_data folder and outputs the result in agg_data.

# In[ ]:


labels = OrderedDict([('First Party Collection/Use', 0),
             ('Third Party Sharing/Collection', 1),
             ('User Access, Edit and Deletion', 2),
             ('Data Retention', 3),
             ('Data Security', 4),
             ('International and Specific Audiences', 5),
             ('Do Not Track', 6),
             ('Policy Change', 7),
             ('User Choice/Control', 8),
             ('Introductory/Generic', 9),
             ('Practice not covered', 10),
             ('Privacy contact information', 11)])

with open("labels.pkl", "wb") as f:
    pickle.dump(labels, f)


dp.aggregate_data(read = False)


# Now that we have the aggregated data in agg_data we will process all the sentences and transform them into a list of integers. The integers will refer to the position inside the word2idx dictionary. The labels will also be transformed into an n-dimensinal vector with 1s if a sentence has that label and 0s if it doesn't. All the data will be placed in the corresponding folder inside processed_data. We load the labels with which we want to perform the classification. We will also show them so that it is clearer to the user.

# In[ ]:


labels_file = open("labels.pkl","rb")

labels = pickle.load(labels_file)

labels_file.close()

for label, index in labels.items():
    
    print(str(index) + '. ' + label)


# In[ ]:


sentence_matrices_all, labels_matrices_all = dp.process_dataset(labels, dictionary, read = False)


# We now create an PPD which stands for PrivacyPoliciesDataset containing the training and testing dataset. We will need to split the data in two to get the test training data and the data that will be used for training and validation. The function split_dataset_randomly is spliting the dataset 90/10 by default.

# In[ ]:


dataset = PPD(sentence_matrices_all, labels_matrices_all, labels)

# test_dataset, train_validation_dataset = dataset.split_dataset_randomly(0.2)
# 
# validation_dataset, train_dataset = train_validation_dataset.split_dataset_randomly(0.2)
# 
# test_dataset.pickle_dataset("datasets/test_dataset_label6.pkl")
# 
# train_dataset.pickle_dataset("datasets/train_dataset_label6.pkl")
# 
# validation_dataset.pickle_dataset("datasets/validation_dataset_label6.pkl")


# In case we aready had all the data split and prepared we can load it like this: 

# In[ ]:


test_dataset = PPD.unpickle_dataset("datasets/test_dataset_label6.pkl")

train_dataset = PPD.unpickle_dataset("datasets/train_dataset_label6.pkl")

validation_dataset = PPD.unpickle_dataset("datasets/validation_dataset_label6.pkl")


# In[ ]:


train_dataset.labels_stats()
print("-" * 35 * 3)
validation_dataset.labels_stats()
print("-" * 35 * 3)
test_dataset.labels_stats()
print("-" * 35 * 3)


# ## 3. Creation of CNN and training

# Now we set the 7 main parameters of the CNN we are using:
# 1. Number of words in the dictionary
# 2. Embeddings dimension
# 3. Number of filters per kernel size
# 4. Number of hidden units
# 5. Number of labels to classify
# 6. List of all the kernels sizes we want to use
# 7. Name of the model

# In[ ]:


train_dataset.labels_tensor.shape


# In[ ]:


weights_matrix.shape


# In[ ]:


model = CNN(weights_matrix.shape[0], weights_matrix.shape[1], 200, [100], 12, [3], name = 'zeros_60-20-20_polisis')

model.load_pretrained_embeddings(weights_matrix)


# 
# We will also add the pretrained embeddings to the embedding layer of the CNN through load_pretrained_embeddings. The function called train_cn will need two more parameters:
# 1. number of epochs 
# 2. learning rate
# 3. momentum constant

# In[ ]:


results = model.train_CNN(train_dataset, validation_dataset, lr = 0.01, epochs_num = 20, alpha = 0, momentum = 0.9)

epochs, train_losses, validation_losses, f1_scores_validations, precisions_validations, recalls_validations = results


# In[ ]:


# plt.plot(epochs, train_losses, label = "train")
# plt.plot(epochs, validation_losses, label = "validation")
# plt.legend()
# plt.title("loss vs epoch")
import os

# Ensure the directory exists
output_dir = "trained_models_pics"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the plot
# plt.savefig(join(output_dir, model.cnn_name + '_loss.png'), format='png')

# plt.show()


# plt.plot(epochs, f1_scores_validations, label = "validation")
# plt.legend()
# plt.title("f1 vs epoch")
# #plt.savefig(join("trained_models_pics", model.cnn_name + '_loss.png'), format = 'png')
# plt.show()

# import os
# os.makedirs('trained_models', exist_ok=True)
# os.makedirs('trained_models/Thresholds', exist_ok=True)
# dict_path = join("trained_models", model.cnn_name + "_state.pt")
# torch.save(model.state_dict(), dict_path)
# model.save_cnn_params()


# We save all the parameters used in the CNN (weights of all the layers and the configurations of the CNN)

# ## 4. Evaluation of the CNN results

# We extract the labels true labels from the training and testing data sets and predict the labels using both labels. The predictions are usually refered as y_hat. 

# In[ ]:


model.eval()

y_train = train_dataset.labels_tensor

y_validation = validation_dataset.labels_tensor

x_train = PPD.collate_data(train_dataset)[0]

x_validation = PPD.collate_data(validation_dataset)[0]

y_hat_train = model(x_train)

y_hat_validation = model(x_validation)


# We now show how the F1 score changes for all possible thresholds

# In[ ]:


model.print_results(train_dataset, validation_dataset, 0.5)


# With the following block of code we will find the threshold that with which we obtain the best overall F1 score. 

# In[ ]:


best_f1_score, best_threshold_label = CNN.get_best_thresholds(y_validation, y_hat_validation, labels)


# In[ ]:


model.print_results_best_t(validation_dataset, torch.tensor(best_threshold_label).float())


# In[ ]:


model.f1_score(y_validation, y_hat_validation, torch.tensor(best_threshold_label).float(), macro=True)


# In[ ]:


model.f1_score(y_validation, y_hat_validation, torch.tensor(best_threshold_label).float(), macro=False)


# In[ ]:


metrics_05 = confusion_matrices(model, validation_dataset, 0.5)


# In[ ]:


metrics_best_t = confusion_matrices(model, validation_dataset, torch.tensor(best_threshold_label).float())


# In[ ]:


confusion_matrices_change(metrics_05, metrics_best_t)


# Now we show the results for the best combination of thresholds

# In[ ]:


# We show the F1, precision and recall for the best threshold
#f1, precision, recall = CNN.f1_score(y_validation, y_hat_validation, torch.tensor(best_t_label).float())
f1, precision, recall = CNN.f1_score_per_label(y_validation, y_hat_validation, 0.5)

print("f1        |" + str(f1))

print("precision |" + str(precision))

print("recall    |" + str(recall))


# We show a list of all the possible labels to remind the user which ones are available.

# In[ ]:


for label, i in labels.items():
    
    print("index {}. {}.".format(i, label))


# We can also show how the F1 score changes for all possible thresholds in just one label

# In[ ]:


threshold_list = np.arange(0.0, 1, 0.01)

label = 'User Choice/Control'

f1_scores_per_label = [CNN.f1_score_per_label(y_validation, y_hat_validation, t)[0][labels[label]].item() for t in threshold_list]

# plt.plot(threshold_list, f1_scores_per_label)

# plt.title(label + " f1 score" + " vs threshold")

# plt.show()

f1_label, precision_label, recall_label = CNN.f1_score_per_label(y_validation, y_hat_validation, 0.5)

f1_label = f1_label[labels[label]].item()

precision_label = precision_label[labels[label]].item()

recall_label = recall_label[labels[label]].item()

print("Label: " + label + "\n")

print("f1_label        |" + str(f1_label))

print("precision_label |" + str(precision_label))

print("recall_label    |" + str(recall_label))

# ## 5. Comparison between models

# In[ ]:
import os
from argparse import ArgumentParser

# Function to get default parameters
def get_default_params():
    return {
            "vocab_size": 10000,  
            "emb_dim": 300,      
            "Co": 100,           # Number of convolutional output channels
            "Hu": [200],         # Number of hidden units
            "C": 12,             # Number of classes
            "Ks": [3]            # Kernel sizes
    }

# Load model parameters safely
def load_params(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'rb') as file:
                params = pickle.load(file)
                return {k: v for k, v in params.items() if k in CNN.__init__.__code__.co_varnames}
        except Exception as e:
            print(f"Error reading file '{file_path}': {e}")
    print(f"File '{file_path}' does not exist. Using default parameters.")
    return get_default_params()

# Load model state safely
def load_model(file_path, params):
    model = CNN(**params)
    if os.path.exists(file_path):
        try:
            model.load_state_dict(torch.load(file_path))
        except Exception as e:
            print(f"Error loading model state from '{file_path}': {e}")
    else:
        print(f"File '{file_path}' does not exist. Initializing a new model.")
    return model

# Paths
model1_params_path = 'trained_models/Thresholds/cnn_300_200_[100]_12_[3]_e80_60-20-20_polisis_params.pkl'
model1_state_path = 'trained_models/Thresholds/cnn_300_200_[100]_12_[3]_e80_60-20-20_polisis_state.pt'

model2_params_path = 'trained_models/Thresholds/cnn_300_200_[100]_12_[3]_lr.005_e300_60-20-20_polisis_params.pkl'
model2_state_path = 'trained_models/Thresholds/cnn_300_200_[100]_12_[3]_lr.005_e300_60-20-20_polisis_state.pt'

# Load models
params_model1 = load_params(model1_params_path)
model1 = load_model(model1_state_path, params_model1)

params_model2 = load_params(model2_params_path)
model2 = load_model(model2_state_path, params_model2)

# Evaluation
y_validation = validation_dataset.labels_tensor
x_validation = PPD.collate_data(validation_dataset)[0]

def evaluate_model(model, x_validation, y_validation, validation_dataset):
    y_hat = model(x_validation)
    best_f1_score, best_t_label = CNN.get_best_thresholds(y_validation, y_hat, labels)

    model.print_results_best_t(validation_dataset, torch.tensor(best_t_label).float())

    print('----Macro Averages----')
    model.f1_score(y_validation, y_hat, torch.tensor(best_t_label).float(), macro=True)

    for t in [best_t_label, 0.5]:
        metrics = confusion_matrices(model, validation_dataset, torch.tensor(t).float())
        tp, fp, tn, fn = sum(metrics['TP']), sum(metrics['FP']), sum(metrics['TN']), sum(metrics['FN'])

        print(f'------------------Threshold {t}--------------------')
        print(f'F1: {2 * tp / (2 * tp + fn + fp)}')
        print(f'Precision: {tp / (tp + fp)}')
        print(f'Recall: {tp / (tp + fn)}')

    model.f1_score_per_label(y_validation, y_hat, torch.tensor(best_t_label).float())

# Run evaluation
evaluate_model(model1, x_validation, y_validation, validation_dataset)
evaluate_model(model2, x_validation, y_validation, validation_dataset)
