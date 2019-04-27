# Inverted Index

## Project Description

The goal of this project is to implement a basic Vector Space Retrieval System. 

## Data Description

The Cranfield collection is a standard IR text collection, consisting of 1400 documents from the aerodynamics field, in SGML format. 

The dataset, a list of queries and relevance judgments associated with these queries are available to download from repository.

## How to run?

Make sure your system has all the libraries mentioned in the `requirements.txt` file. 

You just need to run the `main.py` file. The program will guide you throughout. 

It would ask you for the directory where the data files, queries file and relevance file, are stored. 

You have to enter these fields and wait for the program to run. 

The program might take quite a time to run. 

It takes me close to 3-4 minutes to run the complete program. But that is again subjective the type of system you use.


### Example run:

```
python main.py
```

### Implementation Algorithm

The program accepts the required directory and filenames from the user and fetched the same. 

From the documents inside, the SGML tags are removed and then, text preprocessing is done. 

The preprocessed documents go through the process of TF-IDF weights generation. 

The queries are being read and they too go through the same process as the documents except the part where SGML tags are removed. 

Then, cosine similarities are calculated between the queries and the documents, based on which the documents are ranked for each query. 

Then, the precision and recall score are calculated from the contents of the retrieved documents and the contents of the relevance file. 

The results are then displayed for the user.


### Sample Results

1) Average Scores for Top-10 documents :

	1) Recall : 0.23000000000000004

	2) Precision : 0.21497076023391815


2) Average Scores for Top-50 documents :

	1) Recall : 0.098

	2) Precision : 0.4120614035087719


3) Average Scores for Top-100 documents :

	1) Recall : 0.06700000000000002

	2) Precision : 0.528421052631579


4) Average Scores for Top-500 documents : 

	1) Recall : 0.0228

	2) Precision : 0.9152777777777779

Clearly, with 100 documents, the Precision and Recall are better.
