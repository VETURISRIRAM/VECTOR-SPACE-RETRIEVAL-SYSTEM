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


Recall Score for all the queries : [
										0.0, 
										0.13333333333333333, 
										0.13333333333333333, 
										0.05555555555555555, 
										0.10526315789473684, 
										0.2222222222222222, 
										0.6666666666666666, 
										0.5, 
										0.25, 
										0.08333333333333333
									]

Precision Score for all the queries : [0.0, 0.2, 0.2, 0.1, 0.2, 0.4, 0.6, 0.2, 0.2, 0.2] 


2) Average Scores for Top-50 documents :

	1) Recall : 0.098

	2) Precision : 0.4120614035087719

Recall Score for all the queries : [
										0.0, 
										0.26666666666666666, 
										0.4, 
										0.1111111111111111, 
										0.5789473684210527, 
										0.3333333333333333, 
										0.8888888888888888, 
										0.75, 
										0.625, 
										0.16666666666666666

									]

Precision Score for all the queries : [0.0, 0.08, 0.12, 0.04, 0.22, 0.12, 0.16, 0.06, 0.1, 0.08]


3) Average Scores for Top-100 documents :

	1) Recall : 0.06700000000000002

	2) Precision : 0.528421052631579

Recall Score for all the queries : [
										0.0, 
										0.6666666666666666, 
										0.6, 
										0.3333333333333333, 
										0.6842105263157895, 
										0.4444444444444444, 
										0.8888888888888888, 
										0.75, 
										0.75, 
										0.16666666666666666
									]

Precision Score for all the queries : [0.0, 0.1, 0.09, 0.06, 0.13, 0.08, 0.08, 0.03, 0.06, 0.04]


4) Average Scores for Top-500 documents : 

	1) Recall : 0.0228

	2) Precision : 0.9152777777777779

Recall Score for all the queries : [
										1.0, 
										1.0, 
										1.0, 
										0.8888888888888888, 
										1.0, 
										0.8888888888888888, 
										1.0, 
										1.0, 
										0.875, 
										0.5
									]
									
Precision Score for all the queries : [0.002, 0.03, 0.03, 0.032, 0.038, 0.032, 0.018, 0.008, 0.014, 0.024]