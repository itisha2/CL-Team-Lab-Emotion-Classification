# CL-Team-Lab-Emotion-Classification

#### Baseline-Classifier: 
  ##### Folder Description:
    	1. data: isear, ssec data (for this project, we use isear data)
    	2. intermediate_results: has dictionaries / pickle files for unique words in a particular emotion(ex: guilt)
   		3. baseline_model: this folder has all the codes for the baseline classifier: Naive Bayes
				 1. evaluation.py
				 2. load_data.py
				 3. baseline_classifier.py
				 4. pipeline.py: binds all the above scripts
				 5. Implementation_and_results.py: main file (or the starting file, from where all the above methods are called)
				 
#### Approach 2 (Additional work, not a part of baseline): With Feature Space: Naive Bayes CLassifier, with manually derived features instead of bag-of-words:
	Just for your reference.
	File_name: NaiveBayesWithFeatureSpace.ipynb
