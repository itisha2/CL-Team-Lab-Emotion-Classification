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
	Just for your reference. However, the baseline proved to be better than this approach.
	File_name: NaiveBayesWithFeatureSpace.ipynb


#### Advance Approach (soft_decision_strategy):

   ##### Folder Description:
            1. finetuned_BERT_epoch_2.model = trained BERT model
            2. classifier_models: trained classifiers:
                    a) Overlapping Region Detector
                    b) Soft decision maker: classifier_1, classifier_2
            3. data: isear_bert, ssec_bert: dataset for training the ORD and soft decision maker and evaluation of ssec data.
            4. BERT Models:
                a) BertForMultilabelSequenceClassification.py
                b) utilities_bert.py
                c) inference_bert.py
                d) main_bert.py
            5. Overlapping Region Detector and Soft Decision Maker Codes:
                a) utilities_ord.py
                b) mlp_classifier_ord.py
                c) train_ord.py
            6. training_demo.ipynb: demonstrates the training of the ORD and soft_decision_maker and shows some results analysis.
            6. Main class: EmotionClassifier.py
            7. Main classifier running file, with a demo: run_classifier.ipynb


