SHELL := /bin/bash
PACHCTL := pachctl
KUBECTL := kubectl

test-train:
	$(PACHCTL) create repo dataset
	$(PACHCTL) create repo sentiment_words
	$(PACHCTL) create pipeline -f pachyderm/train_model.json
	$(PACHCTL) put file dataset@master:/FPB_dataset.txt -f data/FinancialPhraseBank-v1.0/Sentences_75Agree.txt
	$(PACHCTL) put file sentiment_words@master:/LoughranMcDonald_SentimentWordLists_2018.csv -f resources/LoughranMcDonald_SentimentWordLists_2018.csv

30-example: 
	$(PACHCTL) create repo raw_data
	$(PACHCTL) create repo labeled_data
	$(PACHCTL) create branch labeled_data@master
	$(PACHCTL) put file raw_data@master:/FinancialPhraseBank.jsonl -f data/sample_data/sentence_sample.jsonl --split json --target-file-datums 1 --overwrite
	# Inserting labeled data manually to treat it as 1 commit (for simplicity)
	$(PACHCTL) put file labeled_data@master:/ -r -f data/sample_data/completions/
	cp data/sample_data/source.json label-studio-project/source.json
	$(PACHCTL) create pipeline -f pachyderm/dataset.json

	$(PACHCTL) create repo sentiment_words
	$(PACHCTL) put file sentiment_words@master:/LoughranMcDonald_SentimentWordLists_2018.csv -f resources/LoughranMcDonald_SentimentWordLists_2018.csv

	$(PACHCTL) create pipeline -f pachyderm/visualizations.json
	$(PACHCTL) create pipeline -f pachyderm/train_model.json

delete:
	$(PACHCTL) delete pipeline train_model
	$(PACHCTL) delete pipeline visualizations
	$(PACHCTL) delete pipeline dataset
	$(PACHCTL) delete repo sentiment_words
	$(PACHCTL) delete repo raw_data
	$(PACHCTL) delete repo labeled_data
	rm -r label-studio-project/source.json label-studio-project/target.json label-studio-project/completions/* label-studio-project/tabs.json 

clean: delete