SHELL := /bin/bash
PACHCTL := pachctl
KUBECTL := kubectl

test-train:
	$(PACHCTL) create repo dataset
	$(PACHCTL) create repo sentiment_words
	$(PACHCTL) create pipeline -f pachyderm/train_model.json
	$(PACHCTL) put file dataset@master:/FPB_dataset.txt -f data/FinancialPhraseBank-v1.0/Sentences_75Agree.txt
	$(PACHCTL) put file sentiment_words@master:/LoughranMcDonald_SentimentWordLists_2018.csv -f resources/LoughranMcDonald_SentimentWordLists_2018.csv

test-ls-ingest: 
	$(PACHCTL) create repo raw_data
	$(PACHCTL) create repo labeled_data
	$(PACHCTL) create branch labeled_data@master
	mkdir -p output
	python label-studio-ingest.py --dataset-file data/FinancialPhraseBank-v1.0/Sentences_75Agree.txt --output-file output/FinancialPhraseBank.jsonl
	# $(PACHCTL) put file raw_data@master:/FinancialPhraseBank.jsonl -f output/FinancialPhraseBank.jsonl --split json --target-file-datums 1 --overwrite

100-example: 
	$(PACHCTL) create repo raw_data
	$(PACHCTL) create repo labeled_data
	$(PACHCTL) create branch labeled_data@master
	$(PACHCTL) put file raw_data@master:/FinancialPhraseBank.jsonl -f output/raw100_FinancialPhraseBank.jsonl --split json --target-file-datums 1 --overwrite
	$(PACHCTL) put file labeled_data@master:/ -r -f output/labeled-data/
	cp output/source.json label-studio-project/source.json
	$(PACHCTL) create pipeline -f pachyderm/create_dataset.json

	$(PACHCTL) create repo sentiment_words
	$(PACHCTL) put file sentiment_words@master:/LoughranMcDonald_SentimentWordLists_2018.csv -f resources/LoughranMcDonald_SentimentWordLists_2018.csv
	$(PACHCTL) create pipeline -f pachyderm/train_model.json

delete:
	$(PACHCTL) delete pipeline train_model
	$(PACHCTL) delete repo dataset
	$(PACHCTL) delete repo sentiment_words
	$(PACHCTL) delete repo raw_data
	$(PACHCTL) delete repo labeled_data

clean: delete