
[Technique](https://github.com/yuki678/financial-phrase-bert/blob/master/SA_Model_Comparison_Finphrase.ipynb)

[Data](https://www.researchgate.net/profile/Pekka_Malo/publication/251231364_FinancialPhraseBank-v10/data/0c96051eee4fb1d56e000000/FinancialPhraseBank-v10.zip?origin=publication_list)

1. Get token: 
pachctl auth get-auth-token --ttl "624h" |     grep Token | awk '{print $2}'
1. make setup
2. docker run --env-file .env -v $(pwd)/label-studio-project:/my_text_project -p 8080:8080 jimmywhitaker/label-studio:pach-ls0.9