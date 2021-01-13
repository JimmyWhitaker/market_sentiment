# Market Sentiment Example
The purpose of this example is to illustrate how the two, independent yet symbiotic loops in the [machine learning loop](https://jimmymwhitaker.medium.com/completing-the-machine-learning-loop-e03c784eaab4) can work together practically. The code loop is managed by git+unittest+GitHub Actions, while the data loop and the data+code interactiona are managed by Pachyderm. 

The original technique is a [sentiment analysis classifier](https://github.com/yuki678/financial-phrase-bert/blob/master/SA_Model_Comparison_Finphrase.ipynb) that uses the [Financial Phrase Bank Dataset](https://www.researchgate.net/profile/Pekka_Malo/publication/251231364_FinancialPhraseBank-v10). In this example, we use a reduced version of this blog post and dataset for simplicity and transparency to show the interactions more than the techniques themselves. 


## Running the example
The easiest way to run this example is to use the [Makefile](./Makefile) once the cluster is configured (steps below).


### Setup Pachyderm, S3 gateway, and Label Studio
1. Start a pachyderm cluster - get the endpoint address (if using hub, just look at the address where the dash is being served or if minikube, run `minikube ip`)
2. Create a pachyderm token (details)
```bash
pachctl auth get-auth-token --ttl "624h" | grep Token | awk '{print $2}'
```
3. Create a `.env` file inserting the endpoint address and token. 
```
ENDPOINT_URL=https://<pachyderm_endpoint_address>:30600
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=<pachyderm_token>
AWS_SECRET_ACCESS_KEY=<pachyderm_token>
```
4. Run label studio with the following commands (you won't see any tasks until you add data):
```bash
pachctl create repo raw_data
pachctl create repo labeled_data
pachctl create branch labeled_data@master
pachctl create branch raw_data@master

# Start a local instance of Label Studio (needs the configured .env for the Pach s3 gateway)
docker run -it --env-file .env -v $(pwd)/label-studio-project:/my_text_project -p 8080:8080 jimmywhitaker/label-studio:pach-ls0.9

# Navigate to http://localhost:8080/tasks
```

### Setup GitHub Action
Details on the [Pachyderm GitHub Actions](https://github.com/pachyderm/pachyderm-actions)
1. Get the Pachyderm cluster URL (the same address as the s3 gateway, but with port 31400)
2. Create another Pachyderm token like in the previous step.
```bash
pachctl auth get-auth-token --ttl "624h" | grep Token | awk '{print $2}'
```
3. Create Pachyderm token, [DockerHub username, and DockerHub token](https://docs.docker.com/docker-hub/access-tokens/) secrets in GitHub (see [Managing Access Tokens](https://docs.docker.com/docker-hub/access-tokens/)). See our [GitHub Actions Example](https://github.com/pachyderm/pachyderm-gha#running-this-example) for details. 
4. Once these tokens are in place, the pipelines will be pushed each time code is merged to the master branch.

## Repository Structure

### Data-tests
TODO

### market_sentiment
The main code of the project. This includes the python files needed to load data, etc. 

### Pachyderm
This directory holds all the pachyderm pipelines. These pipelines define the code that will be run on our data in our Pachyderm cluster. Once deployed, they will automatically process any data changes, such as, when new data is labeled, it will automatically create a new dataset and train a model when that dataset is ready. 

#### **pachyderm-github-action**

The Pachyderm GitHub Action is used to deploy our pipelines when code is pushed to our repository. It handles the building of the Docker container, pushing it to our Docker registry, and updates our pipelines with the new version of this container. 

### tests
Unit tests for our code that will be run before building our Docker container. 

### label-studio-project
Project configuration for a sentiment analysis in Label Studio using Pachyderm's s3 gateway as the backend to add versioning to the labeling environment. Data is read from Pachyderm and written back to Pachyderm, which adds versioning automatically. 

## TODO and Known Issues

* There's currently some lag in the s3 gateway communications (likely because test cluster is very small). More investigation needed on this. 
* Add data exploration visualizations
* Makefile for common functions and deployments (integration and prod testing)
* Create staging branch for deployment that can be migrated into production. 
* Add more unit tests
* Add data tests