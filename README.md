
# Contemporary attitudes and beliefs on coronary artery calcium from social media using artificial intelligence

This repository contains the code for ([this paper](https://www.nature.com/articles/s41746-024-01077-w)). 

### Development setup
```sh
conda create -n cac_reddit python=3.11
conda activate cac_reddit
conda install pip
pip install -f requirements.txt
```

### Scraping Reddit data

```sh
python code/scrape_reddit.py <cac_db_path>
```

### Topic modeling

```sh
python code/topic_modeling.py <cac_db_path> <output_topic_model_file> 
```

### Sentiment analysis

```sh
python code/sentiment_analysis <cac_db_path> <output_sentiment_analysis_path>
```
