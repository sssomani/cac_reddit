import pandas as pd
import json

import requests
from urllib.parse import quote as url_parse
import argparse

def create_base_url(discussion, search_string):
    """
    Function to create the base URL for scraping Reddit.
    """
    
    # Ensure specified discussion is either a post (submission) or comment.
    assert discussion in ['submission', 'comment']
    
    # Craft our URL.
    return f'https://api.pushshift.io/reddit/{discussion}/search?html_decode=true&q={search_string}&size=1000'

def scrape_to_json(url):
    """
    Function that scrapes the Pushshift API based on a specific URL and converts those results to a Pandas dataframe.
    """
    
    data = requests.get(url)
    data_json = json.loads(data.content)
    print(f'{len(data_json["data"])} new discussions scraped using Requests!')
    data_pd = pd.DataFrame(data_json['data'])
    return data_pd

def scrape_reddit(base_url, secondary_search=False, keyword=None):
    
    keys_to_keep = [
        'subreddit',
        'author',
        'utc_datetime_str',
        'body',
        'id'
    ]
    
    discussions = scrape_to_json(base_url)
    n_discussions = len(discussions)
    
    while n_discussions > 0:
        last_utc = discussions['created_utc'].iloc[-1]
        url = base_url + f'&after=0&before={last_utc}'
        
        next_discs = scrape_to_json(url)
        n_discussions = len(next_discs)
                
        discussions = pd.concat([discussions, next_discs])
            
    if 'submission' in base_url:
        discussions['body'] = discussions['title'] + '. ' + discussions['selftext']
    
    return discussions[keys_to_keep]

def get_reddit_data(search_strings):
    '''
    
    Scrape Reddit for all discussions related to a set of search strings.         

    Parameters
    ----------
    search_string : str
        Search string to query.
    output_fn : str, optional
        Name of the local Excel database to save data. 

    Returns
    -------
    posts_df : Pandas dataframe
        Database

    '''
    
    discussions = pd.DataFrame(columns=[
        'subreddit',
        'author',
        'utc_datetime_str',
        'body',
        'type'
        'id',
    ])
    
    for search_string in search_strings:

        search_string = url_parse(search_string)

        # First, let's start by searching all posts.
        post_url = create_base_url('submission', search_string)
        posts = scrape_reddit(post_url)
        posts['type'] = 'post'
        posts['search_string'] = search_string

        # Now, let's search all comments.
        comment_url = create_base_url('comment', search_string)
        comments = scrape_reddit(comment_url)
        comments['type'] = 'comment'
        comments['search_string'] = search_string
        
        discussions = pd.concat([discussions, posts, comments])
    
    discussions.drop_duplicates('id', inplace=True)
    print(f'Total of {discussions.shape[0]} found!')

    return discussions

if __name__ == '__main__':
    """
    Scrape Reddit for all CAC-related keywords.
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('output_file_path', type='str', default='data/cac_db.xlsx', help='Path to save dataset')
    args = parser.parse_args()

    cac_db_strings = ['coronary artery calcium', 'coronary calcium', 'cac score', 'calcium score', 'heart scan']
    cac_db = get_reddit_data(cac_db_strings)
    cac_db.to_excel(args.output_file_path)