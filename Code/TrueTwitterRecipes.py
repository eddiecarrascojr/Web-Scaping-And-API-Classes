import twitter
from flask import Flask, request
import multiprocessing
from threading import Timer
import json, io
from functools import partial
import pymongo
import sys
import datetime
import time

def oauth_login():
    # XXX: Go to http://twitter.com/apps/new to create an app and get values
    # for these credentials that you'll need to provide in place of these
    # empty string values that are defined as placeholders.
    # See https://developer.twitter.com/en/docs/basics/authentication/overview/oauth
    # for more information on Twitter's OAuth implementation.

    CONSUMER_KEY = 'pLsJ6GRWMW1ef0P7u7JqYTnBU'
    CONSUMER_SECRET = 'kxJhJejYp5YOZLXCCjfnu1GiLRJ9qGGCnO47j6v1MSxA5qF57a'
    OAUTH_TOKEN = '849327279532539904-5oIs6Po6uRqnGTBxsDSYWPoXMNVmIRL'
    OAUTH_TOKEN_SECRET = 'aQAEA0KPlfciOnNuSG70kEzXCUTsxcdBSR41g1BHTYjX1'

    auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET,
                               CONSUMER_KEY, CONSUMER_SECRET)

    twitter_api = twitter.Twitter(auth=auth)
    return twitter_api

# Sample usage
twitter_api = oauth_login()

# Nothing to see by displaying twitter_api except that it's now a
# defined variable

def twitter_trends(twitter_api, woe_id):
    # Prefix ID with the underscore for query string parameterization.
    # Without the underscore, the twitter package appends the ID value
    # to the URL itself as a special-case keyword argument.
    return twitter_api.trends.place(_id=woe_id)

# Sample usage
# See https://developer.twitter.com/en/docs/trends/trends-for-location/api-reference/get-trends-place
# and http://www.woeidlookup.com to look up different Yahoo! Where On Earth IDs

WORLD_WOE_ID = 1
world_trends = twitter_trends(twitter_api, WORLD_WOE_ID)
print(json.dumps(world_trends, indent=1))

US_WOE_ID = 23424977
us_trends = twitter_trends(twitter_api, US_WOE_ID)
print(json.dumps(us_trends, indent=1))


def twitter_search(twitter_api, q, max_results=200, **kw):

    # See https://developer.twitter.com/en/docs/tweets/search/api-reference/get-search-tweets
    # and https://developer.twitter.com/en/docs/tweets/search/guides/standard-operators
    # for details on advanced search criteria that may be useful for
    # keyword arguments

    # See https://dev.twitter.com/docs/api/1.1/get/search/tweets
    search_results = twitter_api.search.tweets(q=q, count=100, **kw)

    statuses = search_results['statuses']

    # Iterate through batches of results by following the cursor until we
    # reach the desired number of results, keeping in mind that OAuth users
    # can "only" make 180 search queries per 15-minute interval. See
    # https://developer.twitter.com/en/docs/basics/rate-limits
    # for details. A reasonable number of results is ~1000, although
    # that number of results may not exist for all queries.

    # Enforce a reasonable limit
    max_results = min(1000, max_results)

    for _ in range(10): # 10*100 = 1000
        try:
            next_results = search_results['search_metadata']['next_results']
        except KeyError as e: # No more results when next_results doesn't exist
            break

        # Create a dictionary from next_results, which has the following form:
        # ?max_id=313519052523986943&q=NCAA&include_entities=1
        kwargs = dict([ kv.split('=')
                        for kv in next_results[1:].split("&") ])

        search_results = twitter_api.search.tweets(**kwargs)
        statuses += search_results['statuses']

        if len(statuses) > max_results:
            break

    return statuses

# Sample usage

q = "CrossFit"
results = twitter_search(twitter_api, q, max_results=10)

# Show one sample search result by slicing the list...
print(json.dumps(results[0], indent=1))

pp = partial(json.dumps, indent=1)

twitter_world_trends = partial(twitter_trends, twitter_api, WORLD_WOE_ID)

print(pp(twitter_world_trends()))

authenticated_twitter_search = partial(twitter_search, twitter_api)
results = authenticated_twitter_search("iPhone")
print(pp(results))

authenticated_iphone_twitter_search = partial(authenticated_twitter_search, "iPhone")
results = authenticated_iphone_twitter_search()
print(pp(results))

def save_json(filename, data):
    with open('resources/ch09-twittercookbook/{0}.json'.format(filename),
              'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


def load_json(filename):
    with open('resources/ch09-twittercookbook/{0}.json'.format(filename),
              'r', encoding='utf-8') as f:
        return json.load(f)

# Sample usage

q = 'CrossFit'

twitter_api = oauth_login()
results = twitter_search(twitter_api, q, max_results=10)

save_json(q, results)
results = load_json(q)

print(json.dumps(results, indent=1, ensure_ascii=False))

def save_to_mongo(data, mongo_db, mongo_db_coll, **mongo_conn_kw):

    # Connects to the MongoDB server running on
    # localhost:27017 by default

    client = pymongo.MongoClient(**mongo_conn_kw)

    # Get a reference to a particular database

    db = client[mongo_db]

    # Reference a particular collection in the database

    coll = db[mongo_db_coll]

    # Perform a bulk insert and  return the IDs
    try:
        return coll.insert_many(data)
    except:
        return coll.insert_one(data)

def load_from_mongo(mongo_db, mongo_db_coll, return_cursor=False,
                    criteria=None, projection=None, **mongo_conn_kw):

    # Optionally, use criteria and projection to limit the data that is
    # returned as documented in
    # http://docs.mongodb.org/manual/reference/method/db.collection.find/

    # Consider leveraging MongoDB's aggregations framework for more
    # sophisticated queries.

    client = pymongo.MongoClient(**mongo_conn_kw)
    db = client[mongo_db]
    coll = db[mongo_db_coll]

    if criteria is None:
        criteria = {}

    if projection is None:
        cursor = coll.find(criteria)
    else:
        cursor = coll.find(criteria, projection)

    # Returning a cursor is recommended for large amounts of data

    if return_cursor:
        return cursor
    else:
        return [ item for item in cursor ]

# Sample usage

q = 'CrossFit'

twitter_api = oauth_login()
results = twitter_search(twitter_api, q, max_results=10)

ids = save_to_mongo(results, 'search_results', q, host='mongodb://172.16.0.1:27017')

load_from_mongo('search_results', q, host='mongodb://172.16.0.1:27017')

def get_time_series_data(api_func, mongo_db_name, mongo_db_coll,
                         secs_per_interval=60, max_intervals=15, **mongo_conn_kw):

    # Default settings of 15 intervals and 1 API call per interval ensure that
    # you will not exceed the Twitter rate limit.

    interval = 0

    while True:

        # A timestamp of the form "2013-06-14 12:52:07"
        now = str(datetime.datetime.now()).split(".")[0]

        response = save_to_mongo(api_func(), mongo_db_name, \
                                 mongo_db_coll + "-" + now, **mongo_conn_kw)

        print("Write {0} trends".format(len(response.inserted_ids)), file=sys.stderr)
        print("Zzz...", file=sys.stderr)
        sys.stderr.flush()

        time.sleep(secs_per_interval) # seconds
        interval += 1

        if interval >= 15:
            break
def find_popular_tweets(twitter_api, statuses, retweet_threshold=3):

    # You could also consider using the favorite_count parameter as part of
    # this  heuristic, possibly using it to provide an additional boost to
    # popular tweets in a ranked formulation

    return [ status
                for status in statuses
                    if status['retweet_count'] > retweet_threshold ]
# Sample usage

get_time_series_data(twitter_world_trends, 'time-series', 'twitter_world_trends', host='mongodb://172.16.0.1:27017')
