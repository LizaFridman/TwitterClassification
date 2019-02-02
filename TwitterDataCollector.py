import tweepy
from tweepy import OAuthHandler


class TwitterDataCollector:
    def __init__(self, consumer_key, consumer_secret, access_token, access_secret):
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token = access_token
        self.access_secret = access_secret
        self.authenticate()

    def authenticate(self):
        """The oAuth 'dance':
    Get a request token from twitter
    Redirect user to twitter.com to authorize our application
    If using a callback, twitter will redirect the user to us. Otherwise the user must manually supply us with the verifier code.
    Exchange the authorized request token for an access token
"""
        auth = OAuthHandler(self.consumer_key, self.consumer_secret)
        auth.set_access_token(self.access_token, self.access_secret)
        api = tweepy.API(auth, wait_on_rate_limit=True)
