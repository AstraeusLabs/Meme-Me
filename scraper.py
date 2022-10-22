import pandas as pd
import praw
import datetime as dt
import credentials
import webbrowser
import time

keys = credentials.Credentials()
goodMemesDict = { "id":[], \
                "upvote_ratio":[], \
                "url":[], \
                "top_comment":[], \
                "pass":[]}
badMemesDict = { "id":[], \
                "upvote_ratio":[], \
                "url":[], \
                "top_comment":[], \
                "pass":[]}

try:
	reddit = praw.Reddit(client_id='V3v6UZi47nyL6Q', \
	                     client_secret='DjlXlznm0PWeXUEF1ZwSiVsWy0Q', \
	                     user_agent='MemeApp', \
	                     username=keys.username, \
	                     password=keys.password)
except:
	print("ERROR: Failed Login")
	print("Exiting")
	exit(1)

subreddit = reddit.subreddit('Memes')
#hot_memes = subreddit.hot(limit=500)
print("Displaying Memes")
print("Use enter to display title and then meme")
print("Press L if you like a meme")
print("Press enter if you don't like a meme")

for submission in subreddit.new():
        print("Title: "+submission.title)
        input()
        webbrowser.open(submission.url)
        response = input("Rating (L for like nothing for bad meme)")
        if response == '':
                badMemesDict["id"].append(submission.id)
                badMemesDict["upvote_ratio"].append(submission.upvote_ratio)
                badMemesDict["url"].append(submission.url)
                #comment_body = [comment.body for comment in submission.comments if hasattr(comment, "body")]
                #top_comment = comment_body[0]
                #goodMemesDict["top_comment"].append(top_comment)
                badMemesDict["pass"].append("0")
        elif response=="s":
                continue
                print("Post Skipped")
        elif response!="~":
                badMemesDict["id"].append(submission.id)
                badMemesDict["upvote_ratio"].append(submission.upvote_ratio)
                badMemesDict["url"].append(submission.url)
                #comment_body = [comment.body for comment in submission.comments if hasattr(comment, "body")]
                #top_comment = comment_body[0]
                #goodMemesDict["top_comment"].append(top_comment)
                badMemesDict["pass"].append("1")
        else:
                break


goodMemesPd = pd.DataFrame(goodMemesDict)
badMemesPd = pd.DataFrame(badMemesDict)

time = time.time()

goodMemesPd.to_csv('goodmemes'+str(time)+'.csv', index=False)
badMemesPd.to_csv('badmemes'+str(time)+'.csv', index=False)