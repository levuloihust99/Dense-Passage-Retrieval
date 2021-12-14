import requests
import json
import logging

logger = logging.getLogger(__name__)

with open('libs/term_search/vietnamese-stopwords.txt', 'r') as reader:
    stopwords = reader.read().split('\n')

if stopwords[-1] == '':
    stopwords.pop()

segmented_stopwords = []
headers = {'Content-Type': 'application/json'}
counter = 0
for word in stopwords:
    payload = {
        'sentence': word
    }
    resp = requests.post('http://localhost:8080/segment', headers=headers, data=json.dumps(payload))
    segmented_stopwords.append(resp.json()['sentence'])

    if (counter + 1) % 100 == 0:
        logger.info("Done segmented {} words".format(counter + 1))
    counter += 1

with open("libs/term_search/vietnamese-stopwords_segmented.txt", 'w') as writer:
    writer.write('\n'.join(segmented_stopwords))
