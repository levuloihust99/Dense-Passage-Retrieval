from typing import Text
import requests
import json

URL = "http://103.252.1.223:5051/retrieve"


def search(query, top_docs: int):
    headers = {'Content-Type': 'application/json'}
    data = {'query': query, 'top_docs': top_docs}
    resp = requests.post(url=URL, headers=headers, data=json.dumps(data))
    return resp.json()
