import json

with open('dataset/dummy.json') as json_file:  
    data = json.load(json_file)
    print data