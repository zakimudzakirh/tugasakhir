#import json
#Amazon_Instant_Video_5.json
books = open('data_id_book', 'r')
data = books.read().splitlines()
books.close()
print len(data)
data2 = []
data3 = [] 
for num in data: 
    if num not in data2: 
        data2.append(num) 
    else:
        data3.append(num)
print len(data2)
print len(data3)
for num in data3:
    print num

#print data
    # #data = json.load(json_file)
    # f = open('result3.txt', 'w')
    # # print data
    # for d in data:
    #     f.write(d['reviewerID'] + '::' + d['asin'] + '::' + str(d['overall']) + '\n')
    #     # print(d['reviewerID'] + '::' + d['asin'] + '::' + parseDoub)