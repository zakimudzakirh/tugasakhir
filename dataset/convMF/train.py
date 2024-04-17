import numpy as np
import re

data = open('dataset/datasetall.txt', 'r')

array = eval(data.read());


#matching = [s for s in array if "0::171::" in s]

ratings = open('dataset/ratingset2.txt', 'w')

for x in range(204, 493):
	print('user '+ str(x))
	for y in range(0, 6593):
		r = re.compile(str(x) + "::" + str(y) + "::.*")
		newlist = list(filter(r.match, array))
		if not newlist:
			ratings.write("0.0")
		else:
			rating = newlist[0].split("::")[2]
			ratings.write(rating)
		if y < 6592:
			ratings.write(" ")
	ratings.write('\n')