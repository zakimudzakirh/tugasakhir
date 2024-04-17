import numpy as np
from sklearn.metrics import mean_absolute_error
x = np.array(['1.1', '2.2', '3.3'])
y = x.astype(np.float)

resultpmf = open('result_rating2.txt','r')
realrating = open('dataset/ratingset.txt', 'r')

ratings = realrating.read().splitlines()
real = []
for x in range(len(ratings)):
	real.append(np.array((ratings[x]).split(' ')).astype(np.float))

pmf = resultpmf.read().split('[')

print(len(pmf))

array = []
for i in range(1, len(pmf)):
	array.append(np.array(eval('[' + (pmf[i]).rstrip())).astype(np.float))

#array = eval('[' + (pmf.split('[')[1]).rstrip())

print(mean_absolute_error(real, array))

#print(len(array))
