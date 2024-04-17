import json
import numpy as np
#Amazon_Instant_Video_5.json
bookfile = open('result/V.dat', 'r')
vector_file = open('result/bookVectors.dat', 'w')
all_line = bookfile.read().splitlines()
i = 0
for x in range(0, len(all_line)):
    print('book ' + str(x))
    for y in range(0, len(all_line)):
        bookVector1 = all_line[x].split(" ")
        bookVector1 = np.array(bookVector1)
        bookVector1 = bookVector1.astype(np.float)
        bookVector2 = all_line[y].split(" ")
        bookVector2 = np.array(bookVector2)
        bookVector2 = bookVector2.astype(np.float)
        similar = (np.dot(bookVector1, bookVector2) / (np.sqrt(np.dot(bookVector1, bookVector1)) * np.sqrt(np.dot(bookVector2, bookVector2))))
        vector_file.write(str(similar) + " ")
        
    vector_file.write('\n')

