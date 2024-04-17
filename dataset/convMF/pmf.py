from mf import MatrixFactorization
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

def merge_sort(arr, arrx):
    # The last array split
    if len(arr) <= 1:
        return arr, arrx
    mid = len(arr) // 2
    midx = len(arrx) // 2
    # Perform merge_sort recursively on both halves
    left, leftx = merge_sort(arr[:mid], arrx[:midx])
    right, rightx =  merge_sort(arr[mid:], arrx[midx:])

    # Merge each side together
    return merge(left, right, arr.copy(), leftx, rightx, arrx.copy())


def merge(left, right, merged, leftx, rightx, mergedx):

    left_cursor, right_cursor = 0, 0
    while left_cursor < len(left) and right_cursor < len(right):
      
        # Sort each one and place into the result
        if left[left_cursor] >= right[right_cursor]:
            merged[left_cursor+right_cursor]=left[left_cursor]
            mergedx[left_cursor+right_cursor]=leftx[left_cursor]
            left_cursor += 1
        else:
            merged[left_cursor + right_cursor] = right[right_cursor]
            mergedx[left_cursor + right_cursor] = rightx[right_cursor]
            right_cursor += 1
            
    for left_cursor in range(left_cursor, len(left)):
        merged[left_cursor + right_cursor] = left[left_cursor]
        mergedx[left_cursor + right_cursor] = leftx[left_cursor]
        
    for right_cursor in range(right_cursor, len(right)):
        merged[left_cursor + right_cursor] = right[right_cursor]
        mergedx[left_cursor + right_cursor] = rightx[right_cursor]

    return merged, mergedx

ratingset = open('trainrating.txt', 'r')

ratings = ratingset.read().splitlines()

data = []

for rating in ratings:
    data.append(np.array(rating.split(' ')).astype(np.float))
#data = [user_11, user_2]
datax = []
for i in range(6593):
	datax.append(i)

modelA = MatrixFactorization()
modelA.fit(data)

# print(modelA.predict_instance(1))
resrating = open('result_rating.txt', 'w')
resrecommend = open('result_recommend', 'w')
for i in range(len(data)):
    print('user -'+str(i))
    resrating.write(np.array2string(modelA.predict_instance(i), precision=2, separator=',', suppress_small=True))
    res, resx = merge_sort(modelA.predict_instance(1), datax)
    for x in range(30):
        resrecommend.write(str(resx[x]) + ", ")

    resrating.write('\n')
    resrecommend.write('\n')
