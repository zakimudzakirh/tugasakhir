ratings = open('dataset/ratingset.txt', 'r')
validrating = open('data/data/valid_user.txt', 'r')
testrating = open('data/data/test_user.txt', 'r')

ratingtrain = open('trainratings2.txt', 'w')

ratinglines = ratings.read().splitlines()
#validlines = validrating.read().splitlines()
testlines = testrating.read().splitlines()

for i in range(len(ratinglines)):
    #valids = validlines[i].split(" ")
    #valids.pop(0)
    tests = testlines[i].split(" ")
    tests.pop(0)
    ratings = ratinglines[i].split(" ")
    print("user ke-" + str(i))
    for j in range(len(ratings)):
    	#if str(j) in valids:
    	#	ratingtrain.write('0.0')
    	#elif str(j) in tests:
    	if str(j) in tests:
    		ratingtrain.write('0.0')
    	else:
    		ratingtrain.write(ratings[j])
    	if j < 6592:
    		ratingtrain.write(' ')
    ratingtrain.write('\n')