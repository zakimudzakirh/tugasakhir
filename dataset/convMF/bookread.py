import urllib.request
import random


def downloader(image_url, file_name):
    full_file_name = 'images/' +str(file_name) + '.jpg'
    urllib.request.urlretrieve(image_url,full_file_name)


#downloader("https://images-na.ssl-images-amazon.com/images/I/41rSaQpivNL.jpg")

file = open('book_filter.txt', 'r')

# def downloadimg(self):
# 	imgurl = self.getdailyimg();
# 	imgfilename = datetime.datetime.today().strftime('%Y%m%d') + '_' + imgurl.split('/')[-1]
# 	with open('images/' + imgfilename, 'wb') as f:
# 		f.write(self.readimg(imgurl))

books = file.read()

books = eval(books)
for book in books:
	downloader("http://images.amazon.com/images/P/"+book+".01.MZZZZZZZ.jpg", book)