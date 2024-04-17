orders = []
try:
    with open('reviews_Books_5_.json') as data:
        for each_line in data:
            try:
                order = each_line.replace("\n", ",").strip()
                orders.append(order)

            except ValueError:
                pass

    f = open('dataset.json', 'w')
    f.write(orders)
except IOError as err:
    print('File Error: ' + str(err))
