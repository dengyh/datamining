file1 = open('result.csv', 'r')
file2 = open('result_correct.csv', 'r')

data1 = file1.readlines()
data2 = file2.readlines()
count = 0
for index in xrange(len(data1)):
    if data1[index] == data2[index]:
        count += 1

print count * 1.0 / len(data1), count, len(data1)