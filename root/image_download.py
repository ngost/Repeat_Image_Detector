from skimage import io
import csv


# for i in range(113):
#     filename = url_parents+'/'+str(i+1)
#     print(filename)
#     io.imread(filename)
#io.imread("")

f = open('output.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)
wr.writerow([1, 'mkblog'])
wr.writerow([2, 'co'])
wr.writerow([3, 'kr'])
f.close()
