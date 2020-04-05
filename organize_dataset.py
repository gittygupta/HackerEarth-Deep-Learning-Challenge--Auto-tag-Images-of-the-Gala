from fastai.vision import *
import cv2

classes = ['Food', 'Attire', 'Decorationandsignage', 'misc']
f = open('original_dataset/train.csv', 'r')
arr = f.read()
data = list(map(str, arr.split('\n')))
data = data[1:-1]
print('Number of training samples : ', len(data))

filename = []
class_val = []
for i in range(len(data)):
    file_val, class_name = data[i].split(',')
    filename.append(file_val)
    class_val.append(class_name)
directory = 'dataset'

for i in classes:
    path = Path(directory)
    dest = path/i
    dest.mkdir(parents=True, exist_ok=True)

for i in range(len(data)):
    a = cv2.imread("original_dataset/Train Images/" + filename[i])
    cv2.imwrite(directory + '/' + class_val[i] + '/' + filename[i], a)

total = 0
i = 0
for i in range(len(data)):
    a = cv2.imread("original_dataset/Train Images/" + filename[i])
    total += a.shape[1]
    
avg_size = total/len(data)
print('Average Size : ', avg_size)