from fastai.vision import *

path = Path('dataset')
learn = load_learner(path)
f = open('original_dataset/test.csv', 'r')
arr = f.read()
data = list(map(str, arr.split('\n')))
data = data[1:-1]
print('Number of testing samples : ', len(data))
datalen = len(data)

preds = []
for i in range(datalen):
    test_img = open_image('original_dataset/Test Images/' + data[i])
    pred_class,pred_idx,outputs = learn.predict(test_img)
    preds.append(str(pred_class))

f = open('Outputs/output.csv', 'w')
string = 'Image,Class\n'
for i in range(datalen):
    string += (data[i] + ',' + preds[i] + '\n')
f.write(string)
f.close()