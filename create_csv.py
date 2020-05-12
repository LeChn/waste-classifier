import pandas as pd
import numpy as np
import os
import json
import collections
import pdb


numGrids = 12
y = []
for file in os.listdir("./data/."):
    file, index = file[:-4].split("_")
    index = int(index)
    data = json.loads(open('imageLabel/A (' + file + ")/data_file.json").read())
    imgPath = "./imagesDataset/A (" + file + ").jpg"
    if os.path.isfile(imgPath) and 'labels' in data:
        gridLabel = data['labels'][index]
        if gridLabel:
            y.append([f'{file}_{index}.jpg', gridLabel])

df = pd.DataFrame(y, columns=['Image', 'label'])

cnt = collections.Counter(df.label)
rareClasses = set([k for k, v in cnt.items() if v < 100])
for cls in rareClasses:
    df = df[df.label != cls]

df.label.str.get_dummies()
df = df.join(df.label.str.get_dummies())
pdb.set_trace()
df.to_csv(r'train.csv')
