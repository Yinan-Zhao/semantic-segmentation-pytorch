import numpy as np
from scipy.spatial import distance_matrix

train_input = '/mnt/lustre/zhaoyinan/semantic-segmentation-pytorch/feature/feat_train.npy'
val_input = '/mnt/lustre/zhaoyinan/semantic-segmentation-pytorch/feature/feat_val.npy'
feat_train = np.load(train_input)
feat_val = np.load(val_input)

print('computing distance matrix')
dis_val = distance_matrix(feat_val, feat_train)
print('done')
with open('/mnt/lustre/zhaoyinan/semantic-segmentation-pytorch/data/ref_real_val.txt', 'w') as f:
    for i in range(dis_val.shape[0]):
        print(i)
        dis = dis_val[i]
        order = np.argsort(dis)
        for k, item in enumerate(order):
            if k != len(order)-1:
                f.write('%d '%(item))
            else:
                f.write('%d\n'%(item))

'''print('computing distance matrix')
dis_train = distance_matrix(feat_train, feat_train)
print('done')
with open('/mnt/lustre/zhaoyinan/semantic-segmentation-pytorch/data/ref_real_training.txt', 'w') as f:
    for i in range(dis_train.shape[0]):
        print(i)
        dis = dis_train[i]
        order = np.argsort(dis)
        for k, item in enumerate(order):
            if k != len(order)-1:
                f.write('%d '%(item))
            else:
                f.write('%d\n'%(item))'''



