from configs.xception_bsl import *


# print('Start train...')
# for epoch in range(100):
#     train.train(train_loader)
#     train.val(val_loader)
train.test(test_loader)