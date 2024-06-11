from easydict import EasyDict as edict

cfg = edict()

cfg.epochs = 10000
# cfg.one_batch_size = 625
# cfg.row = cfg.one_batch_size
# 求能量做为fft的输入
cfg.one_batch_size = 125 * 6
cfg.row = 6
cfg.column = 6
# cfg.category = 4
cfg.category = 3    #孕妇的睡姿只有三类

cfg.batch_size = 32
cfg.train_learning_rate = 1e-4
cfg.val_rate = 0.2
cfg.train_dataset_path = "../process_data/train.tfrecords"
cfg.val_dataset_path = "../process_data/val.tfrecords"
cfg.test_dataset_path = "../process_data/test.tfrecords"
cfg.people_num = 30
cfg.simple_num_file = cfg.people_num * cfg.category
cfg.train_num_samples = int(75000/125/6 * (cfg.people_num * cfg.category) * (1 - cfg.val_rate))
cfg.val_num_samples = int(75000/125/6 * (cfg.people_num * cfg.category) * cfg.val_rate)
cfg.test_num_samples = (75000/125/6 * cfg.category)

