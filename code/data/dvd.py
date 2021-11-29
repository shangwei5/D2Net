import os
from data import videodata


class DVD(videodata.VIDEODATA):
    def __init__(self, args, name='DVD', train=True):
        super(DVD, self).__init__(args, name=name, train=train)

    def _set_filesystem(self, dir_data):
        print("Loading {} => {} DataSet".format("train" if self.train else "test", self.name))
        self.apath = dir_data
        self.dir_gt = os.path.join(self.apath, 'gt')
        self.dir_input = os.path.join(self.apath, 'blur')
        self.dir_bm = os.path.join(self.apath, 'Blur_map')
        self.dir_label = os.path.join(self.apath, 'label')
        print("DataSet GT path:", self.dir_gt)
        print("DataSet INPUT path:", self.dir_input)
        print("DataSet Blur_map path:", self.dir_bm)
        print("DataSet label path:", self.dir_label)
