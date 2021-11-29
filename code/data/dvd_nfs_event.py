import os
from data import videodata_nfs_event


class DVD_NFS_EVENT(videodata_nfs_event.VIDEODATA):
    def __init__(self, args, name='DVD', train=True):
        super(DVD_NFS_EVENT, self).__init__(args, name=name, train=train)

    def _set_filesystem(self, dir_data):
        print("Loading {} => {} DataSet".format("train" if self.train else "test", self.name))
        self.apath = dir_data
        self.dir_gt = os.path.join(self.apath, 'gt')
        self.dir_input = os.path.join(self.apath, 'blur')
        self.dir_label = os.path.join(self.apath, 'label')
        self.dir_event = os.path.join(self.apath, 'Event_npy')
        print("DataSet GT path:", self.dir_gt)
        print("DataSet INPUT path:", self.dir_input)
        print("DataSet label path:", self.dir_label)
        print("DataSet event path:", self.dir_event)
