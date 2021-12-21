def set_template(args):
    if args.template == 'D2NET':
        args.task = "VideoDeblur"
        args.model = "D2NET"
        args.n_sequence = 3
        args.n_frames_per_video = 200
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 4
        args.loss = '1*L1+2*HEM'
        args.lr = 1e-4
        args.lr_decay = 200
        args.data_train = 'DVD_NFS'
        args.data_test = 'DVD_NFS'
        args.dir_data = '../dataset/GOPRO_Random/train'
        args.dir_data_test = '../dataset/GOPRO_Random/test'
        args.epochs = 500
        args.batch_size = 8
        args.n_GPUs = 2
    elif args.template == 'D2NET_EVENT':
        args.task = "VideoDeblur"
        args.model = "D2NET_EVENT"
        args.n_sequence = 3
        args.n_frames_per_video = 200
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 4
        args.loss = '1*L1+2*HEM'
        args.lr = 1e-4
        args.lr_decay = 200
        args.data_train = 'DVD_NFS_EVENT'
        args.data_test = 'DVD_NFS_EVENT'
        args.dir_data = '../dataset/GOPRO_Random/train'
        args.dir_data_test = '../dataset/GOPRO_Random/test'
        args.epochs = 500
        args.batch_size = 8
        args.n_GPUs = 2
        args.pre_train = './logs/model_best.pt'
    elif args.template == 'D2NET_Offical':
        args.task = "VideoDeblur"
        args.model = "D2NET_Offical"
        args.n_sequence = 5
        args.n_frames_per_video = 300
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 4
        args.loss = '1*L1+2*HEM'
        args.lr = 1e-4
        args.lr_decay = 200
        args.dir_data = '../dataset/GOPRO_Random/train'
        args.dir_data_test = '../dataset/GOPRO_Random/test'
        args.epochs = 500
        args.batch_size = 8
        args.n_GPUs = 2
        args.pre_train = './logs/model_best.pt'
    else:
        raise NotImplementedError('Template [{:s}] is not found'.format(args.template))
