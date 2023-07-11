class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/raid/Mixformer/work_dirs'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/raid/Mixformer/work_dirs/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/raid/Mixformer/work_dirs/pretrained_networks'
        self.lasot_dir = '/lasot/LaSOT/LaSOT_benchmark'
        self.got10k_dir = '/tracking_data/got10k/train'
        self.lasot_lmdb_dir = '/lasot_lmdb'
        self.got10k_lmdb_dir = '/got10k_lmdb'
        self.trackingnet_dir = '/tracking_data/trackingnet'
        self.trackingnet_lmdb_dir = '/trackingnet_lmdb'
        self.coco_dir = '/tracking_data/coco'
        self.coco_lmdb_dir = '/coco_lmdb'
        self.tnl2k_dir = '/tracking_data/tnl2k'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/vid'
        self.imagenet_lmdb_dir = '/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
