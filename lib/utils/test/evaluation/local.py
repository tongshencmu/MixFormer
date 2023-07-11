from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/got10k_lmdb'
    settings.got10k_path = '/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = '/lasot_lmdb'
    settings.lasot_path = '/lasot_new/LaSOTBenchmark'
    settings.network_path = '/raid/Mixformer/work_dirs/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/nfs'
    settings.otb_path = '/OTB2015'
    settings.prj_dir = '/raid/Mixformer/work_dirs'
    settings.result_plot_path = '/raid/Mixformer/work_dirs/test/result_plots'
    settings.results_path = '/raid/Mixformer/work_dirs/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/raid/Mixformer/work_dirs'
    settings.segmentation_path = '/raid/Mixformer/work_dirs/test/segmentation_results'
    settings.tc128_path = '/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/trackingNet'
    settings.uav_path = '/UAV123'
    settings.vot_path = '/VOT2019'
    settings.youtubevos_dir = ''

    return settings

