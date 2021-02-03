import traceback

# try:
import os
import sys
import time
import copy
import shutil
import os.path as osp

import torch
import mmcv
from mmcv.runner import init_dist, load_checkpoint
from mmcv.utils import Config, DictAction, get_git_hash
from mmcv.parallel import MMDataParallel

from mmdet import __version__
from mmdet.apis import set_random_seed, trainer_detector
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger

# except Exception as ex:
#     ex_type, ex_val, ex_stack = sys.exc_info()
#     print('ex_type:',ex_type)
#     print('ex_val:',ex_val)
#     for stack in traceback.extract_tb(ex_stack):
#         print(stack)


def merge_to_mmcfg_from_mvcfg(mmcfg, mvcfg):
    return mmcfg
    def modify_if_exist(mmpara, mmfields, mvpara, mvfields):
        for i in range(len(mvfields)):
            mmfield = mmfields[i]
            mvfield = mvfields[i]
            if mvpara.get(mvfield, None):
                mmpara[mmfield] = mvpara.get(mvfield)
    ## model
    mmcfg.img_norm_cfg.type = "BN"
    for module, config in mmcfg.model.items():
        if isinstance(config, dict) and 'img_norm_cfg' in config.keys():
            mmcfg._cfg_dict['model'][module]['img_norm_cfg']['type'] = mmcfg.img_norm_cfg.type

    ## num_classes
    mmcfg.classes = mvcfg.DATASETS.CLASSES
    mmcfg.labels = mvcfg.DATASETS.LABELS

    num_classes = max(mmcfg.labels) + 1
    if len(mmcfg.classes) < num_classes:
        for i in range(num_classes - len(mmcfg.classes)):
            mmcfg.classes.append('unnamed_cls{}'.format(i+len(mmcfg.classes)+1))
    elif len(mmcfg.classes) > num_classes:
        mmcfg.classes = mmcfg.classes[:len(mmcfg.classes) - num_classes]
    # modify num_classes in model
    mmcfg.num_classes = num_classes
    for key, val in mmcfg.model.items():
        if 'num_classes' in val:
            mmcfg.model[key]['num_classes'] = num_classes

    ## dataset
    if mmcfg.data_root != mvcfg.DATASETS.ROOT:
        mmcfg.data_root = mvcfg.DATASETS.ROOT
    for mode in ['train', 'val', 'test']:
        modify_if_exist(mmcfg._cfg_dict['data'][mode], ['type'],
                        mmcfg._cfg_dict, ['dataset_type'])
        for para in ['data_root', 'dataset', 'classes', 'labels']:
            modify_if_exist(mmcfg._cfg_dict['data'][mode], [para],
                            mmcfg._cfg_dict, [para])

    ## pipeline train
    # mmcfg.crop_size = mvcfg.DATASETS.AUGMENT.CROP_SIZE
    # option_para = {'Relabel': ['labels'],
    #                'MVCrop': ['crop_size'],}
    # for i, trans_dict in enumerate(mmcfg.train_pipeline):
    #     trans_type = trans_dict.type
    #     if trans_type in option_para.keys():
    #         modify_if_exist(mmcfg._cfg_dict['train_pipeline'][i],
    #                         option_para[trans_type],
    #                         mmcfg._cfg_dict,
    #                         option_para[trans_type])
    # mmcfg.data.train.pipeline = mmcfg.train_pipeline

    ## pipeline val
    # option_para = {'Relabel': ['labels'],
    #                'MVCrop': ['crop_size'],}
    # for i, trans_dict in enumerate(mmcfg.val_pipeline):
    #     if trans_dict.type != 'MultiScaleFlipAug':
    #         continue
    #     for j, aug_dict in enumerate(mmcfg.val_pipeline[i].transforms):
    #         if aug_dict.type in option_para.keys():
    #             modify_if_exist(mmcfg._cfg_dict['val_pipeline'][i].transforms[j],
    #                             option_para[aug_dict.type],
    #                             mmcfg._cfg_dict,
    #                             option_para[aug_dict.type])
    # mmcfg.data.val.pipeline = mmcfg.val_pipeline

    ## pipeline test
    # option_para = {'MVCrop': ['crop_size'],}
    # for i, trans_dict in enumerate(mmcfg.test_pipeline):
    #     trans_type = trans_dict.type
    #     if trans_type in option_para.keys():
    #         modify_if_exist(mmcfg._cfg_dict['test_pipeline'][i],
    #                         option_para[trans_type],
    #                         mmcfg._cfg_dict,
    #                         option_para[trans_type])
    # mmcfg.data.test.pipeline = mmcfg.test_pipeline

    # mmcfg.data.samples_per_gpu = mvcfg.TRAIN.BATCH_SIZE
    # mmcfg.data.workers_per_gpu = 0

    ## schedule
    mmcfg.total_epochs = mvcfg.TRAIN.END_EPOCH
    ## runtime
    # if mvcfg.TRAIN.FT.RESUME:
    #     mmcfg.load_from = mvcfg.TRAIN.FT.CHECKPATH
    mmcfg.work_dir = osp.join(mmcfg.data_root, 'models')

    return mmcfg


class ainnovision():
    def init(self):
        self.py_dir = os.path.split(__file__)[0]
        print ("python ainnovision init")

    def train(self, runstate):
        try:
            self.train_py(runstate)
        except Exception as ex:
            ex_type, ex_val, ex_stack = sys.exc_info()
            print('ex_type:',ex_type)
            print('ex_val:',ex_val)
            for stack in traceback.extract_tb(ex_stack):
                print(stack)

    def train_py(self, runstate):
        # manuvision config
        mv_config_file = "ainnovision_train.yaml"
        mv_config_path = os.path.join(self.py_dir, mv_config_file)
        mvcfg = Config.fromfile(mv_config_path)
        # mmseg config
        # mm_config_path = '../configs/pspnet/pspnet_r50-d8_yantai_st12.py'
        mm_config_file = "mm_det.py"
        mm_config_path = os.path.join(self.py_dir, mm_config_file)
        mmcfg = Config.fromfile(mm_config_path)
        cfg = merge_to_mmcfg_from_mvcfg(mmcfg, mvcfg)

        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        # init distributed env first, since logger depends on the dist info.
        if cfg.get('launcher', 'none') == 'none' or len(cfg.gpu_ids) == 1:
            distributed = False
        else:
            distributed = True
            init_dist(cfg.launcher, **cfg.dist_params)

        # create work_dir
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        cfg.trainer_csv_path = osp.join(cfg.data_root, 'train_log.csv')
        # dump config
        cfg.dump(osp.join(cfg.work_dir, osp.basename(mm_config_path)))
        # init the logger before other steps
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(cfg.work_dir, 'train.log')
        logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

        # init the meta dict to record some important information such as
        # environment info and seed, which will be logged
        meta = dict()
        # log env info
        env_info_dict = collect_env()
        env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                    dash_line)
        meta['env_info'] = env_info

        # log some basic info
        logger.info(f'Distributed training: {distributed}')
        logger.info(f'Config:\n{cfg.pretty_text}')

        # set random seeds
        cfg.seed = cfg.get('seed', None)
        if cfg.seed is not None:
            logger.info(f'Set random seed to {cfg.seed}, deterministic: '
                        f'{cfg.deterministic}')
            set_random_seed(cfg.seed, deterministic=cfg.deterministic)

        meta['seed'] = cfg.seed
        meta['exp_name'] = osp.basename(mm_config_path)

        # validate
        cfg.validate = cfg.get('validate', True)

        model = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))

        logger.info(model)

        datasets = [build_dataset(cfg.data.train)]
        if len(cfg.workflow) == 2:
            val_dataset = copy.deepcopy(cfg.data.val)
            val_dataset.pipeline = cfg.data.val.pipeline
            datasets.append(build_dataset(val_dataset))
        if cfg.checkpoint_config is not None:
            # save mmseg version, config file content and class names in
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(
                mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
                config=cfg,
                CLASSES=datasets[0].CLASSES,
                # PALETTE=datasets[0].PALETTE,
            )
        # add an attribute for visualization convenience
        model.CLASSES = datasets[0].CLASSES
        trainer_detector(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=cfg.validate,
            timestamp=timestamp,
            meta=meta,
            runstate=runstate)

    def inference(self, runstate):
        try:
            self.inference_py(runstate)
        except Exception as ex:
            ex_type, ex_val, ex_stack = sys.exc_info()
            print('ex_type:',ex_type)
            print('ex_val:',ex_val)
            for stack in traceback.extract_tb(ex_stack):
                print(stack)

    def inference_py(self, runstate):
        # manuvision config
        mv_config_file = "ainnovision_train.yaml"
        mv_config_path = os.path.join(self.py_dir, mv_config_file)
        mvcfg = Config.fromfile(mv_config_path)
        # mmseg config
        # mm_config_path = '../configs/pspnet/pspnet_r50-d8_yantai_st12.py'
        mm_config_file = "mm_det.py"
        mm_config_path = os.path.join(self.py_dir, mm_config_file)
        mmcfg = Config.fromfile(mm_config_path)
        cfg = merge_to_mmcfg_from_mvcfg(mmcfg, mvcfg)

        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        cfg.model.pretrained = None
        cfg.data.test.test_mode = True

        # init distributed env first, since logger depends on the dist info.
        if cfg.get('launcher', 'none') == 'none' or len(cfg.gpu_ids) == 1:
            distributed = False
        else:
            distributed = True
            init_dist(cfg.launcher, **cfg.dist_params)

        # build the dataloader
        # TODO: support multiple images per gpu (only minor changes are needed)
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=0,
            dist=distributed,
            shuffle=False)

        # build the model and load checkpoint
        model = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        model_path = osp.join(cfg.work_dir, 'F1_best_model.pth.tar')
        checkpoint = load_checkpoint(model, model_path, map_location='cpu')
        config = checkpoint['meta']['config']

        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            mv_single_gpu_test(model, data_loader, runstate=runstate,
                               draw_contours=True, out_dir=cfg.data_root)

    def convert(self, runstate, mode=0):
        try:
            self.convert_py(runstate, mode)
        except Exception as ex:
            ex_type, ex_val, ex_stack = sys.exc_info()
            print('ex_type:',ex_type)
            print('ex_val:',ex_val)
            for stack in traceback.extract_tb(ex_stack):
                print(stack)

    def convert_py(self, runstate, mode=0):
        # manuvision config
        mv_config_file = "ainnovision_train.yaml"
        mv_config_path = os.path.join(self.py_dir, mv_config_file)
        mvcfg = Config.fromfile(mv_config_path)
        # mmseg config
        # mm_config_path = '../configs/pspnet/pspnet_r50-d8_yantai_st12.py'
        mm_config_file = "mm_det.py"
        mm_config_path = os.path.join(self.py_dir, mm_config_file)
        mmcfg = Config.fromfile(mm_config_path)
        cfg = merge_to_mmcfg_from_mvcfg(mmcfg, mvcfg)

        checkpath = osp.join(cfg.work_dir, 'F1_best_model.pth.tar')
        checkpoint = torch.load(checkpath, map_location='cpu')
        model_cfg = checkpoint['meta']["config"].model

        # build the model and load checkpoint
        model_cfg.pretrained = None
        segmentor = build_segmentor(
            model_cfg, train_cfg=None, test_cfg=cfg.test_cfg)
        # convert SyncBN to BN
        segmentor = _convert_batchnorm(segmentor)
        checkpoint = load_checkpoint(segmentor, checkpath, map_location='cpu')

        # conver model to onnx file
        input_shape = (1, 3, cfg.convert_size[0], cfg.convert_size[1])
        output_file = osp.join(cfg.work_dir, 'F1_best_model.onnx')
        pytorch2onnx(
            segmentor,
            input_shape,
            opset_version=10,
            show=True,
            output_file=output_file,
            verify=True)

    def uninit(self):
        print("python ainnovision uninit")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    runstate = np.array([1])

    mv = ainnovision()
    mv.init()
    mv.train_py(runstate)
    # mv.inference_py(runstate)
    # mv.convert_py(runstate, 0)
