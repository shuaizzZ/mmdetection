
import sys
import os

from yutils.csv.csv import CSV

from mmcv.runner import HOOKS, Hook
from mmcv.runner.checkpoint import get_state_dict, weights_to_cpu
from mmcv.runner.dist_utils import allreduce_params, master_only


class CheckRunstateHook(Hook):
    def __init__(self, runstate):
        self.runstate = runstate

    def before_iter(self, runner):
        if self.runstate[0] == 0:
            sys.exit(0)

    def after_train_epoch(self, runner):
        # print('before', runner.data_loader.dataset.img_infos[1]['filename'])
        if hasattr(runner.data_loader.dataset, 'epoch_ops'):
            runner.data_loader.dataset.epoch_ops()
        # print('after', runner.data_loader.dataset.img_infos[1]['filename'])

class TrainerLogHook(Hook):
    def __init__(self, trainer_csv_path, num_classes=None, ndigits=3):
        self.trainer_csv_path = trainer_csv_path
        self.ndigits = ndigits
        assert isinstance(num_classes, int)
        self.num_classes = num_classes

    @master_only
    def before_run(self, runner):
        runner.best_metrics = ['mAP']
        runner.best_type = 'sum'
        if os.path.isfile(self.trainer_csv_path):
            os.remove(self.trainer_csv_path)
        self.log_csv = CSV(self.trainer_csv_path)

        log_head = ['epoch']
        for name in runner.best_metrics:
            log_head.append('{}_{}'.format(name, runner.best_type))
        # for c in range(1, self.num_classes):
        #     for name in runner.best_metrics:
        #         log_head.append('{}_{}'.format(name, c))
        self.log_csv.append(log_head)

    @master_only
    def after_train_epoch(self, runner):
        log_info = [runner.epoch]
        for name in runner.best_metrics:
            log_info.append(round(runner.log_buffer.output[name], self.ndigits))
        # for c in range(1, self.num_classes):
        #     for name in runner.best_metrics:
        #         log_info.append(round(runner.log_buffer.output[name]['class'][c], self.ndigits))
        self.log_csv.append(log_info)


@HOOKS.register_module()
class TrainerCheckpointHook(Hook):
    """Save checkpoints periodically.

    Args:
        interval (int): The saving period. If ``by_epoch=True``, interval
            indicates epochs, otherwise it indicates iterations.
            Default: -1, which means "never".
        by_epoch (bool): Saving checkpoints by epoch or by iteration.
            Default: True.
        save_optimizer (bool): Whether to save optimizer state_dict in the
            checkpoint. It is usually used for resuming experiments.
            Default: True.
        out_dir (str, optional): The directory to save checkpoints. If not
            specified, ``runner.work_dir`` will be used by default.
        max_keep_ckpts (int, optional): The maximum checkpoints to keep.
            In some cases we want only the latest few checkpoints and would
            like to delete old ones to save the disk space.
            Default: -1, which means unlimited.
        sync_buffer (bool): Whether to synchronize buffers in different
            gpus. Default: False.
    """

    def __init__(self,
                 interval=-1,
                 by_epoch=True,
                 save_optimizer=True,
                 out_dir=None,
                 max_keep_ckpts=-1,
                 sync_buffer=False,
                 best_type='sum',
                 **kwargs):
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.max_keep_ckpts = max_keep_ckpts
        self.args = kwargs
        self.sync_buffer = sync_buffer
        self.best_type = best_type

    def before_run(self, runner):
        runner.best_type = self.best_type
        runner.best_metrics = ['mAP']
        runner.best_eval_res = {}
        runner.cur_eval_res = {}
        for name in runner.best_metrics:
            runner.best_eval_res[name] = [0, 0]
            runner.cur_eval_res[name] = 0

    def after_train_epoch(self, runner):
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        if self.sync_buffer:
            allreduce_params(runner.model.buffers())
        self._save_checkpoint(runner)

    @master_only
    def _save_checkpoint(self, runner):
        """Save the current checkpoint and delete unwanted checkpoint."""
        if not self.out_dir:
            self.out_dir = runner.work_dir
        runner.save_checkpoint(
            self.out_dir, save_optimizer=self.save_optimizer, **self.args)

        for name, best_val in runner.best_eval_res.items():
            if name not in runner.best_metrics:
                continue
            cur_val = runner.log_buffer.output[name][runner.best_type]
            runner.cur_eval_res[name] = cur_val
            # if cur_val==1, that is mean : value = (0+smooth)/(0+smooth) = 1,
            # we thank that this should't be the best value.
            if cur_val >= best_val[0] and cur_val < 1:
                runner.best_eval_res[name] = [cur_val, runner.epoch+ 1]
                runner.save_checkpoint(
                    self.out_dir, save_optimizer=self.save_optimizer,
                    filename_tmpl=f'{name}_best_model.pth.tar', **self.args)

                runner.logger.info(f'Saving {name}_best checkpoint at {runner.epoch + 1} epochs')
            runner.log_buffer.output['best_pred_'+name] = runner.best_eval_res[name]

        if runner.meta is not None:
            if self.by_epoch:
                cur_ckpt_filename = self.args.get(
                    'filename_tmpl', 'epoch_{}.pth').format(runner.epoch + 1)
            else:
                cur_ckpt_filename = self.args.get(
                    'filename_tmpl', 'iter_{}.pth').format(runner.iter + 1)
            runner.meta.setdefault('hook_msgs', dict())
            runner.meta['hook_msgs']['last_ckpt'] = os.path.join(
                self.out_dir, cur_ckpt_filename)
        # remove other checkpoints
        if self.max_keep_ckpts > 0:
            if self.by_epoch:
                name = 'epoch_{}.pth'
                current_ckpt = runner.epoch + 1
            else:
                name = 'iter_{}.pth'
                current_ckpt = runner.iter + 1
            redundant_ckpts = range(
                current_ckpt - self.max_keep_ckpts * self.interval, 0,
                -self.interval)
            filename_tmpl = self.args.get('filename_tmpl', name)
            for _step in redundant_ckpts:
                ckpt_path = os.path.join(self.out_dir,
                                         filename_tmpl.format(_step))
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
                else:
                    break

    def after_train_iter(self, runner):
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return

        runner.logger.info(
            f'Saving checkpoint at {runner.iter + 1} iterations')
        if self.sync_buffer:
            allreduce_params(runner.model.buffers())
        self._save_checkpoint(runner)