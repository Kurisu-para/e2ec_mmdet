from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class E2ECModeSwitchHook(Hook):
    def __init__(self, start_epochs=10):
        self.start_epochs = start_epochs

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        if epoch >= self.start_epochs:
            runner.logger.info('Add additional DM loss now!')
            model.mask_head.use_dm = True
