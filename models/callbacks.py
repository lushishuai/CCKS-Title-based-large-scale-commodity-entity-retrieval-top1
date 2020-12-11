
import numpy as np
import torch
class LRWarmup(object):
    '''
    Bert模型内定的学习率变化机制
    Example:
    '''

    def __init__(self, optimizer,num_warmup_steps):

        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, training_step):
        if training_step<self.num_warmup_steps:
            for param_group, lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = lr * (training_step / self.num_warmup_steps)



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self,model=None, patience=2,mode='min', verbose=False,min_delta=0, restore_best_weights=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 3
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.model = model
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf

        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

        self.counter = 0
        self.best_score = self.best
        self.early_stop = False


    def step(self, current):

        if self.monitor_op(current- self.min_delta, self.best_score):

            self.best_score = current
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

            if self.restore_best_weights:
                if self.verbose > 0:
                    print('Restoring model weights from the end of '
                          'the best epoch')
                self.model.set_weights(self.best_weights)

class ModelCheckpoint(object):
    '''
    模型保存，两种模式：
    1. 直接保存最好模型
    2. 按照epoch频率保存模型
    '''
    def __init__(self,model, checkpoint_dir,

                 # monitor,
                 mode='min',
                 epoch_freq=1,
                 best = None,
                 save_best_only = True):
        # if isinstance(checkpoint_dir,Path):
        #     checkpoint_dir = checkpoint_dir
        # else:
        #     checkpoint_dir = Path(checkpoint_dir)
        # assert checkpoint_dir.is_dir()
        # checkpoint_dir.mkdir(exist_ok=True)
        self.model = model
        self.base_path = checkpoint_dir
        # self.arch = arch
        # self.monitor = monitor
        self.epoch_freq = epoch_freq
        self.save_best_only = save_best_only

        # 计算模式
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf

        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        # 这里主要重新加载模型时候
        #对best重新赋值
        if best:
            self.best = best

        # if save_best_only:
        #     self.model_name = f"BEST_{arch}_MODEL.pth"

    def epoch_step(self, current):
        '''
        正常模型
        :param state: 需要保存的信息
        :param current: 当前判断指标
        :return:
        '''
        # 是否保存最好模型


        if self.monitor_op(current, self.best):
            # logger.info(f"\nEpoch {state['epoch']}: {self.monitor} improved from {self.best:.5f} to {current:.5f}")
            self.best = current
            # state['best'] = self.best
            # best_path = self.base_path/ self.model_name
            # torch.save(state, str(best_path))
            torch.save(self.model.state_dict(), self.base_path)
        # if self.save_best_only:
        #
        # if self.monitor_op(current, self.best):
        #     logger.info(f"\nEpoch {state['epoch']}: {self.monitor} improved from {self.best:.5f} to {current:.5f}")
        #     self.best = current
        #     state['best'] = self.best
        #     best_path = self.base_path/ self.model_name
        #     torch.save(state, str(best_path))


        # if self.save_best_only:
        #     if self.monitor_op(current, self.best):
        #         logger.info(f"\nEpoch {state['epoch']}: {self.monitor} improved from {self.best:.5f} to {current:.5f}")
        #         self.best = current
        #         state['best'] = self.best
        #         best_path = self.base_path/ self.model_name
        #         torch.save(state, str(best_path))
        # # 每隔几个epoch保存下模型
        # else:
        #     filename = self.base_path / f"EPOCH_{state['epoch']}_{state[self.monitor]}_{self.arch}_MODEL.pth"
        #     if state['epoch'] % self.epoch_freq == 0:
        #         logger.info(f"\nEpoch {state['epoch']}: save model to disk.")
        #         torch.save(state, str(filename))

    def bert_epoch_step(self, state,current):
        '''
        适合bert类型模型，适合pytorch_transformer模块
        :param state:
        :param current:
        :return:
        '''
        model_to_save = state['model']
        if self.save_best_only:
            if self.monitor_op(current, self.best):
                logger.info(f"\nEpoch {state['epoch']}: {self.monitor} improved from {self.best:.5f} to {current:.5f}")
                self.best = current
                state['best'] = self.best
                model_to_save.save_pretrained(str(self.base_path))
                output_config_file = self.base_path / 'configs.json'
                with open(str(output_config_file), 'w') as f:
                    f.write(model_to_save.config.to_json_string())
                state.pop("model")
                torch.save(state,self.base_path / 'checkpoint_info.bin')
        else:
            if state['epoch'] % self.epoch_freq == 0:
                save_path = self.base_path / f"checkpoint-epoch-{state['epoch']}"
                save_path.mkdir(exist_ok=True)
                logger.info(f"\nEpoch {state['epoch']}: save model to disk.")
                model_to_save.save_pretrained(save_path)
                output_config_file = save_path / 'configs.json'
                with open(str(output_config_file), 'w') as f:
                    f.write(model_to_save.config.to_json_string())
                state.pop("model")
                torch.save(state, save_path / 'checkpoint_info.bin')
