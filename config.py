import os
import math

class Config():
    def __init__(self) -> None:
        self.sys_home_dir = [os.path.expanduser('~'), '/adasnet'][1]  
        self.data_root_dir = '..'
        self.task = 'COD'
        self.testsets = ','.join(['CHAMELEON', 'NC4K', 'TEST-CAMO', 'TEST-COD10K'])
        self.training_set = 'TR-COD10K+TR-CAMO'

        self.size = (1024, 1024)
        self.dynamic_size = [None, ((512-256, 2048+256), (512-256, 2048+256))][0]
        self.background_color_synthesis = False

        self.load_all = False and self.dynamic_size is None
        self.compile = True
        self.precisionHigh = True

        self.ms_supervision = True
        self.cxt_num = [0, 3][1]
        self.mul_scl_ipt = ['', 'add', 'cat'][2]
        self.dec_att = ['', 'SAP'][1]
        self.squeeze_block = ['', 'SAPDBasic_x1'][1]
        self.dec_blk = ['SAPDBasic', 'SAPD'][1]

        self.batch_size = 4
        self.finetune_last_epochs = -20
        self.lr = 1e-5 * math.sqrt(self.batch_size / 4)
        self.num_workers = max(4, self.batch_size)

        self.bb = [
            'vgg16', 'vgg16bn', 'resnet50',
            'swin_v1_t', 'swin_v1_s',
            'swin_v1_b', 'swin_v1_l',
            'pvt_v2_b0', 'pvt_v2_b1',
            'pvt_v2_b2', 'pvt_v2_b5',
            'mamba_vision_t_1k',
            'res2net50_26w_4s',
        ][6]    
 
        self.lateral_channels_in_collection = {
            'vgg16': [512, 256, 128, 64], 'vgg16bn': [512, 256, 128, 64], 'resnet50': [1024, 512, 256, 64],
            'pvt_v2_b2': [512, 320, 128, 64], 'pvt_v2_b5': [512, 320, 128, 64],
            'swin_v1_b': [1024, 512, 256, 128], 'swin_v1_l': [1536, 768, 384, 192],
            'swin_v1_t': [768, 384, 192, 96], 'swin_v1_s': [768, 384, 192, 96],
            'pvt_v2_b0': [256, 160, 64, 32], 'pvt_v2_b1': [512, 320, 128, 64],
            'mamba_vision_t_1k': [640, 320, 160, 80],
            'res2net50_26w_4s': [2048, 1024, 512, 256],
        }[self.bb]
        if self.mul_scl_ipt == 'cat':
            self.lateral_channels_in_collection = [channel * 2 for channel in self.lateral_channels_in_collection]
        self.cxt = self.lateral_channels_in_collection[1:][::-1][-self.cxt_num:] if self.cxt_num else []

        self.lat_blk = ['FEM'][0]
        self.dec_channels_inter = ['fixed', 'adap'][0]
        self.freeze_bb = False
        self.model = [
            'adasnet',
        ][0]

        self.preproc_methods = ['flip', 'enhance', 'rotate', 'pepper', 'crop'][:4 if not self.background_color_synthesis else 1]
        self.optimizer = ['Adam', 'AdamW'][1]
        self.lr_decay_epochs = [1e5]
        self.lr_decay_rate = 0.5
        self.lambdas_pix_last = {
            'bce': 60,
            'iou': 1,
            'dice': 2,
            'ssim': 20,
        }

        self.weights_root_dir = os.path.join(self.sys_home_dir, 'weights/backbone')
        self.weights = {
            'pvt_v2_b2': os.path.join(self.weights_root_dir, 'pvt_v2_b2.pth'),
            'pvt_v2_b5': os.path.join(self.weights_root_dir, ['pvt_v2_b5.pth', 'pvt_v2_b5_22k.pth'][0]),
            'swin_v1_b': os.path.join(self.weights_root_dir, ['swin_base_patch4_window12_384_22kto1k.pth', 'swin_base_patch4_window12_384_22k.pth'][0]),
            'swin_v1_l': os.path.join(self.weights_root_dir, ['swin_large_patch4_window12_384_22kto1k.pth', 'swin_large_patch4_window12_384_22k.pth'][0]),
            'swin_v1_t': os.path.join(self.weights_root_dir, ['swin_tiny_patch4_window7_224_22kto1k_finetune.pth'][0]),
            'swin_v1_s': os.path.join(self.weights_root_dir, ['swin_small_patch4_window7_224_22kto1k_finetune.pth'][0]),
            'pvt_v2_b0': os.path.join(self.weights_root_dir, ['pvt_v2_b0.pth'][0]),
            'pvt_v2_b1': os.path.join(self.weights_root_dir, ['pvt_v2_b1.pth'][0]),
            'mamba_vision_t_1k': 'mamba_vision_t_1k.pt',
            'res2net50_26w_4s': 'mamba_vision_t_1k.pt',
        }

        self.verbose_eval = True
        self.only_S_MAE = False
        self.SDPA_enabled = False

        self.device = [0, 'cpu'][0]

        self.batch_size_valid = 1
        self.rand_seed = 7
        run_sh_file = [f for f in os.listdir('.') if 'train.sh' == f] + [os.path.join('..', f) for f in os.listdir('..') if 'train.sh' == f]
        if run_sh_file:
            with open(run_sh_file[0], 'r') as f:
                lines = f.readlines()
                self.save_last = int([l.strip() for l in lines if "'{}')".format(self.task) in l and 'val_last=' in l][0].split('val_last=')[-1].split()[0])
                self.save_step = int([l.strip() for l in lines if "'{}')".format(self.task) in l and 'step=' in l][0].split('step=')[-1].split()[0])


if __name__ == '__main__':
    import argparse


    parser = argparse.ArgumentParser(description='Only choose one argument to activate.')
    parser.add_argument('--print_task', action='store_true', help='print task name')
    parser.add_argument('--print_testsets', action='store_true', help='print validation set')
    args = parser.parse_args()

    config = Config()
    for arg_name, arg_value in args._get_kwargs():
        if arg_value:
            print(config.__getattribute__(arg_name[len('print_'):]))

