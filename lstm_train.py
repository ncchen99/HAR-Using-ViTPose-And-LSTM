# %%
DATASET_PATH = "./swenbao_jump_data/"

from argparse import ArgumentParser

def configuration_parser(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--data_root', type=str, default=DATASET_PATH)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--num_class', type=int, default=2)
    return parser

# %%

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from src.lstm_vitpose import ActionClassificationLSTM, PoseDataModule

# %%



def do_training_validation(datapath):
    pl.seed_everything(21)    
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = configuration_parser(parser)
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    # init model    
    data_module = PoseDataModule(data_root=args.data_root,
                                        batch_size=args.batch_size, datapath=datapath) 
    model = ActionClassificationLSTM(34, 50, learning_rate=args.learning_rate)
    #save only the top 1 model based on val_loss
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor='val_loss')
    lr_monitor = LearningRateMonitor(logging_interval='step')  
    #trainer
    trainer = pl.Trainer.from_argparse_args(args,
        # fast_dev_run=True,
        max_epochs=args.epochs, 
        deterministic=True, 
        gpus=1, 
        progress_bar_refresh_rate=1, 
        callbacks=[EarlyStopping(monitor='train_loss', patience=15), checkpoint_callback, lr_monitor])    
    trainer.fit(model, data_module)    
    return model

"""
# To reload tensorBoard
%load_ext tensorboard

# logs folder path
%tensorboard --logdir=lightning_logs
"""
import os
def get_latest_run_version_ckpt_epoch_no(lightning_logs_dir='lightning_logs', run_version=None):
    if run_version is None:
        run_version = 0
        for dir_name in os.listdir(lightning_logs_dir):
            if 'version' in dir_name:
                if int(dir_name.split('_')[1]) > run_version:
                    run_version = int(dir_name.split('_')[1])                
    checkpoints_dir = os.path.join(lightning_logs_dir, 'version_{}'.format(run_version), 'checkpoints')    
    files = os.listdir(checkpoints_dir)
    ckpt_filename = None
    for file in files:
        print(file)
        if file.endswith('.ckpt'):
            ckpt_filename = file        
    if ckpt_filename is not None:
        ckpt_path = os.path.join(checkpoints_dir, ckpt_filename)
    else:
        print('CKPT file is not present')    
    return ckpt_path

# %%



for i in range(0, 7):
    datapath = {
        "train_data": f"{DATASET_PATH}train/data_{i}.csv",
        "train_info": f"{DATASET_PATH}train/info_{i}.csv",
        "test_data": f"{DATASET_PATH}test/data_{i}.csv",
        "test_info": f"{DATASET_PATH}test/info_{i}.csv",
    }
    do_training_validation(datapath)
    ckpt_path = get_latest_run_version_ckpt_epoch_no()
    print('The latest model path: {}'.format(ckpt_path))


# %%

