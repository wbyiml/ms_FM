import os
import shutil


from mindspore import Tensor, nn, Model, context
from mindspore.train.callback import LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint

from dataset import create_dataset
from model import FactorizationMachineModel

import mindspore.dataset.core.config as config  # ds.config 或 ds.core.config
config.set_num_parallel_workers(2)


# CUDA_VISIBLE_DEVICES=1 python train.py --dataset_path datas/miniset/train.txt --device_target GPU --rebuild_cache


# TODO：优化dataset，数据预处理
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', help='criteo, datas/train.txt, datas/miniset/train.txt')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=1e-6)  # 1e-6
    parser.add_argument('--save_dir', default='outputs')
    parser.add_argument('--device_target', type=str, default="CPU", choices=['GPU', 'CPU'])
    parser.add_argument('--rebuild_cache', action='store_true', help='rebuild cache data')
    args = parser.parse_args()

    context.set_context(
        # mode=context.GRAPH_MODE,
        mode=context.PYNATIVE_MODE,
        save_graphs=False,
        device_target=args.device_target)

    if os.path.exists('outputs'):
        shutil.rmtree('outputs')


    dataset,field_dims = create_dataset(args.dataset_path, args.batch_size,is_train=True,rebuild_cache=args.rebuild_cache)


    network = FactorizationMachineModel(field_dims, embed_dim=16) # 16


    loss = nn.BCELoss(reduction='mean')
    # optimizer = nn.Adam(network.trainable_params(), args.learning_rate, weight_decay=args.weight_decay)
    optimizer = nn.Momentum(network.trainable_params(), args.learning_rate, 0.9)

    model = Model(network, loss, optimizer, {'acc': nn.Accuracy()})
    print("============== Starting Training ==============")
    config_ck = CheckpointConfig(save_checkpoint_steps=2000,
                                 keep_checkpoint_max=5)
    ckpoint_cb = ModelCheckpoint(prefix="fm", directory=args.save_dir, config=config_ck)
    time_cb = TimeMonitor(data_size=dataset.get_dataset_size())
    loss_cb = LossMonitor()
    if args.device_target == "CPU":
        model.train(args.epoch, dataset, callbacks=[time_cb, loss_cb, ckpoint_cb], dataset_sink_mode=False)
    else:
        model.train(args.epoch, dataset, callbacks=[time_cb, loss_cb, ckpoint_cb])
    print("============== Training Success ==============")
