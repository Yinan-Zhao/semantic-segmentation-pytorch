# System libs
import os
import time
# import math
import random
import argparse
from distutils.version import LooseVersion
# Numerical libs
import torch
import torch.nn as nn
import numpy as np
# Our libs
from config import cfg
from dataset_memory_separate import TrainDataset
from models import ModelBuilder, SegmentationAttentionSeparateModule
from utils import AverageMeter, parse_devices, setup_logger
from lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback


# train one epoch
def train(segmentation_module, iterator, optimizers, history, epoch, cfg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()

    segmentation_module.train(not cfg.TRAIN.fix_bn)

    # main loop
    tic = time.time()
    qread_all = []
    qval_all = []
    for i in range(100):
        # load a batch of data
        batch_data = next(iterator)
        data_time.update(time.time() - tic)
        segmentation_module.zero_grad()

        # adjust learning rate
        cur_iter = i + (epoch - 1) * cfg.TRAIN.epoch_iters
        adjust_learning_rate(optimizers, cur_iter, cfg)

        # forward pass
        #print(batch_data)
        loss, acc, qread, qval, mv_b, p = segmentation_module(batch_data)
        loss = loss.mean()
        acc = acc.mean()

        qread_all.append(qread.detach().cpu().float().numpy())
        qval_all.append(qval.detach().cpu().float().numpy())
        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()
        np.save('debug/qread_%03d.npy'%(i), qread_all[i])
        np.save('debug/qval_%03d.npy'%(i), qval_all[i])
        np.save('debug/mv_b_%03d.npy'%(i), mv_b)
        np.save('debug/p_%03d.npy'%(i), p)


def checkpoint(nets, history, cfg, epoch):
    print('Saving checkpoints...')
    (net_enc_query, net_enc_memory, net_att_query, net_att_memory, net_decoder, crit) = nets

    dict_enc_query = net_enc_query.state_dict()
    dict_enc_memory = net_enc_memory.state_dict()
    dict_att_query = net_att_query.state_dict()
    dict_att_memory = net_att_memory.state_dict()
    dict_decoder = net_decoder.state_dict()

    torch.save(
        history,
        '{}/history_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(
        dict_enc_query,
        '{}/enc_query_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(
        dict_enc_memory,
        '{}/enc_memory_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(
        dict_att_query,
        '{}/att_query_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(
        dict_att_memory,
        '{}/att_memory_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(
        dict_decoder,
        '{}/decoder_epoch_{}.pth'.format(cfg.DIR, epoch))


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def create_optimizers(nets, cfg):
    (net_enc_query, net_enc_memory, net_att_query, net_att_memory, net_decoder, crit) = nets
    optimizer_enc_query = torch.optim.SGD(
        group_weight(net_enc_query),
        lr=cfg.TRAIN.lr_encoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    optimizer_enc_memory = torch.optim.SGD(
        group_weight(net_enc_memory),
        lr=cfg.TRAIN.lr_encoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    optimizer_att_query = torch.optim.SGD(
        group_weight(net_att_query),
        lr=cfg.TRAIN.lr_encoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    optimizer_att_memory = torch.optim.SGD(
        group_weight(net_att_memory),
        lr=cfg.TRAIN.lr_encoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    optimizer_decoder = torch.optim.SGD(
        group_weight(net_decoder),
        lr=cfg.TRAIN.lr_decoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    return (optimizer_enc_query, optimizer_enc_memory, optimizer_att_query, optimizer_att_memory, optimizer_decoder)


def adjust_learning_rate(optimizers, cur_iter, cfg):
    scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.max_iters) ** cfg.TRAIN.lr_pow)
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder * scale_running_lr
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder * scale_running_lr

    (optimizer_enc_query, optimizer_enc_memory, optimizer_att_query, optimizer_att_memory, optimizer_decoder) = optimizers
    for param_group in optimizer_enc_query.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_encoder
    for param_group in optimizer_enc_memory.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_encoder
    for param_group in optimizer_att_query.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_encoder
    for param_group in optimizer_att_memory.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_encoder
    for param_group in optimizer_decoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_decoder


def main(cfg, gpus):
    # Network Builders
    net_enc_query = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_enc_query)
    if cfg.MODEL.memory_encoder_noBN:
        net_enc_memory = ModelBuilder.build_encoder_memory_separate(
            arch=cfg.MODEL.arch_encoder.lower()+'_nobn',
            fc_dim=cfg.MODEL.fc_dim,
            weights=cfg.MODEL.weights_enc_memory,
            num_class=cfg.DATASET.num_class)
    else:
        net_enc_memory = ModelBuilder.build_encoder_memory_separate(
            arch=cfg.MODEL.arch_encoder.lower(),
            fc_dim=cfg.MODEL.fc_dim,
            weights=cfg.MODEL.weights_enc_memory,
            num_class=cfg.DATASET.num_class,
            pretrained=cfg.memory_enc_pretrained)
    net_att_query = ModelBuilder.build_encoder(
        arch='attention',
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_att_query)
    net_att_memory = ModelBuilder.build_encoder(
        arch='attention',
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_att_memory)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder)
    

    crit = nn.NLLLoss(ignore_index=-1)

    if cfg.MODEL.arch_decoder.endswith('deepsup'):
        segmentation_module = SegmentationAttentionSeparateModule(
            net_enc_query, net_enc_memory, net_att_query, net_att_memory, net_decoder, crit, cfg.TRAIN.deep_sup_scale, zero_memory=cfg.MODEL.zero_memory, random_memory_bias=cfg.MODEL.random_memory_bias, random_memory_nobias=cfg.MODEL.random_memory_nobias, random_scale=cfg.MODEL.random_scale, qval_qread_BN=cfg.MODEL.qval_qread_BN, debug=True)
    else:
        segmentation_module = SegmentationAttentionSeparateModule(
            net_enc_query, net_enc_memory, net_att_query, net_att_memory, net_decoder, crit, zero_memory=cfg.MODEL.zero_memory, random_memory_bias=cfg.MODEL.random_memory_bias, random_memory_nobias=cfg.MODEL.random_memory_nobias, random_scale=cfg.MODEL.random_scale, qval_qread_BN=cfg.MODEL.qval_qread_BN, debug=True)

    # Dataset and Loader
    dataset_train = TrainDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_train,
        cfg.DATASET,
        cfg.DATASET.ref_path, 
        cfg.DATASET.ref_start, 
        cfg.DATASET.ref_end,
        batch_per_gpu=cfg.TRAIN.batch_size_per_gpu)

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=len(gpus),  # we have modified data_parallel
        shuffle=False,  # we do not use this param
        collate_fn=user_scattered_collate,
        num_workers=cfg.TRAIN.workers,
        drop_last=True,
        pin_memory=True)
    print('1 Epoch = {} iters'.format(cfg.TRAIN.epoch_iters))

    # create loader iterator
    iterator_train = iter(loader_train)

    # load nets into gpu
    '''if len(gpus) > 1:
        segmentation_module = UserScatteredDataParallel(
            segmentation_module,
            device_ids=gpus)
        # For sync bn
        patch_replication_callback(segmentation_module)'''
    segmentation_module = UserScatteredDataParallel(
        segmentation_module,
        device_ids=gpus)
    # For sync bn
    patch_replication_callback(segmentation_module)
    segmentation_module.cuda()

    # Set up optimizers
    nets = (net_enc_query, net_enc_memory, net_att_query, net_att_memory, net_decoder, crit)
    optimizers = create_optimizers(nets, cfg)

    for epoch in range(0, 1):
        train(segmentation_module, iterator_train, optimizers, None, epoch+1, cfg)

    print('Training Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        default="0-3",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--memory_enc_pretrained",
        action='store_true',
        help="use a pretrained memory encoder",
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.memory_enc_pretrained = args.memory_enc_pretrained
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # Output directory
    if not os.path.isdir(cfg.DIR):
        os.makedirs(cfg.DIR)
    logger.info("Outputing checkpoints to: {}".format(cfg.DIR))
    with open(os.path.join(cfg.DIR, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))

    # Start from checkpoint
    if cfg.TRAIN.start_epoch > 0:
        cfg.MODEL.weights_enc_query = os.path.join(
            cfg.DIR, 'enc_query_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
        cfg.MODEL.weights_enc_memory = os.path.join(
            cfg.DIR, 'enc_memory_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
        cfg.MODEL.weights_att_query = os.path.join(
            cfg.DIR, 'att_query_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
        cfg.MODEL.weights_att_memory = os.path.join(
            cfg.DIR, 'att_memory_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
        cfg.MODEL.weights_decoder = os.path.join(
            cfg.DIR, 'decoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
        assert os.path.exists(cfg.MODEL.weights_enc_query) and os.path.exists(cfg.MODEL.weights_enc_memory) and \
            os.path.exists(cfg.MODEL.weights_att_query) and os.path.exists(cfg.MODEL.weights_att_memory) and \
            os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    # Parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]
    num_gpus = len(gpus)
    cfg.TRAIN.batch_size = num_gpus * cfg.TRAIN.batch_size_per_gpu

    cfg.TRAIN.max_iters = cfg.TRAIN.epoch_iters * cfg.TRAIN.num_epoch
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder

    random.seed(cfg.TRAIN.seed)
    torch.manual_seed(cfg.TRAIN.seed)

    main(cfg, gpus)
