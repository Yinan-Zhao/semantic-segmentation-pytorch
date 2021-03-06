# System libs
import os
import argparse
from distutils.version import LooseVersion
from multiprocessing import Queue, Process
# Numerical libs
import numpy as np
import math
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from config import cfg
from dataset_memory import ValDataset
from models import ModelBuilder, SegmentationAttentionModule
from utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, parse_devices, setup_logger
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm
from scipy.misc import imresize

colors = loadmat('data/color150.mat')['colors']


def visualize_result(data, pred, dir_result):
    (img, seg, info) = data

    # segmentation
    seg_color = colorEncode(seg, colors)

    # prediction
    pred_color = colorEncode(pred, colors)

    # aggregate images and save
    im_vis = np.concatenate((img, seg_color, pred_color),
                            axis=1).astype(np.uint8)

    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(os.path.join(dir_result, img_name.replace('.jpg', '.png')))


def evaluate(segmentation_module, loader, cfg, gpu_id, result_queue):
    segmentation_module.eval()

    for i, batch_data in enumerate(loader):
        # process data
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data['seg_label'][0])
        img_resized_list = batch_data['img_data']
        img_ref_resized_list = batch_data['img_refs']

        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu_id)

            zip_list = zip(img_resized_list, img_ref_resized_list)

            for img, img_refs in zip_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                feed_dict['img_refs'] = img_refs
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu_id)

                scores_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)
                #scores = scores_tmp

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

        # calculate accuracy and SEND THEM TO MASTER
        acc, pix = accuracy(pred, seg_label)
        intersection, union = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class)
        result_queue.put_nowait((acc, pix, intersection, union))

        # visualization
        if cfg.VAL.visualize:
            visualize_result(
                (batch_data['img_ori'], seg_label, batch_data['info']),
                pred,
                os.path.join(cfg.DIR, 'result')
            )


def worker(cfg, gpu_id, start_idx, end_idx, result_queue):
    torch.cuda.set_device(gpu_id)

    # Dataset and Loader
    dataset_val = ValDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_val,
        cfg.DATASET,
        start_idx=start_idx, end_idx=end_idx)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.VAL.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=2)

    # Network Builders
    net_enc_query = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_enc_query)
    net_enc_memory = ModelBuilder.build_encoder_memory(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_enc_memory,
        num_class=cfg.DATASET.num_class)
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
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationAttentionModule(net_enc_query, net_enc_memory, net_att_query, net_att_memory, net_decoder, crit, normalize_key=cfg.MODEL.normalize_key, p_scalar=cfg.MODEL.p_scalar)

    segmentation_module.cuda()

    # Main loop
    evaluate(segmentation_module, loader_val, cfg, gpu_id, result_queue)


def main(cfg, gpus):
    if cfg.eval_with_train:
        num_files = 100
    else:
        with open(cfg.DATASET.list_val, 'r') as f:
            lines = f.readlines()
            num_files = len(lines)

    num_files_per_gpu = math.ceil(num_files / len(gpus))

    pbar = tqdm(total=num_files)

    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    result_queue = Queue(500)
    procs = []
    for idx, gpu_id in enumerate(gpus):
        start_idx = idx * num_files_per_gpu
        end_idx = min(start_idx + num_files_per_gpu, num_files)
        proc = Process(target=worker, args=(cfg, gpu_id, start_idx, end_idx, result_queue))
        print('gpu:{}, start_idx:{}, end_idx:{}'.format(gpu_id, start_idx, end_idx))
        proc.start()
        procs.append(proc)

    # master fetches results
    processed_counter = 0
    while processed_counter < num_files:
        if result_queue.empty():
            continue
        (acc, pix, intersection, union) = result_queue.get()
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)
        processed_counter += 1
        pbar.update(1)

    for p in procs:
        p.join()

    # summary
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {:.4f}'.format(i, _iou))

    print('[Eval Summary]:')
    print('Mean IoU: {:.4f}, Accuracy: {:.2f}%'
          .format(iou.mean(), acc_meter.average()*100))

    print('Evaluation Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Validation"
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
        "--debug_with_gt",
        action='store_true',
        help="put gt in the memory",
    )
    parser.add_argument(
        "--debug_with_random",
        action='store_true',
        help="put gt in the memory",
    )
    parser.add_argument(
        "--debug_with_double_random",
        action='store_true',
        help="put gt in the memory",
    )
    parser.add_argument(
        "--debug_with_double_complete_random",
        action='store_true',
        help="put gt in the memory",
    )
    parser.add_argument(
        "--debug_with_translated_gt",
        action='store_true',
        help="put gt in the memory",
    )
    parser.add_argument(
        "--debug_with_randomSegNoise",
        action='store_true',
        help="put gt in the memory",
    )
    parser.add_argument(
        "--eval_with_train",
        action='store_true',
        help="evaluate with the training set",
    )
    parser.add_argument(
        "--zero_qval",
        action='store_true',
        help="zero qval",
    )
    parser.add_argument(
        "--is_debug",
        action='store_true',
        help="store intermediate results, such as probability",
    )
    parser.add_argument(
        "--eval_from_scratch",
        action='store_true',
        help="store intermediate results, such as probability",
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.DATASET.debug_with_gt = args.debug_with_gt
    cfg.DATASET.debug_with_random = args.debug_with_random
    cfg.DATASET.debug_with_translated_gt = args.debug_with_translated_gt
    cfg.DATASET.debug_with_double_random = args.debug_with_double_random
    cfg.DATASET.debug_with_double_complete_random = args.debug_with_double_complete_random
    cfg.DATASET.debug_with_randomSegNoise = args.debug_with_randomSegNoise
    cfg.zero_qval = args.zero_qval
    cfg.eval_with_train = args.eval_with_train
    cfg.is_debug = args.is_debug
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # absolute paths of model weights
    if not args.eval_from_scratch:
        cfg.MODEL.weights_enc_query = os.path.join(
            cfg.DIR, 'enc_query_' + cfg.VAL.checkpoint)
        cfg.MODEL.weights_enc_memory = os.path.join(
            cfg.DIR, 'enc_memory_' + cfg.VAL.checkpoint)
        cfg.MODEL.weights_att_query = os.path.join(
            cfg.DIR, 'att_query_' + cfg.VAL.checkpoint)
        cfg.MODEL.weights_att_memory = os.path.join(
            cfg.DIR, 'att_memory_' + cfg.VAL.checkpoint)
        cfg.MODEL.weights_decoder = os.path.join(
            cfg.DIR, 'decoder_' + cfg.VAL.checkpoint)
        assert os.path.exists(cfg.MODEL.weights_enc_query) and os.path.exists(cfg.MODEL.weights_enc_memory) and \
                os.path.exists(cfg.MODEL.weights_att_query) and os.path.exists(cfg.MODEL.weights_att_memory) and \
                os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    if not os.path.isdir(os.path.join(cfg.DIR, "result")):
        os.makedirs(os.path.join(cfg.DIR, "result"))

    # Parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]

    main(cfg, gpus)
