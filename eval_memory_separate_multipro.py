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
from dataset_memory_separate import ValDataset
from models import ModelBuilder, SegmentationAttentionSeparateModule
from utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, parse_devices, setup_logger
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm
from scipy.misc import imresize

colors = loadmat('data/color150.mat')['colors']

def segm_one_hot(segm, num_class):
    size = segm.size()
    oneHot_size = (num_class+1, size[1], size[2])
    segm_oneHot = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
    segm_oneHot = segm_oneHot.scatter_(0, segm, 1.0)
    return segm_oneHot[0]


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
        img_ref_rgb_resized_list = batch_data['img_refs_rgb']
        img_ref_mask_resized_list = batch_data['img_refs_mask']

        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu_id)

            if cfg.is_debug:
                zip_list = zip(img_resized_list[-2:-1], img_ref_rgb_resized_list[-2:-1], img_ref_mask_resized_list[-2:-1])
            else:
                zip_list = zip(img_resized_list, img_ref_rgb_resized_list, img_ref_mask_resized_list)

            for img, img_refs_rgb, img_refs_mask in zip_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                feed_dict['img_refs_rgb'] = img_refs_rgb
                feed_dict['img_refs_mask'] = img_refs_mask
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu_id)

                # forward pass
                if cfg.is_debug:
                    scores_tmp, qread, qval, qk_b, mk_b, mv_b, p, feature_enc, feature_memory = segmentation_module(feed_dict, segSize=segSize)
                    np.save('debug/qread_%03d.npy'%(i), qread.detach().cpu().float().numpy())
                    np.save('debug/qval_%03d.npy'%(i), qval.detach().cpu().float().numpy())
                    np.save('debug/qk_b_%03d.npy'%(i), qk_b.detach().cpu().float().numpy())
                    np.save('debug/mk_b_%03d.npy'%(i), mk_b.detach().cpu().float().numpy())
                    np.save('debug/mv_b_%03d.npy'%(i), mv_b.detach().cpu().float().numpy())
                    np.save('debug/p_%03d.npy'%(i), p.detach().cpu().float().numpy())
                    np.save('debug/feature_enc_%03d.npy'%(i), feature_enc[-1].detach().cpu().float().numpy())
                    np.save('debug/feature_memory_%03d.npy'%(i), feature_memory[-1].detach().cpu().float().numpy())
                    print(batch_data['info'])
                else:
                    if cfg.eval_att_voting:
                        scores_tmp, qread, qval, qk_b, mk_b, mv_b, p, feature_enc, feature_memory = segmentation_module(feed_dict, segSize=segSize)
                        height, width = qread.shape[-2], qread.shape[-1]
                        assert p.shape[0] == height*width
                        img_refs_mask_resize = nn.functional.interpolate(img_refs_mask[0].cuda(), size=(height, width), mode='nearest')
                        img_refs_mask_resize_flat = img_refs_mask_resize[:,0,:,:].view(img_refs_mask_resize.shape[0], -1)
                        mask_voting_flat = torch.mm(img_refs_mask_resize_flat, p)
                        mask_voting = mask_voting_flat.view(mask_voting_flat.shape[0], height, width)
                        mask_voting = torch.unsqueeze(mask_voting, 0)
                        scores_tmp = nn.functional.interpolate(mask_voting[:,:cfg.DATASET.num_class], size=segSize, mode='bilinear', align_corners=False)

                        np.save('debug/p_%03d.npy'%(i), p.detach().cpu().float().numpy())
                        np.save('debug/img_refs_mask_%03d.npy'%(i), img_refs_mask.cuda().detach().cpu().float().numpy())
                        np.save('debug/img_refs_mask_resize_%03d.npy'%(i), img_refs_mask_resize.detach().cpu().float().numpy())
                        np.save('debug/img_refs_mask_resize_flat_%03d.npy'%(i), img_refs_mask_resize_flat.detach().cpu().float().numpy())
                        np.save('debug/mask_voting_flat_%03d.npy'%(i), mask_voting_flat.detach().cpu().float().numpy())
                        np.save('debug/mask_voting_%03d.npy'%(i), mask_voting.detach().cpu().float().numpy())

                    else:
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
    if cfg.eval_with_train:
        dataset_val = ValDataset(
            cfg.DATASET.root_dataset,
            cfg.DATASET.list_train,
            cfg.DATASET,
            start_idx=start_idx, end_idx=end_idx)
    else:
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
    if cfg.MODEL.memory_encoder_arch:
        net_enc_memory = ModelBuilder.build_encoder_memory_separate(
            arch=cfg.MODEL.memory_encoder_arch.lower(),
            fc_dim=cfg.MODEL.fc_dim,
            weights=cfg.MODEL.weights_enc_memory,
            num_class=cfg.DATASET.num_class,
            RGB_mask_combine_val=cfg.DATASET.RGB_mask_combine_val,
            segm_downsampling_rate=cfg.DATASET.segm_downsampling_rate)
    else:
        if cfg.MODEL.memory_encoder_noBN:
            net_enc_memory = ModelBuilder.build_encoder_memory_separate(
                arch=cfg.MODEL.arch_encoder.lower()+'_nobn',
                fc_dim=cfg.MODEL.fc_dim,
                weights=cfg.MODEL.weights_enc_memory,
                num_class=cfg.DATASET.num_class,
                RGB_mask_combine_val=cfg.DATASET.RGB_mask_combine_val,
                segm_downsampling_rate=cfg.DATASET.segm_downsampling_rate)
        else:
            net_enc_memory = ModelBuilder.build_encoder_memory_separate(
                arch=cfg.MODEL.arch_encoder.lower(),
                fc_dim=cfg.MODEL.fc_dim,
                weights=cfg.MODEL.weights_enc_memory,
                num_class=cfg.DATASET.num_class,
                RGB_mask_combine_val=cfg.DATASET.RGB_mask_combine_val,
                segm_downsampling_rate=cfg.DATASET.segm_downsampling_rate)
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

    segmentation_module = SegmentationAttentionSeparateModule(net_enc_query, net_enc_memory, net_att_query, net_att_memory, net_decoder, crit, zero_memory=cfg.MODEL.zero_memory, zero_qval=cfg.MODEL.zero_qval, qval_qread_BN=cfg.MODEL.qval_qread_BN, normalize_key=cfg.MODEL.normalize_key, p_scalar=cfg.MODEL.p_scalar, memory_feature_aggregation=cfg.MODEL.memory_feature_aggregation, memory_noLabel=cfg.MODEL.memory_noLabel, debug=cfg.is_debug or cfg.eval_att_voting)

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
        "--is_debug",
        action='store_true',
        help="store intermediate results, such as probability",
    )
    parser.add_argument(
        "--eval_from_scratch",
        action='store_true',
        help="evaluate from scratch",
    )
    parser.add_argument(
        "--eval_att_voting",
        action='store_true',
        help="evaluate with attention-based voting",
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
    cfg.eval_with_train = args.eval_with_train
    cfg.is_debug = args.is_debug
    cfg.eval_att_voting = args.eval_att_voting
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
