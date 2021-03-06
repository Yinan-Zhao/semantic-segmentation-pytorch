import os
import json
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import copy


def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, odgt, opt, **kwargs):
        # parse options
        self.imgSizes = opt.imgSizes
        self.imgMaxSize = opt.imgMaxSize
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant
        self.num_class = opt.num_class

        # parse the input list
        self.parse_input_list(odgt, **kwargs)

        # mean and std
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.list_sample_orig = copy.deepcopy(self.list_sample)

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def img_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    def segm_transform(self, segm):
        # to tensor, -1 to 149
        segm = torch.from_numpy(np.array(segm)).long() - 1
        return segm

    def segm_one_hot(self, segm):
        segm = torch.from_numpy(np.array(segm)).long().unsqueeze(0)
        size = segm.size()
        oneHot_size = (self.num_class+1, size[1], size[2])
        segm_oneHot = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
        segm_oneHot = segm_oneHot.scatter_(0, segm, 1.0)
        return segm_oneHot

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p


class TrainDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, ref_path, ref_start=0, ref_end=3, batch_per_gpu=1, **kwargs):
        super(TrainDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset
        # down sampling rate of segm labe
        self.segm_downsampling_rate = opt.segm_downsampling_rate
        self.batch_per_gpu = batch_per_gpu

        # classify images into two classes: 1. h > w and 2. h <= w
        self.batch_record_list = [[], []]
        self.batch_ref_list = [[], []]

        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = 0
        self.if_shuffled = False

        self.ref_start = ref_start
        self.ref_end = ref_end

        with open(ref_path, 'r') as f:
            lines = f.readlines()
        self.ref_list = [[int(item) for item in line.strip().split()] for line in lines]
        assert len(self.ref_list) == len(self.list_sample)

    def _get_sub_batch(self):
        while True:
            # get a sample record
            this_sample = self.list_sample[self.cur_idx]
            this_ref_list = self.ref_list[self.cur_idx]
            if this_sample['height'] > this_sample['width']:
                self.batch_record_list[0].append(this_sample) # h > w, go to 1st class
                self.batch_ref_list[0].append(this_ref_list)
            else:
                self.batch_record_list[1].append(this_sample) # h <= w, go to 2nd class
                self.batch_ref_list[1].append(this_ref_list)

            # update current sample pointer
            self.cur_idx += 1
            if self.cur_idx >= self.num_sample:
                self.cur_idx = 0
                permutation = np.random.permutation(len(self.list_sample))
                self.list_sample = [self.list_sample[i] for i in permutation]
                self.ref_list = [self.ref_list[i] for i in permutation]

            if len(self.batch_record_list[0]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[0]
                self.batch_record_list[0] = []
                ref_lists = self.batch_ref_list[0]
                self.batch_ref_list[0] = []
                break
            elif len(self.batch_record_list[1]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[1]
                self.batch_record_list[1] = []
                ref_lists = self.batch_ref_list[1]
                self.batch_ref_list[1] = []
                break
        return batch_records, ref_lists

    def __getitem__(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        if not self.if_shuffled:
            np.random.seed(index)
            permutation = np.random.permutation(len(self.list_sample))
            self.list_sample = [self.list_sample[i] for i in permutation]
            self.ref_list = [self.ref_list[i] for i in permutation]
            self.if_shuffled = True

        # get sub-batch candidates
        batch_records, ref_lists = self._get_sub_batch()

        # resize all images' short edges to the chosen size
        if isinstance(self.imgSizes, list) or isinstance(self.imgSizes, tuple):
            this_short_size = np.random.choice(self.imgSizes)
        else:
            this_short_size = self.imgSizes

        # calculate the BATCH's height and width
        # since we concat more than one samples, the batch's h and w shall be larger than EACH sample
        batch_widths = np.zeros(self.batch_per_gpu, np.int32)
        batch_heights = np.zeros(self.batch_per_gpu, np.int32)
        for i in range(self.batch_per_gpu):
            img_height, img_width = batch_records[i]['height'], batch_records[i]['width']
            this_scale = min(
                this_short_size / min(img_height, img_width), \
                self.imgMaxSize / max(img_height, img_width))
            batch_widths[i] = img_width * this_scale
            batch_heights[i] = img_height * this_scale

        # Here we must pad both input image and segmentation map to size h' and w' so that p | h' and p | w'
        batch_width = np.max(batch_widths)
        batch_height = np.max(batch_heights)
        batch_width = int(self.round2nearest_multiple(batch_width, self.padding_constant))
        batch_height = int(self.round2nearest_multiple(batch_height, self.padding_constant))

        assert self.padding_constant >= self.segm_downsampling_rate, \
            'padding constant must be equal or large than segm downsamping rate'
        batch_images = torch.zeros(
            self.batch_per_gpu, 3, batch_height, batch_width)
        batch_segms = torch.zeros(
            self.batch_per_gpu,
            batch_height // self.segm_downsampling_rate,
            batch_width // self.segm_downsampling_rate).long()
        batch_refs = torch.zeros(
            self.batch_per_gpu, 3+1+self.num_class, self.ref_end-self.ref_start, batch_height, batch_width)

        for i in range(self.batch_per_gpu):
            this_record = batch_records[i]

            # load image and label
            image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
            segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])

            img = Image.open(image_path).convert('RGB')
            segm = Image.open(segm_path)
            assert(segm.mode == "L")
            assert(img.size[0] == segm.size[0])
            assert(img.size[1] == segm.size[1])

            # random_flip
            if np.random.choice([0, 1]):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                segm = segm.transpose(Image.FLIP_LEFT_RIGHT)

            # note that each sample within a mini batch has different scale param
            img = imresize(img, (batch_widths[i], batch_heights[i]), interp='bilinear')
            segm = imresize(segm, (batch_widths[i], batch_heights[i]), interp='nearest')

            # further downsample seg label, need to avoid seg label misalignment
            segm_rounded_width = self.round2nearest_multiple(segm.size[0], self.segm_downsampling_rate)
            segm_rounded_height = self.round2nearest_multiple(segm.size[1], self.segm_downsampling_rate)
            segm_rounded = Image.new('L', (segm_rounded_width, segm_rounded_height), 0)
            segm_rounded.paste(segm, (0, 0))
            segm = imresize(
                segm_rounded,
                (segm_rounded.size[0] // self.segm_downsampling_rate, \
                 segm_rounded.size[1] // self.segm_downsampling_rate), \
                interp='nearest')

            # image transform, to torch float tensor 3xHxW
            img = self.img_transform(img)

            # segm transform, to torch long tensor HxW
            segm = self.segm_transform(segm)

            # put into batch arrays
            batch_images[i][:, :img.shape[1], :img.shape[2]] = img
            batch_segms[i][:segm.shape[0], :segm.shape[1]] = segm

            # prepare the references
            this_ref_list = ref_lists[i]
            for k in range(self.ref_end - self.ref_start):
                ref_record = self.list_sample_orig[this_ref_list[k+self.ref_start]]
                image_path = os.path.join(self.root_dataset, ref_record['fpath_img'])
                segm_path = os.path.join(self.root_dataset, ref_record['fpath_segm'])

                img = Image.open(image_path).convert('RGB')
                segm = Image.open(segm_path)
                assert(segm.mode == "L")
                assert(img.size[0] == segm.size[0])
                assert(img.size[1] == segm.size[1])

                if np.random.choice([0, 1]):
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    segm = segm.transpose(Image.FLIP_LEFT_RIGHT)
                img = imresize(img, (batch_widths[i], batch_heights[i]), interp='bilinear')
                segm = imresize(segm, (batch_widths[i], batch_heights[i]), interp='nearest')

                img = self.img_transform(img)
                segm = self.segm_one_hot(segm)

                batch_refs[i][0:3, k, :img.shape[1], :img.shape[2]] = img
                batch_refs[i][3:, k, :segm.shape[1], :segm.shape[2]] = segm

        output = dict()
        output['img_data'] = batch_images
        output['seg_label'] = batch_segms
        output['img_refs'] = batch_refs
        return output

    def __len__(self):
        return int(1e10) # It's a fake length due to the trick that every loader maintains its own list
        #return self.num_sampleclass


class ValDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, **kwargs):
        super(ValDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset

        self.train_list_sample = [json.loads(x.rstrip()) for x in open(opt.list_train, 'r')]

        self.train_num_sample = len(self.train_list_sample)
        assert self.train_num_sample > 0
        print('# training samples: {}'.format(self.train_num_sample))

        self.debug_with_gt = opt.debug_with_gt
        self.debug_with_translated_gt = opt.debug_with_translated_gt
        self.debug_with_random = opt.debug_with_random
        self.debug_with_double_random = opt.debug_with_double_random
        self.debug_with_double_complete_random = opt.debug_with_double_complete_random
        self.debug_with_randomSegNoise = opt.debug_with_randomSegNoise

        self.ref_start = opt.ref_val_start
        self.ref_end = opt.ref_val_end

        with open(opt.ref_val_path, 'r') as f:
            lines = f.readlines()
        self.ref_list = [[int(item) for item in line.strip().split()] for line in lines]

        start_idx = kwargs['start_idx']
        end_idx = kwargs['end_idx']
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.ref_list = self.ref_list[start_idx:end_idx]
        assert len(self.ref_list) == len(self.list_sample)

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        this_ref_list = self.ref_list[index]
        # load image and label
        image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
        segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
        img = Image.open(image_path).convert('RGB')
        segm = Image.open(segm_path)
        assert(segm.mode == "L")
        assert(img.size[0] == segm.size[0])
        assert(img.size[1] == segm.size[1])

        # segm transform, to torch long tensor HxW
        segm = self.segm_transform(segm)
        batch_segms = torch.unsqueeze(segm, 0)

        ori_width, ori_height = img.size

        img_resized_list = []
        ref_resized_list = []
        for this_short_size in self.imgSizes:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_width = self.round2nearest_multiple(target_width, self.padding_constant)
            target_height = self.round2nearest_multiple(target_height, self.padding_constant)

            # resize images
            img_resized = imresize(img, (target_width, target_height), interp='bilinear')

            # image transform, to torch float tensor 3xHxW
            img_resized = self.img_transform(img_resized)
            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

            batch_refs = torch.zeros(
                3+1+self.num_class, self.ref_end-self.ref_start, target_height, target_width)

            for k in range(self.ref_end - self.ref_start):
                ref_record = self.train_list_sample[this_ref_list[k+self.ref_start]]
                image_ref_path = os.path.join(self.root_dataset, ref_record['fpath_img'])
                segm_ref_path = os.path.join(self.root_dataset, ref_record['fpath_segm'])

                img_ref = Image.open(image_ref_path).convert('RGB')
                segm_ref = Image.open(segm_ref_path)
                assert(segm_ref.mode == "L")
                assert(img_ref.size[0] == segm_ref.size[0])
                assert(img_ref.size[1] == segm_ref.size[1])

                img_ref = imresize(img_ref, (target_width, target_height), interp='bilinear')
                segm_ref = imresize(segm_ref, (target_width, target_height), interp='nearest')

                img_ref = self.img_transform(img_ref)
                segm_ref = self.segm_one_hot(segm_ref)

                batch_refs[0:3, k, :img_ref.shape[1], :img_ref.shape[2]] = img_ref
                batch_refs[3:, k, :segm_ref.shape[1], :segm_ref.shape[2]] = segm_ref

                if self.debug_with_gt:
                    img_resized_gt = img_resized[0]
                    batch_refs[0:3, k, :img_resized_gt.shape[1], :img_resized_gt.shape[2]] = img_resized_gt
                    
                    segm_ref = Image.open(os.path.join(self.root_dataset, this_record['fpath_segm']))
                    segm_ref = imresize(segm_ref, (target_width, target_height), interp='nearest')
                    segm_ref = self.segm_one_hot(segm_ref)
                    batch_refs[3:, k, :segm_ref.shape[1], :segm_ref.shape[2]] = segm_ref
                elif self.debug_with_translated_gt:
                    img_resized_gt = img_resized[0]
                    translation = 20
                    batch_refs[0:3, k, translation:img_resized_gt.shape[1], translation:img_resized_gt.shape[2]] = img_resized_gt[:,:img_resized_gt.shape[1]-translation, :img_resized_gt.shape[2]-translation]
                    
                    segm_ref = Image.open(os.path.join(self.root_dataset, this_record['fpath_segm']))
                    segm_ref = imresize(segm_ref, (target_width, target_height), interp='nearest')
                    segm_ref = self.segm_one_hot(segm_ref)
                    batch_refs[3:, k, translation:segm_ref.shape[1], translation:segm_ref.shape[2]] = segm_ref[:, :segm_ref.shape[1]-translation, :segm_ref.shape[2]-translation]
                elif self.debug_with_random:
                    img_resized_gt = img_resized[0]
                    batch_refs[0:3, k, :img_resized_gt.shape[1], :img_resized_gt.shape[2]] = img_resized_gt
                    
                    segm_ref = Image.open(os.path.join(self.root_dataset, ref_record['fpath_segm']))
                    segm_ref = imresize(segm_ref, (target_width, target_height), interp='nearest')
                    segm_ref = self.segm_one_hot(segm_ref)
                    batch_refs[3:, k, :segm_ref.shape[1], :segm_ref.shape[2]] = segm_ref
                elif self.debug_with_double_random:
                    ref_record_tmp = self.train_list_sample[this_ref_list[k+100+self.ref_start]]
                    segm_ref = Image.open(os.path.join(self.root_dataset, ref_record_tmp['fpath_segm']))
                    segm_ref = imresize(segm_ref, (target_width, target_height), interp='nearest')
                    segm_ref = self.segm_one_hot(segm_ref)
                    batch_refs[3:, k, :segm_ref.shape[1], :segm_ref.shape[2]] = segm_ref
                elif self.debug_with_double_complete_random:
                    ref_record_tmp = self.train_list_sample[np.random.randint(0, len(self.train_list_sample))]
                    segm_ref = Image.open(os.path.join(self.root_dataset, ref_record_tmp['fpath_segm']))
                    segm_ref = imresize(segm_ref, (target_width, target_height), interp='nearest')
                    segm_ref = self.segm_one_hot(segm_ref)
                    batch_refs[3:, k, :segm_ref.shape[1], :segm_ref.shape[2]] = segm_ref
                elif self.debug_with_randomSegNoise:
                    batch_refs[3:, k, :segm_ref.shape[1], :segm_ref.shape[2]] = torch.rand_like(segm_ref)

            batch_refs = torch.unsqueeze(batch_refs, 0)
            ref_resized_list.append(batch_refs)


        output = dict()
        output['img_ori'] = np.array(img)
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        output['seg_label'] = batch_segms.contiguous()
        output['img_refs'] = [x. contiguous() for x in ref_resized_list]
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample


class TestDataset(BaseDataset):
    def __init__(self, odgt, opt, **kwargs):
        super(TestDataset, self).__init__(odgt, opt, **kwargs)

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image
        image_path = this_record['fpath_img']
        img = Image.open(image_path).convert('RGB')

        ori_width, ori_height = img.size

        img_resized_list = []
        for this_short_size in self.imgSizes:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_width = self.round2nearest_multiple(target_width, self.padding_constant)
            target_height = self.round2nearest_multiple(target_height, self.padding_constant)

            # resize images
            img_resized = imresize(img, (target_width, target_height), interp='bilinear')

            # image transform, to torch float tensor 3xHxW
            img_resized = self.img_transform(img_resized)
            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        output = dict()
        output['img_ori'] = np.array(img)
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample
