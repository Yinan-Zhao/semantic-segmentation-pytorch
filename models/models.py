import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import numpy as np
from . import resnet, resnext, mobilenet, hrnet
from lib.nn import SynchronizedBatchNorm2d
BatchNorm2d = SynchronizedBatchNorm2d


class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc


class SegmentationModule(SegmentationModuleBase):
    def __init__(self, net_enc, net_dec, crit, deep_sup_scale=None):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale

    def forward(self, feed_dict, *, segSize=None):
        # training
        if segSize is None:
            if self.deep_sup_scale is not None: # use deep supervision technique
                (pred, pred_deepsup) = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))
            else:
                pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))

            loss = self.crit(pred, feed_dict['seg_label'])
            if self.deep_sup_scale is not None:
                loss_deepsup = self.crit(pred_deepsup, feed_dict['seg_label'])
                loss = loss + loss_deepsup * self.deep_sup_scale

            acc = self.pixel_acc(pred, feed_dict['seg_label'])
            return loss, acc
        # inference
        else:
            pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True), segSize=segSize)
            return pred

class SegmentationAttentionModule(SegmentationModuleBase):
    def __init__(self, net_enc_query, net_enc_memory, net_att_query, net_att_memory, net_dec, crit, deep_sup_scale=None):
        super(SegmentationAttentionModule, self).__init__()
        self.encoder_query = net_enc_query
        self.encoder_memory = net_enc_memory
        self.attention_query = net_att_query
        self.attention_memory = net_att_memory
        self.decoder = net_dec
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale

    def maskRead(self, qkey, qval, qmask, mkey, mval, mmask):
        '''
        read for *mask area* of query from *mask area* of memory
        '''
        B, Dk, _, H, W = mkey.size()
        _, Dv, _, _, _ = mval.size()
        qread = torch.zeros_like(qval)
        # key: b,dk,t,h,w
        # value: b,dv,t,h,w
        # mask: b,1,t,h,w
        for b in range(B):
            # exceptions
            if qmask[b,0].sum() == 0 or mmask[b,0].sum() == 0: 
                # print('skipping read', qmask[b,0].sum(), mmask[b,0].sum())
                # no query or mask pixels -> skip read
                continue
            qk_b = qkey[b,:,qmask[b,0]] # dk, Nq
            mk_b = mkey[b,:,mmask[b,0]] # dk, Nm
            mv_b = mval[b,:,mmask[b,0]] # dv, Nm 
            # print(mv_b.shape)

            p = torch.mm(torch.transpose(mk_b, 0, 1), qk_b) # Nm, Nq
            p = p / math.sqrt(Dk)
            p = F.softmax(p, dim=0)

            read = torch.mm(mv_b, p) # dv, Nq
            # qval[b,:,qmask[b,0]] = read # dv, Nq
            qread[b,:,qmask[b,0]] = qread[b,:,qmask[b,0]] + read # dv, Nq
            
        return qread

    def memoryEncode(self, encoder, img_refs, return_feature_maps=True):
        # encoding into memory
        feat_ = []
        batch_size, _, num_frames, height, width = img_refs.size()
        for t in range(num_frames):
            feat = encoder(img_refs[:,:,t], return_feature_maps=return_feature_maps)
            feat_.append(feat)

        feats = []
        for i in range(len(feat_[0])):
            tmp = []
            for j in range(len(feat_)):
                tmp.append(feat_[j][i])
            feats.append(torch.stack(tmp, dim=2))
        return feats

    def memoryAttention(self, att_module, feat):
        key_ = []
        val_ = []
        batch_size, _, num_frames, height, width = feat[-1].size()
        for t in range(num_frames):
            key, val = att_module([feat_item[:,:,t] for feat_item in feat])
            key_.append(key)
            val_.append(val)

        keys = torch.stack(key_, dim=2)
        vals = torch.stack(val_, dim=2)
        return keys, vals

    def forward(self, feed_dict, *, segSize=None):
        # training
        if segSize is None:
            if self.deep_sup_scale is not None: # use deep supervision technique
                raise Exception('deep_sup_scale is not implemented') 
            else:
                feature_enc = self.encoder_query(feed_dict['img_data'], return_feature_maps=True)
                qkey, qval = self.attention_query(feature_enc)
                feature_memory = self.memoryEncode(self.encoder_memory, feed_dict['img_refs'], return_feature_maps=True)
                mkey, mval = self.memoryAttention(self.attention_memory, feature_memory)
                qmask = torch.ones_like(qkey)[:,0:1] > 0.
                mmask = torch.ones_like(mkey)[:,0:1] > 0.
                qread = self.maskRead(qkey, qval, qmask, mkey, mval, mmask)
                feature = torch.cat((qval, qread), dim=1)
                pred = self.decoder([feature])

            loss = self.crit(pred, feed_dict['seg_label'])
            '''if self.deep_sup_scale is not None:
                loss_deepsup = self.crit(pred_deepsup, feed_dict['seg_label'])
                loss = loss + loss_deepsup * self.deep_sup_scale'''

            acc = self.pixel_acc(pred, feed_dict['seg_label'])
            return loss, acc
        # inference
        else:
            feature_enc = self.encoder_query(feed_dict['img_data'], return_feature_maps=True)
            qkey, qval = self.attention_query(feature_enc)
            feature_memory = self.memoryEncode(self.encoder_memory, feed_dict['img_refs'], return_feature_maps=True)
            mkey, mval = self.memoryAttention(self.attention_memory, feature_memory)
            qmask = torch.ones_like(qkey)[:,0:1] > 0.
            mmask = torch.ones_like(mkey)[:,0:1] > 0.
            qread = self.maskRead(qkey, qval, qmask, mkey, mval, mmask)
            feature = torch.cat((qval, qread), dim=1)
            pred = self.decoder([feature], segSize=segSize)
            return pred 

class SegmentationAttentionSeparateModule(SegmentationModuleBase):
    def __init__(self, net_enc_query, net_enc_memory, net_att_query, net_att_memory, net_dec, crit, deep_sup_scale=None, zero_memory=False):
        super(SegmentationAttentionSeparateModule, self).__init__()
        self.encoder_query = net_enc_query
        self.encoder_memory = net_enc_memory
        self.attention_query = net_att_query
        self.attention_memory = net_att_memory
        self.decoder = net_dec
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale
        self.zero_memory = zero_memory

    def maskRead(self, qkey, qval, qmask, mkey, mval, mmask):
        '''
        read for *mask area* of query from *mask area* of memory
        '''
        B, Dk, _, H, W = mkey.size()
        _, Dv, _, _, _ = mval.size()
        qread = torch.zeros_like(qval)
        # key: b,dk,t,h,w
        # value: b,dv,t,h,w
        # mask: b,1,t,h,w
        for b in range(B):
            # exceptions
            if qmask[b,0].sum() == 0 or mmask[b,0].sum() == 0: 
                # print('skipping read', qmask[b,0].sum(), mmask[b,0].sum())
                # no query or mask pixels -> skip read
                continue
            qk_b = qkey[b,:,qmask[b,0]] # dk, Nq
            mk_b = mkey[b,:,mmask[b,0]] # dk, Nm
            mv_b = mval[b,:,mmask[b,0]] # dv, Nm 
            # print(mv_b.shape)

            p = torch.mm(torch.transpose(mk_b, 0, 1), qk_b) # Nm, Nq
            p = p / math.sqrt(Dk)
            p = F.softmax(p, dim=0)

            read = torch.mm(mv_b, p) # dv, Nq
            # qval[b,:,qmask[b,0]] = read # dv, Nq
            qread[b,:,qmask[b,0]] = qread[b,:,qmask[b,0]] + read # dv, Nq
        #np.save('debug/mv_b.npy', mv_b.detach().cpu().float().numpy())
        #np.save('debug/p.npy', p.detach().cpu().float().numpy())
        #np.save('debug/mval.npy', mval.detach().cpu().float().numpy())
        #np.save('debug/qread.npy', qread.detach().cpu().float().numpy())
        return qread

    def memoryEncode(self, encoder, img_refs, return_feature_maps=True):
        # encoding into memory
        feat_ = []
        batch_size, _, num_frames, height, width = img_refs.size()
        for t in range(num_frames):
            feat = encoder(img_refs[:,:,t], return_feature_maps=return_feature_maps)
            feat_.append(feat)

        feats = []
        for i in range(len(feat_[0])):
            tmp = []
            for j in range(len(feat_)):
                tmp.append(feat_[j][i])
            feats.append(torch.stack(tmp, dim=2))
        return feats

    def memoryAttention(self, att_module, feat):
        key_ = []
        val_ = []
        batch_size, _, num_frames, height, width = feat[-1].size()
        for t in range(num_frames):
            key, val = att_module([feat_item[:,:,t] for feat_item in feat])
            key_.append(key)
            val_.append(val)

        keys = torch.stack(key_, dim=2)
        vals = torch.stack(val_, dim=2)
        return keys, vals

    def forward(self, feed_dict, *, segSize=None):
        # training
        if segSize is None:
            if self.deep_sup_scale is not None: # use deep supervision technique
                raise Exception('deep_sup_scale is not implemented') 
            else:
                feature_enc = self.encoder_query(feed_dict['img_data'], return_feature_maps=True)                
                qkey, qval = self.attention_query(feature_enc)
                
                feature_memory = self.memoryEncode(self.encoder_query, feed_dict['img_refs_rgb'], return_feature_maps=True)
                mkey, _ = self.memoryAttention(self.attention_query, feature_memory)

                mask_feature_memory = self.memoryEncode(self.encoder_memory, feed_dict['img_refs_mask'], return_feature_maps=True)
                #np.save('debug/img_refs_mask.npy', feed_dict['img_refs_mask'].detach().cpu().float().numpy())
                #for idx, feat in enumerate(mask_feature_memory):
                #    print(feat.shape)
                #    np.save('debug/mask_feature_memory_%d.npy'%(idx), feat.detach().cpu().float().numpy())
                _, mval = self.memoryAttention(self.attention_memory, mask_feature_memory)

                qmask = torch.ones_like(qkey)[:,0:1] > 0.
                mmask = torch.ones_like(mkey)[:,0:1] > 0.
                qread = self.maskRead(qkey, qval, qmask, mkey, mval, mmask)
                if self.zero_memory:
                    feature = torch.cat((qval, torch.zeros_like(qread)), dim=1)
                else:
                    feature = torch.cat((qval, qread), dim=1)
                pred = self.decoder([feature])

            loss = self.crit(pred, feed_dict['seg_label'])
            '''if self.deep_sup_scale is not None:
                loss_deepsup = self.crit(pred_deepsup, feed_dict['seg_label'])
                loss = loss + loss_deepsup * self.deep_sup_scale'''

            acc = self.pixel_acc(pred, feed_dict['seg_label'])
            return loss, acc
        # inference
        else:
            feature_enc = self.encoder_query(feed_dict['img_data'], return_feature_maps=True)                
            qkey, qval = self.attention_query(feature_enc)
            
            feature_memory = self.memoryEncode(self.encoder_query, feed_dict['img_refs_rgb'], return_feature_maps=True)
            mkey, _ = self.memoryAttention(self.attention_query, feature_memory)

            mask_feature_memory = self.memoryEncode(self.encoder_memory, feed_dict['img_refs_mask'], return_feature_maps=True)
            _, mval = self.memoryAttention(self.attention_memory, mask_feature_memory)

            qmask = torch.ones_like(qkey)[:,0:1] > 0.
            mmask = torch.ones_like(mkey)[:,0:1] > 0.
            qread = self.maskRead(qkey, qval, qmask, mkey, mval, mmask)
            if self.zero_memory:
                feature = torch.cat((qval, torch.zeros_like(qread)), dim=1)
            else:
                feature = torch.cat((qval, qread), dim=1)
            pred = self.decoder([feature], segSize=segSize)

            return pred 


class ModelBuilder:
    # custom weights initialization
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        #elif classname.find('Linear') != -1:
        #    m.weight.data.normal_(0.0, 0.0001)

    @staticmethod
    def build_encoder_memory(arch='resnet50dilated', fc_dim=512, weights='', num_class=150):
        pretrained = True if len(weights) == 0 else False
        arch = arch.lower()
        if arch == 'resnet18dilated':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = ResnetDilated_Memory(orig_resnet, dilate_scale=8, num_class=num_class)
        else:
            raise Exception('Architecture undefined!')

        # encoders are usually pretrained
        #net_encoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    @staticmethod
    def build_encoder_memory_separate(arch='resnet50dilated', fc_dim=512, weights='', num_class=150, pretrained=True):
        arch = arch.lower()
        if arch == 'resnet18dilated':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = ResnetDilated_Memory_Separate(orig_resnet, dilate_scale=8, num_class=num_class)
        else:
            raise Exception('Architecture undefined!')

        # encoders are usually pretrained
        if not pretrained:
            net_encoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    @staticmethod
    def build_encoder(arch='resnet50dilated', fc_dim=512, weights=''):
        pretrained = True if len(weights) == 0 else False
        arch = arch.lower()
        if arch == 'mobilenetv2dilated':
            orig_mobilenet = mobilenet.__dict__['mobilenetv2'](pretrained=pretrained)
            net_encoder = MobileNetV2Dilated(orig_mobilenet, dilate_scale=8)
        elif arch == 'resnet18':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet18dilated':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet34':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet34dilated':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet50':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet50dilated':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet101':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet101dilated':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnext101':
            orig_resnext = resnext.__dict__['resnext101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnext) # we can still use class Resnet
        elif arch == 'hrnetv2':
            net_encoder = hrnet.__dict__['hrnetv2'](pretrained=pretrained)
        elif arch == 'attention':
            net_encoder = AttModule(fc_dim=fc_dim)
        else:
            raise Exception('Architecture undefined!')

        # encoders are usually pretrained
        # net_encoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    @staticmethod
    def build_decoder(arch='ppm_deepsup',
                      fc_dim=512, num_class=150,
                      weights='', use_softmax=False):
        arch = arch.lower()
        if arch == 'c1_deepsup':
            net_decoder = C1DeepSup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'c1':
            net_decoder = C1(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm':
            net_decoder = PPM(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm_deepsup':
            net_decoder = PPMDeepsup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'upernet_lite':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=256)
        elif arch == 'upernet':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=512)
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]

class ResnetDilated_Memory(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8, num_class=150):
        super(ResnetDilated_Memory, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        #self.conv1 = orig_resnet.conv1
        self.conv1 = conv3x3(3+1+num_class, 64, stride=2)
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

        nn.init.kaiming_normal_(self.conv1.weight.data)
        self.conv1.weight.data[:,0:3,:,:] = orig_resnet.conv1.weight.data

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]

class ResnetDilated_Memory_Separate(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8, num_class=150):
        super(ResnetDilated_Memory_Separate, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        #self.conv1 = orig_resnet.conv1
        self.conv1 = conv3x3(1+num_class, 64, stride=2)
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

        nn.init.kaiming_normal_(self.conv1.weight.data)

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


class MobileNetV2Dilated(nn.Module):
    def __init__(self, orig_net, dilate_scale=8):
        super(MobileNetV2Dilated, self).__init__()
        from functools import partial

        # take pretrained mobilenet features
        self.features = orig_net.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if dilate_scale == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif dilate_scale == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        if return_feature_maps:
            conv_out = []
            for i in range(self.total_idx):
                x = self.features[i](x)
                if i in self.down_idx:
                    conv_out.append(x)
            conv_out.append(x)
            return conv_out

        else:
            return [self.features(x)]


# last conv, deep supervision
class C1DeepSup(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1DeepSup, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)


# last conv
class C1(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax: # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)

        return x

class AttModule(nn.Module):
    def __init__(self, fc_dim=2048):
        super(AttModule, self).__init__()

        self.key_conv = nn.Conv2d(fc_dim, fc_dim//8, kernel_size=3,
                      stride=1, padding=1, bias=False)
        self.value_conv = nn.Conv2d(fc_dim, fc_dim//2, kernel_size=3,
                      stride=1, padding=1, bias=False)

    def forward(self, conv_out):
        conv5 = conv_out[-1]
        key = self.key_conv(conv5)
        value = self.value_conv(conv5)

        return key, value


# pyramid pooling
class PPM(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x


# pyramid pooling, deep supervision
class PPMDeepsup(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMDeepsup, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)


# upernet
class UPerNet(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256, 512, 1024, 2048), fpn_dim=256):
        super(UPerNet, self).__init__()
        self.use_softmax = use_softmax

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:   # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):  # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x) # lateral branch

            f = nn.functional.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse() # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        x = nn.functional.log_softmax(x, dim=1)

        return x
