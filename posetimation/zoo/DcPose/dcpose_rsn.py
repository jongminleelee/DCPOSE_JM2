#!/usr/bin/python
# -*- coding:utf8 -*-


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from collections import OrderedDict

from .ConvVideoTransformer import ConvTransformer
from ..base import BaseModel

from thirdparty.deform_conv import DeformConv, ModulatedDeformConv
from posetimation.layers import BasicBlock, ChainOfBasicBlocks, DeformableCONV, PAM_Module, CAM_Module
from posetimation.layers import RSB_BLOCK, CHAIN_RSB_BLOCKS
from ..backbones.hrnet import HRNet
from utils.common import TRAIN_PHASE
from utils.utils_registry import MODEL_REGISTRY
from .blocks import FlowLayer

# import sys
# sys.path.append('/home/nsml/hrnet_stage/posetimation/zoo/DcPose/')
# from jre_module import *


@MODEL_REGISTRY.register()
class DcPose_RSN(BaseModel):

    def __init__(self, cfg, phase, **kwargs):
        super(DcPose_RSN, self).__init__()
        self.logger = logging.getLogger(__name__)

        self.inplanes = 64
        self.batch_size = cfg['TRAIN']['BATCH_SIZE_PER_GPU']
        self.use_warping_train = cfg['MODEL']['USE_WARPING_TRAIN']
        self.use_warping_test = cfg['MODEL']['USE_WARPING_TEST']
        self.freeze_weights = cfg['MODEL']['FREEZE_WEIGHTS']
        self.use_gt_input_train = cfg['MODEL']['USE_GT_INPUT_TRAIN']
        self.use_gt_input_test = cfg['MODEL']['USE_GT_INPUT_TEST']
        self.warping_reverse = cfg['MODEL']['WARPING_REVERSE']
        self.cycle_consistency_finetune = cfg['MODEL']['CYCLE_CONSISTENCY_FINETUNE']

        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']

        self.is_train = True if phase == TRAIN_PHASE else False
        # define rough_pose_estimation
        self.use_prf = cfg.MODEL.USE_PRF
        self.use_ptm = cfg.MODEL.USE_PTM
        self.use_pcn = cfg.MODEL.USE_PCN

        self.freeze_hrnet_weights = cfg.MODEL.FREEZE_HRNET_WEIGHTS
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.use_rectifier = cfg.MODEL.USE_RECTIFIER
        self.use_margin = cfg.MODEL.USE_MARGIN
        self.use_group = cfg.MODEL.USE_GROUP

        self.deformable_conv_dilations = cfg.MODEL.DEFORMABLE_CONV.DILATION
        self.deformable_aggregation_type = cfg.MODEL.DEFORMABLE_CONV.AGGREGATION_TYPE
        ####
        self.rough_pose_estimation_net = HRNet(cfg, phase)

        self.pretrained = cfg.MODEL.PRETRAINED

        k = 3

        prf_inner_ch = cfg.MODEL.PRF_INNER_CH
        prf_basicblock_num = cfg.MODEL.PRF_BASICBLOCK_NUM

        ptm_inner_ch = cfg.MODEL.PTM_INNER_CH
        ptm_basicblock_num = cfg.MODEL.PTM_BASICBLOCK_NUM

        prf_ptm_combine_inner_ch = cfg.MODEL.PRF_PTM_COMBINE_INNER_CH
        prf_ptm_combine_basicblock_num = cfg.MODEL.PRF_PTM_COMBINE_BASICBLOCK_NUM
        hyper_parameters = OrderedDict({
            "k": k,
            "prf_basicblock_num": prf_basicblock_num,
            "prf_inner_ch": prf_inner_ch,
            "ptm_basicblock_num": ptm_basicblock_num,
            "ptm_inner_ch": ptm_inner_ch,
            "prf_ptm_combine_basicblock_num": prf_ptm_combine_basicblock_num,
            "prf_ptm_combine_inner_ch": prf_ptm_combine_inner_ch,
        }
        )
        self.logger.info("###### MODEL {} Hyper Parameters ##########".format(self.__class__.__name__))
        self.logger.info(hyper_parameters)

        assert self.use_prf and self.use_ptm and self.use_pcn and self.use_margin and self.use_margin and self.use_group

        prf_ptm_combine_ch = prf_inner_ch + ptm_inner_ch

        ####### PTM #######
        if ptm_basicblock_num > 0:

            self.support_temporal_fuse = CHAIN_RSB_BLOCKS(self.num_joints * 5, ptm_inner_ch, ptm_basicblock_num,
                                                          )

            # self.support_temporal_fuse = ChainOfBasicBlocks(self.num_joints * 3, ptm_inner_ch, 1, 1, 2,
            #                                                 ptm_basicblock_num, groups=self.num_joints)
        else:
            self.support_temporal_fuse = nn.Conv2d(self.num_joints * 5, ptm_inner_ch, kernel_size=3, padding=1,
                                                   groups=self.num_joints)

        self.offset_mask_combine_conv = CHAIN_RSB_BLOCKS(prf_ptm_combine_ch, prf_ptm_combine_inner_ch,
                                                         prf_ptm_combine_basicblock_num)
        # self.offset_mask_combine_conv = ChainOfBasicBlocks(prf_ptm_combine_ch, prf_ptm_combine_inner_ch, 1, 1, 2,
        #                                                    prf_ptm_combine_basicblock_num)

        ####### PCN #######
        # self.offsets_list, self.masks_list, self.modulated_deform_conv_list, self.modulated_deform_conv_list2, self.modulated_deform_conv_list3 = [], [], [], [], []
        self.offsets_list, self.masks_list, self.modulated_deform_conv_list = [], [], []
        for d_index, dilation in enumerate(self.deformable_conv_dilations):
            # offsets
            offset_layers, mask_layers = [], []
            offset_layers.append(self._offset_conv(prf_ptm_combine_inner_ch, k, k, dilation, self.num_joints).cuda())
            mask_layers.append(self._mask_conv(prf_ptm_combine_inner_ch, k, k, dilation, self.num_joints).cuda())
            self.offsets_list.append(nn.Sequential(*offset_layers))
            self.masks_list.append(nn.Sequential(*mask_layers))
            self.modulated_deform_conv_list.append(DeformableCONV(self.num_joints, k, dilation))
            # self.modulated_deform_conv_list2.append(DeformableCONV(self.num_joints, k, dilation))
            # self.modulated_deform_conv_list3.append(DeformableCONV(self.num_joints, k, dilation))

        self.offsets_list = nn.ModuleList(self.offsets_list)
        self.masks_list = nn.ModuleList(self.masks_list)
        self.modulated_deform_conv_list = nn.ModuleList(self.modulated_deform_conv_list)
        # self.modulated_deform_conv_list2 = nn.ModuleList(self.modulated_deform_conv_list2)
        # self.modulated_deform_conv_list3 = nn.ModuleList(self.modulated_deform_conv_list3)

        self.motion_layer_48 = FlowLayer(48, self.batch_size, 20)
        self.motion_layer_96 = FlowLayer(96, self.batch_size, 20)
        self.motion_layer_192 = FlowLayer(192, self.batch_size, 20)
        self.motion_layer_384 = FlowLayer(384, self.batch_size, 20)
        
        BN_MOMENTUM = 0.1

        self.bn_1 = nn.BatchNorm2d(48, momentum=BN_MOMENTUM)
        self.bn_2 = nn.BatchNorm2d(96, momentum=BN_MOMENTUM)
        self.bn_3 = nn.BatchNorm2d(192, momentum=BN_MOMENTUM)
        self.bn_4 = nn.BatchNorm2d(384, momentum=BN_MOMENTUM)        

        #self.past_output_layer = CHAIN_RSB_BLOCKS(self.num_joints * 3, self.num_joints, 1)
        #self.next_output_layer = CHAIN_RSB_BLOCKS(self.num_joints * 3, self.num_joints, 1)

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()
        
        #print(gird)

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = grid + flo
        
        #print(flo)

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid, align_corners=False)
        mask = (torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=False)

        # if W==128:
            # np.save('mask.npy', mask.cpu().data.numpy())
            # np.save('warp.npy', output.cpu().data.numpy())
        
        #mask[mask<0.5] = 0
        #mask[mask>0] = 1
        
        #return output*mask
        return output


    def _offset_conv(self, nc, kh, kw, dd, dg):
        conv = nn.Conv2d(nc, dg * 2 * kh * kw, kernel_size=(3, 3), stride=(1, 1), dilation=(dd, dd),
                         padding=(1 * dd, 1 * dd), bias=False)
        return conv

    def _mask_conv(self, nc, kh, kw, dd, dg):
        conv = nn.Conv2d(nc, dg * 1 * kh * kw, kernel_size=(3, 3), stride=(1, 1), dilation=(dd, dd),
                         padding=(1 * dd, 1 * dd), bias=False)
        return conv

    # def forward(self, x, margin, debug=False, vis=False):
    def forward(self, x, **kwargs):
        num_color_channels = 3
        assert "margin" in kwargs
        margin = kwargs["margin"]
        if not x.is_cuda or not margin.is_cuda:
            x.cuda()
            margin.cuda()

        if not self.use_rectifier:
            target_image = x[:, 0:num_color_channels, :, :]
            rough_x = self.rough_pose_estimation_net(target_image)
            return rough_x

        # current / previous / next
        # rough_pose_estimation is hrnet

        # 왜 batchsize 방향으로 처음부터 작업을 하는게 아니라?
        # 네트워크해서 채널방향으로 쪼개고 나서 그것을 batchsize방향으로 합치는 것일까?
        # rough_heatmaps, hrnet_stage3_outputs = self.rough_pose_estimation_net(torch.cat(x.split(num_color_channels, dim=1), 0))
        # hrnet_stage3_outputs = self.rough_pose_estimation_net.stage123_forward(torch.cat(x.split(num_color_channels, dim=1), 0))
        rough_heatmap, feature1, feature2, feature3, feature4 = self.rough_pose_estimation_net(torch.cat(x.split(num_color_channels, dim=1), 0))

        # true_batch_size = int(rough_heatmaps.shape[0] / 5)
        true_batch_size = int(feature1.shape[0] / 5)
        
        
        current_rough_heatmaps, previous_rough_heatmaps, next_rough_heatmaps, previous2_rough_heatmaps, next2_rough_heatmaps = rough_heatmap.split(true_batch_size, dim=0)

        current_hrnet_feature1, previous_hrnet_feature1, next_hrnet_feature1, previous2_hrnet_feature1, next2_hrnet_feature1 = feature1.split(true_batch_size, dim=0)
        current_hrnet_feature2, previous_hrnet_feature2, next_hrnet_feature2, previous2_hrnet_feature2, next2_hrnet_feature2 = feature2.split(true_batch_size, dim=0)
        current_hrnet_feature3, previous_hrnet_feature3, next_hrnet_feature3, previous2_hrnet_feature3, next2_hrnet_feature3 = feature3.split(true_batch_size, dim=0)
        current_hrnet_feature4, previous_hrnet_feature4, next_hrnet_feature4, previous2_hrnet_feature4, next2_hrnet_feature4 = feature4.split(true_batch_size, dim=0)

        # motion_module_flowlayer
        # [b, 48, 96, 72] == [b, 48, h, w]
        with torch.no_grad():
        # u1, u2 형태로 값이 나오도록 수정
        # 현재 -> 미래 or 현재 -> 과거가 아니라
        # 과거 -> 현재 or 미래 -> 현재 형태로 수정을 함. 
            
            # 과거1 -> 미래1
            # ratio 1/2 
            
            pc_u1_1,pc_u2_1 = self.motion_layer_48(previous_hrnet_feature1, next_hrnet_feature1)
            #nc_u1,nc_u2 = self.motion_layer_48(next2_hrnet_feature1, current_hrnet_feature1)
            #p2c_u1_1,p2c_u2_1 = self.motion_layer_48(previous2_hrnet_feature1, current_hrnet_feature1)
            #n2c_u1,n2c_u2 = self.motion_layer_48(next2_hrnet_feature1, current_hrnet_feature1)
        
            temp_mag= torch.sqrt(pc_u1_1*pc_u1_1+pc_u2_1*pc_u2_1)
            temp_mask = temp_mag<0.5
            
            pn_mask_1 = (torch.ones(temp_mag.size())).cuda()
            pn_mask_1[temp_mask] = 0
            
            # 과거1 -> 미래2 
            # ratio 1/3         
            
            pc_u1_2,pc_u2_2 = self.motion_layer_96(previous_hrnet_feature2, next2_hrnet_feature2)
            #nc_u1,nc_u2 = self.motion_layer_96(next2_hrnet_feature2, current_hrnet_feature2)
            #p2c_u1_2,p2c_u2_2 = self.motion_layer_96(previous2_hrnet_feature2, current_hrnet_feature2)
            #n2c_u1,n2c_u2 = self.motion_layer_96(next2_hrnet_feature2, current_hrnet_feature2)

            temp_mag= torch.sqrt(pc_u1_2*pc_u1_2+pc_u2_2*pc_u2_2)
            temp_mask = temp_mag<0.5
            
            pn_mask_2 = (torch.ones(temp_mag.size())).cuda()
            pn_mask_2[temp_mask] = 0            
            
            
            # 과거2 -> 미래1 
            # ratio 2/3 
            
            pc_u1_3,pc_u2_3 = self.motion_layer_192(previous2_hrnet_feature3, next_hrnet_feature3)
            #nc_u1,nc_u2 = self.motion_layer_192(next2_hrnet_feature3, current_hrnet_feature3)
            #p2c_u1_3,p2c_u2_3 = self.motion_layer_192(previous2_hrnet_feature3, current_hrnet_feature3)
            #n2c_u1,n2c_u2 = self.motion_layer_192(next2_hrnet_feature3, current_hrnet_feature3)    
            
            temp_mag= torch.sqrt(pc_u1_3*pc_u1_3+pc_u2_3*pc_u2_3)
            temp_mask = temp_mag<0.5
            
            pn_mask_3 = (torch.ones(temp_mag.size())).cuda()
            pn_mask_3[temp_mask] = 0   
            
            
            # 과거2 -> 미래2 
            # ratio 1/2 
            
            pc_u1_4,pc_u2_4 = self.motion_layer_384(previous2_hrnet_feature4, next2_hrnet_feature4)        
            
            temp_mag= torch.sqrt(pc_u1_4*pc_u1_4+pc_u2_4*pc_u2_4)
            temp_mask = temp_mag<0.5
            
            pn_mask_4 = (torch.ones(temp_mag.size())).cuda()
            pn_mask_4[temp_mask] = 0   

        # ===========================================================================================================================================================
            
             # 과거1 -> 미래1
            ratio = (1/2)**(1/2)
            
            input_feature1 = []
            for temp_m,output1,output2,output3 in zip(pn_mask_1.split(1, dim=1),previous_hrnet_feature1.split(1, dim=1),pc_u1_1.split(1, dim=1),pc_u2_1.split(1, dim=1)):
                flo = torch.cat([ratio*output2*temp_m,ratio*output3*temp_m],dim=1)
                temp = self.warp(output1*temp_m,flo)
                input_feature1.append(temp)
            
            input_feature1 = torch.cat(input_feature1,dim=1)
            input_feature1 = input_feature1 + current_hrnet_feature1
            input_feature1 = self.bn_1(input_feature1) 
    
             # 과거1 -> 미래2
            ratio = (1/3)**(1/2)
    
            input_feature2 = []
            for temp_m,output1,output2,output3 in zip(pn_mask_2.split(1, dim=1),previous_hrnet_feature2.split(1, dim=1),pc_u1_2.split(1, dim=1),pc_u2_2.split(1, dim=1)):
                flo2 = torch.cat([ratio*output2*temp_m,ratio*output3*temp_m],dim=1)
                temp2 = self.warp(output1*temp_m,flo2)
                input_feature2.append(temp2)
            
            input_feature2 = torch.cat(input_feature2,dim=1)
            input_feature2 = input_feature2 + current_hrnet_feature2
            input_feature2 = self.bn_2(input_feature2) 
            
            # 과거2 -> 미래1
            ratio = (2/3)**(1/2)
            
            input_feature3 = []
            for temp_m,output1,output2,output3 in zip(pn_mask_3.split(1, dim=1),previous_hrnet_feature3.split(1, dim=1),pc_u1_3.split(1, dim=1),pc_u2_3.split(1, dim=1)):
                flo3 = torch.cat([ratio*output2*temp_m,ratio*output3*temp_m],dim=1)
                temp3 = self.warp(output1*temp_m,flo3)
                input_feature3.append(temp3)
            
            input_feature3 = torch.cat(input_feature3,dim=1) 
            input_feature3 = input_feature3 + current_hrnet_feature3    
            input_feature3 = self.bn_3(input_feature3) 
            
            # 과거2 -> 미래2
            ratio = (1/2)**(1/2)
            
            input_feature4 = []
            for temp_m,output1,output2,output3 in zip(pn_mask_4.split(1, dim=1),previous_hrnet_feature4.split(1, dim=1),pc_u1_4.split(1, dim=1),pc_u2_4.split(1, dim=1)):
                flo4 = torch.cat([ratio*output2*temp_m,ratio*output3*temp_m],dim=1)
                temp4 = self.warp(output1*temp_m,flo4)
                input_feature4.append(temp4)
            
            input_feature4 = torch.cat(input_feature4,dim=1)     
            input_feature4 = input_feature4 + current_hrnet_feature4 
            input_feature4 = self.bn_4(input_feature4)               
                   


        stage4_2_input = []
        stage4_2_input.append(input_feature1)
        stage4_2_input.append(input_feature2)
        stage4_2_input.append(input_feature3)
        stage4_2_input.append(input_feature4)

        motion_heatmap = self.rough_pose_estimation_net.stage4_2_forward(stage4_2_input)

        '''
        prev_next_heatmaps = torch.cat([pre_motion_heatmap, next_motion_heatmap], dim=1)

        # hrnet + posetrack2018기반으로 fintuning pretrained된 값을 사용 
        # validation 80 정도의 값을 가지고 있음.
        current_rough_heatmaps_list = current_rough_heatmaps.split(1, dim=1)
        previous_rough_heatmaps_list = previous_rough_heatmaps.split(1, dim=1)
        next_rough_heatmaps_list = next_rough_heatmaps.split(1, dim=1)
        previous2_rough_heatmaps_list = previous2_rough_heatmaps.split(1, dim=1)
        next2_rough_heatmaps_list = next2_rough_heatmaps.split(1, dim=1)

        temp_support_fuse_list = []
        for joint_index in range(self.num_joints):
            temp_support_fuse_list.append(current_rough_heatmaps_list[joint_index])
            temp_support_fuse_list.append(previous_rough_heatmaps_list[joint_index]*0.5)
            temp_support_fuse_list.append(next_rough_heatmaps_list[joint_index]*0.5)
            temp_support_fuse_list.append(previous2_rough_heatmaps_list[joint_index]*0.5)
            temp_support_fuse_list.append(next2_rough_heatmaps_list[joint_index]*0.5)

        support_heatmaps = torch.cat(temp_support_fuse_list, dim=1)

        # self.support_temporal_fuse = CHAIN_RSB_BLOCKS(self.num_joints * 3, ptm_inner_ch, ptm_basicblock_num,)
        # 해당 위 layer는 3*3 stack layer 부분이다.
        # 이 때 왜? ptm의 결과를
        support_heatmaps = self.support_temporal_fuse(support_heatmaps).cuda()

        # 3*3 conv stack conv 처리 !!
        prev_next_combine_featuremaps = self.offset_mask_combine_conv(torch.cat([prev_next_heatmaps, support_heatmaps], dim=1))

        # DEFORMABLE_CONV:
        # DILATION:
        # - 3
        # - 6
        # - 9
        # - 12
        # - 15

        warped_heatmaps_list = []
        for d_index, dilation in enumerate(self.deformable_conv_dilations):
            offsets = self.offsets_list[d_index](prev_next_combine_featuremaps)
            masks = self.masks_list[d_index](prev_next_combine_featuremaps)

            warped_heatmaps1 = self.modulated_deform_conv_list[d_index](support_heatmaps, offsets, masks)
            # warped_heatmaps2 = self.modulated_deform_conv_list2[d_index](prev_heatmaps, offsets, masks)
            # warped_heatmaps3 = self.modulated_deform_conv_list3[d_index](next_heatmaps, offsets, masks)
            warped_heatmaps_list.append(warped_heatmaps1)
            # warped_heatmaps_list.append(warped_heatmaps2)
            # warped_heatmaps_list.append(warped_heatmaps3)

        if self.deformable_aggregation_type == "weighted_sum":

            # 5개의 dilations가 있기 때문에 해당 부분을 균등하게 1/5 씩 weight값을 부여함.
            warper_weight = 1 / len(self.deformable_conv_dilations)
            output_heatmaps = warper_weight * warped_heatmaps_list[0]
            for warper_heatmaps in warped_heatmaps_list[1:]:
                output_heatmaps += warper_weight * warper_heatmaps

        else:
            output_heatmaps = self.deformable_aggregation_conv(torch.cat(warped_heatmaps_list, dim=1))
            # elif self.deformable_aggregation_type == "conv":

        if not self.freeze_hrnet_weights:
            return current_rough_heatmaps, output_heatmaps
        else:
            return output_heatmaps, current_rough_heatmaps,pre_motion_heatmap, next_motion_heatmap 
        '''
        return motion_heatmap, current_rough_heatmaps, previous_rough_heatmaps

    def init_weights(self):
        logger = logging.getLogger(__name__)
        ## init_weights
        rough_pose_estimation_name_set = set()
        for module_name, module in self.named_modules():
            # rough_pose_estimation_net 单独判断一下
            if module_name.split('.')[0] == "rough_pose_estimation_net":
                rough_pose_estimation_name_set.add(module_name)
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.001)
                for name, _ in module.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, std=0.001)

                for name, _ in module.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(module.bias, 0)
            elif isinstance(module, DeformConv):
                filler = torch.zeros(
                    [module.weight.size(0), module.weight.size(1), module.weight.size(2), module.weight.size(3)],
                    dtype=torch.float32, device=module.weight.device)
                for k in range(module.weight.size(0)):
                    filler[k, k, int(module.weight.size(2) / 2), int(module.weight.size(3) / 2)] = 1.0
                module.weight = torch.nn.Parameter(filler)
                # module.weight.requires_grad = True
            elif isinstance(module, ModulatedDeformConv):
                filler = torch.zeros(
                    [module.weight.size(0), module.weight.size(1), module.weight.size(2), module.weight.size(3)],
                    dtype=torch.float32, device=module.weight.device)
                for k in range(module.weight.size(0)):
                    filler[k, k, int(module.weight.size(2) / 2), int(module.weight.size(3) / 2)] = 1.0
                module.weight = torch.nn.Parameter(filler)
                # module.weight.requires_grad = True
            else:
                for name, _ in module.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(module.bias, 0)
                    if name in ['weights']:
                        nn.init.normal_(module.weight, std=0.001)
                        
# =====================================================================================================
        parameters_names = set()
        for name, _ in self.named_parameters():
            parameters_names.add(name)

        buffers_names = set()
        for name, _ in self.named_buffers():
            buffers_names.add(name)

        if os.path.isfile(self.pretrained):
            pretrained_state_dict = torch.load(self.pretrained)
            if 'state_dict' in pretrained_state_dict.keys():
                pretrained_state_dict = pretrained_state_dict['state_dict']
            logger.info('=> loading pretrained model {}'.format(self.pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                '''
                # self.pretrained_layers => * 이 정의되어 있음.!!
                if name.split('.')[0] in self.pretrained_layers or self.pretrained_layers[0] is '*':
                    layer_name = name.split('.')[0]
                    if layer_name in rough_pose_estimation_name_set:
                        need_init_state_dict[name] = m
                    else:
                        # 为了适应原本hrnet得预训练网络
                        # 이 과정이 있는 이유는 ... coco로 학습된 pretrained 모델을 불러오기 때문에 
                        # 해당 과정에서 layer이름에 rough_pose_estimation_net가 앞에 더 붙게 된다. 
                        # 즉 이 부분을 보완하기 위해 작업이 된 것이라고 보면 좋다. 
                        new_layer_name = "rough_pose_estimation_net.{}".format(layer_name)
                        if new_layer_name in rough_pose_estimation_name_set:
                            parameter_name = "rough_pose_estimation_net.{}".format(name)
                            need_init_state_dict[parameter_name] = m
                '''            
                # if name.split('.')[0] in self.pretrained_layers or self.pretrained_layers[0] is '*':
                if name in parameters_names or name in buffers_names:
                        # logger.info('=> init {} from {}'.format(name, pretrained))
                        print('=> init {}'.format(name))
                        need_init_state_dict[name] = m 
                ## stage4_2 분기하는 기존 stage4의 weight값을 불러올 수 있도록 세팅하는 부분이다.         
                if name.split('.')[0] == "rough_pose_estimation_net":
                    if name.split('.')[1] == "stage4":
                        new_name = name.split('.')
                        new_name[1] = "stage4_2"
                        layer_name = '.'.join(new_name)
                        need_init_state_dict[layer_name] = m
                    if name.split('.')[1] == "final_layer":
                        new_name = name.split('.')
                        new_name[1] = "final_layer_2"
                        layer_name = '.'.join(new_name)
                        need_init_state_dict[layer_name] = m                              
            self.load_state_dict(need_init_state_dict, strict=False)
            
            
        elif self.pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(self.pretrained))

        # rough_pose_estimation
        if self.freeze_hrnet_weights:
            self.rough_pose_estimation_net.freeze_weight()

    @classmethod
    def get_model_hyper_parameters(cls, args, cfg):
        prf_inner_ch = cfg.MODEL.PRF_INNER_CH
        prf_basicblock_num = cfg.MODEL.PRF_BASICBLOCK_NUM
        ptm_inner_ch = cfg.MODEL.PTM_INNER_CH
        ptm_basicblock_num = cfg.MODEL.PTM_BASICBLOCK_NUM
        prf_ptm_combine_inner_ch = cfg.MODEL.PRF_PTM_COMBINE_INNER_CH
        prf_ptm_combine_basicblock_num = cfg.MODEL.PRF_PTM_COMBINE_BASICBLOCK_NUM
        if "DILATION" in cfg.MODEL.DEFORMABLE_CONV:
            dilation = cfg.MODEL.DEFORMABLE_CONV.DILATION
            dilation_str = ",".join(map(str, dilation))
        else:
            dilation_str = ""
        hyper_parameters_setting = "chPRF_{}_nPRF_{}_chPTM_{}_nPTM_{}_chComb_{}_nComb_{}_D_{}".format(
            prf_inner_ch, prf_basicblock_num, ptm_inner_ch, ptm_basicblock_num, prf_ptm_combine_inner_ch,
            prf_ptm_combine_basicblock_num,
            dilation_str)

        return hyper_parameters_setting

    @classmethod
    def get_net(cls, cfg, phase, **kwargs):
        model = DcPose_RSN(cfg, phase, **kwargs)
        return model
