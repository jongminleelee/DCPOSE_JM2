#!/usr/bin/python
# -*- coding:utf8 -*-
import time
import torch
import numpy as np
import os.path as osp
import logging
from torchvision.utils import make_grid

from tensorboardX import SummaryWriter
from .evaludate import accuracy, pck_accuracy, pck_accuracy_origin_image, save_result_images, save_fusion_images
from engine.core import CORE_FUNCTION_REGISTRY, BaseFunction, AverageMeter

# from engine.evaludate import accuracy
from engine.defaults import VAL_PHASE, TEST_PHASE, TRAIN_PHASE
from datasets.process.heatmaps_process import get_max_preds
from datasets.process import get_final_preds
from datasets.transforms import reverse_transforms

from utils.utils_bbox import cs2box
from utils.utils_folder import create_folder
from utils.utils_image_tensor import tensor2im

from tabulate import tabulate



@CORE_FUNCTION_REGISTRY.register()
class CommonFunction(BaseFunction):

    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.output_dir = cfg.OUTPUT_DIR

        if "criterion" in kwargs.keys():
            self.criterion = kwargs["criterion"]
        if "tb_log_dir" in kwargs.keys():
            self.tb_log_dir = kwargs["tb_log_dir"]
        if "writer_dict" in kwargs.keys():
            self.writer_dict = kwargs["writer_dict"]

        self.PE_Name = kwargs.get("PE_Name", "DCPOSE")
        ##
        self.max_iter_num = 0
        self.dataloader_iter = None
        self.tb_writer = None
        self.global_steps = 0
        self.DataSetName = str(self.cfg.DATASET.NAME).upper()

    def train(self, model, epoch, optimizer, dataloader, tb_writer_dict, **kwargs):
        self.tb_writer = tb_writer_dict["writer"]
        self.global_steps = tb_writer_dict["global_steps"]
        logger = logging.getLogger(__name__)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()
        acc2 = AverageMeter()
        acc3 = AverageMeter()
        acc4 = AverageMeter()
        
        # switch to train mode
        model.train()

        self.max_iter_num = len(dataloader)
        self.dataloader_iter = iter(dataloader)
        end = time.time()

        for iter_step in range(self.max_iter_num):
            input_x, input2_sup_A, input2_sup_B, target_heatmaps, target_heatmaps_weight, meta, input_sup_A, input_sup_B, \
            prev2_target_motion_heatmaps, next2_target_motion_heatmaps, prev_target_motion_heatmaps, next_target_motion_heatmaps \
                = next(self.dataloader_iter)
            self._before_train_iter(input_x)

            data_time.update(time.time() - end)

            target_heatmaps = target_heatmaps.cuda(non_blocking=True)
            target_heatmaps_weight = target_heatmaps_weight.cuda(non_blocking=True)
            if prev_target_motion_heatmaps is not None and next_target_motion_heatmaps is not None \
                    and prev2_target_motion_heatmaps is not None and next2_target_motion_heatmaps is not None:
                prev_target_motion_heatmaps = prev_target_motion_heatmaps.cuda(non_blocking=True)
                next_target_motion_heatmaps = next_target_motion_heatmaps.cuda(non_blocking=True)
                prev2_target_motion_heatmaps = prev2_target_motion_heatmaps.cuda(non_blocking=True)
                next2_target_motion_heatmaps = next2_target_motion_heatmaps.cuda(non_blocking=True)

            if self.PE_Name == "HRNET":
                outputs = model(input_x.cuda())
            elif self.PE_Name == "DCPOSE":
                margin_left, margin_right = meta["margin_left"], meta["margin_right"]
                margin = torch.stack([margin_left, margin_right], dim=1).cuda()
                concat_input = torch.cat((input_x, input_sup_A, input_sup_B, input2_sup_A, input2_sup_B), 1).cuda()

                outputs = model(concat_input, margin=margin)
            else:
                outputs = model(input_x.cuda())

            if isinstance(outputs, list) or isinstance(outputs, tuple):
                pred_heatmaps = outputs[0]
                pred_loss = self.criterion(pred_heatmaps, target_heatmaps, target_heatmaps_weight)

                currnet_rough_heatmaps = outputs[1]
                pre_motion_heatmap = outputs[2]
                #next_motion_heatmap = outputs[3]
                
                #pre_motion_loss = self.criterion(pre_motion_heatmap, prev2_target_motion_heatmaps, target_heatmaps_weight)
                #next_motion_loss = self.criterion(next_motion_heatmap, next2_target_motion_heatmaps, target_heatmaps_weight)
                #current_loss = self.criterion(current_heatmaps, target_heatmaps, target_heatmaps_weight)
                #prev_loss = self.criterion(prev_heatmaps, prev_target_motion_heatmaps, target_heatmaps_weight)
                #prev2_loss = self.criterion(prev2_heatmaps, prev2_target_motion_heatmaps, target_heatmaps_weight)
                #next_loss = self.criterion(next_heatmaps, next_target_motion_heatmaps, target_heatmaps_weight)
                #next2_loss = self.criterion(next2_heatmaps, next2_target_motion_heatmaps, target_heatmaps_weight)
               
                loss = pred_loss # + pre_motion_loss + next_motion_loss
                
                # for pred_heatmaps in outputs[1:]:
                #     loss += self.criterion(pred_heatmaps, target_heatmaps, target_heatmaps_weight)
            else:
                pred_heatmaps = outputs
                loss = self.criterion(pred_heatmaps, target_heatmaps, target_heatmaps_weight)

            # compute gradient and do update step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure accuracy and record loss
            losses.update(loss.item(), input_x.size(0))

            _, avg_acc, cnt, _ = accuracy(pred_heatmaps.detach().cpu().numpy(), target_heatmaps.detach().cpu().numpy())
            acc.update(avg_acc, cnt)
            
            _, avg_acc2, cnt2, _ = accuracy(currnet_rough_heatmaps.detach().cpu().numpy(), target_heatmaps.detach().cpu().numpy())
            acc2.update(avg_acc2, cnt2)
            
            _, avg_acc3, cnt3, _ = accuracy(pre_motion_heatmap.detach().cpu().numpy(), target_heatmaps.detach().cpu().numpy())
            acc3.update(avg_acc3, cnt3)
            '''
            _, avg_acc4, cnt4, _ = accuracy(next_motion_heatmap.detach().cpu().numpy(), next2_target_motion_heatmaps.detach().cpu().numpy())
            acc4.update(avg_acc4, cnt4)
            '''
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if iter_step % self.cfg.PRINT_FREQ == 0 or iter_step >= self.max_iter_num - 1:

                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                      'pre_final_acc {acc.val:.3f} ({acc.avg:.3f})\t'\
                      'c_acc {acc2.val:.3f} ({acc2.avg:.3f})\t'\
                      'pre_acc {acc3.val:.3f} ({acc3.avg:.3f})\t'.format(epoch, iter_step, self.max_iter_num, batch_time=batch_time,
                                                                        speed=input_x.size(0) / batch_time.val,
                                                                        data_time=data_time, loss=losses, acc=acc, acc2=acc2, acc3=acc3)
                      #
                      #'next_motion {acc4.val:.3f} ({acc4.avg:.3f})\t'\


                logger.info(msg)

                '''
                B, C, H, W = target_heatmaps.shape
                ridx = np.random.randint(0, B)

                support_heatmaps = outputs[2].detach().cpu().numpy()
                pred_heatmaps_print = outputs[0].detach().cpu().numpy()
                pred_prev_heatmaps = outputs[3].detach().cpu().numpy()
                pred_next_heatmaps = outputs[4].detach().cpu().numpy()
                gt_heatmaps = target_heatmaps.detach().cpu().numpy()
                prev_gt_heatmaps = prev_target_motion_heatmaps.detach().cpu().numpy()
                next_gt_heatmaps = next_target_motion_heatmaps.detach().cpu().numpy()
                prev2_gt_heatmaps = prev2_target_motion_heatmaps.detach().cpu().numpy()
                next2_gt_heatmaps = next2_target_motion_heatmaps.detach().cpu().numpy()
                current_heatmaps = current_heatmaps.detach().cpu().numpy()
                prev_heatmaps = prev_heatmaps.detach().cpu().numpy()
                next_heatmaps = next_heatmaps.detach().cpu().numpy()
                prev2_heatmaps = prev2_heatmaps.detach().cpu().numpy()
                next2_heatmaps = next2_heatmaps.detach().cpu().numpy()

                support_heatmap = support_heatmaps[ridx]
                pred_heatmap_print = pred_heatmaps_print[ridx]
                pred_prev_heatmap = pred_prev_heatmaps[ridx]
                pred_next_heatmap = pred_next_heatmaps[ridx]
                gt_heatmap = gt_heatmaps[ridx]
                prev_gt_heatmap = prev_gt_heatmaps[ridx]
                next_gt_heatmap = next_gt_heatmaps[ridx]
                prev2_gt_heatmap = prev2_gt_heatmaps[ridx]
                next2_gt_heatmap = next2_gt_heatmaps[ridx]
                current_heatmap = current_heatmaps[ridx]
                prev_heatmap = prev_heatmaps[ridx]
                next_heatmap = next_heatmaps[ridx]
                prev2_heatmap = prev2_heatmaps[ridx]
                next2_heatmap = next2_heatmaps[ridx]

                support_pose, support_conf = get_max_preds(support_heatmaps)
                pose, pred_conf = get_max_preds(pred_heatmaps_print)
                pred_prev_pose, pred_prev_conf = get_max_preds(pred_prev_heatmaps)
                pred_next_pose, pred_next_conf = get_max_preds(pred_next_heatmaps)
                cpose, cpred_conf = get_max_preds(current_heatmaps)
                ppose, ppred_conf = get_max_preds(prev_heatmaps)
                npose, npred_conf = get_max_preds(next_heatmaps)
                p2pose, p2pred_conf = get_max_preds(prev2_heatmaps)
                n2pose, n2pred_conf = get_max_preds(next2_heatmaps)
                gt_pose, gt_conf = get_max_preds(gt_heatmaps)
                prev_gt_pose, prev_gt_conf = get_max_preds(prev_gt_heatmaps)
                next_gt_pose, next_gt_conf = get_max_preds(next_gt_heatmaps)
                prev2_gt_pose, prev2_gt_conf = get_max_preds(prev2_gt_heatmaps)
                next2_gt_pose, next2_gt_conf = get_max_preds(next2_gt_heatmaps)

                img = np.transpose(input_x[ridx].detach().cpu().numpy(), (1, 2, 0))
                pimg = np.transpose(input_sup_A[ridx].detach().cpu().numpy(), (1, 2, 0))
                nimg = np.transpose(input_sup_B[ridx].detach().cpu().numpy(), (1, 2, 0))
                p2img = np.transpose(input2_sup_A[ridx].detach().cpu().numpy(), (1, 2, 0))
                n2img = np.transpose(input2_sup_B[ridx].detach().cpu().numpy(), (1, 2, 0))

                save_result_images(osp.join(self.output_dir, 'img'), img, support_pose[ridx], support_conf[ridx],
                                   heatmaps=support_heatmap, name='support_')
                save_result_images(osp.join(self.output_dir, 'img'), img, pose[ridx], pred_conf[ridx],
                                   heatmaps=pred_heatmap_print, name='pred_')
                save_result_images(osp.join(self.output_dir, 'img'), img, gt_pose[ridx], gt_conf[ridx],
                                   heatmaps=gt_heatmap, name='gt_')
                save_result_images(osp.join(self.output_dir, 'img'), img, prev_gt_pose[ridx], prev_gt_conf[ridx],
                                   heatmaps=prev_gt_heatmap, name='gt_prev_')
                save_result_images(osp.join(self.output_dir, 'img'), img, next_gt_pose[ridx], next_gt_conf[ridx],
                                   heatmaps=next_gt_heatmap, name='gt_next_')
                save_result_images(osp.join(self.output_dir, 'img'), img, prev2_gt_pose[ridx], prev2_gt_conf[ridx],
                                   heatmaps=prev2_gt_heatmap, name='gt_prev2_')
                save_result_images(osp.join(self.output_dir, 'img'), img, next2_gt_pose[ridx], next2_gt_conf[ridx],
                                   heatmaps=next2_gt_heatmap, name='gt_next2_')
                save_result_images(osp.join(self.output_dir, 'img'), img, pred_prev_pose[ridx], pred_prev_conf[ridx],
                                   heatmaps=pred_prev_heatmap, name='pred_prev_')
                save_result_images(osp.join(self.output_dir, 'img'), img, pred_next_pose[ridx], pred_next_conf[ridx],
                                   heatmaps=pred_next_heatmap, name='pred_next_')
                save_result_images(osp.join(self.output_dir, 'img', 'backbone'), img, cpose[ridx], cpred_conf[ridx],
                                   heatmaps=current_heatmap, name='h_c_')
                save_result_images(osp.join(self.output_dir, 'img', 'backbone'), img, ppose[ridx], ppred_conf[ridx],
                                   heatmaps=prev_heatmap, name='h_p_')
                save_result_images(osp.join(self.output_dir, 'img', 'backbone'), img, npose[ridx], npred_conf[ridx],
                                   heatmaps=next_heatmap, name='h_n_')
                save_result_images(osp.join(self.output_dir, 'img', 'backbone'), img, p2pose[ridx], p2pred_conf[ridx],
                                   heatmaps=prev2_heatmap, name='h_p2_')
                save_result_images(osp.join(self.output_dir, 'img', 'backbone'), img, n2pose[ridx], n2pred_conf[ridx],
                                   heatmaps=next2_heatmap, name='h_n2_')
                save_fusion_images(osp.join(self.output_dir, 'img', 'backbone'), img, heatmaps=current_heatmap,
                                   name='h_c_')
                save_fusion_images(osp.join(self.output_dir, 'img', 'backbone'), img, heatmaps=prev_heatmap,
                                   name='h_p_')
                save_fusion_images(osp.join(self.output_dir, 'img', 'backbone'), img, heatmaps=next_heatmap,
                                   name='h_n_')
                save_fusion_images(osp.join(self.output_dir, 'img', 'backbone'), img, heatmaps=prev2_heatmap,
                                   name='h_p2_')
                save_fusion_images(osp.join(self.output_dir, 'img', 'backbone'), img, heatmaps=next2_heatmap,
                                   name='h_n2_')
                '''
            # For Tensorboard
            self.tb_writer.add_scalar('train_loss', losses.val, self.global_steps)
            self.tb_writer.add_scalar('train_acc', acc.val, self.global_steps)
            self.global_steps += 1

        tb_writer_dict["global_steps"] = self.global_steps

    def eval(self, model, dataloader, tb_writer_dict, **kwargs):

        logger = logging.getLogger(__name__)

        self.tb_writer = tb_writer_dict["writer"]
        self.global_steps = tb_writer_dict["global_steps"]

        batch_time, data_time, losses, acc = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()


        phase = kwargs.get("phase", VAL_PHASE)
        epoch = kwargs.get("epoch", "specified_model")
        # switch to evaluate mode
        model.eval()

        self.max_iter_num = len(dataloader)
        self.dataloader_iter = iter(dataloader)
        dataset = dataloader.dataset
        # prepare data fro validate
        num_samples = len(dataset)
        all_preds = np.zeros((num_samples, self.cfg.MODEL.NUM_JOINTS, 3), dtype=np.float)
        all_boxes = np.zeros((num_samples, 6))
        image_path = []
        filenames = []
        filenames_map = {}
        filenames_counter = 0
        imgnums = []
        idx = 0
        acc_threshold = 0.7

        ###
        result_output_dir, vis_output_dir = self.vis_setup(logger, phase, epoch)
        ###

        with torch.no_grad():
            end = time.time()
            num_batch = len(dataloader)
            for iter_step in range(self.max_iter_num):
                input_x, input2_sup_A, input2_sup_B, target_heatmaps, target_heatmaps_weight, meta, input_sup_A, input_sup_B = next(self.dataloader_iter)
                if phase == VAL_PHASE:
                    self._before_val_iter(input_x)

                data_time.update(time.time() - end)
                # prepare model input
                margin_left = meta["margin_left"]
                margin_right = meta["margin_right"]
                margin = torch.stack([margin_left, margin_right], dim=1).cuda()
                target_heatmaps = target_heatmaps.cuda(non_blocking=True)

                # outputs = model(concat_input, margin)
                if self.PE_Name == 'DCPOSE':
                    concat_input = torch.cat((input_x, input_sup_A, input_sup_B, input2_sup_A, input2_sup_B), 1).cuda()
                    outputs = model(concat_input, margin=margin)
                else:
                    outputs = model(input_x.cuda())

                if phase == VAL_PHASE:
                    if isinstance(model, torch.nn.DataParallel):
                        vis_dict = getattr(model.module, "vis_dict", None)
                    else:
                        vis_dict = getattr(model, "vis_dict", None)

                    if vis_dict:
                        self._running_val_iter(vis_dict=vis_dict, model_input=[input_x, input_sup_A, input_sup_B])

                if isinstance(outputs, list) or isinstance(outputs, tuple):
                    pred_heatmaps = 0.5*outputs[0]+0.5*outputs[1]
                else:
                    pred_heatmaps = outputs


                _, avg_acc, cnt, _ = accuracy(pred_heatmaps.detach().cpu().numpy(), target_heatmaps.detach().cpu().numpy())

                acc.update(avg_acc, cnt)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if iter_step % self.cfg.PRINT_FREQ == 0 or iter_step >= (num_batch - 1):
                    msg = '{}: [{}/{}]\t' \
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                          'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                          'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(phase, iter_step, num_batch, batch_time=batch_time,
                                                                          data_time=data_time, acc=acc)
                    logger.info(msg)



                #### for eval ####
                for ff in range(len(meta['image'])):
                    cur_nm = meta['image'][ff]
                    if not cur_nm in filenames_map:
                        filenames_map[cur_nm] = [filenames_counter]
                    else:
                        filenames_map[cur_nm].append(filenames_counter)
                    filenames_counter += 1

                center = meta['center'].numpy()
                scale = meta['scale'].numpy()
                score = meta['score'].numpy()
                num_images = input_x.size(0)

                preds, maxvals = get_final_preds(pred_heatmaps.clone().cpu().numpy(), center, scale)
                all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
                all_preds[idx:idx + num_images, :, 2:3] = maxvals
                all_boxes[idx:idx + num_images, 0:2] = center[:, 0:2]
                all_boxes[idx:idx + num_images, 2:4] = scale[:, 0:2]
                all_boxes[idx:idx + num_images, 4] = np.prod(scale * 200, 1)
                all_boxes[idx:idx + num_images, 5] = score
                image_path.extend(meta['image'])
                idx += num_images

                # tensorboard writ
                self.global_steps += 1
                #
                self._after_val_iter(meta["image"], preds, maxvals, vis_output_dir, center, scale)


        logger.info('########################################')
        logger.info('{}'.format(self.cfg.EXPERIMENT_NAME))

        name_values, perf_indicator = dataset.evaluate(self.cfg, all_preds, result_output_dir, all_boxes,
                                                       filenames_map, filenames, imgnums)

        model_name = self.cfg.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                self._print_name_value(name_value, model_name)
        else:
            self._print_name_value(name_values, model_name)

        tb_writer_dict["global_steps"] = self.global_steps

    def _before_train_iter(self, batch_x):
        if not self.cfg.DEBUG.VIS_TENSORBOARD:
            return

        show_image_num = min(6, len(batch_x))
        batch_x = batch_x[:show_image_num]
        label_name = "train_{}_x".format(self.global_steps)
        save_image = []
        for x in batch_x:
            x = tensor2im(x)
            save_image.append(x)
        save_image = np.stack(save_image, axis=0)
        self.tb_writer.add_images(label_name, save_image, global_step=self.global_steps, dataformats="NHWC")

    def _before_val_iter(self, batch_x):
        if not self.cfg.DEBUG.VIS_TENSORBOARD:
            return

        show_image_num = min(6, len(batch_x))
        batch_x = batch_x[:show_image_num]
        label_name = "val_{}_x".format(self.global_steps)
        save_image = []
        for x in batch_x:
            x = tensor2im(x)
            save_image.append(x)
        save_image = np.stack(save_image, axis=0)
        self.tb_writer.add_images(label_name, save_image, global_step=self.global_steps, dataformats="NHWC")

    def _running_val_iter(self, **kwargs):
        if not self.cfg.DEBUG.VIS_TENSORBOARD:
            return

        vis_dict = kwargs.get("vis_dict")
        #
        show_image_num = min(3, len(vis_dict["current_x"]))
        current_x = vis_dict["current_x"][0:show_image_num]  # [N,3,384,288]
        previous_x = vis_dict["previous_x"][0:show_image_num]  # [N,3,384,288]
        next_x = vis_dict["next_x"][0:show_image_num]  # [N,3,384,288]
        current_rough_heatmaps = vis_dict["current_rough_heatmaps"][0:show_image_num]  # [N,17,96,72]
        previous_rough_heatmaps = vis_dict["previous_rough_heatmaps"][0:show_image_num]  # [N,17,96,72]
        next_rough_heatmaps = vis_dict["next_rough_heatmaps"][0:show_image_num]  # [N,17,96,72]
        diff_A = vis_dict["diff_A"][0:show_image_num]  # [N,17,96,72]
        diff_B = vis_dict["diff_B"][0:show_image_num]  # [N,17,96,72]
        diff_heatmaps = vis_dict["diff_heatmaps"][0:show_image_num]  # [N,34,96,72]
        support_heatmaps = vis_dict["support_heatmaps"][0:show_image_num]  # [N,17,96,72]
        prf_ptm_combine_featuremaps = vis_dict["prf_ptm_combine_featuremaps"][0:show_image_num]  # [N,96,96,72]
        warped_heatmaps_list = [warped_heatmaps[0:show_image_num] for warped_heatmaps in vis_dict["warped_heatmaps_list"]]  # [N,17,96,72]
        output_heatmaps = vis_dict["output_heatmaps"][0:show_image_num]  # [N,17,96,72]

        show_three_input_image = make_grid(reverse_transforms(torch.cat([previous_x, current_x, next_x], dim=0)), nrow=show_image_num)
        self.tb_writer.add_image('01_three_input_image', show_three_input_image, global_step=self.global_steps)

        # show 2.
        three_rough_heatmaps_channels = []
        current_rough_heatmap_channels = current_rough_heatmaps.split(1, dim=1)
        previous_rough_heatmap_channels = previous_rough_heatmaps.split(1, dim=1)
        next_rough_heatmap_channels = next_rough_heatmaps.split(1, dim=1)
        num_channel = current_rough_heatmaps.shape[1]
        for i in range(num_channel):
            three_rough_heatmaps_channels.append(current_rough_heatmap_channels[i])
            three_rough_heatmaps_channels.append(previous_rough_heatmap_channels[i])
            three_rough_heatmaps_channels.append(next_rough_heatmap_channels[i])

        three_heatmaps_tensor = torch.clamp_min(torch.cat(three_rough_heatmaps_channels, dim=0), 0)
        three_heatmaps_image = make_grid(three_heatmaps_tensor, nrow=show_image_num)
        self.tb_writer.add_image('02_three_heatmaps_image', three_heatmaps_image, global_step=self.global_steps)

        # show 3.
        two_diff_channels = []
        diff_A_channels = diff_A.split(1, dim=1)
        diff_B_channels = diff_B.split(1, dim=1)
        num_channel = current_rough_heatmaps.shape[1]
        for i in range(num_channel):
            two_diff_channels.append(diff_A_channels[i])
            two_diff_channels.append(diff_B_channels[i])

        two_diff_channels_tensor = torch.clamp_min(torch.cat(two_diff_channels, dim=0), 0)
        two_diff_image = make_grid(two_diff_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('03_two_diff_image', two_diff_image, global_step=self.global_steps)

        # show4.
        diff_heatmaps_channels_tensor = torch.clamp_min(torch.cat(torch.split(diff_heatmaps, 1, dim=1), dim=0), 0)
        diff_heatmaps_channels_image = make_grid(diff_heatmaps_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('04_diff_heatmaps_channels_image', diff_heatmaps_channels_image,
                                 global_step=self.global_steps)

        # show5.
        support_heatmaps_channels_tensor = torch.clamp_min(torch.cat(torch.split(support_heatmaps, 1, dim=1), dim=0), 0)
        support_heatmaps_channels_image = make_grid(support_heatmaps_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('05_support_heatmaps_channels_image', support_heatmaps_channels_image,
                                 global_step=self.global_steps)

        # show6.
        prf_ptm_combine_featuremaps_channels_tensor = torch.clamp_min(torch.cat(torch.split(prf_ptm_combine_featuremaps, 1, dim=1), dim=0),
                                                                      0)
        prf_ptm_combine_featuremaps_channels_image = make_grid(prf_ptm_combine_featuremaps_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('06_prf_ptm_combine_featuremaps_channels_image',
                                 prf_ptm_combine_featuremaps_channels_image, global_step=self.global_steps)

        # show7.
        warped_heatmaps_1_channels_tensor = torch.clamp_min(torch.cat(torch.split(warped_heatmaps_list[0], 1, dim=1), dim=0), 0)
        warped_heatmaps_1_channels_image = make_grid(warped_heatmaps_1_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('07_warped_heatmaps_1_channels_image', warped_heatmaps_1_channels_image,
                                 global_step=self.global_steps)
        warped_heatmaps_2_channels_tensor = torch.clamp_min(torch.cat(torch.split(warped_heatmaps_list[1], 1, dim=1), dim=0), 0)
        warped_heatmaps_2_channels_image = make_grid(warped_heatmaps_2_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('08_warped_heatmaps_2_channels_image', warped_heatmaps_2_channels_image,
                                 global_step=self.global_steps)
        warped_heatmaps_3_channels_tensor = torch.clamp_min(torch.cat(torch.split(warped_heatmaps_list[2], 1, dim=1), dim=0), 0)
        warped_heatmaps_3_channels_image = make_grid(warped_heatmaps_3_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('09_warped_heatmaps_3_channels_image', warped_heatmaps_3_channels_image,
                                 global_step=self.global_steps)
        warped_heatmaps_4_channels_tensor = torch.clamp_min(torch.cat(torch.split(warped_heatmaps_list[3], 1, dim=1), dim=0), 0)
        warped_heatmaps_4_channels_image = make_grid(warped_heatmaps_4_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('10_warped_heatmaps_4_channels_image', warped_heatmaps_4_channels_image,
                                 global_step=self.global_steps)
        warped_heatmaps_5_channels_tensor = torch.clamp_min(torch.cat(torch.split(warped_heatmaps_list[4], 1, dim=1), dim=0), 0)
        warped_heatmaps_5_channels_image = make_grid(warped_heatmaps_5_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('11_warped_heatmaps_5_channels_image', warped_heatmaps_5_channels_image,
                                 global_step=self.global_steps)

        # show8.
        output_heatmaps_channels_tensor = torch.clamp_min(torch.cat(torch.split(output_heatmaps, 1, dim=1), dim=0), 0)
        output_heatmaps_channels_image = make_grid(output_heatmaps_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('12_output_heatmaps_channels_image', output_heatmaps_channels_image,
                                 global_step=self.global_steps)

    def _after_val_iter(self, image, preds_joints, preds_confidence, vis_output_dir, center, scale):
        cfg = self.cfg
        # prepare data
        coords = np.concatenate([preds_joints, preds_confidence], axis=-1)
        bboxes = []
        for index in range(len(center)):
            xyxy_bbox = cs2box(center[index], scale[index], pattern="xyxy")
            bboxes.append(xyxy_bbox)

        if cfg.DEBUG.VIS_SKELETON or cfg.DEBUG.VIS_BBOX:
            from .vis_helper import draw_skeleton_in_origin_image
            draw_skeleton_in_origin_image(image, coords, bboxes, vis_output_dir, vis_skeleton=cfg.DEBUG.VIS_SKELETON,
                                          vis_bbox=cfg.DEBUG.VIS_BBOX)

    def vis_setup(self, logger, phase, epoch):
        if phase == TEST_PHASE:
            prefix_dir = "test"
        elif phase == TRAIN_PHASE:
            prefix_dir = "train"
        elif phase == VAL_PHASE:
            prefix_dir = "validate"
        else:
            prefix_dir = "inference"

        if isinstance(epoch, int):
            epoch = "model_{}".format(str(epoch))

        output_dir_base = osp.join(self.output_dir, epoch, prefix_dir, "use_gt_box" if self.cfg.VAL.USE_GT_BBOX else "use_precomputed_box")
        vis_output_dir = osp.join(output_dir_base, "vis")
        result_output_dir = osp.join(output_dir_base, "prediction_result")
        create_folder(vis_output_dir)
        create_folder(result_output_dir)
        logger.info("=> Vis Output Dir : {}".format(vis_output_dir))
        logger.info("=> Result Output Dir : {}".format(result_output_dir))

        if phase == VAL_PHASE:
            tensorboard_log_dir = osp.join(self.output_dir, epoch, prefix_dir, "tensorboard")
            self.tb_writer = SummaryWriter(log_dir=tensorboard_log_dir)

        if self.cfg.DEBUG.VIS_SKELETON:
            logger.info("=> VIS_SKELETON")
        if self.cfg.DEBUG.VIS_BBOX:
            logger.info("=> VIS_BBOX")
        return result_output_dir, vis_output_dir
