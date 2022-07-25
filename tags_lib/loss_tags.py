# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import yaml
from scipy import ndimage
import itertools,operator

with open("./config/anet.yaml", 'r', encoding='utf-8') as f:
        tmp = f.read()
        config = yaml.load(tmp, Loader=yaml.FullLoader)


ce = nn.CrossEntropyLoss()
cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
lambda_1 = config['loss']['lambda_1']
lambda_2 = config['loss']['lambda_2']
lambda_3 = config['loss']['lambda_3']

def top_lr_loss(target,pred):

    gt_action = target
    pred_action = pred
    topratio = 0.6
    num_classes = 200
    alpha = 10

    pmask = (gt_action == 1).float()
    nmask = (gt_action == 0).float()
    nmask = nmask 

    num_positive = 10 + torch.sum(pmask) # in case of nan
    num_entries = 10 + num_positive + torch.sum(nmask)
    ratio = num_entries / num_positive
    coef_0 = 0.5 * ratio / (ratio - 1)
    coef_1 = 0.5 * ratio
    eps = 0.000001
    pred_p = torch.log(pred_action + eps)
    pred_n = torch.log(1.0 - pred_action + eps)


    topk = int(num_classes * topratio)
    # targets = targets.cuda()
    count_pos = num_positive
    hard_neg_loss = -1.0 * (1.0-gt_action) * pred_n
    topk_neg_loss = -1.0 * hard_neg_loss.topk(topk, dim=1)[0]#topk_neg_loss with shape batchsize*topk

    loss = (gt_action * pred_p).sum() / count_pos + alpha*(topk_neg_loss.cuda()).mean()

    return -1*loss


class BinaryDiceLoss(nn.Module):
   
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

dice = BinaryDiceLoss()



def top_ce_loss(gt_cls, pred_cls):

    ce_loss = F.cross_entropy(pred_cls,gt_cls)
    pt = torch.exp(-ce_loss)
    focal_loss = ((1 - pt) **2 * ce_loss).mean()
    loss = focal_loss 

    return loss

def get_mask_score(seq,tscale):
    thres = [0.45,0.6,0.8]
    seq = seq.detach().cpu().numpy()
    # print(seq)
    max_mask_score = 0
    score_list = []
    for j in thres:
        filtered_seq = seq > j
        # print(seq)
    
        integer_map = map(int,filtered_seq)
        filtered_seq_int = list(integer_map)
        filtered_seq_int2 = ndimage.binary_fill_holes(filtered_seq_int).astype(int).tolist()
        
        if 1 in filtered_seq_int:

            #### getting start and end point of mask from mask branch ####

            start_pt1 = filtered_seq_int2.index(1)
            end_pt1 = len(filtered_seq_int2) - 1 - filtered_seq_int2[::-1].index(1)  
            r = max((list(y) for (x,y) in itertools.groupby((enumerate(filtered_seq_int)),operator.itemgetter(1)) if x == 1), key=len)
            start_pt = r[0][0]
            end_pt = r[-1][0]
            if (end_pt - start_pt)/tscale > 0.02 : 
                actual_start = start_pt
                actual_end = end_pt
                seg_len = actual_end - actual_start + 1
                delta = 0.25
                win_off = int(delta*seg_len)
                action_score = np.mean(seq[actual_start:actual_end])
                # print("action",action_score)
                win_start = max(0,actual_start-win_off)
                win_end = min(tscale,actual_end+win_off)
                bkg_score = (1-np.mean(seq[win_start:actual_start+win_off]) + 1-np.mean(seq[actual_end-win_off:win_end])) / (win_off)
                # print(bkg_score)
                # print("win-start",win_start,actual_start)
                # print("win-end",win_end,actual_end)
                final_score = max(0,action_score - bkg_score)
                # if final_score < 0:

                # print(final_score)
                score_list.append(final_score)
    max_mask_score = max(score_list)
    # print(max_mask_score)
    return max_mask_score


# General util function to get the boundary of a binary mask.
def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    # h, w = mask.shape
    # print("mask",mask.size())
    b,h,w = mask.size()
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1

    # Pad image so mask truncated by the image border is also considered as boundary.
    # new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    new_mask = F.pad(mask,(1,1,1,1), 'constant', 0)
    kernel = torch.ones((3,3), dtype=torch.float64)
    new_mask_erode = kornia.morphology.erosion(new_mask.unsqueeze(1),kernel.type(torch.cuda.DoubleTensor)).squeeze(1)
    mask_erode = new_mask_erode[:,1 : h + 1, 1 : w + 1]
    return mask - mask_erode


def boundary_iou(gt, dt, dilation_ratio=0.02):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    boundary_iou = intersection / union
    return boundary_iou


def bottom_branch_loss(gt_action, pred_action):

    pmask = (gt_action == 1).float()
    nmask = (gt_action == 0).float()
    nmask = nmask 
    num_positive = 10 + torch.sum(pmask) # in case of nan
    num_entries = 10 + num_positive + torch.sum(nmask)
    ratio = num_entries / num_positive
    coef_0 = 0.5 * ratio / (ratio - 1)
    coef_1 = 0.5 * ratio
    epsilon = 0.000001
    loss_pos = coef_1 * torch.log(pred_action + epsilon) * pmask
    loss_neg = coef_0 * torch.log(1.0 - pred_action + epsilon) * nmask
    w_bce_loss = -1 * torch.sum(loss_pos + loss_neg) / num_entries
    BCE_loss = F.binary_cross_entropy(pred_action,gt_action,reduce=False)
    pt = torch.exp(-BCE_loss)
    # F_loss = 0.4*loss2 + 0.6*dice(pred_action,gt_action)
    F_loss = lambda_2*w_bce_loss + (1 - lambda_2)*dice(pred_action,gt_action)
    
    return F_loss

def redundancy_loss(gt_action , pred_action, gt_cls, pred_cls, features):
    ### inter-branch consistency loss ## 
    mask_fg = torch.ones_like(gt_cls).cuda()
    mask_bg = torch.zeros_like(gt_cls).cuda()
    sim_loss = 0
    B,K,T = pred_cls.size()
    if T == 100 :
        for i in range(B):
            val_top,_ = torch.max(torch.softmax(pred_cls[i,:200,:],dim=0),dim=0)
            val_bot ,_ = torch.max(pred_action[i,:,:], dim=1)
            cls_thres = float(torch.mean(val_top,dim=0).detach().cpu().numpy())
            # cls_thres = 0.0
            # print(cls_thres)
            mask_thres = float(torch.mean(val_bot,dim=0).detach().cpu().numpy())
            top_mask = torch.where(val_top >= cls_thres, mask_fg[i,:], mask_bg[i,:]).cuda(0)
            # print(top_mask)
            bot_mask = torch.where(val_bot >= mask_thres, mask_fg[i,:], mask_bg[i,:]).cuda(0)
            top_loc = (top_mask==1).nonzero().squeeze().cuda(0)
            bot_loc = (bot_mask==1).nonzero().squeeze().cuda(0)
            top_feat = torch.mean(features[i,:,top_loc],dim=1).cuda(0)
            bot_feat = torch.mean(features[i,:,bot_loc],dim=1).cuda(0)

            sim_loss += (1-cos_sim(top_feat,bot_feat))
        # torch.cuda.empty_cache()
    const_loss = sim_loss / B

    ##### mask redundancy loss ######
    fin_mask = torch.where(gt_cls == 200, mask_bg, mask_fg)
    # loss = 0 
    loss = 0
    B,_ = fin_mask.size()
    _,_,tscale = gt_action.size()
    for i in range(B):
        gt_mask = fin_mask[i,:]
        gt_loc = (gt_mask==1).nonzero().squeeze()
        # loss = 0
        loss += F.mse_loss(pred_action[i,:,gt_loc],gt_action[i,:,gt_loc])
        # if tscale == 100:
        #     loc_list = gt_loc.tolist()
        #     for j in loc_list:
        #         max_score = get_mask_score(pred_action[i,:,j],tscale)
        #         red_term = (1 - max_score)**2
        #         loss += (red_term*F.mse_loss(pred_action[i,:,j],gt_action[i,:,j]))
        #     red_loss = loss / (len(loc_list))
        # else:
        #     red_loss += F.mse_loss(pred_action[i,:,gt_loc],gt_action[i,:,gt_loc])

    mask_red_loss = loss / B

    fin_loss =  mask_red_loss + lambda_3*const_loss
   
    return fin_loss


def top_branch_loss(gt_cls, pred_cls, mask_gt):

    loss = lambda_1*top_ce_loss(gt_cls.cuda(), pred_cls) + (1 - lambda_1)*top_lr_loss(mask_gt.cuda(), torch.sigmoid(pred_cls))

    return loss

def tags_loss(gt_cls, pred_cls ,gt_action , pred_action, mask_gt , label_gt, features):

    top_loss = top_branch_loss(gt_cls, pred_cls, mask_gt)
    bottom_loss = bottom_branch_loss(gt_action.cuda(), pred_action) 
    redun_loss = redundancy_loss(gt_action.cuda() , pred_action.cuda(), gt_cls.cuda(), pred_cls.cuda(), features.cuda())

    tot_loss = top_loss + bottom_loss + redun_loss

    return tot_loss, top_loss, bottom_loss, redun_loss
