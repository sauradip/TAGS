# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import yaml

with open("./config/anet.yaml", 'r', encoding='utf-8') as f:
        tmp = f.read()
        config = yaml.load(tmp, Loader=yaml.FullLoader)


ce = nn.CrossEntropyLoss()

lambda_1 = config['loss']['lambda_1']
lambda_2 = config['loss']['lambda_2']

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


def to_one_hot(tensor,nClasses):
    
    n,h,w = tensor.size()
    one_hot = torch.zeros(n,nClasses,h,w).scatter_(1,tensor.view(n,1,h,w),1)
    return one_hot

class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=2):
        super(mIoULoss, self).__init__()
        self.classes = n_classes

    def forward(self, inputs, target_oneHot):
    	# inputs => N x Classes x H x W
    	# target_oneHot => N x Classes x H x W

    	N = inputs.size()[0]

    	# predicted probabilities for each pixel along channel
    	# inputs = F.softmax(inputs,dim=1)
    	
    	# Numerator Product
    	inter = inputs * target_oneHot
    	## Sum over all pixels N x C x H x W => N x C
    	inter = inter.view(N,self.classes,-1).sum(2)

    	#Denominator 
    	union= inputs + target_oneHot - (inputs*target_oneHot)
    	## Sum over all pixels N x C x H x W => N x C
    	union = union.view(N,self.classes,-1).sum(2)

    	loss = inter/union

    	## Return average loss over classes and batch
    	return (1-loss.mean())

# class boundaryIoULoss(nn.Module):
#     def __init__(self, weight=None, size_average=True, n_classes=2):
#         super(mIoULoss, self).__init__()
#         self.classes = n_classes

#     def forward(self, inputs, target_oneHot):
#     	# inputs => N x Classes x H x W
#     	# target_oneHot => N x Classes x H x W

#     	N = inputs.size()[0]

#     	# predicted probabilities for each pixel along channel
#     	# inputs = F.softmax(inputs,dim=1)
    	
#     	# Numerator Product
#     	inter = inputs * target_oneHot
#     	## Sum over all pixels N x C x H x W => N x C
#     	inter = inter.view(N,self.classes,-1).sum(2)

#     	#Denominator 
#     	union= inputs + target_oneHot - (inputs*target_oneHot)
#     	## Sum over all pixels N x C x H x W => N x C
#     	union = union.view(N,self.classes,-1).sum(2)

#     	loss = inter/union

#     	## Return average loss over classes and batch
#     	return (1-loss.mean())

mIOU_loss = mIoULoss()

import cv2
# import numpy as np
import kornia


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
    # img_diag = torch.sqrt(h ** 2 + w ** 2)
    # dilation = int(round(dilation_ratio * img_diag))
    dilation = int(round(dilation_ratio * img_diag))

    if dilation < 1:
        dilation = 1

    # Pad image so mask truncated by the image border is also considered as boundary.
    # new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    new_mask = F.pad(mask,(1,1,1,1), 'constant', 0)

    # kernel = np.ones((3, 3), dtype=np.uint8)
    kernel = torch.ones((3,3), dtype=torch.float64)
    # new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    new_mask_erode = kornia.morphology.erosion(new_mask.unsqueeze(1),kernel.type(torch.cuda.DoubleTensor)).squeeze(1)
    
    mask_erode = new_mask_erode[:,1 : h + 1, 1 : w + 1]
    # print("erosion",mask_erode.size())
    # G_d intersects G in the paper.
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
    # F_loss = lambda_2*w_bce_loss + (1 - lambda_2)*dice(pred_action,gt_action)
    pred_bg = 1.0 - pred_action
    gt_bg = 1.0 - gt_action
    gt_iou = torch.cat((gt_action,gt_bg),1)
    pred_iou = torch.cat((pred_action,pred_bg),1)
    # F_loss = lambda_2*w_bce_loss + (1 - lambda_2)*mIOU_loss(pred_iou,gt_iou)
    # F_loss = lambda_2*dice(pred_action,gt_action) + (1 - lambda_2)*mIOU_loss(pred_iou,gt_iou)
    # F_loss = lambda_2*w_bce_loss + (1 - lambda_2)*dice(pred_action,gt_action)
    F_loss = boundary_iou(gt_action,pred_action)
    # F_loss = iou_pytorch(pred_action,gt_action)
    return F_loss

def top_branch_loss(gt_cls, pred_cls, mask_gt):

    # loss = lambda_1*top_ce_loss(gt_cls.cuda(), pred_cls) + (1 - lambda_1)*top_lr_loss(mask_gt.cuda(), torch.sigmoid(pred_cls))
    loss = lambda_1*top_ce_loss(gt_cls.cuda(), pred_cls) + (1 - lambda_1)*boundary_iou(mask_gt.cuda(), torch.sigmoid(pred_cls))

    return loss

def gsm_loss(gt_cls, pred_cls ,gt_action , pred_action, mask_gt , label_gt):

    top_loss = top_branch_loss(gt_cls, pred_cls, mask_gt)
    bottom_loss = bottom_branch_loss(gt_action.cuda(), pred_action) 

    tot_loss = top_loss + bottom_loss 

    return tot_loss, top_loss, bottom_loss



