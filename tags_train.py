import os
import torch
import torch.nn.parallel
import torch.optim as optim
from torch import autograd
import numpy as np
# from gsm_lib import opts
from tags_model import TAGS
import yaml
import tags_lib.tags_dataloader as tags_dataset
from tags_lib.loss_tags import tags_loss
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()




with open("./config/anet.yaml", 'r', encoding='utf-8') as f:
        tmp = f.read()
        config = yaml.load(tmp, Loader=yaml.FullLoader)

output_path=config['dataset']['training']['output_path']
num_gpu = config['training']['num_gpu']
batch_size = config['training']['batch_size']
learning_rate = config['training']['learning_rate']
decay = config['training']['weight_decay']
epoch = config['training']['max_epoch']
num_batch = config['training']['batch_size']
step_train = config['training']['step']
gamma_train = config['training']['gamma']
fix_seed = config['training']['random_seed']


################## fix everything ##################
import random
seed = fix_seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#######################################################


def get_mem_usage():
    GB = 1024.0 ** 3
    output = ["device_%d = %.03fGB" % (device, torch.cuda.max_memory_allocated(torch.device('cuda:%d' % device)) / GB) for device in range(num_gpu)]
    return ' '.join(output)[:-1]


# training
def train(data_loader, model, optimizer, epoch,scheduler):
    model.train()
    for n_iter, (input_data, top_br_gt, bottom_br_gt, action_gt, label_gt) in enumerate(data_loader):
        # forward pass
        top_br_pred, bottom_br_pred, features = model(input_data.cuda())
        loss = tags_loss(top_br_gt,top_br_pred,bottom_br_gt,bottom_br_pred, action_gt,label_gt, features)
        # update step
        optimizer.zero_grad()
        loss[0].backward()
        writer.add_scalar("Total_Loss-train", loss[0], epoch)
        writer.add_scalar("Top_Branch_Loss-train", loss[1], epoch)
        writer.add_scalar("Bottom_Branch_Loss-train", loss[2], epoch)
        optimizer.step()
    print("[Epoch {0:03d}] Total-Loss {1:.2f} = T-Loss {2:.2f} + B-Loss {3:.2f} + R-Loss {4:.2f}  (train)".format(
    epoch, loss[0],loss[1],loss[2], loss[3]))

# validation
def test(data_loader, model, epoch, best_loss):
    model.eval()
    with torch.no_grad():
      for n_iter, (input_data, top_br_gt, bottom_br_gt, action_gt, label_gt) in enumerate(data_loader):

        # forward pass
        top_br_pred, bottom_br_pred, features = model(input_data.cuda())
        loss = tags_loss(top_br_gt,top_br_pred,bottom_br_gt,bottom_br_pred, action_gt,label_gt, features)
        writer.add_scalar("Total_Loss-validation", loss[0], epoch)
    print("[Epoch {0:03d}] Total-Loss {1:.2f} = T-Loss {2:.2f} + B-Loss {3:.2f} + R-Loss {4:.2f}  (val)".format(
    epoch, loss[0],loss[1],loss[2],loss[3]))

    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict()}
    torch.save(state, output_path + "/TAGS_checkpoint.pth.tar")
    if loss[0] < best_loss:
        best_loss = loss[0]
        torch.save(state, output_path + "/TAGS_best.pth.tar")

    return best_loss

if __name__ == '__main__':

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    model = TAGS()

    # print(model)
    model = torch.nn.DataParallel(model, device_ids=list(range(num_gpu))).cuda()

    for param in model.parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nTotal Number of Learnable Paramters (in M) : ",total_params/1000000)
    print('No of Gpus using to Train :  {} '.format(num_gpu))
    print(" Saving all Checkpoints in path : "+ output_path )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=decay)
    train_loader = torch.utils.data.DataLoader(tags_dataset.TAGSDataset(subset="train"),
                                               batch_size=num_batch, shuffle=True,
                                               num_workers=8, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(tags_dataset.TAGSDataset(subset="validation"),
                                              batch_size=num_batch, shuffle=False,
                                              num_workers=8, pin_memory=True)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_train, gamma=gamma_train)
    best_loss = 1e10

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for epoch in range(epoch):
      with autograd.detect_anomaly():
        train(train_loader, model, optimizer, epoch,scheduler)
        best_loss = test(test_loader, model, epoch, best_loss)
        scheduler.step()
    writer.flush()
    end.record()
    torch.cuda.synchronize()

    print("Total Time taken for Running "+str(epoch)+" epoch is :"+ str(start.elapsed_time(end)/1000) + " secs")  # milliseconds



