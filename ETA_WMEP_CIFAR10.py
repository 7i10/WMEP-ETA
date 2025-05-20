import argparse
import copy
import logging
import os
import time
from torchvision.utils import make_grid, save_image
import numpy as np
import torch
from torch.nn import functional as F
from CIFAR10_models import *
import random
from torch.autograd import Variable
from utils import *
import math

num_of_example = 50000
num_of_class = 10

# 日本時間での出力用
from datetime import datetime, timedelta, timezone
JST = timezone(timedelta(hours=9))

logger = logging.getLogger(__name__)

class JSTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, JST)
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.isoformat()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ResNet18', type=str, help='model name')
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='CIFAR10', type=str)
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--lr_schedule', default='multistep', choices=['cyclic', 'multistep'])
    parser.add_argument('--milestone1', default=100, type=int)
    parser.add_argument('--milestone2', default=105, type=int)
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    # ETA
    parser.add_argument('--inner-gamma', default=0.15, type=float, help='Label relaxation factor')
    parser.add_argument('--outer-gamma', default=0.15, type=float)
    parser.add_argument('--beta', default=0.6)
    parser.add_argument('--lamda', default=0.65, type=float, help='Penalize regularization term')
    parser.add_argument('--batch-m', default=0.75, type=float)
    parser.add_argument('--eta', default=0.75)
    # WMEP
    parser.add_argument('--epochs_reset', default=10, type=int)
    parser.add_argument('--momentum_decay', default=0.3, type=float, help='momentum_decay')
    parser.add_argument('--EMA_value', default=0.82, type=float)
    # FGSM attack
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=8, type=float, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous', 'normal'], help='Perturbation initialization method')
    # ouput
    parser.add_argument('--out-dir', default='CIFAR10', type=str, help='Output directory')
    parser.add_argument('--out-path', default='log', type=str, help='Output path')
    parser.add_argument('--log', default="test.log", type=str)
    return parser.parse_args()


def label_relaxation(label, factor, num_of_class):
    one_hot = np.eye(num_of_class)[label.cuda().data.cpu().numpy()] 
    result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(num_of_class - 1))
    return result

def atta_aug(input_tensor, rst):
        batch_size = input_tensor.shape[0]
        x = torch.zeros(batch_size)
        y = torch.zeros(batch_size)
        flip = [False] * batch_size

        for i in range(batch_size):
            flip_t = bool(random.getrandbits(1))
            x_t = random.randint(0, 8)
            y_t = random.randint(0, 8)

            rst[i, :, :, :] = input_tensor[i, :, x_t:x_t + 32, y_t:y_t + 32]
            if flip_t:
                rst[i] = torch.flip(rst[i], [2])
            flip[i] = flip_t
            x[i] = x_t
            y[i] = y_t

        return rst, {"crop": {'x': x, 'y': y}, "flipped": flip}

class EMA(object):
    def __init__(self, model, alpha=0.999, buffer_ema=True):
        self.step = 0
        self.model = copy.deepcopy(model)
        self.alpha = alpha
        self.buffer_ema = buffer_ema
        self.shadow = self.get_model_state()
        self.backup = {}
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]

    def update_params(self, model):
        decay = min(self.alpha, (self.step + 1) / (self.step + 10))
        state = model.state_dict()
        for name in self.param_keys:
            self.shadow[name].copy_(decay * self.shadow[name] + (1 - decay) * state[name])
        for name in self.buffer_keys:
            if self.buffer_ema:
                self.shadow[name].copy_(decay * self.shadow[name] + (1 - decay) * state[name])
            else:
                self.shadow[name].copy_(state[name])
        self.step += 1

    def apply_shadow(self):
        self.backup = self.get_model_state()
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self.backup)

    def get_model_state(self):
        return {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }

def main():
    args = get_args()

    output_path = os.path.join(args.out_dir, args.out_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logfile = os.path.join(output_path, args.log)
    if os.path.exists(logfile):
        os.remove(logfile)
    
    # ログ設定
    logger.setLevel(logging.INFO)

    # ファイル出力用のハンドラ
    file_handler = logging.FileHandler(logfile)
    file_handler.setFormatter(JSTFormatter(
        fmt='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S'
    ))
    logger.addHandler(file_handler)

    # コンソール出力用のハンドラ
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSTFormatter(
        fmt='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S'
    ))
    logger.addHandler(console_handler)

    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_all_loaders(args.data_dir, args.batch_size)
    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std

    if args.model == "VGG":
        model = VGG('VGG19')
    elif args.model == "ResNet18":
        model = ResNet18()
    elif args.model == "PreActResNet18":
        model = PreActResNet18()
    elif args.model == "WideResNet":
        model = WideResNet()
    elif args.model == "ResNet34":
        model = ResNet34()
    # model=torch.nn.DataParallel(model)
    model = model.cuda()
    
    model.train()
    teacher_model = EMA(model)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss()
    batch_size = args.batch_size

    iter_num = num_of_example // batch_size + (0 if num_of_example % batch_size == 0 else 1)

    lr_steps = args.epochs * iter_num
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                                                      step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps * args.milestone1/args.epochs, lr_steps * args.milestone2/args.epochs],
                                                         gamma=0.1)

    # Training
    best_result = 0
    total_time = 0
    epoch_clean_list = []
    epoch_pgd_list = []

    for i, (X, y) in enumerate(train_loader):
        cifar_x, cifar_y = X.cuda(), y.cuda()

    for epoch in range(args.epochs):

        start_epoch_time = time.time()
        cur_order = np.random.permutation(num_of_example)
        batch_size = args.batch_size
        batch_idx = -batch_size
        inner_loss = 0
        train_loss = 0
        train_acc = 0
        train_n = 0

        count = 0

        teacher_model.model.eval()

        # リセットエポックごとの摂動初期化
        if epoch %args.epochs_reset == 0:
            temp = torch.rand(50000,3,32,32)
            all_delta = torch.zeros_like(temp).cuda()
            all_momentum = torch.zeros_like(temp).cuda()
            for j in range(len(epsilon)):
                all_delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
            all_delta.data = clamp(alpha * torch.sign(all_delta), -epsilon, epsilon)
        
        idx = torch.randperm(cifar_x.shape[0])
        cifar_x = cifar_x[idx, :,:,:].view(cifar_x.size())
        cifar_y = cifar_y[idx].view(cifar_y.size())
        all_delta = all_delta[idx, :, :, :].view(all_delta.size())
        all_momentum = all_momentum[idx, :, :, :].view(all_delta.size())       

        
        ## DL: 動的ラベル緩和
        inner_gammas = math.tan(1 - (epoch/args.epochs)) * args.beta
        outer_gammas = math.tan(1 - (epoch/args.epochs)) * args.beta
        if inner_gammas < args.inner_gamma:
            inner_gammas = args.inner_gamma
            outer_gammas = args.outer_gamma

        for i in range(iter_num):
            
            batch_idx = (batch_idx + batch_size) if batch_idx + batch_size < num_of_example else 0
            X = cifar_x[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()
            y = cifar_y[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()
            delta = all_delta[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()
            next_delta = all_delta[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()

            momentum = all_momentum[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()
            X = X.cuda()
            y = y.cuda()
            batch_size = X.shape[0]
            rst = torch.zeros(batch_size, 3, 32, 32).cuda()
            X, transform_info = atta_aug(X, rst)
            
            relaxtion_label = torch.tensor(label_relaxation(y, inner_gammas, num_of_class)).cuda()
            delta.requires_grad = True
            ori_output = model(X + delta[:X.size(0)])
            clean_acc = (ori_output.max(1)[1] == y).sum().item()      
            
            ori_loss = nn.CrossEntropyLoss()(ori_output, relaxtion_label.float())  
            ori_loss.backward(retain_graph=True)
            x_grad = delta.grad.detach()

            adv_delta=delta.detach().clone()
            adv_delta.data = clamp(delta + alpha * torch.sign(x_grad), -epsilon, epsilon)
            adv_delta.data[:X.size(0)] = clamp(adv_delta[:X.size(0)], lower_limit - X, upper_limit - X)
            adv_delta = adv_delta.detach()
            
            output = model(X + adv_delta[:X.size(0)])
            adv_acc = (output.max(1)[1] == y).sum().item()
            grad_norm = torch.norm(x_grad, p=1)
            
            attack_value = 2 - (adv_acc / clean_acc)
            momentum = (x_grad / grad_norm) * attack_value + momentum * args.momentum_decay
            
            next_delta.data = clamp(delta + alpha * torch.sign(momentum), -epsilon, epsilon)
            next_delta.data[:X.size(0)] = clamp(next_delta[:X.size(0)], lower_limit - X, upper_limit - X)
            
            loss_adv = nn.CrossEntropyLoss(reduction='none', label_smoothing=(1.0-outer_gammas))(output, y)

            ## TD: 分類駆動型損失
            nat_probs = F.softmax(ori_output, dim=1)
            true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()
            loss_robust = torch.sum(torch.square(torch.sub(ori_output, output)),dim=1) * torch.tanh(1.0000001 - true_probs)

            loss = loss_adv + float(args.lamda) * loss_robust

            ## COLA: 壊滅的過学習対策    
            adjustment = torch.where(output.max(1)[1] == y, args.eta, 1.0)
            loss = torch.dot(loss, adjustment) / batch_size
            loss = loss.mean()
            
            opt.zero_grad()
            loss.backward()

            opt.step()
            inner_loss += ori_loss.item() * batch_size
            train_loss += loss.item() * batch_size
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += batch_size
            scheduler.step()

            if adv_acc / clean_acc < args.EMA_value:
                count += 1
                teacher_model.update_params(model)
                teacher_model.apply_shadow()
            
            all_momentum[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]] = momentum
            all_delta[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]]=next_delta
            

        epoch_time = time.time()
        total_time += epoch_time - start_epoch_time
        lr = scheduler.get_last_lr()[0]
        
        logger.info('Epoch \t Seconds \t LR \t \t Update \t Gamma \t \t Inner Loss \t Train Loss \t Train Acc')
        logger.info('%d \t %.1f \t \t %.4f \t %.4f  \t %.4f \t %.4f \t %.4f \t %.4f',
                    epoch, epoch_time - start_epoch_time, lr, count/iter_num, inner_gammas, inner_loss/train_n, train_loss/train_n, train_acc/train_n)
        
        if args.model == "VGG":
            model_test = VGG('VGG19').cuda()
        elif args.model == "ResNet18":
            model_test = ResNet18().cuda()
        elif args.model == "PreActResNet18":
            model_test = PreActResNet18().cuda()
        elif args.model == "WideResNet":
            model_test = WideResNet().cuda()
        elif args.model == "ResNet34":
            model_test = ResNet34().cuda()
        # model_test = torch.nn.DataParallel(model_test)
        model_test.load_state_dict(teacher_model.model.state_dict())
        model_test.float()
        model_test.eval()

        pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 10, 1)
        test_loss, test_acc = evaluate_standard(test_loader, model_test)
        epoch_clean_list.append(test_acc)
        epoch_pgd_list.append(pgd_acc)

        logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
        logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)            
        if best_result <= pgd_acc:
            best_result = pgd_acc
            torch.save(model_test.state_dict(), os.path.join(output_path, 'test_best_model.pth'))

    torch.save(model_test.state_dict(), os.path.join(output_path, 'test_last_model.pth'))
    logger.info(epoch_clean_list)
    logger.info(epoch_pgd_list)
    logger.info(total_time)

if __name__ == "__main__":
    main()
