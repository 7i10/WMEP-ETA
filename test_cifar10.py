from autoattack import AutoAttack
from CIFAR10_models import *
from utils import *
import argparse
import sys
import os
import logging

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

def setup_logger(logfile):
    """モデルごとに個別のロガーを設定"""
    logger = logging.getLogger(logfile)
    logger.setLevel(logging.INFO)
    
    # ファイルハンドラ
    file_handler = logging.FileHandler(logfile)
    file_handler.setFormatter(JSTFormatter(fmt='[%(asctime)s] - %(message)s', datefmt='%Y/%m/%d %H:%M:%S'))
    logger.addHandler(file_handler)
    
    # コンソールハンドラ
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSTFormatter(fmt='[%(asctime)s] - %(message)s', datefmt='%Y/%m/%d %H:%M:%S'))
    logger.addHandler(console_handler)
    return logger

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='CIFAR10')
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--epsilon', type=int, default=8)
    parser.add_argument('--best_model_path', type=str, default='CIFAR10_test_models/_best_model.pth')
    parser.add_argument('--last_model_path', type=str, default='CIFAR10_test_models/_last_model.pth')
    parser.add_argument('--model', default='ResNet18', type=str, help='model name')
    parser.add_argument('--n_ex', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--out_dir', type=str, default='CIFAR10_test_results')
    parser.add_argument('--out_best_file', type=str, default='_best_log.txt')
    parser.add_argument('--out_last_file', type=str, default='_last_log.txt')


    arguments = parser.parse_args()
    return arguments

# 引数取得
args = get_args()

# ログファイルのパス設定
best_model_logfile = os.path.join(args.out_dir, args.out_best_file)
last_model_logfile = os.path.join(args.out_dir, args.out_last_file)
if os.path.exists(best_model_logfile):
    os.remove(best_model_logfile)
if os.path.exists(last_model_logfile):
    os.remove(last_model_logfile)

# 各ロガーを作成
best_model_logger = setup_logger(best_model_logfile)
last_model_logger = setup_logger(last_model_logfile)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# モデルの初期化関数
def load_model(model_name, model_path, logger):
    if model_name == "ResNet18":
        model = ResNet18()
    elif model_name == "VGG":
        model = VGG('VGG19')
    elif model_name == "PreActResNet18":
        model = PreActResNet18()
    elif model_name == "WideResNet":
        model = WideResNet()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    ###
    # model=torch.nn.DataParallel(model)
    model = model.to(device)
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint)
    # num_params = sum(p.numel() for p in model.parameters())
    # print(f"モデルの総パラメータ数: {num_params}")
    model.eval()
    logger.info(f"Loaded model from {model_path}")
    return model

# モデルをロード
best_model = load_model(args.model, args.best_model_path, best_model_logger)
last_model = load_model(args.model, args.last_model_path, last_model_logger)

# データローダーを準備
train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

def evaluate_and_log(model, test_loader, epsilon, logger, n_ex, batch_size, norm):
    epsilon=float(epsilon)/255.
    logger.info(f"Starting evaluation with epsilon={epsilon:.6f}")

    # FGSM, PGD, CW, Standard の評価
    AT_fgsm_loss, AT_fgsm_acc = evaluate_fgsm(test_loader, model, 1)
    AT_pgd_loss_10, AT_pgd_acc_10 = evaluate_pgd(test_loader, model, 10, 1, epsilon / std)
    AT_pgd_loss_20, AT_pgd_acc_20 = evaluate_pgd(test_loader, model, 20, 1, epsilon / std)
    AT_pgd_loss_50, AT_pgd_acc_50 = evaluate_pgd(test_loader, model, 50, 1, epsilon / std)
    AT_CW_loss_20, AT_pgd_cw_acc_20 = evaluate_pgd_cw(test_loader, model, 20, 1)
    AT_models_test_loss, AT_models_test_acc = evaluate_standard(test_loader, model)

    # 結果をログに記録
    logger.info(f"AT_models_test_acc: {AT_models_test_acc:.4f}")
    logger.info(f"AT_fgsm_acc: {AT_fgsm_acc:.4f}")
    logger.info(f"AT_pgd_acc_10: {AT_pgd_acc_10:.4f}")
    logger.info(f"AT_pgd_acc_20: {AT_pgd_acc_20:.4f}")
    logger.info(f"AT_pgd_acc_50: {AT_pgd_acc_50:.4f}")
    logger.info(f"AT_pgd_cw_acc_20: {AT_pgd_cw_acc_20:.4f}")

    # AutoAttack の評価
    adversary = AutoAttack(model, norm=norm, eps=epsilon, version='standard', log_path=logger.handlers[0].baseFilename)

    # テストデータを準備
    x_test = torch.cat([x for x, y in test_loader], 0)
    y_test = torch.cat([y for x, y in test_loader], 0)

    # AutoAttack を実行
    adv_complete = adversary.run_standard_evaluation(x_test[:n_ex], y_test[:n_ex], bs=batch_size)

# 各モデルの評価
evaluate_and_log(
    model=best_model,
    test_loader=test_loader,
    epsilon=args.epsilon,
    logger=best_model_logger,
    n_ex=args.n_ex,
    batch_size=args.batch_size,
    norm=args.norm
)

evaluate_and_log(
    model=last_model,
    test_loader=test_loader,
    epsilon=args.epsilon,
    logger=last_model_logger,
    n_ex=args.n_ex,
    batch_size=args.batch_size,
    norm=args.norm
)