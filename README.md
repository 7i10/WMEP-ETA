# WMEP-ETA: Fast Adversarial Training with Efficient Techniques

This repository provides the full code, models, and logs used in our graduation research on adversarial robustness, especially focusing on the WMEP-ETA method for CIFAR-10/CIFAR-100.

---

## 🧠 Overview

- 🎯 Goal: Improve adversarial robustness
- 🧪 Dataset: CIFAR-10, CIFAR-100
- ⚔️ Attacks: FGSM, PGD, AutoAttack, CW
- 🛡 Methods: WMEP-ETA
- 📊 Evaluation: Clean accuracy, robust accuracy under multiple attacks

---

## 🗂 Repository Structure

WMEP-ETA/
├── CIFAR10/ # CIFAR-10における実験結果
│ ├── log/ # 実験ログ（.logファイル）
│ ├── test_model/ # テスト用学習済みモデル（チェックポイント）
│ └── test_result/ # 評価結果（CSVやグラフなど）
│
├── CIFAR100/ # CIFAR-100における実験結果（CIFAR10と同構成）
│ ├── log/
│ ├── test_model/
│ └── test_result/
│
├── CIFAR10_models/ # CIFAR-10用モデル定義（例：EfficientNet, ResNetなど）
│ ├── efficientnet.py
│ ├── resnet.py
│ └── ...
│
├── CIFAR100_models/ # CIFAR-100用モデル定義（CIFAR10と同様）
│ ├── efficientnet.py
│ ├── resnet.py
│ └── ...
│
├── autoattack/ # AutoAttack評価用のスクリプト群
│ ├── autoattack.py
│ ├── autopgd_pt.py
│ ├── fab_pt.py
│ ├── square.py
│ └── ...
│
├── ETA_WMEP_CIFAR10.py # CIFAR-10用のWMEP+ETA学習スクリプト
├── ETA_WMEP_CIFAR100.py # CIFAR-100用のWMEP+ETA学習スクリプト
│
├── test_cifar10.py # CIFAR-10の評価実行スクリプト
├── test_cifar100.py # CIFAR-100の評価実行スクリプト
│
└── README.md # このファイル


WMEP-ETA/
├── CIFAR10/                     # CIFAR-10における実験結果
│   ├── log/                     # 実験ログ（.logファイル）
│   ├── test_model/              # テスト用学習済みモデル（チェックポイント）
│   └── test_result/             # 評価結果（CSVやグラフなど）
│
├── CIFAR100/                    # CIFAR-100における実験結果（CIFAR10と同構成）
│   ├── log/
│   ├── test_model/
│   └── test_result/
│
├── CIFAR10_models/              # CIFAR-10用モデル定義（例：EfficientNet, ResNetなど）
│   ├── efficientnet.py
│   ├── resnet.py
│   └── ...
│
├── CIFAR100_models/             # CIFAR-100用モデル定義（CIFAR10と同様）
│   ├── efficientnet.py
│   ├── resnet.py
│   └── ...
│
├── autoattack/                  # AutoAttack評価用のスクリプト群
│   ├── autoattack.py
│   ├── autopgd_pt.py
│   ├── fab_pt.py
│   ├── square.py
│   └── ...
│
├── ETA_WMEP_CIFAR10.py          # CIFAR-10用のWMEP+ETA学習スクリプト
├── ETA_WMEP_CIFAR100.py         # CIFAR-100用のWMEP+ETA学習スクリプト
│
├── test_cifar10.py              # CIFAR-10の評価実行スクリプト
├── test_cifar100.py             # CIFAR-100の評価実行スクリプト
│
└── README.md                    # このファイル

---

## 🚀 How to Reproduce

---

## 📈 Sample Results

---

## 📦 Pretrained Models
CIFAR10/test_model/
CIFAR100/test_model/

---

## 📃 License
MIT License

---

## ✉️ Contact
