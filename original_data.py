import argparse
import os
import pickle
import random
from collections import OrderedDict
from random import sample

import cv2
from PIL import Image

try:
    import matplotlib

    matplotlib.use("Agg")  # "Fail to allocate bitmap" 対策
finally:
    import matplotlib.pyplot as plt

import datetime

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import mahalanobis
from skimage import morphology
from skimage.segmentation import mark_boundaries
from sklearn.covariance import LedoitWolf
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, wide_resnet50_2
from tqdm import tqdm  # 進捗状況や処理状況をプログレスバー（ステータスバー）として表示

import datasets.mvtec as mvtec

THRESHOLD_IMAGE = 0.4  # 画像を異常と判断する閾値(0～1の間で設定、この値より大きいと異常)
THRESHOLD_PIXCEL = 0.4  # 画素を異常と判断する閾値(0～1の間で設定、この値より大きいと異常)

# デバイスの設定 device setup
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# パスの設定
def parse_args():
    parser = argparse.ArgumentParser("PaDiM")
    parser.add_argument("--data_path", type=str, default="C:/Users/Tokuda/Python/PaDiM/original_data")
    parser.add_argument("--save_path", type=str, default="C:/Users/Tokuda/Python/PaDiM/original_result")
    parser.add_argument("--arch", type=str, choices=["resnet18", "wide_resnet50_2"], default="wide_resnet50_2")
    return parser.parse_args()


def train():
    """step1 学習"""

    args = parse_args()

    # モデル読み込み load model
    if args.arch == "resnet18":
        # pretrained:学習済みパラメータ progress:転移学習
        model = resnet18(pretrained=True, progress=True)
        t_d = 448
        d = 100
    elif args.arch == "wide_resnet50_2":
        # pretrained:学習済みパラメータ progress:転移学習
        model = wide_resnet50_2(pretrained=True, progress=True)
        t_d = 1792
        d = 550
    model.to(device)
    model.eval()

    # シードの固定(やっておかないと結果がおかしくなる)
    random.seed(1024)
    torch.manual_seed(1024)
    if use_cuda:
        torch.cuda.manual_seed_all(1024)

    idx = torch.tensor(sample(range(0, t_d), d))

    # モデルの中間出力を設定 set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)

    os.makedirs(os.path.join(args.save_path, "temp_%s" % args.arch), exist_ok=True)

    train_dataset = mvtec.MVTecDataset(args.data_path, class_name="original_class", is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)

    train_outputs = OrderedDict([("layer1", []), ("layer2", []), ("layer3", [])])

    # 学習セットの特徴を抽出 extract train set features
    train_feature_filepath = os.path.join(args.save_path, "temp_%s" % args.arch, "train_%s.pkl" % "original_class")
    if not os.path.exists(train_feature_filepath):
        for (x, _, _) in tqdm(train_dataloader, "| feature extraction | train | %s |" % "original_class"):
            # モデル予測 model prediction
            with torch.no_grad():
                _ = model(x.to(device))
            # 中間層出力を取得 get intermediate layer outputs
            for k, v in zip(train_outputs.keys(), outputs):
                train_outputs[k].append(v.cpu().detach())
            # フック出力を初期化 initialize hook outputs
            outputs = []
        for k, v in train_outputs.items():
            train_outputs[k] = torch.cat(v, 0)

        # 埋め込み結合 Embedding concat
        embedding_vectors = train_outputs["layer1"]
        for layer_name in ["layer2", "layer3"]:
            embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])

        # ランダムにDディメンションを選択　randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
        # 多変量ガウス分布を計算　calculate multivariate Gaussian distribution
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W)
        mean = torch.mean(embedding_vectors, dim=0).numpy()
        cov = torch.zeros(C, C, H * W).numpy()
        I = np.identity(C)
        for i in range(H * W):
            # cov[:, :, i] = LedoitWolf().fit(embedding_vectors[:, :, i].numpy()).covariance_
            cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
        # 学んだ分布を保存 save learned distribution  # 学習結果は各画素ごとの平均と共分散行列
        train_outputs = [mean, cov]
        with open(train_feature_filepath, "wb") as f:
            pickle.dump(train_outputs, f)

    print("================train conmpleted================")


def test():
    """テスト関数"""
    args = parse_args()

    # モデル読み込み load model
    if args.arch == "resnet18":
        # pretrained:学習済みパラメータ progress:転移学習
        model = resnet18(pretrained=True, progress=True)
        t_d = 448
        d = 100
    elif args.arch == "wide_resnet50_2":
        # pretrained:学習済みパラメータ progress:転移学習
        model = wide_resnet50_2(pretrained=True, progress=True)
        t_d = 1792
        d = 550
    model.to(device)
    model.eval()
    # シードの固定(やっておかないと結果がおかしくなる)
    random.seed(1024)
    torch.manual_seed(1024)
    if use_cuda:
        torch.cuda.manual_seed_all(1024)

    idx = torch.tensor(sample(range(0, t_d), d))

    # モデルの中間出力を設定 set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    total_roc_auc = []
    total_pixel_roc_auc = []

    test_dataset = mvtec.MVTecDataset(args.data_path, class_name="original_class", is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)

    test_outputs = OrderedDict([("layer1", []), ("layer2", []), ("layer3", [])])

    # 学習セットで学んだ特徴の分布を読み込み
    train_feature_filepath = os.path.join(args.save_path, "temp_%s" % args.arch, "train_%s.pkl" % "original_class")
    if os.path.exists(train_feature_filepath):
        print("load train set feature from: %s" % train_feature_filepath)
        with open(train_feature_filepath, "rb") as f:
            train_outputs = pickle.load(f)

    gt_list = []
    gt_mask_list = []
    test_imgs = []

    # テストセットの特徴を抽出 extract test set features
    for (x, y, mask) in tqdm(test_dataloader, "| feature extraction | test | %s |" % "original_class"):
        test_imgs.extend(x.cpu().detach().numpy())
        gt_list.extend(y.cpu().detach().numpy())
        gt_mask_list.extend(mask.cpu().detach().numpy())  # tensor to numpy
        # モデル予測 model prediction
        with torch.no_grad():
            _ = model(x.to(device))
        # 中間層出力を取得  get intermediate layer outputs
        for k, v in zip(test_outputs.keys(), outputs):
            test_outputs[k].append(v.cpu().detach())
        # フック出力を初期化  initialize hook outputs
        outputs = []
    for k, v in test_outputs.items():
        test_outputs[k] = torch.cat(v, 0)

    # 埋め込み結合 Embedding concat
    embedding_vectors = test_outputs["layer1"]
    for layer_name in ["layer2", "layer3"]:
        embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])

    # ランダムにDディメンションを選択　randomly select d dimension
    embedding_vectors = torch.index_select(embedding_vectors, 1, idx)

    # 距離行列を計算 calculate distance matrix
    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
    dist_list = []
    for i in range(H * W):
        mean = train_outputs[0][:, i]
        conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
        dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
        dist_list.append(dist)

    dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

    # アップサンプル upsample
    dist_list = torch.tensor(dist_list)
    score_map = (
        F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode="bilinear", align_corners=False).squeeze().numpy()
    )

    # スコアマップにGaussianスムージングを適用 apply gaussian smoothing on the score map
    for i in range(score_map.shape[0]):
        score_map[i] = gaussian_filter(score_map[i], sigma=4)

    # 正規化 Normalization
    max_score = score_map.max()
    min_score = score_map.min()
    scores = (score_map - min_score) / (max_score - min_score)

    # 画像レベルのROC AUCスコアを計算 calculate image-level ROC AUC score
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    gt_list = np.asarray(gt_list)
    # print("gt_list:", gt_list) # ラベル
    # print("img_scores:", img_scores) # 予測値
    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    total_roc_auc.append(img_roc_auc)
    print("image ROCAUC: %.3f" % (img_roc_auc))
    fig_img_rocauc.plot(fpr, tpr, label="%s img_ROCAUC: %.3f" % ("original_class", img_roc_auc))

    # 最適なしきい値を取得 get optimal threshold
    gt_mask = np.asarray(gt_mask_list)
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]
    # print("threshold:", threshold) #0.4前後

    # ピクセルごとのレベルのRocaucを計算 calculate per-pixel level ROCAUC
    fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
    total_pixel_roc_auc.append(per_pixel_rocauc)
    print("pixel ROCAUC: %.3f" % (per_pixel_rocauc))

    # 画像保存(5枚セット)
    fig_pixel_rocauc.plot(fpr, tpr, label="%s ROCAUC: %.3f" % ("original_class", per_pixel_rocauc))
    save_dir = args.save_path + "/" + f"pictures_{args.arch}"
    os.makedirs(save_dir, exist_ok=True)
    plot_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, "original_class")

    # 全部終わったら Average image ROCAUC Average pixel ROCAUC を保存
    fig_img_rocauc.title.set_text("Average image ROCAUC: %.3f" % np.mean(total_roc_auc))
    fig_img_rocauc.legend(loc="lower right")

    fig_pixel_rocauc.title.set_text("Average pixel ROCAUC: %.3f" % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(args.save_path, "roc_curve.png"), dpi=100)

    print("================test conmpleted================")


def capture():
    """テスト関数"""
    args = parse_args()

    # モデル読み込み load model
    if args.arch == "resnet18":
        # pretrained:学習済みパラメータ progress:転移学習
        model = resnet18(pretrained=True, progress=True)
        t_d = 448
        d = 100
    elif args.arch == "wide_resnet50_2":
        # pretrained:学習済みパラメータ progress:転移学習
        model = wide_resnet50_2(pretrained=True, progress=True)
        t_d = 1792
        d = 550
    model.to(device)
    model.eval()
    # シードの固定(やっておかないと結果がおかしくなる)
    random.seed(1024)
    torch.manual_seed(1024)
    if use_cuda:
        torch.cuda.manual_seed_all(1024)

    idx = torch.tensor(sample(range(0, t_d), d))

    # モデルの中間出力を設定 set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)

    # 学習セットで学んだ特徴の分布を読み込み
    train_feature_filepath = os.path.join(args.save_path, "temp_%s" % args.arch, "train_%s.pkl" % "original_class")
    if os.path.exists(train_feature_filepath):
        print("load train set feature from: %s" % train_feature_filepath)
        with open(train_feature_filepath, "rb") as f:
            train_outputs = pickle.load(f)

    # ImageNetに合う形に正規化
    transform = transforms.Compose(
        [
            transforms.Resize(256),  # (256, 256) で切り抜く。
            transforms.CenterCrop(224),  # 画像の中心に合わせて、(224, 224) で切り抜く
            transforms.ToTensor(),  # テンソルにする。
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 標準化
        ]
    )

    # VideoCapture オブジェクトを取得
    capture = cv2.VideoCapture(0)

    test_outputs = OrderedDict([("layer1", []), ("layer2", []), ("layer3", [])])

    while capture.isOpened():
        ret, frame = capture.read()
        # resize the window
        windowsize = (480, 480)
        frame = cv2.resize(frame, windowsize)
        cv2.imshow("q:end s:capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 入力画像を読み込む [h,w,c]

            # plt.imshow(img)  # 入力画像を表示
            # plt.show()

            test_imgs = []
            image_pil = Image.fromarray(img)
            image_pil = image_pil.convert("RGB")
            inputs = transform(image_pil)
            inputs = inputs.unsqueeze(0).to(device)
            test_imgs.extend(inputs.cpu().detach().numpy())  # tensor2numpy

            # モデル予測 model prediction
            print("model prediction")
            with torch.no_grad():
                _ = model(inputs)
            # 中間層出力を取得  get intermediate layer outputs
            for k, v in zip(test_outputs.keys(), outputs):
                test_outputs[k].append(v.cpu().detach())
            # フック出力を初期化  initialize hook outputs
            outputs = []

            for k, v in test_outputs.items():
                test_outputs[k] = torch.cat(v, 0)

            # 埋め込み結合 Embedding concat
            embedding_vectors = test_outputs["layer1"]
            for layer_name in ["layer2", "layer3"]:
                embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])

            # ランダムにDディメンションを選択　randomly select d dimension
            embedding_vectors = torch.index_select(embedding_vectors, 1, idx)

            # 距離行列を計算 calculate distance matrix
            print("calculate distance matrix")
            B, C, H, W = embedding_vectors.size()
            embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
            dist_list = []
            for i in range(H * W):
                mean = train_outputs[0][:, i]
                conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
                dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
                dist_list.append(dist)

            dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

            # アップサンプル upsample
            print("upsample")
            dist_list = torch.tensor(dist_list)
            score_map = (
                F.interpolate(dist_list.unsqueeze(1), size=inputs.size(2), mode="bilinear", align_corners=False)
                .squeeze()
                .numpy()
            )

            # スコアマップにGaussianスムージングを適用 apply gaussian smoothing on the score map
            print("apply gaussian smoothing on the score map")
            score_map = gaussian_filter(score_map, sigma=4)  # 1枚だけ
            # for i in range(score_map.shape[0]):
            # score_map[i] = gaussian_filter(score_map[i], sigma=4)

            # 正規化 Normalization
            max_score = score_map.max()
            min_score = score_map.min()
            scores = (score_map - min_score) / (max_score - min_score)

            # 画像用閾値
            threshold_image = THRESHOLD_IMAGE
            # 画像レベルのスコアを計算 calculate image-level score 0.25以上を異常とする
            print("calculate image-level score")
            img_scores = scores.reshape(1, -1).max(axis=1)
            if img_scores > threshold_image:
                print("Abnormal: score=", img_scores)  # 異常
            else:
                print("Normal: score=", img_scores)  # 正常

            # 画素用しきい値
            threshold_pixel = THRESHOLD_PIXCEL
            # 画像保存(3枚セット)
            print("save images(3-mai)")
            save_dir = args.save_path + "/" + f"pictures_{args.arch}"
            os.makedirs(save_dir, exist_ok=True)
            plot_fig_cap(test_imgs, scores, threshold_pixel, save_dir, "original_class")

            test_outputs = OrderedDict([("layer1", []), ("layer2", []), ("layer3", [])])  # リセット

    # 終了処理
    capture.release()
    cv2.destroyAllWindows()

    print("================capture conmpleted================")


def plot_fig_cap(test_img, scores, threshold, save_dir, class_name):
    """画像出力

    Args:
        test_img ([type]): 画像
        scores ([type]): スコア
        threshold ([type]): [description]
        save_dir ([type]): [description]
        class_name ([type]): [description]
    """

    vmax = scores.max() * 255.0
    vmin = scores.min() * 255.0

    # print(test_img)
    img = test_img[0]
    img = denormalization(img)

    heat_map = scores * 255
    mask = scores
    mask[mask > threshold] = 1
    mask[mask <= threshold] = 0
    kernel = morphology.disk(4)
    mask = morphology.opening(mask, kernel)
    mask *= 255
    vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode="thick")
    fig_img, ax_img = plt.subplots(1, 3, figsize=(12, 3))
    fig_img.subplots_adjust(right=0.9)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    for ax_i in ax_img:
        ax_i.axes.xaxis.set_visible(False)
        ax_i.axes.yaxis.set_visible(False)
    ax_img[0].imshow(img)
    ax_img[0].title.set_text("Image")

    ax = ax_img[1].imshow(heat_map, cmap="jet", norm=norm)
    ax_img[1].imshow(img, cmap="gray", interpolation="none")
    ax_img[1].imshow(heat_map, cmap="jet", alpha=0.5, interpolation="none")
    ax_img[1].title.set_text("Predicted heat map")

    ax_img[2].imshow(vis_img)
    ax_img[2].title.set_text("Segmentation result")
    left = 0.92
    bottom = 0.15
    width = 0.015
    height = 1 - 2 * bottom
    rect = [left, bottom, width, height]
    cbar_ax = fig_img.add_axes(rect)
    cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
    cb.ax.tick_params(labelsize=8)
    font = {
        "family": "serif",
        "color": "black",
        "weight": "normal",
        "size": 8,
    }
    cb.set_label("Anomaly Score", fontdict=font)

    now = datetime.datetime.now()
    fig_img.savefig(os.path.join(save_dir, class_name + "{0}_{1:%Y%m%d%H%M%S}".format(0, now)), dpi=100)
    plt.close()


def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
    """画像出力

    Args:
        test_img ([type]): 画像
        scores ([type]): スコア
        gts ([type]): [description]
        threshold ([type]): [description]
        save_dir ([type]): [description]
        class_name ([type]): [description]
    """
    num = len(scores)
    vmax = scores.max() * 255.0
    vmin = scores.min() * 255.0
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode="thick")
        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text("Image")
        ax_img[1].imshow(gt, cmap="gray")
        ax_img[1].title.set_text("GroundTruth")
        ax = ax_img[2].imshow(heat_map, cmap="jet", norm=norm)
        ax_img[2].imshow(img, cmap="gray", interpolation="none")
        ax_img[2].imshow(heat_map, cmap="jet", alpha=0.5, interpolation="none")
        ax_img[2].title.set_text("Predicted heat map")
        ax_img[3].imshow(mask, cmap="gray")
        ax_img[3].title.set_text("Predicted mask")
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text("Segmentation result")
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            "family": "serif",
            "color": "black",
            "weight": "normal",
            "size": 8,
        }
        cb.set_label("Anomaly Score", fontdict=font)

        fig_img.savefig(os.path.join(save_dir, class_name + "_{}".format(i)), dpi=100)
        plt.close()


def denormalization(x):
    """非正規化

    Args:
        x ([type]): [description]

    Returns:
        [type]: [description]
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.0).astype(np.uint8)

    return x


def embedding_concat(x, y):
    """埋め込み結合

    Args:
        x ([type]): [description]
        y ([type]): [description]

    Returns:
        [type]: [description]
    """
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


if __name__ == "__main__":
    """メイン"""
    # train() # 学習用
    # test() # 画像を使って予測
    capture()  # USBカメラで撮影して予測
