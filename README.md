# CADSeg — CAD図面のルール違反を見つけるための画像セグメンテーション

このリポジトリは、**CAD図面上の複数種類のルール違反を同時にハイライト**するための学習・推論テンプレートです。大きな図面をタイルに分割して学習/推論し、最後に滑らかに\*\*スティッチ（合成）\*\*します。モデルは **UNet++（ResNet-34エンコーダ）** を既定に採用し、**マルチラベル**（クラスごとにチャネルを1つずつ）で出力します。

---

## 1) 画像セグメンテーションとは？ そして違反検出にどう使うの？

**画像セグメンテーション**は、画像の各ピクセルが「何に属するか」を分類する技術です。たとえば道路の画像なら「車」「歩道」「標識」などをピクセル単位で塗り分けます。

このプロジェクトでは、対象が写真ではなく**CAD図面**です。ルール違反（例：矢印が対象に接していない、文字が線と重なっている、寸法が欠けている…）は、**細い線・矢印・文字の重なり**など、ピクセルレベルで判断したい現象です。そこで、各ルール違反を**1つのチャネル**として持ち、モデルは **「このピクセルはルールAの違反か？」** をチャネルごとに確率で出します（**マルチラベル**方式。複数違反が同じ場所に重なってもOK）。

### ざっくりイメージ

* 入力：CAD図面（超高解像度）
* 出力：ルールA/B/C…に対応した**確率マップ**（同じサイズ）
* しきい値で二値化 → 違反領域を色付きで重ねて表示 → レビューや自動チェックに活用

**ポイント**

* データが少なくても学習しやすいよう、**ImageNetで事前学習済みのエンコーダ**を使います
* 図面が大きいので**タイル分割**→推論→**重み付きで合成**し、**継ぎ目のない**結果を作ります
* 各ルールはデータ数が少なめでも、**共通のエッジ/ライン特徴**をエンコーダが共有学習できるので効率的です

---

## 2) 全体アーキテクチャ（超ざっくり）

```
[データ]  図面画像 + 各ルールの二値マスク（1クラス=1枚）
    │
    ├─(前処理) タイル化(1024×1024, オーバーラップ有) + CADに安全な水増し
    │
    ├─(学習) UNet++（ResNet-34, ImageNet初期化）
    │      └─ マルチラベル出力（チャネル数 = ルール数, 活性化は学習時ナシ/評価時Sigmoid）
    │      └─ 損失: Dice + BCE（クラス重み対応）/ 目的によりTverskyやFocalも選択可
    │      └─ サンプラ: クラスごとの**陽性タイル**を多めに混ぜるバランス戦略
    │
    ├─(検証) フル画像でタイル推論→スティッチ→IoU/Dice（クラス別/マクロ平均）
    │
    ├─(キャリブレーション) クラスごとに確率の**しきい値**を探索（F1/IoU最大化）
    │
    └─(推論) TTA（左右/上下反転）+ 融合（mean/gmean/max）
           └─ 小領域除去などの後処理 → **二値マスク**と**オーバーレイ画像**を保存
           └─ 必要なら TorchScript/ONNX でエクスポート
```

### 主要コンポーネント

* **データI/O**：タイル生成・スティッチ、CADに“安全”なAugmentation
* **モデル**：UNet++（ResNet-34）を既定。DeepLabV3+等に差し替えも可能
* **損失**：`Dice + BCEWithLogits`（クラス重み/陽性重み対応）。TverskyやFocalも選べます
* **サンプラ**：各クラスの陽性タイルを均等に引く**クラスバランス・バッチサンプラ**
* **検証/チューニング**：フル画像で正しく評価し、**クラス別しきい値**を自動探索
* **推論/後処理**：TTA・小ブロブ除去・カラーマップでの可視化
* **エクスポート**：TorchScript / ONNX（動的サイズ対応）

---

## クイックスタート（最短ルート）

> 事前に Python 仮想環境を作成し、`requirements.txt` をインストールしてください。

```bash
# 1) データの整合チェック（画像とマスクのサイズ一致/二値性など）
python -m cadseg.cli.prepare_dataset --configs configs

# 2) 学習（自動split: 80/20。必要なら data/splits/*.txt を用意）
python -m cadseg.cli.train --configs configs --run_dir runs/exp1

# 3) バリデーション（任意：学習済みモデルでフル画像評価）
python -m cadseg.cli.validate --configs configs \
  --ckpt runs/exp1/checkpoints/best.pth --limit 10

# 4) しきい値キャリブレーション（クラスごとに最適化: 既定はF1）
python -m cadseg.cli.calibrate --configs configs \
  --ckpt runs/exp1/checkpoints/best.pth --metric f1

# 5) 推論（オーバーレイと二値マスクを保存）
python -m cadseg.cli.infer --configs configs \
  --ckpt runs/exp1/checkpoints/best.pth --out_dir outputs/infer

# 6) エクスポート（TorchScript/ONNX）
python -m cadseg.cli.export --configs configs \
  --ckpt runs/exp1/checkpoints/best.pth --out_dir runs/exp1/export \
  --jit trace --onnx --dynamic --check_onnx
```

---

## すぐに“見て”確認したい人向け：タイル＋GTオーバーレイの簡易プレビュー（ワンライナー）

> **目的**：学習前に「タイル切り出し」「マスク重ね合わせ」が期待どおりかを**目視**でチェック。
>
> 以下は“シェルから1コマンド”で、ランダムなタイルの**オーバーレイグリッド**画像を `outputs/preview/train_tile_preview.png` に出力します。

```bash
python - <<'PY'
from cadseg.config import load_configs
from cadseg.dataio.datasets import TiledDataset
import numpy as np, cv2, os, random, torch
cfg = load_configs('configs'); ds = TiledDataset(cfg.dataset, cfg.dataset.classes, stage='train', aug_cfg=cfg.aug, preload_index=True)
K = min(12, len(ds)); idxs = random.sample(range(len(ds)), K)
imgs=[]
for i in idxs:
    s = ds[i]
    im = (s['image'].permute(1,2,0).numpy()*255).astype(np.uint8)
    mk = s['mask'].numpy()  # (C,H,W) 0/1
    m = (mk.max(axis=0)*255).astype(np.uint8)
    m = cv2.cvtColor(m, cv2.COLOR_GRAY2RGB)
    over = np.clip(0.6*im + 0.4*m, 0, 255).astype(np.uint8)
    imgs.append(over)
h,w,_=imgs[0].shape; cols=4; rows=(len(imgs)+cols-1)//cols
canvas=np.zeros((rows*h, cols*w, 3), np.uint8)
for j,im in enumerate(imgs): r=j//cols; c=j%cols; canvas[r*h:(r+1)*h, c*w:(c+1)*w]=im
os.makedirs('outputs/preview', exist_ok=True)
cv2.imwrite('outputs/preview/train_tile_preview.png', canvas[:,:,::-1])
print('saved: outputs/preview/train_tile_preview.png')
PY
```

* **何を見ればよい？**

  * タイルの境界で不自然な歪みがないこと（Augmentationが強すぎない）
  * マスクの塗りが**画像と正しく重なっている**こと（座標/サイズのずれがない）
  * クラスが極端に偏っていないか（陽性タイルがゼロばかりになっていないか）

---

## よくある質問（超要約）

* **なぜ1モデルに全ルールをまとめるの？**

  * エンコーダがエッジ/ライン/テキストといった**共通特徴**を共有学習でき、**データ効率が高い**ため（各ルール50枚程度でも学習が安定）。
* **しきい値は0.5固定でいい？**

  * クラスごとに分布が違うので、**検証セットで最適化**するとF1/IoUが上がります。
* **線が細くて途切れがち**

  * UNet++の**濃いスキップ接続**と、**タイルのオーバーラップ＋ハン窓合成**で境界を滑らかにします。必要なら**小ブロブ除去**でノイズも抑制。

---

## 連絡先 / Issue

セットアップで詰まった場合は Issue を立てるか、`prepare_dataset` の出力（`data/meta/dataset_summary.json`）を共有してください。クラス名の綴りやマスクの配置ミスなど、**初期のデータ契約のズレ**が原因のことがよくあります。
