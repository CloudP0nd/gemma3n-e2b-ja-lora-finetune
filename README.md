# Gemma 3n-E2B Japanese LoRA Fine-tuning（破滅的忘却対策版）

<p align="center">
  <img src="https://img.shields.io/badge/Model-unsloth/gemma--3n--E2B--it-blue" alt="Model">
  <img src="https://img.shields.io/badge/Framework-Unsloth-green" alt="Framework">
  <img src="https://img.shields.io/badge/API-FastLanguageModel-teal" alt="API">
  <img src="https://img.shields.io/badge/Method-LoRA_(attn+MLP)-orange" alt="Method">
  <img src="https://img.shields.io/badge/Precision-FP16-yellow" alt="Precision">
  <img src="https://img.shields.io/badge/LR-2e--4-yellow" alt="LR">
  <img src="https://img.shields.io/badge/Dataset-oasst1--21k--ja_(8k_filtered)-red" alt="Dataset">
  <img src="https://img.shields.io/badge/Platform-Google_Colab-F9AB00" alt="Platform">
</p>

Unsloth 最適化版 `unsloth/gemma-3n-E2B-it` の日本語性能を **破滅的忘却（catastrophic forgetting）を回避** しながら底上げするための Google Colab ノートブックです。Unsloth フレームワークによる高速 LoRA ファインチューニングと、**学習前後の定量評価** による忘却モニタリングを統合しています。

> ⚠️ **モデル名について**: 仕様書に記載の「Gemma4-E2B」は、現時点で Google が公開している **Gemma 3n-E2B-it**（Effective 2B パラメータ・モバイル向け軽量モデル）と解釈して実装しています。Gemma 4 シリーズがリリースされた際は `MODEL_NAME` を差し替えるだけで対応可能です。

---

## 🎯 目的

Gemma 3n-E2B は Google が 2025 年にリリースしたモバイル/エッジ向けの軽量マルチモーダルモデルで、約 2B の有効パラメータ（総パラメータは約 5B）を持ちます。多言語対応を謳っていますが、日本語の指示追従・応答品質には改善の余地があります。本ノートブックは、高品質な日本語対話データを用いて日本語性能を強化しつつ、既存の英語能力・推論能力を保持することを目指します。

---

## 🛡️ 破滅的忘却への対策

日本語データだけでファインチューニングすると、英語能力や推論能力が急激に劣化する「破滅的忘却」が起こり得ます。本ノートブックでは以下の **9 つの施策** を組み合わせてこれを防ぎます：

| # | 施策 | 内容 | 効果 |
|---|------|------|------|
| 1 | **LoRA（フルチューンではない）** | rank=16 / alpha=32 のアダプタのみ学習 | 元重みを凍結し最小限の差分更新 |
| 2 | **全 attention + MLP 層へ適用** | `q,k,v,o,gate,up,down_proj`（7 層） | 適応範囲を十分に確保しつつ、フリーズ部分で知識保持 |
| 3 | **学習率 2e-4** | LoRA 標準的な中庸な値 | 過度な重み更新を回避しつつ日本語パターンを獲得 |
| 4 | **エポック数 2** | 多すぎず少なすぎず | 過学習による能力崩壊を防止 |
| 5 | **Cosine LR Schedule** | 学習率をコサインカーブで漸減 | 学習終盤の急激な更新を防止 |
| 6 | **Warmup (5%)** | 最初の 5% ステップは線形増加 | 学習初期の不安定性を抑制 |
| 7 | **Weight Decay (0.01)** | L2 正則化を適用 | 過学習・過度な適応を抑制 |
| 8 | **LoRA Dropout (0.05)** | アダプタに入れるドロップアウト | アダプタの過適合を防止 |
| 9 | **学習前後の PPL 評価** | 日本語/英語の Perplexity を比較 | 忘却を定量モニタリング |

---

## ⚙️ ハイパーパラメータ

| 項目 | 値 |
|---|---|
| 対象モデル | `unsloth/gemma-3n-E2B-it` |
| Precision | FP16（QLoRA ではない） |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Gradient Accumulation Steps | 4 |
| Learning Rate | 2e-4 |
| Optimizer | `paged_adamw_8bit` |
| LR Scheduler | Cosine |
| Warmup Ratio | 0.05 |
| Weight Decay | 0.01 |
| Epochs | 2 |
| Max Sequence Length | 2048 |
| Gradient Checkpointing | Unsloth (有効) |

---

## 📊 データセット

[`llm-jp/oasst1-21k-ja`](https://huggingface.co/datasets/llm-jp/oasst1-21k-ja) は、OpenAssistant コーパス (oasst1) を日本語に翻訳した約 21,000 件の対話データです。本ノートブックでは、以下の基準でフィルタリングを行い **8,000 件の高品質サンプル** を抽出します：

1. **日本語含有率 30% 以上**（ひらがな/カタカナ/漢字の割合）
2. **応答長 50〜1,500 文字**（長すぎず短すぎず）
3. **重複排除**（同一 instruction は 1 件のみ）
4. **ノイズ除去**（URL のみ/コードのみ/記号のみの応答を除外）
5. **シャッフル後に先頭 8,000 件を採用**

---

## 🚀 使い方

### 1. Colab でノートブックを開く

[`Gemma3n-E2B_JA_LoRA_FineTuning.ipynb`](./Gemma3n-E2B_JA_LoRA_FineTuning.ipynb) をダウンロードし、[Google Colab](https://colab.research.google.com/) にアップロードします。

### 2. HuggingFace トークンの準備

Gemma は gated model のため、事前に以下を行ってください：

1. [HuggingFace](https://huggingface.co/) でアカウント作成
2. [google/gemma-3n-E2B-it](https://huggingface.co/google/gemma-3n-E2B-it) のライセンス同意
3. [Access Token](https://huggingface.co/settings/tokens) を発行（read 権限）
4. Colab の **Secrets** に `HF_TOKEN` として登録

### 3. GPU ランタイムを選択

`ランタイム > ランタイムのタイプを変更 > T4 GPU`（以上推奨）

### 4. 上から順に実行

セルを上から順に実行してください。A100 で約 1 時間、T4 で 2-3 時間で完了します。

### 5. 学習済みモデルの取得

- **LoRA アダプタ**（軽量・数十 MB）: Step 15-1 で保存
- **マージ済み 16bit モデル**（数 GB）: Step 15-2 で保存（コメントイン）
- **HuggingFace Hub へ公開**: Step 15-3 で `push_to_hub_merged`（コメントイン）

---

## 📈 評価指標

学習前後で以下を計測し、忘却をモニタリングします：

- **Japanese Perplexity**（下降期待）
- **English Perplexity**（維持期待＝忘却なしの証拠）
- **生成サンプルの定性比較**（日本語プロンプト 3 件 + 英語プロンプト 1 件）

---

## 📁 リポジトリ構成

```
.
├── Gemma3n-E2B_JA_LoRA_FineTuning.ipynb   # メインの Colab ノートブック
├── README.md                              # このファイル
└── LICENSE                                # Apache 2.0
```

---

## ⚠️ 注意事項

- **Gemma ライセンス**: Gemma は [Gemma Terms of Use](https://ai.google.dev/gemma/terms) に基づき使用してください。
- **データセットライセンス**: `llm-jp/oasst1-21k-ja` は [Apache 2.0](https://huggingface.co/datasets/llm-jp/oasst1-21k-ja) で公開されています。
- **GPU メモリ**: T4 (16GB) で動作確認済みの設定にしていますが、メモリ不足の場合は `per_device_train_batch_size` を 1 に下げてください。
- **Unsloth バージョン**: Gemma 3n サポートが含まれる最新版を使用してください（`pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"`）。

---

## 📝 ライセンス

本リポジトリのコードは [MIT License](./LICENSE) のもとで公開します。ただし、学習済みモデルの利用に関しては元の Gemma ライセンス（[Gemma Terms of Use](https://ai.google.dev/gemma/terms)）に従ってください。
