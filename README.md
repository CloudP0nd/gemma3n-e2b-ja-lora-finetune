# Gemma 4 E2B Japanese LoRA Fine-tuning（破滅的忘却対策版）

<p align="center">
  <img src="https://img.shields.io/badge/Model-unsloth/gemma--4--E2B--it-blue" alt="Model">
  <img src="https://img.shields.io/badge/Framework-Unsloth-green" alt="Framework">
  <img src="https://img.shields.io/badge/API-FastModel-teal" alt="API">
  <img src="https://img.shields.io/badge/Method-LoRA_(attn+MLP)-orange" alt="Method">
  <img src="https://img.shields.io/badge/Precision-FP16-yellow" alt="Precision">
  <img src="https://img.shields.io/badge/LR-2e--4-yellow" alt="LR">
  <img src="https://img.shields.io/badge/Dataset-oasst1--21k--ja_(8k_filtered)-red" alt="Dataset">
  <img src="https://img.shields.io/badge/Platform-Google_Colab-F9AB00" alt="Platform">
</p>

Unsloth 最適化版 `unsloth/gemma-4-E2B-it` の日本語性能を **破滅的忘却（catastrophic forgetting）を回避** しながら底上げするための Google Colab ノートブックです。Unsloth フレームワークによる高速 LoRA ファインチューニングと、**学習前後の定量評価** による忘却モニタリングを統合しています。

---

## 🆕 Gemma 4 について

Gemma 4 は Google DeepMind がリリースした新しいオープンモデルファミリーで、以下のバリアントがあります（Apache-2.0 ライセンス）：

| バリアント | 構造 | コンテキスト | モダリティ | 想定用途 |
|---|---|---|---|---|
| **E2B** | Dense + PLE | 128K | Text/Image/Audio | フォン/エッジ推論・ASR・音声翻訳 |
| **E4B** | Dense + PLE | 128K | Text/Image/Audio | ラップトップ・軽量マルチモーダル |
| **12B Unified** | Dense | 256K | Text/Image/Audio | ラップトップ・ローカルマルチモーダル |
| **26B-A4B** | MoE | 256K | Text/Image | スピード/品質バランス（コンピュータユース） |
| **31B** | Dense | 256K | Text/Image | 最高品質 |

本ノートブックでは最も軽量な **E2B** を使用します。

> ⚠️ Gemma 4 は `FastLanguageModel` ではなく **`FastModel`** API を使用します（マルチモーダル対応のため）。テキストのみ学習する場合でも、`finetune_vision_layers=False` を指定して視覚層をフリーズします。

---

## 🎯 目的

Gemma 4 E2B は多言語（140+ 言語）対応を謳っていますが、日本語の指示追従・応答品質には改善の余地があります。本ノートブックは、高品質な日本語対話データを用いて日本語性能を強化しつつ、既存の英語能力・推論能力・マルチモーダル能力を保持することを目指します。

---

## 🛡️ 破滅的忘却への対策

日本語データだけでファインチューニングすると、英語能力や推論能力が急激に劣化する「破滅的忘却」が起こり得ます。本ノートブックでは以下の **11 の施策** を組み合わせてこれを防ぎます：

| # | 施策 | 内容 | 効果 |
|---|------|------|------|
| 1 | **LoRA（フルチューンではない）** | rank=16 / alpha=32 のアダプタのみ学習 | 元重みを凍結し最小限の差分更新 |
| 2 | **attention + MLP 両方へ適用** | `finetune_attention_modules=True`, `finetune_mlp_modules=True` | 適応範囲を確保しつつ、フリーズ部分で知識保持 |
| 3 | **視覚層はフリーズ** | `finetune_vision_layers=False` | マルチモーダル能力の保持 |
| 4 | **学習率 2e-4** | LoRA 標準的な中庸な値 | 過度な重み更新を回避 |
| 5 | **エポック数 2** | 多すぎず少なすぎず | 過学習による能力崩壊を防止 |
| 6 | **Cosine LR Schedule** | 学習率をコサインカーブで漸減 | 学習終盤の急激な更新を防止 |
| 7 | **Warmup (5%)** | 最初の 5% ステップは線形増加 | 学習初期の不安定性を抑制 |
| 8 | **Weight Decay (0.01)** | L2 正則化を適用 | 過学習・過度な適応を抑制 |
| 9 | **LoRA Dropout (0.05)** | アダプタに入れるドロップアウト | アダプタの過適合を防止 |
| 10 | **`train_on_responses_only`** | ユーザー発言部分をマスク | 応答部分のみ学習し、入力模倣を防止 |
| 11 | **学習前後の PPL 評価** | 日本語/英語の Perplexity を比較 | 忘却を定量モニタリング |

---

## ⚙️ ハイパーパラメータ

| 項目 | 値 |
|---|---|
| 対象モデル | `unsloth/gemma-4-E2B-it` |
| API | `FastModel`（マルチモーダル対応） |
| Precision | FP16（`load_in_4bit=False`） |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| 学習対象 | `language + attention + MLP`（vision はフリーズ） |
| Gradient Accumulation Steps | 4 |
| Learning Rate | 2e-4 |
| Optimizer | `paged_adamw_8bit` |
| LR Scheduler | Cosine |
| Warmup Ratio | 0.05 |
| Weight Decay | 0.01 |
| Epochs | 2 |
| Max Sequence Length | 2048 |
| Chat Template | `gemma-4`（`<\|turn\|>user` / `<\|turn\|>model` 形式） |
| 推論パラメータ | `temperature=1.0, top_p=0.95, top_k=64`（Gemma 4 公式推奨） |

---

## 📊 データセット

[`llm-jp/oasst1-21k-ja`](https://huggingface.co/datasets/llm-jp/oasst1-21k-ja) は、OpenAssistant コーパス (oasst1) を日本語に翻訳した約 21,000 件の対話データです。本ノートブックでは以下の基準でフィルタリングを行い **8,000 件の高品質サンプル** を抽出します：

1. **日本語含有率 30% 以上**（ひらがな/カタカナ/漢字の割合）
2. **応答長 50〜1,500 文字**（長すぎず短すぎず）
3. **重複排除**（同一 instruction は 1 件のみ）
4. **ノイズ除去**（URL のみ/コードのみ/記号のみの応答を除外）
5. **シャッフル後に先頭 8,000 件を採用**

Unsloth の `standardize_data_formats` で `conversations` 形式に正規化した上でフィルタをかけます。

---

## 🚀 使い方

### 1. Colab でノートブックを開く

[`Gemma4-E2B_JA_LoRA_FineTuning.ipynb`](./Gemma4-E2B_JA_LoRA_FineTuning.ipynb) をダウンロードし、[Google Colab](https://colab.research.google.com/) にアップロードします。

### 2. HuggingFace トークンの準備

Gemma 4 は gated model のため、事前に以下を行ってください：

1. [HuggingFace](https://huggingface.co/) でアカウント作成
2. [unsloth/gemma-4-E2B-it](https://huggingface.co/unsloth/gemma-4-E2B-it) のライセンス同意
3. [Access Token](https://huggingface.co/settings/tokens) を発行（read 権限）
4. Colab の **Secrets** に `HF_TOKEN` として登録

### 3. GPU ランタイムを選択

`ランタイム > ランタイムのタイプを変更 > T4 GPU`（以上推奨）。FP16 LoRA のため T4 (16GB) で動作しますが、A100/L4 ならより高速です。

### 4. 上から順に実行

セルを上から順に実行してください。A100 で約 1 時間、T4 で 2-3 時間で完了します。

### 5. 学習済みモデルの取得

- **LoRA アダプタ**（軽量・数十 MB）: Step 19-1 で保存
- **マージ済み 16bit モデル**（数 GB）: Step 19-2 で保存（コメントイン）
- **HuggingFace Hub へ公開**: Step 19-3 で `push_to_hub`（コメントイン）

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
├── Gemma4-E2B_JA_LoRA_FineTuning.ipynb   # メインの Colab ノートブック
├── README.md                             # このファイル
└── LICENSE                               # MIT
```

---

## ⚠️ 注意事項

- **Gemma ライセンス**: Gemma は [Gemma Terms of Use](https://ai.google.dev/gemma/terms) に基づき使用してください。
- **データセットライセンス**: `llm-jp/oasst1-21k-ja` は [Apache 2.0](https://huggingface.co/datasets/llm-jp/oasst1-21k-ja) で公開されています。
- **GPU メモリ**: T4 (16GB) で動作確認済みの設定にしていますが、メモリ不足の場合は `per_device_train_batch_size` を 1 に下げてください。
- **Unsloth バージョン**: Gemma 4 サポートが含まれる最新版を使用してください（インストールセルで公式レシピ通りにインストール）。
- **マルチモーダル**: 本ノートブックはテキストのみ学習します（`finetune_vision_layers=False`）。視覚/音声も学習したい場合はこのフラグを `True` に変更してください。

---

## 📝 ライセンス

本リポジトリのコードは [MIT License](./LICENSE) のもとで公開します。ただし、学習済みモデルの利用に関しては元の Gemma ライセンス（[Gemma Terms of Use](https://ai.google.dev/gemma/terms)）に従ってください。

---

## 🙏 謝辞

- [Unsloth](https://github.com/unslothai/unsloth) — 高速ファインチューニングフレームワーク
- [Google DeepMind](https://deepmind.google/) — Gemma 4 モデルの公開
- [LLM-JP](https://github.com/llm-jp) — 日本語データセットの公開
