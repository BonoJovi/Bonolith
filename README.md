# JaIM - Japanese AI-powered Input Method

<div align="center">

**Japanese Input Method / Japanese a Input Method / Japanese AI Method**

[![Version](https://img.shields.io/badge/Version-1.2.3-blue)](https://github.com/BonoJovi/JaIM/releases)
[![Rust](https://img.shields.io/badge/Rust-2024-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/bonojovi)

**LLM 駆動の次世代日本語入力 for Linux**

</div>

---

## 特徴

- **ローマ字変換** — ローマ字入力からひらがな・カタカナへの変換
- **辞書変換** — IPADIC ベースの 232,000+ エントリ辞書（Trie 検索 + DP 分節）
- **活用形対応** — 動詞・形容詞の活用形を自動生成（食べた、走って、読んだ 等）
- **記号入力** — 矢印（やじるし→）、括弧ペア（かっこ→「」）等の記号辞書
- **LLM リランキング** — Qwen2.5-0.5B + ローカル HTTP サーバーによる文脈を考慮した候補順位付け（バックグラウンド実行）
- **文法スコアリング** — 文法ルールによる候補フィルタリング
- **ユーザ学習** — 選択履歴を学習し、候補順位を最適化
- **文節編集** — 文節の移動・伸縮・候補切替
- **かな変換** — F6〜F10 によるひらがな / カタカナ / ローマ字変換（MS-IME 準拠）
- **IBus / Fcitx5 対応** — IBus は zbus (D-Bus) による Pure Rust 実装、Fcitx5 は FFI 経由の C++ アドオン

## アーキテクチャ

3 段階のハイブリッド変換パイプライン：

```
キー入力 → ローマ字→かな変換 → 辞書分節 (DP)
                                    ↓
                            文法スコアリング → 候補リスト → ユーザ
                                    ↓ (バックグラウンド)
                            LLM リランキング (llama-server HTTP)
                                    ↓
                            候補順位更新（次回操作時に反映）
```

## コアコンポーネント

| コンポーネント | 役割 |
|----------------|------|
| **ローマ字変換器** | ASCII → ひらがな / カタカナのステートマシン |
| **辞書エンジン** | Trie ベースのかな→漢字検索 + DP 分節（232K エントリ） |
| **文法エンジン** | 構造検証とスコアリング（9 ルール） |
| **LLM エンジン** | llama-server (HTTP) 経由の Qwen2.5-0.5B によるバックグラウンドリランキング |
| **ユーザスコアラ** | 選択履歴の対数スケール学習 |
| **変換エンジン** | パイプライン統合と文節編集 |

## IME フレームワーク対応

| フレームワーク | 状態 | 実装方式 |
|----------------|------|----------|
| IBus | 対応済み | Pure Rust（`zbus` による D-Bus 直接実装） |
| Fcitx5 | 対応済み | Rust エンジン (cdylib) + C++ アドオン (FFI) |

## プロジェクト構成

```
JaIM/
  src/
    main.rs                  # エントリポイント（IBus D-Bus サービス）
    lib.rs                   # ライブラリクレート（Fcitx5 用 cdylib エクスポート）
    ffi.rs                   # C FFI レイヤ（Fcitx5 向けキー処理・UI 状態取得）
    core/
      romaji/                # ローマ字→かな変換ステートマシン
      dictionary/            # Trie + DP 分節 + 組み込み辞書 (IPADIC)
      grammar/               # 文法検証
      llm/                   # LLM リランキング (HTTP 経由で llama-server と通信)
      user_scorer.rs         # ユーザ選択履歴学習
    engine/                  # 変換パイプライン統合
    ibus/                    # IBus D-Bus 統合
      engine_impl.rs         # IBus エンジンインターフェース
      factory.rs             # IBus ファクトリインターフェース
      keymap.rs              # キー定数・ヘルパ
      config.rs              # トグルキー設定
    bin/
      generate_dict.rs       # IPADIC → builtin_dict.rs ジェネレータ
  fcitx5/
    jaim_engine.cpp          # Fcitx5 C++ アドオン
    jaim_engine.h            # アドオンヘッダ
    jaim_ffi.h               # Rust FFI ヘッダ
    CMakeLists.txt           # Fcitx5 ビルド設定
    jaim-addon.conf          # アドオン記述ファイル
    jaim-im.conf             # 入力メソッド記述ファイル
  scripts/
    jaim-llm-server.service  # llama-server 用 systemd ユーザーサービス
  data/
    jaim.xml                 # IBus コンポーネント記述ファイル
```

## 動作要件

- [Rust](https://rustup.rs/)（最新 stable）
- Linux + IBus または Fcitx5
- IPADIC 辞書（`sudo apt install mecab-ipadic`）
- LLM 用（オプション）: llama-server + Qwen2.5-0.5B Q4 モデル（約 512MB）

## ビルド

```bash
# 辞書生成（IPADIC から組み込み辞書を生成）
cargo run --bin generate-dict

# リリースビルド
cargo build --release
```

## LLM サーバーセットアップ（オプション）

LLM リランキングを有効にするには、llama-server をセットアップします。
LLM サーバーが起動していなくても JaIM は正常に動作します（辞書＋文法＋ユーザ学習のみで変換）。

### 1. llama-server のインストール

[llama.cpp リリースページ](https://github.com/ggml-org/llama.cpp/releases) から Ubuntu x64 バイナリをダウンロード：

```bash
# ダウンロード・展開
cd /tmp
curl -LO https://github.com/ggml-org/llama.cpp/releases/latest/download/llama-<version>-bin-ubuntu-x64.tar.gz
mkdir llama-extract && cd llama-extract
tar xzf ../llama-*-bin-ubuntu-x64.tar.gz

# インストール
mkdir -p ~/.local/bin ~/.local/lib
cp llama-*/llama-server ~/.local/bin/
cp llama-*/lib*.so* ~/.local/lib/
```

### 2. モデルのダウンロード

```bash
mkdir -p ~/.local/share/jaim/models
cd ~/.local/share/jaim/models
curl -LO https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

### 3. systemd サービスとして登録

```bash
mkdir -p ~/.config/systemd/user/
cp scripts/jaim-llm-server.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now jaim-llm-server
```

動作確認：

```bash
curl http://127.0.0.1:8080/health
# → {"status":"ok"}
```

### LLM ON/OFF の切り替え

導入後はいつでも切り替えできます：

```bash
jaim llm on        # 有効化（systemd サービスを enable + start）
jaim llm off       # 無効化（stop + disable、辞書のみで動作）
jaim llm status    # 現状確認
```

実行中の IBus / Fcitx5 セッションには即座に反映されないので、切り替え後は `ibus-daemon -drx` でリロードするか、再ログインしてください。

`./scripts/install.sh --no-llm` で **最初から LLM を無効にした状態でインストール** することもできます（古い PC や LLM を使いたくないユーザ向け）。

## インストール

### IBus

```bash
# IBus デーモンを停止してバイナリをインストール
sudo pkill -f ibus-daemon
sleep 1
sudo rm -f /usr/bin/ibus-engine-jaim
sudo cp target/release/jaim /usr/bin/ibus-engine-jaim

# IBus コンポーネント記述ファイルをインストール
sudo cp data/jaim.xml /usr/share/ibus/component/jaim.xml

# IBus を再起動
sleep 2 && ibus-daemon -drx
```

### Fcitx5

```bash
# Fcitx5 アドオンをビルド・インストール
cd fcitx5
mkdir -p build && cd build
make clean
cmake .. -DCMAKE_INSTALL_PREFIX=/usr
make
sudo make install && fcitx5 -r -d
```

## セットアップ

### 1. 言語パックのインストール（Ubuntu 22.04 のみ）

Ubuntu 22.04 では言語パックが自動インストールされないため、手動でのインストールが必要です。
24.04 以降は OS インストール時に自動インストールされるため不要です。

設定 → 地域と言語 → インストールされている言語の管理 → 「はい」で言語パックをインストール

### 2. JaIM を登録

#### IBus の場合

設定 → キーボード → 入力ソース → 「+」 → 日本語 → Japanese (JaIM - Japanese AI Input) → 追加

#### Fcitx5 の場合

Fcitx5 設定 → 入力メソッド → 「+」 → JaIM を検索 → 追加

### 3. 確認

トップバー（ディストロによってはタスクバー / タスクトレイ）の入力メソッドアイコンをクリックし、JaIM が表示されていれば登録完了です。PC の再起動が必要な場合があります。

## キーバインド

| キー | 動作 |
|------|------|
| Ctrl+Shift+Space | IME ON/OFF（設定変更可） |
| Space | 漢字変換 |
| Enter | 確定 |
| Escape | キャンセル |
| ← / → | 文節フォーカス移動 |
| Shift+← / Shift+→ | 文節伸縮 |
| ↑ / ↓ / Space | 候補切替 |
| F6 | ひらがなに変換 |
| F7 | 全角カタカナに変換 |
| F8 | 半角カタカナに変換 |
| F9 | 全角英数に変換 |
| F10 | 半角英数に変換 |

## テスト

```bash
cargo test
```

## ライセンス

[MIT](LICENSE) - Copyright (c) 2026 Yoshihiro NAKAHARA

[![Ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/bonojovi)

