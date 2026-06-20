# Bonolith 誤変換ケースデータセット

Bonolithの変換品質を体系的に評価・改善するためのケース集。
品詞ベース文脈解析(IPADIC接続コスト導入)の効果を測る評価セットとして
v2.x 開発で使用する。

## 構成

- `cases.jsonl` — 誤変換ケース本体 (JSON Lines)
- `schema.json` — 1ケースのスキーマ (参照用)
- `README.md` — 本ドキュメント

## スキーマ

1行1ケース。フィールド:

| フィールド | 型 | 必須 | 説明 |
|---|---|---|---|
| `id` | string | ✅ | `case_NNNN` 連番 |
| `input_hiragana` | string | ✅ | 入力ひらがな (確定前) |
| `input_romaji` | string | | 入力ローマ字 (任意) |
| `expected` | string[] | ✅ | 正解の文節区切り + 表記 |
| `expected_readings` | string[] | ✅ | `expected` と1:1対応する文節ひらがな読み。連結すると `input_hiragana` と一致。refinement check で文節境界を導出 |
| `actual` | string[] | | Bonolithの現状出力 (未計測なら省略可) |
| `category` | enum | ✅ | `segmentation` / `word_choice` / `inflection` / `both` |
| `subcategory` | string | | 自由記述 (例: `homophone_misselection`) |
| `pos_solvable` | enum | ✅ | `yes` / `partial` / `no` |
| `pos_hypothesis` | string | | 品詞でどう解けるか/解けないかの仮説 |
| `notes` | string | | 補足説明 |
| `date_collected` | string | ✅ | `YYYY-MM-DD` |
| `bonolith_version` | string | | 計測時のバージョン |
| `source` | enum | | `generated` / `user_reported` / `captured` |

## カテゴリ定義

- **`segmentation`**: 文節区切りが誤り (例: 「わたしはははをたべる」を「私は は はを食べる」に切る)
- **`word_choice`**: 同音異義語の選択誤り (例: 「きしゃ」で「貴社/記者/汽車」誤選)
- **`inflection`**: 活用形誤り (例: 「行って」を「言って」に)
- **`both`**: 区切りも単語選択も誤り

## `pos_solvable` の判定基準

- **`yes`**: 品詞接続コストだけで原理上解ける (例: 「形容詞→名詞」優先で「いい天気」)
- **`partial`**: 品詞 + 共起頻度/前文脈で解ける (LLMなしで届く範囲)
- **`no`**: 文脈意味理解が必要 (LLM領域。品詞解析では原理的に解けない)

## ケース収集方法

### 1. ジェネレート (Claude)

代表的な日本語IMEの古典的問題パターンを `cases.jsonl` の冒頭ブロックに収録。
`source: generated` でマーキング。

### 2. ユーザー手動追加

実使用で遭遇したケースを直接追記。`source: user_reported`。

### 3. ランタイムキャプチャ

`BONOLITH_LOG_CONVERSIONS=1` 環境変数で Bonolith を起動すると、
`~/.local/share/bonolith/conversions.jsonl` に commit 時の変換を記録する。
後で `scripts/curate_cases.sh` で対話的に評価用ケースへ昇格させる。

**プライバシー注意**: ランタイムキャプチャは確定したテキストを生で記録する。
パスワード入力やプライベートな文章を含む可能性があるため、
評価セット作成時のみ短期間有効化し、終わったらログファイルを削除すること。

## ベンチマーク (将来)

`cargo test --test conversion_quality` で評価セット全件を走らせ、
正答率を出すテストランナーを `tests/conversion_quality.rs` に作る予定。
(a)-A2 のPoC評価時に同時整備する。

## バージョン履歴

- 2026-05-17: v2.0.2 ベースで初版作成
