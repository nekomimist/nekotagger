# nekotagger

[Grio43/OppaiOracle](https://huggingface.co/Grio43/OppaiOracle) を使う、拡張可能な
コマンドライン画像taggerです。単一画像は標準出力へ、ディレクトリは画像の相対構造を
保った `.txt` ファイルへタグを出力します。

## セットアップ

Python環境と依存関係はすべて `uv` で管理します。

```bash
uv sync
```

初回のモデル利用時にHugging FaceからONNXモデルを取得します。V1.1は約1GBで、
CUDA 12/cuDNN 9のPythonランタイムも依存に含まれるため、初回セットアップには
数GBの空き容量が必要です。モデルはHugging Faceの共有キャッシュに保存され、
リポジトリ内にはコピーされません。キャッシュ場所を変える場合は `HF_HOME` を設定してください。

## 使い方

単一画像のタグを標準出力へ出します。診断やモデル取得状況はstderrへ出るため、
タグだけを安全にリダイレクトできます。

```bash
uv run nekotagger image.png
uv run nekotagger image.png > tags.txt
```

ディレクトリ直下を処理します。

```bash
uv run nekotagger /data/images
```

再帰的に処理します。

```bash
uv run nekotagger /data/images --recursive
```

入力が `/data/images`、画像が `/data/images/set/a.png`、カレントディレクトリが
`/work` なら、既定の出力は `/work/set/a.txt` です。

出力ルートを指定する場合:

```bash
uv run nekotagger /data/images -r --output-dir /output
```

画像と同じディレクトリへ保存する場合:

```bash
uv run nekotagger /data/images -r --same-dir
```

既存の `.txt` がある場合、対話端末では一度だけ、すべて上書きするか連番を付けるか
中止するか確認します。非対話実行では次のどちらかを指定してください。

```bash
uv run nekotagger /data/images -r --overwrite
uv run nekotagger /data/images -r --numbered
```

`--numbered` は `a_1.txt`, `a_2.txt` のように保存します。同じディレクトリに
`a.jpg` と `a.png` がある場合も、片方を失わないよう自動的に連番を付けます。

## モデルとタグ設定

既定はOppaiOracle V1.1（448×448）、threshold 0.65、confidence順の最大50タグです。

```bash
# 高速な320×320版
uv run nekotagger image.png --variant v1

# 閾値と件数を変更（0は件数制限なし）
uv run nekotagger image.png --threshold 0.76 --max-tags 0

# モデルのraw tag名を維持
uv run nekotagger image.png --keep-underscores
```

通常はunderscoreを空白へ変換し、カンマと空白で区切ります。`<PAD>` と `<UNK>` は
常に除外します。

## タグの除外

characterやartistなどのカテゴリを指定して除外できます。同じオプションを繰り返すと
複数カテゴリを除外します。

```bash
uv run nekotagger image.png \
  --exclude-category character \
  --exclude-category artist

# artist、copyright、characterをまとめて除外
uv run nekotagger image.png --exclude-named
```

利用できるカテゴリは `general`, `artist`, `copyright`, `character`, `meta` です。

カテゴリの意味と代表例:

| カテゴリ | 意味 | 例 |
| --- | --- | --- |
| `general` | 画像に見えている人物、物、特徴、構図 | `1girl`, `long_hair`, `blue_eyes` |
| `artist` | 作者・サークル名 | `kantoku`, `refeia` |
| `copyright` | 作品、シリーズ、フランチャイズ名 | `kantai_collection`, `umamusume`, `onii-chan_wa_oshimai!` |
| `character` | キャラクター名 | `hatsune_miku`, `rem_(re:zero)` |
| `meta` | 画像・投稿・制作方法・管理状態に関する情報 | `highres`, `official_art`, `commentary_request`, `ai-generated` |

Danbooruの `copyright` は法律上の著作権表示という意味ではなく、主に作品・シリーズ名を
表します。`--exclude-named` は `artist`, `copyright`, `character` をまとめて除外し、
`general` と `meta` は残します。

`meta` は範囲が広いため、`--exclude-category meta` を指定すると `ai-generated` だけでなく
`highres`, `official_art`, `translation_request` なども除外されます。AI生成タグだけを
除外したい場合は、カテゴリではなく個別タグを指定します。

```bash
# 作者名、シリーズ名、キャラクター名を除外
uv run nekotagger image.png --exclude-named

# AI生成を示すタグだけを除外
uv run nekotagger image.png --exclude-tag ai-generated
```

カテゴリ情報は
[ThetaCursed/danbooru-2026-clean-metadata](https://huggingface.co/datasets/ThetaCursed/danbooru-2026-clean-metadata)
の `tags_dictionary.parquet`（約7.5MB）を固定revisionから取得します。Danbooru APIへは
アクセスしません。初回だけHugging Faceから共有キャッシュへダウンロードし、2回目以降は
ネットワークへ接続せずキャッシュを使用します。

独自に取得した同じスキーマのファイルを使う場合:

```bash
uv run nekotagger image.png \
  --exclude-category character \
  --tag-metadata /data/tags_dictionary.parquet
```

カテゴリ辞書に存在しないタグは、誤って除外しないよう残します。現在の固定辞書では
OppaiOracle語彙19,294件のうち18,968件を分類できます。

個別タグはCLIまたはUTF-8テキストファイルで除外できます。ファイルは1行1タグで、
空行と `#` から始まる行を無視します。空白区切りとunderscore区切りはどちらも使えます。

```bash
uv run nekotagger image.png --exclude-tag watermark --exclude-tag "blue eyes"
uv run nekotagger image.png --exclude-tags-file exclude-tags.txt
```

除外はraw tag名に対して行い、その後に `--max-tags` とunderscore変換を適用します。
そのため、上位タグが除外されても件数上限までは次の候補で補われます。

## デバイス

```bash
uv run nekotagger image.png --device auto
uv run nekotagger image.png --device cuda:0
uv run nekotagger image.png --device cuda:1
uv run nekotagger image.png --device cpu
```

`auto` はCUDA device 0を試し、利用できなければ警告してCPUへフォールバックします。
`cuda` または `cuda:N` は厳格指定で、CUDAを初期化できなければエラーにします。
ディレクトリ処理の既定バッチサイズはCUDAで4、CPUで1です。必要なら
`--batch-size` で調整できます。

## 対応画像と失敗時の動作

ディレクトリ探索ではJPEG、PNG、WebP、BMP、TIFF、GIFに対応します。GIFは先頭フレームを
使用します。ディレクトリsymlinkは再帰しません。壊れた画像は記録して残りを処理し、
最後に失敗一覧をstderrへ出して終了コード1を返します。

前処理はモデルの学習条件に合わせ、EXIF向き補正、透過部分のRGB `(114, 114, 114)`
への合成、拡大しないbilinear letterbox、padding mask、mean/std 0.5の正規化を行います。

## 開発

```bash
uv run pytest
uv run ruff check .
uv run mypy
```

実モデルを取得してV1/V1.1をスモークテストする場合:

```bash
NEKOTAGGER_TEST_DEVICE=cuda:0 uv run pytest --run-model tests/test_model_smoke.py
```

モデル固有処理は共通CLI・探索・出力処理から分離されています。別taggerを追加するときは、
`Tagger` protocolを実装しregistryへ登録します。

## モデルの注意事項とライセンス

OppaiOracleモデルはApache License 2.0で配布されています。モデル作者は、色、髪の長さ、
身体サイズ、小物などの隣接タグに学習データ由来の混同があり、出力は人間が確認するよう
明記しています。このツールもモデル出力を正解データとはみなしません。

本実装はモデル配布物の公開I/O仕様に基づく独立実装です。別ライセンスで公開されている
作者の学習コードは複製していません。
