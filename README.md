# ディレクトリ構成
```
wiki-rag-tool/
├── README.md                 # このファイル
├── main.py                   # RAGシステムのメインプログラム
├── requirements.txt          # Pythonライブラリの依存関係
├── Modelfile                 # Ollamaモデル登録用の設定ファイル
├── venv/                     # Python仮想環境（作成後）
├── wiki/                     # Wikipediaデータ（転送後）
├── diff/                     # 差分データ（転送後）
├── elyza/                    # ELYZAモデルファイル（ダウンロード後）
│   └── Llama-3-ELYZA-JP-8B-q4_k_m.gguf
├── glucose/                  # GLuCoSE埋め込みモデル（ダウンロード後）
└── vector_db/               # Chromaベクトルデータベース（初回実行後）
```


# ringoからtachyonにwikiデータを転送するコマンド
scp -r -3 [kmcid]@[ringo]:/home/www/inside-cgi/wiki/wiki/ [kmcid]@[tachyon]:~/wiki-rag-tool

scp -r -3 [irom]@[ringo]:/home/www/inside-cgi/wiki/diff/ [kmcid]@[tachyon]:~/wiki-rag-tool

# ollamaのインストール
curl -fsSL https://ollama.com/install.sh | sh

# python　仮想環境の構築
python3 -m venv venv
. venv/bin/activate

# hugging faceからモデルのダウンロード
pip install huggingface-hub
huggingface-cli download elyza/Llama-3-ELYZA-JP-8B-GGUF --local-dir elyza
huggingface-cli download pkshatech/GLuCoSE-base-ja-v2 --local-dir glucose

# ollamaにモデルを登録
pip install langchain langchain_community langchain-huggingface chromadb sentence-transformers sentencepiece  unstructured "unstructured[md]" "unstructured[csv]" pypdf
ollama create elyza-local -f ./Modelfile

# プログラムの実行
python3 main.py
