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