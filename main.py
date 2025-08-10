import os
import shutil

# --- 設定項目 (変更) ---
# 検索対象のドキュメントが格納されているディレクトリのリスト
DATA_PATHS = ["wiki/", "diff/"]
# 検索するファイルのパターンを.txtに限定
GLOB_PATTERN = "**/*.txt"
# ベクトルデータベースの保存先
DB_PATH = "vector_db/"
# ローカルの埋め込みモデルのパス
EMBED_MODEL_PATH = "./glucose"
# Ollamaで登録したローカルLLMの名前
LLM_MODEL_NAME = "elyza-local"


def main():
    """
    RAGシステムのメイン処理
    """
    from langchain_community.document_loaders import DirectoryLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings  # 修正
    from langchain_community.llms import Ollama
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate

    # 1. 埋め込みモデルの準備
    print("埋め込みモデルをロード中...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_PATH)
    print("ロード完了。")

    # 2. ベクトルデータベースの準備
    if not os.path.exists(DB_PATH):
        print(f"データベースが存在しないため、新規作成します。")
        
        # --- ドキュメントの読み込み (変更) ---
        all_documents = []
        print(f"対象ディレクトリ: {DATA_PATHS}")
        for path in DATA_PATHS:
            if not os.path.isdir(path):
                print(f"警告: ディレクトリ '{path}' が見つかりません。スキップします。")
                continue
            
            print(f"'{path}' から.txtファイルを読み込み中...")
            from langchain_community.document_loaders import TextLoader
            loader = DirectoryLoader(
                path,
                glob=GLOB_PATTERN,
                recursive=True,
                show_progress=True,
                use_multithreading=True,
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'}
            )
            documents = loader.load()
            all_documents.extend(documents)

        if not all_documents:
            print("エラー: 対象ディレクトリに.txtファイルが見つかりません。処理を中断します。")
            return

        # テキストの分割
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(all_documents)
        
        # ベクトル化してChromaに保存
        print("データベースを構築中...（時間がかかる場合があります）")
        db = Chroma.from_documents(texts, embeddings, persist_directory=DB_PATH)
        db.persist()
        print("データベースの構築が完了しました。")
    else:
        print(f"既存のデータベースを'{DB_PATH}'からロードします。")
    
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    # 3. LLMとQAチェーンの準備
    llm = Ollama(model=LLM_MODEL_NAME,temperature=0.0)
    
    prompt_template = """
    あなたは、提供された「KMCの内部wikiというコンテキスト情報」だけを知識源とする、非常に厳格で忠実なアシスタントです。
    あなたの唯一のタスクは、以下のルールに従って内部wikiに書かれた内容に関するユーザーの質問に回答することです。

    # ルール
    - 回答は、必ず「コンテキスト情報」に書かれている内容のみを根拠としなければなりません。
    - コンテキスト情報に答えが記載されていない、または関連性が低い場合は、他の知識や推測を一切含めないでください。
    - 自身の意見や、コンテキスト外の情報を付け加えてはいけません。

    ---
    コンテキスト情報:
    {context}
    ---
    質問: {question}
    ---
    上記のルールに厳格に従った回答:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

    # 4. 質問応答ループ
    print("\n--- 質問応答を開始します --- (終了するには 'exit' と入力)")
    while True:
        query = input("\n質問を入力してください: ")
        if query.lower() == 'exit':
            break
        if not query.strip():
            continue

        print("\n回答を生成中...")
        result = qa_chain.invoke(query)
        
        print("\n✅ 回答:")
        print(result["result"])
        
        print("\n📚 参照した情報源:")
        sources = {doc.metadata.get("source", "N/A") for doc in result["source_documents"]}
        for src in sources:
            print(f"- {src}")


if __name__ == "__main__":
    main()