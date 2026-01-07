from dotenv import load_dotenv

load_dotenv()

import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


EXPERT_SYSTEM_MESSAGES: dict[str, str] = {
    "プロダクトマネージャー": "あなたは経験豊富なプロダクトマネージャーです。ユーザー価値、要件の整理、優先度付け、リスクと仮説検証の観点で、簡潔に実行可能な提案をしてください。",
    "ソフトウェアアーキテクト": "あなたは熟練のソフトウェアアーキテクトです。前提の確認、設計方針、トレードオフ、拡張性・保守性・セキュリティを重視して提案してください。必要に応じて疑問点を質問してください。",
    "データサイエンティスト": "あなたは実務経験豊富なデータサイエンティストです。分析設計、特徴量、評価指標、バイアス、運用（MLOps）まで見据えた提案をしてください。",
}


def ask_llm(input_text: str, expert_type: str) -> str:
    """入力テキストと専門家種別を受け取り、LLM からの回答テキストを返す。"""

    if not input_text.strip():
        return "入力テキストが空です。質問を入力してください。"

    system_message = EXPERT_SYSTEM_MESSAGES.get(
        expert_type, "あなたは親切で有能なアシスタントです。"
    )

    # OPENAI_API_KEY は .env から読み込まれる想定
    if not os.getenv("OPENAI_API_KEY"):
        return (
            "OPENAI_API_KEY が未設定です。.env に OPENAI_API_KEY を設定した上で、"
            "アプリを再起動してください。"
        )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm
    result = chain.invoke({"input": input_text})
    return result.content


st.set_page_config(page_title="LangChain x Streamlit チャット", layout="centered")

st.title("LangChain x Streamlit: 専門家に相談")
st.markdown(
    """
このアプリは、入力したテキストを **LangChain** 経由で LLM に渡し、回答を表示します。

### 使い方
1. **専門家の種類**（ラジオボタン）を選びます。
2. 下の **入力フォーム** に質問/相談内容を入力します。
3. **送信** を押すと、選択した専門家として回答します。
"""
)

expert_type = st.radio(
    "専門家の種類",
    options=list(EXPERT_SYSTEM_MESSAGES.keys()),
    horizontal=True,
)

with st.form("question_form"):
    input_text = st.text_area("入力テキスト", height=160, placeholder="例: 新規アプリのMVP要件を整理したい")
    submitted = st.form_submit_button("送信")

if submitted:
    with st.spinner("LLMに問い合わせ中..."):
        answer = ask_llm(input_text=input_text, expert_type=expert_type)

    st.subheader("回答")
    st.write(answer)