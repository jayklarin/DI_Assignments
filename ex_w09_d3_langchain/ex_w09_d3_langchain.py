# ex_w09_d3_langchain.py
# Full working Streamlit chatbot with guaranteed final message display.

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# ---------- Setup ----------
load_dotenv()

model = ChatMistralAI(
    model="open-mistral-7b",
    api_key=os.getenv("MISTRAL_API_KEY")
)
search = TavilySearchResults()
tools = [search]
agent_executor = create_react_agent(model=model, tools=tools)

st.set_page_config(page_title="LangChain AI Chatbot", page_icon="🔗", layout="wide")
st.title("🔗 LangChain AI Chatbot")
st.caption("Mistral + Tavily + LangGraph demo")

user_input = st.text_area("💬 Ask a question:", placeholder="e.g. What are recent AI breakthroughs in medicine?")
run_button = st.button("Send")

# ---------- Main ----------
if run_button and user_input.strip():
    st.write("### 🧠 Response:")
    placeholder = st.empty()
    partial = ""
    messages = [HumanMessage(content=user_input)]

    # --- 1️⃣ Stream intermediate steps for visibility ---
    try:
        for step in agent_executor.stream({"messages": messages}, stream_mode="auto"):
            if "tool_calls" in step:
                st.info(f"🔧 Tool call: {step['tool_calls']}")
            elif "actions" in step:
                st.info(f"🧩 Action: {step['actions']}")
            elif "messages" in step:
                msg = step["messages"][0]
                partial = msg.content
                placeholder.markdown(partial + "▌")
    except Exception as e:
        st.error(f"⚠️ Stream error: {e}")

    # --- 2️⃣ Always fetch final result explicitly ---
    try:
        final_state = agent_executor.invoke({"messages": messages})
        # depending on version this may be a dict or object
        if isinstance(final_state, dict):
            final_msg = final_state.get("messages", [])
            if final_msg:
                text = getattr(final_msg[-1], "content", None) or str(final_msg[-1])
            else:
                text = str(final_state)
        else:
            text = getattr(final_state, "content", str(final_state))

        placeholder.markdown(text)
    except Exception as e:
        st.error(f"⚠️ Final invoke error: {e}")

    st.success("✅ Done!")
else:
    st.write("👆 Enter a question and press **Send** to start chatting.")
