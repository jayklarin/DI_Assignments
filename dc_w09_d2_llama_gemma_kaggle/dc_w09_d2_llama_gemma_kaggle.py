import streamlit as st
from llama_cpp import Llama
import os


# ---------------------------
# Model Configuration
# ---------------------------
MODEL_PATH = "models/gemma/gemma-3-1b-it.Q8_0.gguf"


# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("⚙️ Generation Settings")

temperature = st.sidebar.slider(
    "Temperature (randomness)",
    0.0, 1.5, 0.7, 0.05
)

max_tokens = st.sidebar.slider(
    "Max Tokens",
    50, 1024, 200, 25
)

n_gpu_layers = st.sidebar.slider(
    "GPU Layers",
    0, 40, 0, 1,
    help="Use >0 only on GPU-enabled machines."
)

st.sidebar.markdown("---")

# Clear Chat Button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state["messages"] = []
    st.rerun()



# ---------------------------
# Session State Initialization
# ---------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# ---------------------------
# Streaming Helper
# ---------------------------
def stream_text_streamlit(llm, prompt, max_tokens=200, temperature=0.7):
    placeholder = st.empty()
    full_output = ""

    output_stream = llm(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True
    )

    for token_info in output_stream:
        piece = token_info["choices"][0]["text"]
        full_output += piece
        placeholder.markdown(full_output)

    return full_output


# ---------------------------
# Model Loader
# ---------------------------
@st.cache_resource
def load_model_with_gpu(n_gpu_layers):
    return Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=n_gpu_layers,
        n_threads=os.cpu_count()
    )

llm = load_model_with_gpu(n_gpu_layers)


# ---------------------------
# UI Layout (Header)
# ---------------------------
st.markdown(
    """
    <div style='text-align: center; padding-bottom: 10px;'>
        <h1>💬 Gemma 3B — Local LLM Chat</h1>
        <p style='font-size: 18px; color: #888;'>Powered by <b>llama.cpp</b> and <b>GGUF models</b></p>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------
# Chat History Display
# ---------------------------
st.markdown("### Conversation")
st.markdown("---")

for msg in st.session_state["messages"]:
    role_label = "🧑‍💻 **You**" if msg["role"] == "user" else "🤖 **Assistant**"
    st.markdown(f"{role_label}:")
    st.markdown(msg["content"])
    st.markdown("<br>", unsafe_allow_html=True)

st.markdown("---")


# ---------------------------
# Prompt Input + Buttons
# ---------------------------
st.markdown("### Send a Message")
prompt = st.text_area("Enter your prompt:", height=180)

col1, col2 = st.columns(2)
with col1:
    run_normal = st.button("Generate (non-streaming)")
with col2:
    run_stream = st.button("Generate (streaming)")


# ---------------------------
# Generation Logic
# ---------------------------

# Non-streaming
if run_normal:
    if prompt.strip():
        st.session_state["messages"].append({"role": "user", "content": prompt})

        with st.spinner("Generating response..."):
            output = llm(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            text = output["choices"][0]["text"]

        st.session_state["messages"].append({"role": "assistant", "content": text})
        st.rerun()

    else:
        st.warning("Please enter a prompt first.")

# Streaming
if run_stream:
    if prompt.strip():
        st.session_state["messages"].append({"role": "user", "content": prompt})

        with st.spinner("Streaming response..."):
            text = stream_text_streamlit(
                llm,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )

        st.session_state["messages"].append({"role": "assistant", "content": text})
        st.rerun()

    else:
        st.warning("Please enter a prompt first.")
