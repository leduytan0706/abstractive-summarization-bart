import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_USE_WATCHMAN"] = "false"

import streamlit as st
from transformers import AutoTokenizer, BartForConditionalGeneration
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Text Summarizer", layout="wide")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn", attn_implementation="eager")
    return tokenizer, model

st.title("Tóm tắt văn bản theo hướng tóm lược (Abstractive Summarization) với BART")
tokenizer, model = load_model()
def modify_length(min_length, max_length):
    return int(min_length*1.5), int(max_length*1.5)

input_ids = None
encoder_hidden_states = None
summary_ids = None
summary = None

def summarize_text(text, min_length, max_length, show_process = False):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    input_ids = inputs["input_ids"]
    min_length, max_length = modify_length(min_length, max_length)
    if show_process:
        outputs = model.generate(
            input_ids,
            num_beams=2,
            max_length=max_length,
            min_length=min_length,
            early_stopping=True,
            do_sample=False,
            output_attentions=True,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
        encoder_hidden_states = outputs.encoder_hidden_states
        summary_ids = outputs.sequences
        summary = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        return input_ids, encoder_hidden_states, summary_ids, summary
    else:
        summary_ids = model.generate(
            input_ids,
            num_beams=2,
            max_length=max_length,
            min_length=min_length,
            early_stopping=True,
            do_sample=False,
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

def plot_hidden_states(hidden_states, tokens):
    # Loại bỏ shape thứ nhất (batch size)
    first_state = hidden_states[0].squeeze(0) 
    last_state = hidden_states[-1].squeeze(0) 


    def pca_reduce(hidden_state):
        pca = PCA(n_components=2)
        return pca.fit_transform(hidden_state.numpy())
    
    def plot_state(reduced, title, xlabel="PCA 1", ylabel="PCA 2"):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6)
        for i, token in enumerate(tokens):
            ax.annotate(token, (reduced[i, 0], reduced[i, 1]))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        return fig

    col1, col2 = st.columns(2)
    with col1:
        fig1 = plot_state(pca_reduce(first_state), "Layer Encoder thứ nhất (positional + token embedding)")
        st.pyplot(fig1)
    with col2:
        fig2 = plot_state(pca_reduce(last_state), "Layer Encoder cuối cùng")
        st.pyplot(fig2)
    

button_state = False
max_length = 130
min_length = 30
show_process = True
button_state = False
# Điều chỉnh độ dài tóm tắt
empty_col, button_col = st.columns([1, 1])
with button_col:
    button_state = st.button("Tóm tắt")
col1, col2 = st.columns(2)
with col1:
    text = st.text_area("Nhập nội dung cần tóm tắt:", height=300)
    max_col, min_col, checkbox_col = st.columns(3)
    with max_col: 
        max_length = st.slider("Độ dài tối đa của đoạn tóm tắt", 30, 300, 130)
    with min_col:
        min_length = st.slider("Độ dài tối thiểu của đoạn tóm tắt", 10, 100, 30)
    with checkbox_col:
        show_process = st.checkbox("Hiển thị quá trình", value=True)
with col2:
    if button_state:
        if text.strip() == "":
            st.text_area("⚠️ Vui lòng nhập văn bản trước.", height=300, max_chars=0)
        else:
            if show_process:
                input_ids, encoder_hidden_states, summary_ids, summary = summarize_text(text, min_length, max_length, show_process)
            # print(type(cross_attentions))  # List
            # print(type(cross_attentions[0]))  # Nếu là tuple → chính xác nguồn lỗi
            else:
                summary = summarize_text(text, min_length, max_length, show_process)
            st.text_area("Kết quả tóm tắt", value=summary, height=300, max_chars=len(summary))
    else:
        st.text_area("Kết quả tóm tắt", height=300, disabled=True)
    
if show_process and button_state:
    st.subheader("Quy trình xử lý", divider="gray")
    st.html("<h4>1. Tokenization - Tách văn bản thành các token (từ, kí tự đặc biệt, ...) </h4>")
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    st.text(tokens)
    st.text("Loại bỏ các suffix")
    formatted_tokens = [token.replace("Ġ","") for token in tokens]
    st.text(formatted_tokens[1:-1])
    st.html("<hr />")
    st.html("<h4>2. Encoder - Các token được mã hóa thành các trạng thái ẩn (hidden states)</h4>")
    if encoder_hidden_states is not None:
        st.text("Đồ thị scatter plot biểu diễn các token trước khi vào encoder (sau khi đã qua token embedding và positional encoding) và sau khi ra khỏi layer cuối cùng của encoder")
        st.text("Mỗi token trong văn bản tương ứng với một điểm trên scatter plot. Vị trí của điểm được xác định bởi biểu diễn ngữ nghĩa (semantic vector) của token đó, sau khi giảm từ 1024 chiều còn 2 chiều.")
        plot_hidden_states(encoder_hidden_states, tokens)
        st.text("Các điểm càng gần nhau thì có mối quan hệ ngữ nghĩa càng mật thiết với nhau.")
    else:
        st.text("Có sự cố xảy ra khi biểu diễn dữ liệu bước này.")
    st.html("<hr />")
    st.html("<h4>3. Sinh ra các từ tóm tắt - Summary tokens generation")
    if summary_ids is not None:
        st.text(f"Có tổng cộng {len(summary_ids)} được sinh ra")
        summary_tokens = tokenizer.convert_ids_to_tokens(summary_ids[0], skip_special_tokens=True)
        st.text(summary_tokens)
        formatted_summary_tokens = [token.replace("Ġ","") for token in summary_tokens]
        st.text(formatted_summary_tokens[1:-1])
    else:
        st.text("Có sự cố xảy ra khi biểu diễn dữ liệu bước này.") 
    st.html("<hr />")
    st.html("<h4>4. Bản tóm tắt được sinh ra - Final summary")  
    if summary is not None:
        st.text(summary)
    else:
        st.text("Có sự cố xảy ra khi biểu diễn dữ liệu bước này.") 

            


        
