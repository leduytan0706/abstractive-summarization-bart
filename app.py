import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_USE_WATCHMAN"] = "false"

import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Tên model sử dụng
bart_finetuned_model_name = "leduytan0706/bart-large-cnn-xsum"
bart_model_name = "facebook/bart-large-cnn"
pegasus_model_name = "google/pegasus-cnn_dailymail"

st.set_page_config(page_title="Text Summarizer", layout="wide")

# Load model từ HuggingFace
@st.cache_resource
def load_model():
    finetuned_tokenizer = BartTokenizer.from_pretrained(bart_finetuned_model_name)
    finetuned_model = BartForConditionalGeneration.from_pretrained(bart_finetuned_model_name)
    tokenizer_bart = BartTokenizer.from_pretrained(bart_model_name)
    model_bart = BartForConditionalGeneration.from_pretrained(bart_model_name)
    tokenizer_pegasus = PegasusTokenizer.from_pretrained(pegasus_model_name)
    model_pegasus = PegasusForConditionalGeneration.from_pretrained(pegasus_model_name)
    model_bart.eval()
    model_pegasus.eval()
    return finetuned_tokenizer, finetuned_model, tokenizer_bart, model_bart, tokenizer_pegasus, model_pegasus

# Điều chỉnh số lượng token sinh tỉ lệ với token nhận vào
def modify_length(tokens_length, min_length, max_length):
    min_length_modified = min_length*1.2
    max_length_modified = max_length*1.2
    if (min_length > tokens_length):
        if summary_length == "Ngắn":
            min_length_modified = 30*1.2
            max_length_modified = 55*1.2
        elif summary_length == "Dài":
            min_length_modified = 60*1.2
            max_length_modified = 100*1.2
    return int(min_length_modified), int(max_length_modified)


# Tóm tắt văn bản bằng bart-large-cnn
def summarize_text_bart(text,  min_length, max_length, type="facebook/bart-large-cnn",):
    # Tokenization: Chia văn bản thành các token (từ, kí tự, ...)
    bart_tokenizer = finetuned_tokenizer if type == bart_finetuned_model_name else tokenizer_bart
    bart_model = finetuned_model if type == bart_finetuned_model_name else model_bart

    inputs = bart_tokenizer(text, return_tensors="pt", max_length=2048, truncation=True)
    input_ids = inputs["input_ids"]
    input_tokens = bart_tokenizer.convert_ids_to_tokens(input_ids[0])
    min_length, max_length = modify_length(len(input_ids[0]), min_length, max_length)

    # Sinh tóm tắt
    outputs = bart_model.generate(
        input_ids,
        attention_mask=inputs['attention_mask'],
        num_beams=4,
        max_length=max_length,
        min_length=min_length,
        early_stopping=True,
        output_hidden_states=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer_bart.pad_token_id,
        no_repeat_ngram_size=3,
        repetition_penalty=1.5
    )

    # Lấy trạng thái cuối cùng của encoder
    encoder_hidden_states = outputs.encoder_hidden_states
    summary_ids = outputs.sequences
    # print(summary_ids)
    # Lấy các token tóm tắt được sinh ra
    summary_tokens = bart_tokenizer.convert_ids_to_tokens(summary_ids[0], skip_special_tokens=True)
    # Giải mã/Ghép các token thành văn bản tóm tắt
    summary = bart_tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    return {
            "input_tokens": input_tokens,
            "encoder_hidden_states": encoder_hidden_states,
            "summary_tokens": summary_tokens,
            "summary": summary
        }

def summarize_text_pegasus(text, min_length, max_length):
    # Tokenization: Chia văn bản thành các token (từ, kí tự, ...)
    inputs = tokenizer_pegasus(text, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"]
    input_tokens = tokenizer_pegasus.convert_ids_to_tokens(input_ids[0])
    attention_mask = inputs["attention_mask"]
    min_length, max_length = modify_length(len(input_ids[0]), min_length, max_length)

    with torch.no_grad():
        # Đưa các token vào Encoder
        encoder_outputs = model_pegasus.model.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

    # Lấy trạng thái cuối cùng của encoder
    encoder_hidden_states = encoder_outputs.hidden_states

    # Sinh tóm tắt
    summary_ids = model_pegasus.generate(
        encoder_outputs=encoder_outputs,  # Dùng encoder_outputs này
        attention_mask=attention_mask,                 # Top-p (nucleus) sampling — keeps only the best next words
        do_sample=False,             # Tắt sampling để model nghiêm túc hơn
        num_beams=4,                  # Beam search để chọn câu văn mượt hơn
        no_repeat_ngram_size=3,       # Tránh lặp từ 3 cụm trở lên
        repetition_penalty=2.0,       # Hạn chế lặp
        length_penalty=1.0,           # Không ưu tiên quá dài/quá ngắn
        early_stopping=True,
        max_length=max_length,               # Giới hạn chiều dài hợp lý
        min_length=min_length,                # Ép model viết đoạn tử tế
        pad_token_id=tokenizer_pegasus.pad_token_id
    )
    # print(len(summary_ids[0]))

    # Lấy các token tóm tắt được sinh ra
    summary_tokens = tokenizer_pegasus.convert_ids_to_tokens(summary_ids[0], skip_special_tokens=True)
    # Giải mã/Ghép các token thành văn bản tóm tắt
    summary = tokenizer_pegasus.decode(summary_ids[0], skip_special_tokens=True)

    return {
        "input_tokens": input_tokens,
        "encoder_hidden_states": encoder_hidden_states,
        "summary_tokens": summary_tokens,
        "summary": summary.replace("<n>", "")
    }

# Vẽ biểu đồ thể hệ mối quan hệ giữa các token trước và sau khi vào Encoder
def plot_hidden_states(hidden_states, tokens):
    # Loại bỏ shape thứ nhất (batch size)
    first_state = hidden_states[0].squeeze(0)
    last_state = hidden_states[-1].squeeze(0)


    def pca_reduce(hidden_state):
        pca = PCA(n_components=2)
        return pca.fit_transform(hidden_state.numpy())

    def plot_state(reduced, title, xlabel="PCA 1", ylabel="PCA 2"):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6)
        for i, token in enumerate(tokens):
            ax.annotate(token, (reduced[i, 0], reduced[i, 1]))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        return fig

    fig1 = plot_state(pca_reduce(first_state), "Trước layer Encoder thứ nhất (positional + token embedding)")
    st.pyplot(fig1)

    fig2 = plot_state(pca_reduce(last_state), "Layer Encoder cuối cùng")
    st.pyplot(fig2)

def handle_submit(text, model_name, min_length, max_length):
    if model_name == bart_model_name or model_name == bart_finetuned_model_name:
        return summarize_text_bart(text, min_length, max_length, model_name)
    elif model_name == pegasus_model_name:
        return summarize_text_pegasus(text, min_length, max_length)



st.html("<h1 style='text-align:center; font-size: 2rem;'>Tóm tắt văn bản tiếng Anh hướng tóm lược - Abstractive Summarization</h1>")
st.html("<h3>Các mô hình sử dụng</h3>")
st.html("<p><strong>leduytan0706/bart-large-cnn-xsum</strong>: mô hình bart-large được tôi fine-tuned trên hỗn hợp dữ liệu giữa argilla/cnn-dailymail-summaries và EdinburghNLP/xsum<p>")
st.html("<p><strong>facebook/bart-large-cnn</strong>: Mô hình bart-large được huấn luyện trên tập dữ liệu CNN_Dailymail<p>")
st.html("<p><strong>google/pegasus-cnn_dailymail</strong>: Mô hình pegasus-large được huấn luyện trên tập dữ liệu CNN_Dailymail<p>")
st.html("<p>Các mô hình trên đều có thể được tìm thấy trên HuggingFace<p>")
finetuned_tokenizer, finetuned_model, tokenizer_bart, model_bart, tokenizer_pegasus, model_pegasus = load_model()


input_tokens = None
encoder_hidden_states = None
summary_tokens = None
summary = None

button_state = False
summary_length = "Ngắn"
show_process = False
button_state = False


# Điều chỉnh độ dài tóm tắt
empty_col, button_col = st.columns([1, 1])
with button_col:
    button_state = st.button("Tóm tắt")
col1, col2 = st.columns(2)
with col1:
    text = st.text_area("Nhập nội dung cần tóm tắt:", height=300)
    length_col, model_col, process_col = st.columns(3)
    with length_col:
        summary_length = st.select_slider(
            "Chọn độ dài cho tóm tắt",
            options=[
                "Ngắn",
                "Dài"
            ],
            value=("Ngắn")
        )
    with model_col:
        model_name = st.selectbox("Chọn mô hình tóm tắt", options=[bart_finetuned_model_name, bart_model_name, pegasus_model_name])
    with process_col:
        show_process = st.checkbox("Hiển thị quá trình", value=False)
with col2:
    if button_state:
        if text.strip() == "":
            st.text_area("⚠️ Vui lòng nhập văn bản trước.", height=300, max_chars=0)
        else:
            min_length = 40
            max_length = 80
            if summary_length == "Dài":
                min_length = 70
                max_length = 120
            sum_result = handle_submit(text, model_name, min_length, max_length)
            if show_process:
                input_tokens = sum_result["input_tokens"]
                encoder_hidden_states = sum_result["encoder_hidden_states"]
                summary_tokens = sum_result["summary_tokens"]
            summary = sum_result["summary"]
            st.text_area("Kết quả tóm tắt", value=summary, height=300, max_chars=len(summary))
    else:
        st.text_area("Kết quả tóm tắt", height=300, disabled=True)

if show_process and button_state:
    st.subheader("Quy trình xử lý", divider="gray")
    st.html("<h4>1. Tokenization - Tách văn bản thành các token (từ, kí tự đặc biệt, ...) </h4>")
    if input_tokens:
        st.text(input_tokens)
    st.html("<hr />")
    st.html("<h4>2. Encoder - Các token được mã hóa thành các trạng thái ẩn (hidden states)</h4>")
    if encoder_hidden_states is not None:
        st.text("Đồ thị scatter plot biểu diễn các token trước khi vào encoder (sau khi đã qua token embedding và positional encoding) và sau khi ra khỏi layer cuối cùng của encoder")
        st.text("Mỗi token trong văn bản tương ứng với một điểm trên scatter plot. Vị trí của điểm được xác định bởi biểu diễn ngữ nghĩa (semantic vector) của token đó, sau khi giảm từ 1024 chiều còn 2 chiều.")
        st.text("Trước khi vào Encoder, các token được biểu diễn dựa trên mối quan hệ tương đồng về ngữ nghĩa tự nhiên, chưa bao gồm ngữ cảnh trong văn bản.")
        st.text("Sau khi ra khỏi các layer của Encoder, các token được biểu diễn dựa trên mối quan hệ tương đồng về ngữ nghĩa dựa trên nhiều loại ngữ cảnh trong văn bản.")
        plot_hidden_states(encoder_hidden_states, input_tokens)
        st.text("Các điểm càng gần nhau thì có mối quan hệ ngữ nghĩa càng mật thiết với nhau.")

    else:
        st.text("Có sự cố xảy ra khi biểu diễn dữ liệu bước này.")
    st.html("<hr />")
    st.html("<h4>3. Sinh ra các token tóm tắt - Summary tokens generation")
    if summary_tokens is not None:
        st.text(f"Có tổng cộng {len(summary_tokens)} được sinh ra")
        st.text(summary_tokens)
    else:
        st.text("Có sự cố xảy ra khi biểu diễn dữ liệu bước này.")
    st.html("<hr />")
    st.html("<h4>4. Bản tóm tắt được sinh ra - Final summary")
    if summary is not None:
        st.text(summary)
    else:
        st.text("Có sự cố xảy ra khi biểu diễn dữ liệu bước này.")
            


        
