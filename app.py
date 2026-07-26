import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import shap
import streamlit.components.v1 as components

st.set_page_config(page_title="Review Authenticity Engine", page_icon="🔍", layout="centered")

st.markdown("""
<style>
.main { padding-top: 2rem; }
.stTextArea textarea { font-size: 15px; border-radius: 10px; }
.result-card {
    padding: 1.2rem 1.5rem; border-radius: 12px; margin: 1rem 0;
    font-size: 1.1rem; font-weight: 600; display: flex; align-items: center; gap: 0.6rem;
}
.fake-card { background: linear-gradient(135deg, #3a1010, #2a0a0a); border: 1px solid #ff4b4b; color: #ff6b6b; }
.real-card { background: linear-gradient(135deg, #0f2a15, #0a1f0f); border: 1px solid #2ecc71; color: #4ade80; }
.badge {
    display: inline-block; padding: 0.25rem 0.75rem; border-radius: 20px;
    background: #1e3a5f; color: #7dd3fc; font-size: 0.8rem; font-weight: 500; margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained("Smitvkohale/review-authenticity-engine")
    model = DistilBertForSequenceClassification.from_pretrained("Smitvkohale/review-authenticity-engine")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

def predict_proba(texts):
    enc = tokenizer(list(texts), truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    return torch.softmax(logits, dim=-1).numpy()

st.markdown('<div class="badge">🛡️ DistilBERT · Fine-tuned on synthetic fraud rings</div>', unsafe_allow_html=True)
st.title("Review Authenticity Engine")
st.write("Paste a product review below to check its fake-review probability and see which words influenced the score.")

review_text = st.text_area("Review text", height=150, placeholder="e.g. Amazing product highly recommend buy it now best purchase ever")

col1, col2 = st.columns([1, 4])
with col1:
    analyze = st.button("🔍 Analyze Review", use_container_width=True)

if analyze and review_text.strip():
    with st.spinner("Analyzing..."):
        probs = predict_proba([review_text])[0]
        fake_prob = probs[1]

        st.subheader("Result")
        if fake_prob > 0.5:
            st.markdown(f'<div class="result-card fake-card">⚠️ Likely FAKE — {fake_prob:.1%} confidence</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-card real-card">✅ Likely REAL — {(1-fake_prob):.1%} confidence</div>', unsafe_allow_html=True)

        st.subheader("Why the model made this decision")
        st.caption("Words in red pushed the score toward FAKE; words in blue pushed toward REAL.")
        explainer = shap.Explainer(predict_proba, shap.maskers.Text(tokenizer))
        shap_values = explainer([review_text])
        html_output = shap.plots.text(shap_values[0, :, 1], display=False)
        components.html(html_output, height=200, scrolling=True)

st.markdown("---")
st.caption("Part of **FraudScope** — a two-layer fake review & coordination detection system (text classifier + reviewer network analysis). [GitHub repo link here]")