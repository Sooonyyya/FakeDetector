from flask import Flask, request, jsonify, render_template_string
import pickle
from preprocessing import preprocess_text, preprocess_review_text

app = Flask(__name__)

with open('model_news.pkl', 'rb') as f:
    news_vec, news_model = pickle.load(f)

with open('model_reviews.pkl', 'rb') as f:
    reviews_vec, reviews_model = pickle.load(f)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fake Content Detector</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0f0f13;
    color: #e8e8f0;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 2rem 1rem;
  }
  .header { text-align: center; margin-bottom: 2rem; }
  .header h1 { font-size: 2rem; font-weight: 700; letter-spacing: -0.5px; color: #ffffff; margin-bottom: 0.4rem; }
  .header h1 span { color: #6c63ff; }
  .header p { font-size: 0.9rem; color: #888; }
  .stats { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; justify-content: center; }
  .stat { background: #1a1a24; border: 1px solid #2a2a3a; border-radius: 10px; padding: 0.75rem 1.25rem; text-align: center; min-width: 130px; }
  .stat-val { font-size: 1.3rem; font-weight: 700; color: #6c63ff; }
  .stat-lbl { font-size: 0.75rem; color: #666; margin-top: 2px; }
  .card { background: #1a1a24; border: 1px solid #2a2a3a; border-radius: 16px; padding: 1.75rem; width: 100%; max-width: 640px; }
  .tabs { display: flex; gap: 8px; margin-bottom: 1.25rem; }
  .tab { flex: 1; padding: 10px; border: 1px solid #2a2a3a; border-radius: 10px; font-size: 0.9rem; font-weight: 600; cursor: pointer; background: transparent; color: #666; transition: all 0.2s; }
  .tab:hover { color: #aaa; border-color: #444; }
  .tab.active { background: #6c63ff22; color: #6c63ff; border-color: #6c63ff55; }
  textarea { width: 100%; min-height: 140px; resize: vertical; font-size: 0.9rem; padding: 12px 14px; border: 1px solid #2a2a3a; border-radius: 10px; font-family: inherit; color: #e8e8f0; background: #0f0f13; margin-bottom: 1rem; line-height: 1.6; outline: none; transition: border-color 0.2s; }
  textarea:focus { border-color: #6c63ff55; }
  textarea::placeholder { color: #444; }
  .btn { width: 100%; padding: 12px; font-size: 1rem; font-weight: 600; border-radius: 10px; border: none; background: #6c63ff; color: #fff; cursor: pointer; transition: all 0.2s; letter-spacing: 0.2px; }
  .btn:hover { background: #5a52e0; transform: translateY(-1px); }
  .btn:active { transform: translateY(0); }
  .btn:disabled { background: #333; color: #666; cursor: not-allowed; transform: none; }
  .result { margin-top: 1.25rem; padding: 1.25rem; border-radius: 12px; display: none; align-items: center; gap: 14px; animation: fadeIn 0.3s ease; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
  .result.real { background: #0d2e1a; border: 1px solid #1a5c32; }
  .result.fake { background: #2e0d0d; border: 1px solid #5c1a1a; }
  .result-icon { font-size: 2rem; }
  .result-label { font-size: 1.4rem; font-weight: 700; letter-spacing: 1px; }
  .result-label.real { color: #4ade80; }
  .result-label.fake { color: #f87171; }
  .result-desc { font-size: 0.82rem; color: #888; margin-top: 3px; }
  .confidence-bar { margin-top: 8px; height: 4px; border-radius: 2px; background: #2a2a3a; overflow: hidden; }
  .confidence-fill { height: 100%; border-radius: 2px; transition: width 0.4s ease; }
  .real .confidence-fill { background: #4ade80; }
  .fake .confidence-fill { background: #f87171; }
  .loading { text-align: center; padding: 1rem; color: #666; font-size: 0.9rem; display: none; margin-top: 1rem; }
  .dot { animation: blink 1.4s infinite; }
  .dot:nth-child(2) { animation-delay: 0.2s; }
  .dot:nth-child(3) { animation-delay: 0.4s; }
  @keyframes blink { 0%, 80%, 100% { opacity: 0; } 40% { opacity: 1; } }
  .footer { margin-top: 2rem; font-size: 0.78rem; color: #444; text-align: center; }
</style>
</head>
<body>

<div class="header">
  <h1>Fake Content <span>Detector</span></h1>
  <p>NLP-powered classifier using Naive Bayes &middot; Logistic Regression &middot; SVM</p>
</div>

<div class="stats">
  <div class="stat">
    <div class="stat-val">99.58%</div>
    <div class="stat-lbl">News accuracy</div>
  </div>
  <div class="stat">
    <div class="stat-val">93.14%</div>
    <div class="stat-lbl">Reviews accuracy</div>
  </div>
  <div class="stat">
    <div class="stat-val">44 898</div>
    <div class="stat-lbl">News trained on</div>
  </div>
  <div class="stat">
    <div class="stat-val">40 162</div>
    <div class="stat-lbl">Reviews trained on</div>
  </div>
</div>

<div class="card">
  <div class="tabs">
    <button class="tab active" onclick="setTab('news', this)">📰 News</button>
    <button class="tab" onclick="setTab('review', this)">💬 Review</button>
  </div>

  <textarea id="inputText" placeholder="Paste your news article here..."></textarea>

  <button class="btn" id="classifyBtn" onclick="classify()">Classify text</button>

  <div class="loading" id="loading">
    Analyzing<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span>
  </div>

  <div class="result" id="result">
    <div class="result-icon" id="resultIcon"></div>
    <div style="flex: 1;">
      <div class="result-label" id="resultLabel"></div>
      <div class="result-desc" id="resultDesc"></div>
      <div class="confidence-bar">
        <div class="confidence-fill" id="confidenceFill" style="width: 0%"></div>
      </div>
    </div>
  </div>
</div>

<div class="footer">
  Trained on Kaggle datasets &middot; Naive Bayes &middot; Logistic Regression &middot; SVM
</div>

<script>
  let currentTab = 'news';

  function setTab(tab, el) {
    currentTab = tab;
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    el.classList.add('active');
    document.getElementById('inputText').placeholder =
      tab === 'news' ? 'Paste your news article here...' : 'Paste your review text here...';
    document.getElementById('result').style.display = 'none';
  }

  async function classify() {
    const text = document.getElementById('inputText').value.trim();
    if (!text) return;

    const btn = document.getElementById('classifyBtn');
    const loading = document.getElementById('loading');
    const result = document.getElementById('result');

    btn.disabled = true;
    loading.style.display = 'block';
    result.style.display = 'none';

    try {
      const response = await fetch('/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text, type: currentTab })
      });

      const data = await response.json();
      const isFake = data.result === 'FAKE';
      const conf = data.confidence ? data.confidence : null;

      result.className = 'result ' + (isFake ? 'fake' : 'real');
      document.getElementById('resultIcon').textContent = isFake ? '⚠️' : '✅';
      document.getElementById('resultLabel').className = 'result-label ' + (isFake ? 'fake' : 'real');
      document.getElementById('resultLabel').textContent = data.result;

      const confText = conf ? ' · Confidence: ' + conf + '%' : '';
      document.getElementById('resultDesc').textContent =
        isFake
          ? 'This ' + currentTab + ' was classified as fake or computer-generated.' + confText
          : 'This ' + currentTab + ' was classified as authentic.' + confText;

      if (conf) {
        document.getElementById('confidenceFill').style.width = conf + '%';
      }

      result.style.display = 'flex';
    } catch (e) {
      alert('Error connecting to server.');
    }

    loading.style.display = 'none';
    btn.disabled = false;
  }
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    text = data.get('text', '')
    content_type = data.get('type', 'news')

    if content_type == 'news':
        processed = preprocess_text(text)
        tfidf = news_vec.transform([processed])
        model = news_model
    else:
        processed = preprocess_review_text(text)
        tfidf = reviews_vec.transform([processed])
        model = reviews_model

    prediction = model.predict(tfidf)[0]

    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(tfidf)[0]
        confidence = int(max(proba) * 100)
    elif hasattr(model, 'decision_function'):
        score = abs(model.decision_function(tfidf)[0])
        confidence = min(int(50 + score * 15), 99)
    else:
        confidence = None

    result = 'FAKE' if prediction == 1 else 'REAL'
    return jsonify({'result': result, 'confidence': confidence})

if __name__ == '__main__':
    print("Starting Fake Content Detector...")
    print("Open your browser at: http://127.0.0.1:5000")
    app.run(debug=False)