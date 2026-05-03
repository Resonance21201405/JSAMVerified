"""
app.py — BIS Standards Recommendation Engine · Web Interface (v5 — with moving avatar)
=======================================================================================
Ultra-premium Streamlit app with moving animated avatar that roams the page
"""

import os, sys, json, time, re, logging
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("groq").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BIS Standards Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# MOVING AVATAR COMPONENT
# ══════════════════════════════════════════════════════════════════════════════
MOVING_AVATAR = """
<div id="movingAvatar" style="position:fixed;z-index:2000;pointer-events:all;
  width:80px;height:80px;bottom:30px;right:30px;">
  
  <div style="position:relative;width:100%;height:100%;">
    <!-- Avatar Body -->
    <div id="avatarBody" style="position:absolute;top:0;left:0;width:100%;height:100%;
      background:linear-gradient(135deg, #00d4ff, #0096c7);
      border-radius:50%;display:flex;align-items:center;justify-content:center;
      box-shadow:0 0 30px rgba(0,212,255,0.5);
      animation:avatarFloat 3s ease-in-out infinite;
      cursor:pointer;user-select:none;">
      
      <!-- Avatar Face -->
      <div style="position:relative;width:100%;height:100%;
        background:linear-gradient(135deg, #00d4ff 0%, #0096c7 100%);
        border-radius:50%;display:flex;align-items:center;justify-content:center;">
        
        <!-- Eyes -->
        <div style="position:absolute;width:50%;height:20%;
          display:flex;justify-content:space-around;align-items:center;
          top:30%;left:25%;gap:8px;">
          <div style="width:10px;height:10px;background:#fff;border-radius:50%;
            box-shadow:0 0 5px rgba(255,255,255,0.8);
            animation:eyeBlink 2s ease-in-out infinite;"></div>
          <div style="width:10px;height:10px;background:#fff;border-radius:50%;
            box-shadow:0 0 5px rgba(255,255,255,0.8);
            animation:eyeBlink 2s ease-in-out infinite;animation-delay:0.2s;"></div>
        </div>
        
        <!-- Mouth -->
        <div style="position:absolute;bottom:25%;left:50%;transform:translateX(-50%);
          width:15px;height:8px;border:2px solid #fff;
          border-top:none;border-radius:0 0 15px 15px;
          animation:mouthSmile 2s ease-in-out infinite;"></div>
        
        <!-- Happy indicator particles -->
        <div style="position:absolute;top:-15px;left:10px;width:6px;height:6px;
          background:#ffeb3b;border-radius:50%;
          animation:happyFloat 2s ease-in-out infinite;
          box-shadow:0 0 8px #ffeb3b;"></div>
      </div>
    </div>
  </div>
  
  <!-- Styles for avatar animations -->
  <style>
    @keyframes avatarFloat {
      0%, 100% { transform:translateY(0px) scale(1); }
      50% { transform:translateY(-10px) scale(1.05); }
    }
    
    @keyframes eyeBlink {
      0%, 95%, 100% { transform:scaleY(1); }
      97.5% { transform:scaleY(0.1); }
    }
    
    @keyframes mouthSmile {
      0%, 100% { transform:translateX(-50%) scaleX(1); }
      50% { transform:translateX(-50%) scaleX(1.2); }
    }
    
    @keyframes happyFloat {
      0%, 100% { opacity:0; transform:translateY(0); }
      50% { opacity:1; transform:translateY(-15px); }
    }
    
    @keyframes moveAvatar {
      0% { bottom: 30px; right: 30px; }
      25% { bottom: 50%; right: 100px; }
      50% { bottom: 100px; right: 50%; }
      75% { bottom: 150px; right: 200px; }
      100% { bottom: 30px; right: 30px; }
    }
  </style>
</div>

<script>
(function() {
  const avatar = document.getElementById('movingAvatar');
  const avatarBody = document.getElementById('avatarBody');
  
  let x = window.innerWidth - 110;
  let y = window.innerHeight - 110;
  let vx = (Math.random() - 0.5) * 2;
  let vy = (Math.random() - 0.5) * 2;
  const speed = 0.8;
  
  // Store last position to determine direction
  let lastX = x;
  
  function animate() {
    // Update position
    x += vx * speed;
    y += vy * speed;
    
    // Bounce off edges with smooth movement
    if (x < 0 || x > window.innerWidth - 80) {
      vx *= -1;
      x = Math.max(0, Math.min(window.innerWidth - 80, x));
    }
    if (y < 0 || y > window.innerHeight - 80) {
      vy *= -1;
      y = Math.max(0, Math.min(window.innerHeight - 80, y));
    }
    
    // Random direction change occasionally
    if (Math.random() < 0.01) {
      vx = (Math.random() - 0.5) * 2;
      vy = (Math.random() - 0.5) * 2;
    }
    
    // Flip avatar based on direction
    if (x > lastX) {
      avatarBody.style.transform = 'scaleX(1)';
    } else if (x < lastX) {
      avatarBody.style.transform = 'scaleX(-1)';
    }
    lastX = x;
    
    // Apply position
    avatar.style.bottom = y + 'px';
    avatar.style.right = window.innerWidth - x - 80 + 'px';
    
    requestAnimationFrame(animate);
  }
  
  // Click interaction - avatar reacts
  avatarBody.addEventListener('click', function() {
    // Random celebration animation
    const celebration = [
      'scale(1.2)',
      'rotate(360deg)',
      'translateY(-50px)'
    ];
    const random = celebration[Math.floor(Math.random() * celebration.length)];
    
    avatarBody.style.transition = 'transform 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55)';
    avatarBody.style.transform = random;
    
    setTimeout(() => {
      avatarBody.style.transition = 'none';
      avatarBody.style.transform = 'scaleX(-1)';
    }, 600);
    
    // Change direction
    vx = (Math.random() - 0.5) * 3;
    vy = (Math.random() - 0.5) * 3;
  });
  
  // Responsive handling
  window.addEventListener('resize', () => {
    if (x > window.innerWidth - 80) x = window.innerWidth - 80;
    if (y > window.innerHeight - 80) y = window.innerHeight - 80;
  });
  
  animate();
})();
</script>
"""

components.html(MOVING_AVATAR, height=0, scrolling=False)


# ═══════════════════════════════════════════════════════════════���══════════════
# ENHANCED GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Outfit:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background: linear-gradient(135deg, #03060f 0%, #0a0f1f 50%, #030812 100%);
    background-attachment: fixed;
}

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

section[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #050d1f 0%, #030812 100%);
    border-right: 2px solid;
    border-image: linear-gradient(180deg, #00d4ff, #0077b6) 1;
}

section[data-testid="stSidebar"] * { color: #c8d8ee !important; }

section[data-testid="stSidebar"] label {
    color: #4a6fa5 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'Space Mono', monospace !important;
    transition: color 0.3s;
}

section[data-testid="stSidebar"] label:hover {
    color: #00d4ff !important;
    text-shadow: 0 0 10px rgba(0,212,255,0.3);
}

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 2px solid #0d2040 !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #4a6fa5 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    border-bottom: 3px solid transparent !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.3s !important;
}

.stTabs [aria-selected="true"] {
    color: #00d4ff !important;
    border-bottom-color: #00d4ff !important;
}

.stButton > button {
    background: linear-gradient(135deg, #0077b6, #00b4d8) !important;
    color: #ffffff !important;
    border: 1px solid rgba(0,212,255,0.3) !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    padding: 0.65rem 2rem !important;
    transition: all 0.3s !important;
    box-shadow: 0 0 25px rgba(0,180,216,0.25) !important;
}

.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 35px rgba(0,212,255,0.5) !important;
}

.stTextArea textarea, .stTextInput input {
    background: linear-gradient(135deg, #060f22, #08152e) !important;
    border: 1.5px solid #0d2040 !important;
    border-radius: 10px !important;
    color: #c8d8ee !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.95rem !important;
    transition: all 0.3s !important;
}

.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #00d4ff !important;
    box-shadow: 0 0 0 3px rgba(0,212,255,0.15) !important;
}

.stProgress > div > div > div { 
    background: linear-gradient(90deg, #0077b6, #00d4ff) !important;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff, #00d4ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.05;
    letter-spacing: -0.02em;
    animation: gradientShift 6s ease infinite;
}

@keyframes gradientShift {
    0%, 100% { filter: hue-rotate(0deg); }
    50% { filter: hue-rotate(15deg); }
}

.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #4a6fa5;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 0.8rem;
    animation: fadeInUp 0.8s ease forwards;
    opacity: 0;
    animation-delay: 0.3s;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    font-weight: 700;
    color: #00d4ff;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.7rem;
}

.section-label::before {
    content: '';
    display: inline-block;
    width: 24px;
    height: 2px;
    background: linear-gradient(90deg, #00d4ff, transparent);
    box-shadow: 0 0 10px rgba(0,212,255,0.5);
}

.result-card {
    background: linear-gradient(135deg, #060f22 0%, #08152e 100%);
    border: 1.5px solid #0d2040;
    border-radius: 14px;
    padding: 1.3rem 1.6rem;
    margin-bottom: 0.9rem;
    position: relative;
    overflow: hidden;
    transition: all 0.4s ease;
    animation: slideIn 0.5s ease forwards;
    opacity: 0;
}

@keyframes slideIn {
    from { opacity: 0; transform: translateY(20px) scale(0.95); }
    to { opacity: 1; transform: translateY(0) scale(1); }
}

.result-card:nth-child(1) { animation-delay: 0.05s; border-left: 4px solid #00d4ff; }
.result-card:nth-child(2) { animation-delay: 0.15s; border-left: 4px solid #0096c7; }
.result-card:nth-child(3) { animation-delay: 0.25s; border-left: 4px solid #0077b6; }
.result-card:nth-child(4) { animation-delay: 0.35s; border-left: 4px solid #005f99; }
.result-card:nth-child(5) { animation-delay: 0.45s; border-left: 4px solid #004775; }

.result-card:hover {
    border-color: rgba(0,212,255,0.5);
    box-shadow: 0 12px 48px rgba(0,212,255,0.18);
    transform: translateY(-6px);
}

.result-is {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: #00d4ff;
    margin: 0.4rem 0 0.1rem;
}

.result-title {
    font-size: 0.92rem;
    color: #c8d8ee;
    font-weight: 500;
}

.metric-card {
    background: linear-gradient(135deg, #060f22, #08152e);
    border: 1.5px solid #0d2040;
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    transition: all 0.3s ease;
    animation: bounceIn 0.6s ease;
}

@keyframes bounceIn {
    0% { opacity: 0; transform: scale(0.8); }
    50% { transform: scale(1.05); }
    100% { opacity: 1; transform: scale(1); }
}

.metric-card:hover {
    box-shadow: 0 16px 56px rgba(0,212,255,0.25);
    transform: translateY(-8px);
}

.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #ffffff;
}

.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #4a6fa5;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

.metric-pass { color: #00d4ff !important; }
.metric-fail { color: #ff4757 !important; }

.latency-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(0,212,255,0.12);
    border: 1.5px solid rgba(0,212,255,0.25);
    border-radius: 24px;
    padding: 0.4rem 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #00d4ff;
    font-weight: 700;
    box-shadow: 0 0 20px rgba(0,212,255,0.15);
    animation: glow 2s ease-in-out infinite;
}

@keyframes glow {
    0%, 100% { box-shadow: 0 0 20px rgba(0,212,255,0.15); }
    50% { box-shadow: 0 0 35px rgba(0,212,255,0.3); }
}

hr { 
    border-color: #0d2040 !important;
    margin: 1.5rem 0 !important;
}

.hit-badge {
    display: inline-block;
    background: rgba(0,212,255,0.15);
    color: #00d4ff;
    border: 1.5px solid rgba(0,212,255,0.4);
    font-size: 0.64rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    padding: 0.35rem 0.85rem;
    border-radius: 20px;
    text-transform: uppercase;
}

.miss-badge {
    display: inline-block;
    background: rgba(255,71,87,0.12);
    color: #ff4757;
    border: 1.5px solid rgba(255,71,87,0.3);
    font-size: 0.64rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    padding: 0.35rem 0.85rem;
    border-radius: 20px;
    text-transform: uppercase;
}

.eval-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}

.eval-table th {
    background: linear-gradient(135deg, #060f22, #08152e);
    color: #4a6fa5;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    padding: 0.9rem 1rem;
    text-align: left;
    border-bottom: 2px solid #0d2040;
}

.eval-table td {
    padding: 0.8rem 1rem;
    border-bottom: 1px solid rgba(13,32,64,0.5);
    color: #8baac8;
}

.eval-table tr:hover td {
    background: rgba(0,212,255,0.04);
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ANIMATED DATA FLOW VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
ANIMATED_DATA_FLOW = """
<canvas id="dataFlow" style="position:fixed;top:0;left:0;width:100%;height:100%;
  pointer-events:none;z-index:0;opacity:0.65;"></canvas>

<script>
(function(){
  const c = document.getElementById('dataFlow');
  const ctx = c.getContext('2d');
  let W = window.innerWidth;
  let H = window.innerHeight;
  let animId;
  
  const stages = [
    { x: W*0.1, y: H*0.3, label: 'QUERY', color: '#00d4ff', width: 80 },
    { x: W*0.25, y: H*0.5, label: 'BM25', color: '#0096c7', width: 70 },
    { x: W*0.4, y: H*0.3, label: 'FAISS', color: '#0077b6', width: 70 },
    { x: W*0.55, y: H*0.5, label: 'RRF', color: '#005f99', width: 60 },
    { x: W*0.7, y: H*0.3, label: 'LLM', color: '#004775', width: 60 },
    { x: W*0.85, y: H*0.5, label: 'RESULT', color: '#00d4ff', width: 80 },
  ];
  
  let pathT = 0;
  
  function resize() {
    W = c.width = window.innerWidth;
    H = c.height = window.innerHeight;
  }
  
  function draw() {
    ctx.clearRect(0, 0, W, H);
    
    stages.forEach((stage, idx) => {
      const pulse = Math.sin(Date.now() * 0.005 + idx * 0.5) * 0.3 + 0.7;
      ctx.strokeStyle = stage.color + Math.floor(pulse * 200).toString(16).padStart(2, '0');
      ctx.lineWidth = 2 + pulse;
      ctx.beginPath();
      ctx.arc(stage.x, stage.y, 30, 0, Math.PI * 2);
      ctx.stroke();
      
      ctx.fillStyle = stage.color;
      ctx.font = 'bold 11px "Space Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(stage.label, stage.x, stage.y + 3);
    });
    
    for (let i = 0; i < stages.length; i++) {
      const from = stages[i];
      const to = stages[(i + 1) % stages.length];
      
      const grad = ctx.createLinearGradient(from.x, from.y, to.x, to.y);
      grad.addColorStop(0, from.color + '00');
      grad.addColorStop(0.5, from.color + '44');
      grad.addColorStop(1, to.color + '00');
      
      ctx.strokeStyle = grad;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.stroke();
    }
    
    pathT += 0.01;
    animId = requestAnimationFrame(draw);
  }
  
  window.addEventListener('resize', resize);
  
  function init() {
    resize();
    draw();
  }
  
  if (document.readyState === 'complete') init();
  else window.addEventListener('load', init);
})();
</script>
"""

components.html(ANIMATED_DATA_FLOW, height=0, scrolling=False)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Initialising BIS index…")
def load_agent(index_dir: str, llm_backend: str, groq_model: str, groq_key: str):
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
    from src.retriever import BISRetriever
    from src.agent    import BISAgent
    retriever = BISRetriever(index_dir=index_dir)
    agent     = BISAgent(retriever, llm_backend=llm_backend, groq_model=groq_model)
    return agent


def _norm(s: str) -> str:
    s = re.sub(r":\s*\d{4}", "", s)
    return re.sub(r"[^A-Z0-9]", "", s.upper())

def _match(expected: str, retrieved: str) -> bool:
    if _norm(expected) == _norm(retrieved):
        return True
    e_base = re.sub(r"[^A-Z0-9]", "", re.sub(r"\(.*?\)", "", expected).upper())
    r_base = re.sub(r"[^A-Z0-9]", "", re.sub(r"\(.*?\)", "", retrieved).upper())
    return e_base == r_base and "PART" not in _norm(expected)

def compute_metrics(results: list[dict]) -> dict:
    total = len(results)
    hit3 = mrr5 = lat = 0.0
    for r in results:
        exp = r.get("expected_standards", [])
        ret = r.get("retrieved_standards", [])
        lat += r.get("latency_seconds", 0)
        if exp and any(_match(e, rv) for e in exp for rv in ret[:3]):
            hit3 += 1
        for rank, rv in enumerate(ret[:5], 1):
            if exp and any(_match(e, rv) for e in exp):
                mrr5 += 1 / rank
                break
    return {
        "hit3":    round(hit3/total*100, 2) if total else 0,
        "mrr5":    round(mrr5/total, 4) if total else 0,
        "avg_lat": round(lat/total, 3) if total else 0,
        "total":   total,
        "hits":    int(hit3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="font-size:1.5rem;font-weight:800;background:linear-gradient(135deg, #ffffff, #00d4ff);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
      font-family:'Syne',sans-serif;letter-spacing:-0.02em;">
      BIS<span style="color:#00d4ff;">.</span>ENGINE
    </div>
    <div style="font-family:'Space Mono',monospace;font-size:0.62rem;color:#1a3a5c;
      text-transform:uppercase;letter-spacing:0.14em;margin-top:0.4rem;">
      Premium Recommendation System
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:0.65rem;
      color:#4a6fa5;text-transform:uppercase;letter-spacing:0.14em;margin-bottom:0.8rem;">
      ⚙ Configuration</div>
    """, unsafe_allow_html=True)

    index_dir  = st.text_input("Index directory", value="data/index")
    llm_backend = st.selectbox("LLM backend", ["groq","none","anthropic"])
    groq_model  = st.selectbox("Groq model",
                                ["llama-3.3-70b-versatile","llama-3.1-8b-instant"],
                                disabled=(llm_backend != "groq"))
    groq_key    = st.text_input("Groq API key",
                                 value=os.environ.get("GROQ_API_KEY",""),
                                 type="password",
                                 disabled=(llm_backend != "groq"))
    top_k = st.slider("Results per query", 3, 10, 5)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:0.62rem;
      color:#4a6fa5;text-transform:uppercase;letter-spacing:0.14em;
      margin-bottom:0.8rem;">Pipeline</div>
    """, unsafe_allow_html=True)

    pipeline_steps = ["PDF → JSON", "BM25 Keyword", "FAISS Semantic", "RRF Fusion", "LLM Rerank"]
    for i, step in enumerate(pipeline_steps):
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;">
          <div style="width:7px;height:7px;border-radius:50%;background:#00d4ff;
            animation:pulse {2 + i*0.5}s infinite;opacity:{1 - i*0.15};"></div>
          <div style="font-family:'Space Mono',monospace;font-size:0.62rem;
            color:#4a6fa5;">{step}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HERO SECTION
# ══════════════════════════════════════════════════════════════════════════════
col_hero, col_stats = st.columns([6, 4], gap="large")

with col_hero:
    st.markdown("""
    <div>
      <div class="hero-title">BIS Standards<br><span>Semantic Engine</span></div>
      <div class="hero-sub">🔬 Bureau of Indian Standards · SP 21</div>
      <div class="hero-sub" style="margin-top:0.4rem;color:#00d4ff;">⚡ AI-Powered Retrieval</div>
    </div>
    """, unsafe_allow_html=True)

with col_stats:
    st.markdown("""
    <div class="metric-card" style="margin-bottom:10px;animation-delay:0.1s;">
      <div class="metric-value">567</div>
      <div class="metric-label">IS Standards</div>
    </div>
    <div class="metric-card" style="margin-bottom:10px;animation-delay:0.2s;">
      <div class="metric-value">27</div>
      <div class="metric-label">Sections</div>
    </div>
    <div class="metric-card" style="animation-delay:0.3s;">
      <div class="metric-value">&lt;1s</div>
      <div class="metric-label">Latency</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_single, tab_batch = st.tabs(["  ⚡  Single Query  ", "  📊  Batch & Evaluate  "])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — SINGLE QUERY
# ─────────────────────────────────────────────────────────────────────────────
with tab_single:
    st.markdown('<div class="section-label">Semantic Search Interface</div>', unsafe_allow_html=True)

    col_q, col_ex = st.columns([7, 3], gap="large")

    with col_ex:
        st.markdown("""
        <div style="font-family:'Space Mono',monospace;font-size:0.62rem;
          color:#4a6fa5;text-transform:uppercase;letter-spacing:0.1em;
          margin-bottom:0.6rem;">📚 Examples</div>
        """, unsafe_allow_html=True)

        example_queries = [
            "Select example…",
            "33 Grade OPC — chemical and physical requirements?",
            "Corrugated asbestos cement sheets for roofing.",
            "Hollow and solid lightweight concrete blocks.",
            "TMT ribbed bars for reinforcing concrete.",
            "Bitumen felt as waterproofing layer.",
            "Gun-metal gate valves used in plumbing.",
            "uPVC pipes for potable water supply.",
            "Glass wool slabs — thermal insulation.",
        ]
        selected = st.selectbox("Load example", example_queries, label_visibility="collapsed")

    with col_q:
        query_text = st.text_area(
            "Describe your product or compliance need",
            value="" if selected == example_queries[0] else selected,
            height=120,
            placeholder="Describe material properties…",
            label_visibility="visible",
        )
        run_single = st.button("⚡  Find Standards", key="run_single", use_container_width=True)

    if run_single:
        if not query_text.strip():
            st.warning("Please enter a query to search.")
        else:
            try:
                with st.spinner("🔍 Running retrieval pipeline…"):
                    agent   = load_agent(index_dir, llm_backend, groq_model, groq_key)
                    t0      = time.perf_counter()
                    results = agent.answer(query_text.strip(), top_k=top_k)
                    elapsed = round(time.perf_counter() - t0, 3)

                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:1.2rem;margin:1.4rem 0 1rem;">
                  <div class="latency-badge">⚡ {elapsed:.2f}s</div>
                  <div style="font-family:'Space Mono',monospace;font-size:0.68rem;
                    color:#4a6fa5;background:rgba(0,212,255,0.08);padding:0.4rem 0.9rem;
                    border-radius:20px;">✓ {len(results)} standards retrieved</div>
                </div>
                """, unsafe_allow_html=True)

                results_html = ""
                rank_colors  = ["#00d4ff","#0096c7","#0077b6","#005f99","#004775"]
                for i, r in enumerate(results):
                    color = rank_colors[min(i, len(rank_colors)-1)]
                    rat_html = f'<div style="margin-top:0.7rem;font-size:0.82rem;color:#7a95b5;font-style:italic;">"{r.get("rationale","")}"</div>' if r.get("rationale") else ""
                    results_html += f"""
                    <div class="result-card" style="border-left-color:{color}">
                      <div style="font-size:0.65rem;color:#4a6fa5;">#{i+1}</div>
                      <div class="result-is">{r['std_id']}</div>
                      <div class="result-title">{r.get('title','').title()}</div>
                      {rat_html}
                    </div>"""

                st.markdown(results_html, unsafe_allow_html=True)

                st.markdown("<hr>", unsafe_allow_html=True)
                st.download_button(
                    "⬇  Download Result (JSON)",
                    data=json.dumps([{
                        "id": "Q1",
                        "query": query_text.strip(),
                        "retrieved_standards": [r["std_id"] for r in results],
                        "latency_seconds": elapsed,
                    }], indent=2),
                    file_name="bis_result.json",
                    mime="application/json",
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — BATCH EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown('<div class="section-label">Batch Evaluation & Analysis</div>', unsafe_allow_html=True)

    col_upload, col_paste = st.columns([1, 1], gap="large")

    with col_upload:
        uploaded = st.file_uploader("📤 Upload JSON", type=["json"], label_visibility="collapsed")

    with col_paste:
        pasted = st.text_area("📋 Paste JSON", height=180, label_visibility="collapsed",
                               placeholder='[{"id":"Q1","query":"...","expected_standards":["IS 269"]}]')

    queries_data = None
    parse_error  = None
    if uploaded:
        try:    queries_data = json.loads(uploaded.read().decode("utf-8"))
        except Exception as e: parse_error = str(e)
    elif pasted.strip():
        try:    queries_data = json.loads(pasted.strip())
        except Exception as e: parse_error = str(e)

    if parse_error:
        st.error(f"JSON Parse Error: {parse_error}")

    if queries_data:
        st.markdown(f"""
        <div style="font-family:'Space Mono',monospace;font-size:0.75rem;
          color:#00d4ff;margin:1rem 0;background:rgba(0,212,255,0.08);
          padding:0.6rem 1rem;border-radius:8px;border-left:2px solid #00d4ff;">
          ✓ {len(queries_data)} queries loaded
        </div>
        """, unsafe_allow_html=True)

        with st.expander("👁️  Preview queries", expanded=False):
            for i, q in enumerate(queries_data[:5], 1):
                st.markdown(f"""
                <div style="background:linear-gradient(135deg, #060f22, #08152e);
                  border:1.5px solid #0d2040;border-radius:10px;padding:1rem;
                  margin-bottom:0.6rem;">
                  <div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#00d4ff;">
                    {q.get('id','?')} - {q.get('query','')[:80]}…</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")
        run_batch = st.button("▶  Run Inference + Evaluate", key="run_batch", use_container_width=True)

        if run_batch:
            try:
                with st.spinner("📦 Loading index…"):
                    agent = load_agent(index_dir, llm_backend, groq_model, groq_key)

                results   = []
                progress  = st.progress(0)
                status_ph = st.empty()
                total_q   = len(queries_data)

                for i, item in enumerate(queries_data):
                    status_ph.text(f"[{i+1}/{total_q}] Processing {item.get('id','Q')}")
                    t0 = time.perf_counter()
                    try:    retrieved = agent.answer(item.get("query", ""), top_k=top_k)
                    except: retrieved = []
                    elapsed = round(time.perf_counter()-t0, 3)

                    result = {
                        "id": item.get("id", str(i+1)),
                        "query": item.get("query", ""),
                        "retrieved_standards": [r["std_id"] for r in retrieved],
                        "latency_seconds": elapsed,
                    }
                    if "expected_standards" in item:
                        result["expected_standards"] = item["expected_standards"]
                    results.append(result)
                    progress.progress((i+1)/total_q)

                status_ph.empty()
                progress.empty()
                st.markdown("<hr>", unsafe_allow_html=True)

                has_expected = any("expected_standards" in r for r in results)
                if has_expected:
                    metrics = compute_metrics(results)
                    st.markdown('<div class="section-label">Evaluation Metrics</div>', unsafe_allow_html=True)

                    c1, c2, c3, c4 = st.columns(4)
                    for col, val, label, fmt in [
                        (c1, metrics["hit3"], "Hit Rate @3", f"{metrics['hit3']:.1f}%"),
                        (c2, metrics["mrr5"], "MRR @5", f"{metrics['mrr5']:.4f}"),
                        (c3, metrics["avg_lat"], "Avg Latency", f"{metrics['avg_lat']:.2f}s"),
                        (c4, None, "Queries Hit", f"{metrics['hits']}/{metrics['total']}"),
                    ]:
                        with col:
                            st.markdown(f"""
                            <div class="metric-card">
                              <div class="metric-value">{fmt}</div>
                              <div class="metric-label">{label}</div>
                            </div>""", unsafe_allow_html=True)

                st.markdown("")
                clean_results = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
                st.download_button("⬇  Download Results",
                                   data=json.dumps(clean_results, indent=2),
                                   file_name="output.json",
                                   use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;font-family:'Space Mono',monospace;
  font-size:0.63rem;color:#4a6fa5;padding:1rem 0;letter-spacing:0.14em;">
  ✨ BIS Standards Engine — Ultra Premium Edition ✨<br>
  <span style="color:#1a3a5c;">Bureau of Indian Standards · SP 21 Building Materials</span>
</div>
""", unsafe_allow_html=True)