"""
app.py — BIS Standards Recommendation Engine · Web Interface (v4 — ultra-premium with animations)
=================================================================================================
Ultra-premium Streamlit app with animated visuals, dynamic data flow, and interactive elements
  1. Single query  — type a product description, get top-5 IS standards
  2. JSON batch    — upload / paste a JSON file, run inference + eval

Usage:
    pip install streamlit
    streamlit run app.py
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
    page_title="BIS Standards Engine — Ultra Premium",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# PREMIUM GLOBAL CSS WITH ANIMATIONS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Outfit:wght@300;400;500;600&display=swap');

/* ── Reset & base ── */
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

/* ── Sidebar Enhanced ── */
section[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #050d1f 0%, #030812 100%);
    border-right: 2px solid;
    border-image: linear-gradient(180deg, #00d4ff, #0077b6) 1;
    box-shadow: inset -20px 0 40px rgba(0,212,255,0.05);
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

section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] .stTextInput input {
    background: linear-gradient(135deg, #0a1628, #06101f) !important;
    border: 1px solid #1a3a5c !important;
    border-radius: 8px !important;
    color: #c8d8ee !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    transition: all 0.3s;
    box-shadow: 0 0 15px rgba(0,212,255,0.05) !important;
}

section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div:hover,
section[data-testid="stSidebar"] .stTextInput input:hover {
    border-color: #00d4ff !important;
    box-shadow: 0 0 20px rgba(0,212,255,0.15) !important;
    background: linear-gradient(135deg, #0f1a2e, #0a1628) !important;
}

/* ── Tabs Enhanced ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 2px solid #0d2040 !important;
    gap: 0;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #4a6fa5 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em;
    border-bottom: 3px solid transparent !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.3s !important;
}

.stTabs [aria-selected="true"] {
    color: #00d4ff !important;
    border-bottom-color: #00d4ff !important;
}

/* ── Buttons Enhanced ── */
.stButton > button {
    background: linear-gradient(135deg, #0077b6, #00b4d8, #0096c7) !important;
    background-size: 200% 200%;
    color: #ffffff !important;
    border: 1px solid rgba(0,212,255,0.3) !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.06em !important;
    padding: 0.65rem 2rem !important;
    transition: all 0.3s !important;
    box-shadow: 0 0 25px rgba(0,180,216,0.25), inset 0 1px 0 rgba(255,255,255,0.2) !important;
    position: relative;
    overflow: hidden;
}

.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 35px rgba(0,212,255,0.5), inset 0 1px 0 rgba(255,255,255,0.3) !important;
}

/* ── Text areas / inputs Enhanced ── */
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
    box-shadow: 0 0 0 3px rgba(0,212,255,0.15), inset 0 0 10px rgba(0,212,255,0.05) !important;
    background: linear-gradient(135deg, #08152e, #0a1a35) !important;
}

/* ── Expander Enhanced ── */
.streamlit-expanderHeader {
    background: linear-gradient(135deg, #060f22, #08152e) !important;
    border: 1.5px solid #0d2040 !important;
    border-radius: 10px !important;
    color: #c8d8ee !important;
    transition: all 0.3s !important;
}

.streamlit-expanderHeader:hover {
    border-color: #1a3a5c !important;
    box-shadow: 0 0 15px rgba(0,212,255,0.08) !important;
}

/* ── Progress bar Enhanced ── */
.stProgress > div > div > div { 
    background: linear-gradient(90deg, #0077b6, #00d4ff, #00b4d8) !important;
    background-size: 200% 100%;
    animation: shimmer 2s infinite !important;
}

@keyframes shimmer {
    0%, 100% { background-position: 200% center; }
    50% { background-position: 0% center; }
}

/* ── Alerts Enhanced ── */
.stAlert { 
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
    background: linear-gradient(135deg, rgba(6,15,34,0.8), rgba(8,21,46,0.8)) !important;
}

/* ── HR Enhanced ── */
hr { 
    border-color: #0d2040 !important;
    margin: 1.5rem 0 !important;
}

/* ═══════════════════════════════════════════════════════════════════════════════ */
/* CUSTOM COMPONENTS */
/* ═══════════════��═══════════════════════════════════════════════════════════════ */

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff, #00d4ff, #0096c7);
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
    animation: slideInRight 0.6s ease;
}

@keyframes slideInRight {
    from { opacity: 0; transform: translateX(-20px); }
    to { opacity: 1; transform: translateX(0); }
}

.section-label::before {
    content: '';
    display: inline-block;
    width: 24px;
    height: 2px;
    background: linear-gradient(90deg, #00d4ff, transparent);
    box-shadow: 0 0 10px rgba(0,212,255,0.5);
}

/* ═══════════════════════════════════════════════════════════════════════════════ */
/* RESULT CARDS */
/* ═══════════════════════════════════════════════════════════════════════════════ */

.result-card {
    background: linear-gradient(135deg, #060f22 0%, #08152e 100%);
    border: 1.5px solid #0d2040;
    border-radius: 14px;
    padding: 1.3rem 1.6rem;
    margin-bottom: 0.9rem;
    position: relative;
    overflow: hidden;
    transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
    animation: slideIn 0.5s ease forwards;
    opacity: 0;
    box-shadow: 0 8px 32px rgba(0,212,255,0.08);
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

.result-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.4), transparent);
    box-shadow: 0 0 20px rgba(0,212,255,0.3);
}

.result-card:hover {
    border-color: rgba(0,212,255,0.5);
    box-shadow: 0 12px 48px rgba(0,212,255,0.18);
    transform: translateY(-6px) scale(1.02);
    background: linear-gradient(135deg, #08152e 0%, #0a1a35 100%);
}

.result-rank {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #4a6fa5;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    background: rgba(0,212,255,0.08);
    padding: 0.3rem 0.7rem;
    border-radius: 4px;
    display: inline-block;
}

.result-is {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4ff, #0096c7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0.4rem 0 0.1rem;
    letter-spacing: -0.01em;
}

.result-title {
    font-size: 0.92rem;
    color: #c8d8ee;
    font-weight: 500;
    margin-top: 0.3rem;
}

.result-rationale {
    font-size: 0.82rem;
    color: #7a95b5;
    margin-top: 0.7rem;
    font-style: italic;
    line-height: 1.6;
    border-top: 1px solid rgba(0,212,255,0.1);
    padding-top: 0.7rem;
}

/* ═══════════════════════════════════════════════════════════════════════════════ */
/* METRIC CARDS */
/* ═══════════════════════════════════════════════════════════════════════════════ */

.metric-card {
    background: linear-gradient(135deg, #060f22, #08152e);
    border: 1.5px solid #0d2040;
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
    box-shadow: 0 8px 32px rgba(0,212,255,0.08);
    animation: bounceIn 0.6s ease;
}

@keyframes bounceIn {
    0% { opacity: 0; transform: scale(0.8); }
    50% { transform: scale(1.05); }
    100% { opacity: 1; transform: scale(1); }
}

.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #0077b6, #00d4ff, #0096c7);
}

.metric-card:hover {
    box-shadow: 0 16px 56px rgba(0,212,255,0.25);
    transform: translateY(-8px);
    border-color: rgba(0,212,255,0.3);
}

.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.02em;
    position: relative;
    z-index: 1;
    text-shadow: 0 0 20px rgba(0,212,255,0.2);
}

.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #4a6fa5;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 0.4rem;
}

.metric-pass { color: #00d4ff !important; }
.metric-fail { color: #ff4757 !important; }

/* ═══════════════════════════════════════════════════════════════════════════════ */
/* BADGES ══ */
/* ═══════════════════════════════════════════════════════════════════════════════ */

.hit-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(0,150,199,0.1));
    color: #00d4ff;
    border: 1.5px solid rgba(0,212,255,0.4);
    font-size: 0.64rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    padding: 0.35rem 0.85rem;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    box-shadow: 0 0 15px rgba(0,212,255,0.2);
    transition: all 0.3s;
}

.miss-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(255,71,87,0.12), rgba(239,68,68,0.08));
    color: #ff4757;
    border: 1.5px solid rgba(255,71,87,0.3);
    font-size: 0.64rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    padding: 0.35rem 0.85rem;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ═══════════════════════════════════════════════════════════════════════════════ */
/* TABLE ════ */
/* ═══════════════════════════════════════════════════════════════════════════════ */

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
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-bottom: 2px solid #0d2040;
    sticky top: 0;
    z-index: 10;
}

.eval-table td {
    padding: 0.8rem 1rem;
    border-bottom: 1px solid rgba(13,32,64,0.5);
    color: #8baac8;
    transition: all 0.2s;
}

.eval-table tr:hover td {
    background: rgba(0,212,255,0.04);
}

/* ═══════════════════════════════════════════════════════════════════════════════ */
/* QUERY ROWS */
/* ══════════���════════════════════════════════════════════════════════════════════ */

.query-row {
    background: linear-gradient(135deg, #060f22, #08152e);
    border: 1.5px solid #0d2040;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    transition: all 0.3s;
    box-shadow: 0 4px 16px rgba(0,212,255,0.05);
}

.query-row:hover {
    border-color: rgba(0,212,255,0.3);
    box-shadow: 0 8px 28px rgba(0,212,255,0.12);
    transform: translateX(4px);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* LATENCY BADGE */
/* ─────────────────────────────────────────────────────────────────────────── */

.latency-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(135deg, rgba(0,212,255,0.12), rgba(0,150,199,0.08));
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
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ANIMATED SEMANTIC SEARCH VISUALIZATION - DATA FLOW
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
  
  let particles = [];
  let pathT = 0;
  
  class Particle {
    constructor(startIdx, speed) {
      this.startIdx = startIdx;
      this.progress = 0;
      this.speed = speed || (0.002 + Math.random() * 0.003);
      this.size = 2 + Math.random() * 2;
      this.color = stages[startIdx].color;
      this.glow = Math.random() * 0.5 + 0.3;
      this.wobble = Math.random() * Math.PI * 2;
    }
    
    update() {
      this.progress += this.speed;
      if (this.progress > 1) {
        this.progress = 0;
        this.startIdx = (this.startIdx + 1) % stages.length;
        this.color = stages[this.startIdx].color;
      }
      this.wobble += 0.02;
    }
    
    draw() {
      const from = stages[this.startIdx];
      const to = stages[(this.startIdx + 1) % stages.length];
      
      const x = from.x + (to.x - from.x) * this.progress;
      const y = from.y + (to.y - from.y) * this.progress + Math.sin(this.wobble) * 20;
      
      // Glow
      const grad = ctx.createRadialGradient(x, y, 0, x, y, this.size * 6);
      grad.addColorStop(0, this.color + '40');
      grad.addColorStop(1, this.color + '00');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(x, y, this.size * 6, 0, Math.PI * 2);
      ctx.fill();
      
      // Core
      ctx.fillStyle = this.color;
      ctx.beginPath();
      ctx.arc(x, y, this.size, 0, Math.PI * 2);
      ctx.fill();
      
      ctx.strokeStyle = this.color + '88';
      ctx.lineWidth = 0.8;
      ctx.stroke();
    }
  }
  
  function resize() {
    W = c.width = window.innerWidth;
    H = c.height = window.innerHeight;
    stages[0].x = W*0.1;
    stages[1].x = W*0.25;
    stages[2].x = W*0.4;
    stages[3].x = W*0.55;
    stages[4].x = W*0.7;
    stages[5].x = W*0.85;
  }
  
  function initParticles() {
    particles = [];
    for (let i = 0; i < 15; i++) {
      particles.push(new Particle(i % stages.length, 0.003 + Math.random() * 0.002));
    }
  }
  
  function draw() {
    ctx.clearRect(0, 0, W, H);
    
    // Draw pipeline stages
    stages.forEach((stage, idx) => {
      // Stage circle
      const grad = ctx.createRadialGradient(stage.x, stage.y, 0, stage.x, stage.y, 50);
      grad.addColorStop(0, stage.color + '15');
      grad.addColorStop(1, stage.color + '00');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(stage.x, stage.y, 50, 0, Math.PI * 2);
      ctx.fill();
      
      // Stage border with animation
      const pulse = Math.sin(Date.now() * 0.005 + idx * 0.5) * 0.3 + 0.7;
      ctx.strokeStyle = stage.color + Math.floor(pulse * 200).toString(16).padStart(2, '0');
      ctx.lineWidth = 2 + pulse;
      ctx.beginPath();
      ctx.arc(stage.x, stage.y, 30, 0, Math.PI * 2);
      ctx.stroke();
      
      // Label
      ctx.fillStyle = stage.color;
      ctx.font = 'bold 11px "Space Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(stage.label, stage.x, stage.y + 3);
    });
    
    // Draw connecting lines with flow effect
    for (let i = 0; i < stages.length; i++) {
      const from = stages[i];
      const to = stages[(i + 1) % stages.length];
      
      // Gradient line
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
      
      // Animated flow dots on line
      const dotCount = 4;
      for (let d = 0; d < dotCount; d++) {
        const t = (pathT + d * 0.25) % 1;
        const px = from.x + (to.x - from.x) * t;
        const py = from.y + (to.y - from.y) * t;
        
        ctx.fillStyle = from.color;
        ctx.beginPath();
        ctx.arc(px, py, 2 + Math.sin(pathT * 0.1) * 1, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    
    pathT += 0.01;
    
    // Draw particles
    particles.forEach(p => {
      p.update();
      p.draw();
    });
    
    animId = requestAnimationFrame(draw);
  }
  
  window.addEventListener('resize', resize);
  
  function init() {
    resize();
    initParticles();
    draw();
  }
  
  if (document.readyState === 'complete') init();
  else window.addEventListener('load', init);
})();
</script>
"""

components.html(ANIMATED_DATA_FLOW, height=0, scrolling=False)


# ══════════════════════════════════════════════════════════════════════════════
# FLOATING SEMANTIC NODES VISUALIZATION (Background)
# ══════════════════════════════════════════════════════════════════════════════
FLOATING_NODES = """
<div id="floatingNodes" style="position:fixed;top:0;left:0;width:100%;height:100%;
  pointer-events:none;z-index:1;overflow:hidden;"></div>

<script>
(function(){
  const container = document.getElementById('floatingNodes');
  const standards = [
    'IS 269', 'IS 383', 'IS 455', 'IS 459', 'IS 458', 'IS 1489', 'IS 3466',
    'IS 6909', 'IS 8042', 'IS 2185', 'BM25', 'FAISS', 'RRF', 'LLM'
  ];
  const colors = ['#00d4ff', '#0096c7', '#0077b6', '#005f99', '#004775', '#00b4d8'];
  
  function createNode() {
    const node = document.createElement('div');
    const standard = standards[Math.floor(Math.random() * standards.length)];
    const color = colors[Math.floor(Math.random() * colors.length)];
    const size = 30 + Math.random() * 50;
    const x = Math.random() * window.innerWidth;
    const y = Math.random() * window.innerHeight;
    const duration = 15 + Math.random() * 15;
    const delay = Math.random() * 5;
    
    node.innerHTML = `
      <div style="
        position:absolute;
        left:${x}px;
        top:${y}px;
        width:${size}px;
        height:${size}px;
        border-radius:50%;
        background:radial-gradient(circle, ${color}33 0%, ${color}00 70%);
        border:1.5px solid ${color}66;
        display:flex;
        align-items:center;
        justify-content:center;
        font-family:'Space Mono',monospace;
        font-size:${Math.max(8, size/4)}px;
        font-weight:700;
        color:${color};
        text-shadow:0 0 10px ${color}44;
        animation:float 20s ease-in-out infinite;
        animation-delay:${delay}s;
        filter:drop-shadow(0 0 ${size/3}px ${color}44);
        z-index:${Math.floor(Math.random() * 5)};
      ">
        ${standard}
      </div>
      <style>
        @keyframes float {
          0%, 100% { transform: translate(0, 0) scale(1); opacity:0.3; }
          25% { transform: translate(${Math.sin(0)*50}px, ${-Math.cos(0)*50}px) scale(1.1); opacity:0.6; }
          50% { transform: translate(${Math.sin(Math.PI)*100}px, ${-Math.cos(Math.PI)*100}px) scale(1); opacity:0.8; }
          75% { transform: translate(${Math.sin(3*Math.PI/2)*50}px, ${-Math.cos(3*Math.PI/2)*50}px) scale(1.1); opacity:0.6; }
        }
      </style>
    `;
    return node;
  }
  
  // Create floating nodes
  for (let i = 0; i < 12; i++) {
    container.appendChild(createNode());
  }
})();
</script>
"""

components.html(FLOATING_NODES, height=0, scrolling=False)


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE SEARCH FLOW VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
SEARCH_FLOW_VIZ = """
<div id="searchFlow" style="position:fixed;bottom:20px;right:20px;width:280px;height:200px;
  background:linear-gradient(135deg, rgba(6,15,34,0.8), rgba(8,21,46,0.8));
  border:1.5px solid rgba(0,212,255,0.3);border-radius:12px;padding:16px;
  font-family:'Space Mono',monospace;z-index:999;box-shadow:0 8px 32px rgba(0,212,255,0.15);
  color:#c8d8ee;display:none;">
  <div style="font-size:0.65rem;color:#00d4ff;text-transform:uppercase;
    letter-spacing:0.1em;margin-bottom:12px;font-weight:700;">Search Pipeline Status</div>
  <div id="pipelineStatus" style="display:flex;flex-direction:column;gap:8px;font-size:0.72rem;">
    <div style="display:flex;align-items:center;gap:8px;">
      <div style="width:8px;height:8px;border-radius:50%;background:#00d4ff;animation:pulse 1.5s infinite;"></div>
      <span>Query Processing</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;">
      <div style="width:8px;height:8px;border-radius:50%;background:#0096c7;"></div>
      <span>BM25 Indexing</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;">
      <div style="width:8px;height:8px;border-radius:50%;background:#0077b6;"></div>
      <span>FAISS Search</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;">
      <div style="width:8px;height:8px;border-radius:50%;background:#005f99;"></div>
      <span>RRF Fusion</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;">
      <div style="width:8px;height:8px;border-radius:50%;background:#004775;"></div>
      <span>LLM Ranking</span>
    </div>
  </div>
  <style>
    @keyframes pulse {
      0%, 100% { opacity:0.3;transform:scale(1); }
      50% { opacity:1;transform:scale(1.2); }
    }
  </style>
</div>

<script>
  window.showSearchFlow = function() {
    document.getElementById('searchFlow').style.display = 'block';
    setTimeout(() => { document.getElementById('searchFlow').style.display = 'none'; }, 5000);
  };
</script>
"""

components.html(SEARCH_FLOW_VIZ, height=0, scrolling=False)


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
      BIS<span style="color:#00d4ff;filter:drop-shadow(0 0 8px rgba(0,212,255,0.4));">.</span>ENGINE
    </div>
    <div style="font-family:'Space Mono',monospace;font-size:0.62rem;color:#1a3a5c;
      text-transform:uppercase;letter-spacing:0.14em;margin-top:0.4rem;">
      Premium Recommendation System
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:0.65rem;
      color:#4a6fa5;text-transform:uppercase;letter-spacing:0.14em;
      margin-bottom:0.8rem;display:flex;align-items:center;gap:0.5rem;">
      <span style="display:inline-block;width:8px;height:8px;background:#00d4ff;border-radius:50%;box-shadow:0 0 8px #00d4ff;"></span>
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
    <div style="font-family:'Space Mono',monospace;font-size:0.65rem;
      color:#4a6fa5;text-transform:uppercase;letter-spacing:0.14em;
      margin-bottom:0.8rem;display:flex;align-items:center;gap:0.5rem;">
      <span style="display:inline-block;width:8px;height:8px;background:#0096c7;border-radius:50%;box-shadow:0 0 8px #0096c7;"></span>
      Retrieval Pipeline</div>
    """, unsafe_allow_html=True)

    pipeline_steps = [
        ("PDF → JSON", "#00d4ff"),
        ("BM25 Keyword", "#0096c7"),
        ("FAISS Semantic", "#0077b6"),
        ("RRF Fusion", "#005f99"),
        ("LLM Rerank", "#004775"),
    ]

    for i, (step, color) in enumerate(pipeline_steps):
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;">
          <div style="width:7px;height:7px;border-radius:50%;background:{color};
            animation:pulseDot 2s ease-in-out infinite;animation-delay:{i*0.5}s;
            box-shadow:0 0 12px {color};"></div>
          <div style="font-family:'Space Mono',monospace;font-size:0.63rem;
            color:#4a6fa5;text-transform:uppercase;letter-spacing:0.1em;">{step}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""<style>
    @keyframes pulseDot {
      0%,100% { opacity:0.4; transform:scale(1); } 
      50% { opacity:1; transform:scale(1.4); }
    }
    </style>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HERO SECTION
# ══════════════════════════════════════════════════════════════════════════════
col_hero, col_stats = st.columns([6, 4], gap="large")

with col_hero:
    st.markdown("""
    <div style="animation:fadeInUp 1s ease forwards;">
      <div class="hero-title">BIS Standards<br><span>Semantic</span> Engine</div>
      <div class="hero-sub">🔬 Bureau of Indian Standards · SP 21 · Building Materials</div>
      <div class="hero-sub" style="margin-top:0.4rem;color:#00d4ff;font-size:0.68rem;">
        ⚡ AI-Powered Retrieval · Real-time Search & Analysis
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_stats:
    st.markdown("""
    <style>
    @keyframes countUp {
      from { opacity:0; transform:translateY(12px); }
      to   { opacity:1; transform:translateY(0); }
    }
    .stat-box {
      background:linear-gradient(135deg, #060f22, #08152e);
      border:1.5px solid #0d2040;border-radius:12px;
      padding:1rem 0.8rem;text-align:center;
      animation: countUp 0.6s ease forwards;
      margin-bottom:10px;
    }
    .stat-num {
      font-family:'Syne',sans-serif;font-size:1.9rem;
      font-weight:800;color:#00d4ff;text-shadow: 0 0 15px rgba(0,212,255,0.3);
    }
    .stat-lbl {
      font-family:'Space Mono',monospace;font-size:0.58rem;
      color:#4a6fa5;text-transform:uppercase;letter-spacing:0.12em;margin-top:4px;
    }
    </style>
    <div class="stat-box" style="animation-delay:0.1s;">
      <div class="stat-num">567</div>
      <div class="stat-lbl">IS Standards</div>
    </div>
    <div class="stat-box" style="animation-delay:0.2s;">
      <div class="stat-num">27</div>
      <div class="stat-lbl">Sections</div>
    </div>
    <div class="stat-box" style="animation-delay:0.3s;">
      <div class="stat-num">&lt;1s</div>
      <div class="stat-lbl">Latency</div>
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
          margin-bottom:0.6rem;">📚 Quick Examples</div>
        """, unsafe_allow_html=True)

        example_queries = [
            "Select example…",
            "33 Grade OPC — chemical and physical requirements?",
            "Corrugated asbestos cement sheets for roofing and cladding.",
            "Hollow and solid lightweight concrete masonry blocks.",
            "TMT ribbed bars for reinforcing concrete slabs — yield strength?",
            "Bitumen felt as waterproofing layer under roofing tiles.",
            "Gun-metal gate valves used in plumbing.",
            "uPVC pipes for potable water supply.",
            "Glass wool slabs — thermal insulation in walls and roofs.",
        ]
        selected = st.selectbox("Load example", example_queries, label_visibility="collapsed")

    with col_q:
        query_text = st.text_area(
            "Describe your product or compliance need",
            value="" if selected == example_queries[0] else selected,
            height=120,
            placeholder="e.g. We manufacture 33 Grade Ordinary Portland Cement…",
            label_visibility="visible",
        )
        col_btn, col_info = st.columns([2, 3])
        with col_btn:
            run_single = st.button("⚡  Find Standards", key="run_single", use_container_width=True)
        with col_info:
            st.markdown("""
            <div style="font-family:'Space Mono',monospace;font-size:0.65rem;
              color:#4a6fa5;margin-top:0.8rem;">
            💡 Tip: Describe material properties for best results.
            </div>
            """, unsafe_allow_html=True)

    if run_single:
        if not query_text.strip():
            st.warning("🚨 Please enter a query to search.")
        else:
            try:
                with st.spinner("🔍 Running retrieval pipeline…"):
                    agent   = load_agent(index_dir, llm_backend, groq_model, groq_key)
                    t0      = time.perf_counter()
                    results = agent.answer(query_text.strip(), top_k=top_k)
                    elapsed = round(time.perf_counter() - t0, 3)

                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:1.2rem;margin:1.4rem 0 1rem;flex-wrap:wrap;">
                  <div class="latency-badge">⚡ {elapsed:.2f}s</div>
                  <div style="font-family:'Space Mono',monospace;font-size:0.68rem;
                    color:#4a6fa5;text-transform:uppercase;letter-spacing:0.1em;
                    background:rgba(0,212,255,0.08);padding:0.4rem 0.9rem;border-radius:20px;">
                    ✓ {len(results)} standards retrieved
                  </div>
                </div>
                """, unsafe_allow_html=True)

                results_html = ""
                rank_colors  = ["#00d4ff","#0096c7","#0077b6","#005f99","#004775"]
                for i, r in enumerate(results):
                    color        = rank_colors[min(i, len(rank_colors)-1)]
                    rat_html     = f'<div class="result-rationale">"{r.get("rationale","")}"</div>' if r.get("rationale") else ""
                    results_html += f"""
                    <div class="result-card" style="border-left-color:{color}">
                      <div class="result-rank">#{i+1} Rank</div>
                      <div class="result-is">{r['std_id']}</div>
                      <div class="result-title">{r.get('title','').title()}</div>
                      {rat_html}
                    </div>"""

                st.markdown(results_html, unsafe_allow_html=True)

                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("""
                <div style="font-family:'Space Mono',monospace;font-size:0.65rem;
                  color:#4a6fa5;text-transform:uppercase;letter-spacing:0.1em;
                  margin-bottom:0.8rem;">📥 Export Results</div>
                """, unsafe_allow_html=True)

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
                st.error(f"🚨 Error: {e}")
                st.info("Make sure the index directory contains `chunks.json` and `sparse_merged.pkl`.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — BATCH / JSON EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown('<div class="section-label">Batch Evaluation & Analysis</div>', unsafe_allow_html=True)

    col_upload, col_paste = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("""
        <div style="font-family:'Space Mono',monospace;font-size:0.65rem;
          color:#4a6fa5;text-transform:uppercase;letter-spacing:0.1em;
          margin-bottom:0.6rem;">📤 Upload JSON</div>
        """, unsafe_allow_html=True)
        uploaded = st.file_uploader("Drop JSON here", type=["json"],
                                     label_visibility="collapsed")

    with col_paste:
        st.markdown("""
        <div style="font-family:'Space Mono',monospace;font-size:0.65rem;
          color:#4a6fa5;text-transform:uppercase;letter-spacing:0.1em;
          margin-bottom:0.6rem;">📋 Paste JSON</div>
        """, unsafe_allow_html=True)
        pasted = st.text_area("JSON content", height=180,
                               label_visibility="collapsed",
                               placeholder='[{"id":"Q1","query":"...","expected_standards":["IS 269: 1989"]}]')

    queries_data = None
    parse_error  = None
    if uploaded:
        try:    queries_data = json.loads(uploaded.read().decode("utf-8"))
        except Exception as e: parse_error = str(e)
    elif pasted.strip():
        try:    queries_data = json.loads(pasted.strip())
        except Exception as e: parse_error = str(e)

    if parse_error:
        st.error(f"🚨 JSON Parse Error: {parse_error}")

    if queries_data:
        st.markdown(f"""
        <div style="font-family:'Space Mono',monospace;font-size:0.75rem;
          color:#00d4ff;margin:1rem 0 0.8rem;display:flex;align-items:center;gap:0.5rem;
          background:rgba(0,212,255,0.08);padding:0.6rem 1rem;border-radius:8px;
          border-left:2px solid #00d4ff;">
          ✓ {len(queries_data)} queries loaded for evaluation
        </div>
        """, unsafe_allow_html=True)

        with st.expander("👁️  Preview loaded queries", expanded=False):
            for i, q in enumerate(queries_data[:5], 1):
                exp_str = ", ".join(q.get("expected_standards", [])) or "—"
                st.markdown(f"""
                <div class="query-row">
                  <div style="font-family:'Space Mono',monospace;font-size:0.7rem;
                    color:#00d4ff;letter-spacing:0.1em;background:rgba(0,212,255,0.1);
                    padding:0.2rem 0.6rem;border-radius:4px;display:inline-block;">{q.get('id','?')}</div>
                  <div style="font-size:0.86rem;color:#8baac8;margin:0.5rem 0;line-height:1.5;">
                    {q.get('query','')[:140]}{'…' if len(q.get('query',''))>140 else ''}
                  </div>
                  <div style="font-size:0.73rem;color:#1a3a5c;
                    font-family:'Space Mono',monospace;background:rgba(0,212,255,0.04);
                    padding:0.4rem;border-radius:4px;">Expected: {exp_str}</div>
                </div>
                """, unsafe_allow_html=True)
            if len(queries_data) > 5:
                st.markdown(f"""
                <div style='font-size:0.76rem;color:#1a3a5c;font-family:Space Mono,monospace;
                  background:rgba(0,212,255,0.04);padding:0.5rem;margin-top:0.8rem;
                  border-radius:6px;text-align:center;'>
                  + {len(queries_data)-5} more queries …
                </div>""", unsafe_allow_html=True)

        st.markdown("")
        run_batch = st.button("▶  Run Inference + Evaluate", key="run_batch", use_container_width=True)

        if run_batch:
            try:
                with st.spinner("📦 Loading index…"):
                    agent = load_agent(index_dir, llm_backend, groq_model, groq_key)

                results   = []
                progress  = st.progress(0, text="🔄 Initializing…")
                status_ph = st.empty()
                total_q   = len(queries_data)

                for i, item in enumerate(queries_data):
                    qid   = item.get("id", str(i+1))
                    query = item.get("query", "")

                    status_ph.markdown(f"""
                    <div style="display:flex;align-items:center;gap:0.8rem;
                      font-family:'Space Mono',monospace;font-size:0.72rem;color:#4a6fa5;">
                      <div style="display:inline-flex;align-items:center;justify-content:center;
                        width:24px;height:24px;border-radius:50%;
                        background:linear-gradient(135deg, #00d4ff, #0096c7);
                        color:#fff;font-weight:700;font-size:0.7rem;">
                        {i+1}
                      </div>
                      <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
                        [{qid}] {query[:60]}…
                      </span>
                      <span style="color:#00d4ff;font-weight:700;">{i+1}/{total_q}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    t0 = time.perf_counter()
                    try:    retrieved = agent.answer(query, top_k=top_k)
                    except Exception as e:
                        retrieved = []
                        st.warning(f"[{qid}] failed: {e}")
                    elapsed = round(time.perf_counter()-t0, 3)

                    result = {
                        "id":                  qid,
                        "query":               query,
                        "retrieved_standards": [r["std_id"] for r in retrieved],
                        "latency_seconds":     elapsed,
                        "_retrieved_full":     retrieved,
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
                    metrics_config = [
                        (c1, metrics["hit3"], "Hit Rate @3", f"target > 80%", metrics["hit3"]>80, f"{metrics['hit3']:.1f}%"),
                        (c2, metrics["mrr5"], "MRR @5", f"target > 0.70", metrics["mrr5"]>0.7, f"{metrics['mrr5']:.4f}"),
                        (c3, metrics["avg_lat"], "Avg Latency", f"target < 5s", metrics["avg_lat"]<5, f"{metrics['avg_lat']:.2f}s"),
                        (c4, None, "Queries Hit", f"{metrics['total']} total", True, f"{metrics['hits']}/{metrics['total']}"),
                    ]

                    for col, val, label, target, threshold, fmt in metrics_config:
                        cls = "metric-pass" if threshold else "metric-fail"
                        with col:
                            st.markdown(f"""
                            <div class="metric-card">
                              <div class="metric-value {cls}">{fmt}</div>
                              <div class="metric-label">{label}</div>
                              <div class="metric-sub">{target} {'✓' if threshold else '✗'}</div>
                            </div>""", unsafe_allow_html=True)

                    st.markdown("")

                st.markdown('<div class="section-label">Detailed Results</div>', unsafe_allow_html=True)

                rows_html = ""
                for r in results:
                    exp = r.get("expected_standards", [])
                    ret = r.get("retrieved_standards", [])
                    lat = r.get("latency_seconds", 0)
                    if exp:
                        hit   = any(_match(e, rv) for e in exp for rv in ret[:3])
                        badge = '<span class="hit-badge">✓ HIT@3</span>' if hit else '<span class="miss-badge">✗ MISS</span>'
                    else:
                        badge = ""
                    ret_html = "".join(f'<div style="font-size:0.78rem;color:#4a6fa5;margin:2px 0;font-family:Space Mono,monospace;font-weight:600;">{s}</div>' for s in ret[:3])
                    exp_html = "".join(f'<div style="font-size:0.78rem;color:#00d4ff;margin:2px 0;font-family:Space Mono,monospace;font-weight:600;">{s}</div>' for s in exp) if exp else '<div style="font-size:0.78rem;color:#1a3a5c;font-style:italic;">—</div>'
                    rows_html += f"""
                    <tr>
                      <td><span style="font-family:Space Mono,monospace;font-size:0.7rem;color:#00d4ff;font-weight:700;">{r['id']}</span></td>
                      <td style="max-width:280px;word-break:break-word;color:#8baac8;line-height:1.5;">{r['query'][:110]}{'…' if len(r['query'])>110 else ''}</td>
                      <td>{ret_html}</td>
                      <td>{exp_html}</td>
                      <td>{badge}</td>
                      <td style="text-align:right;font-family:Space Mono,monospace;font-size:0.73rem;color:#4a6fa5;font-weight:700;">{lat:.2f}s</td>
                    </tr>"""

                st.markdown(f"""
                <table class="eval-table">
                  <thead><tr>
                    <th>ID</th><th>Query</th><th>Retrieved (top 3)</th>
                    <th>Expected</th><th>Status</th><th style="text-align:right;">Latency</th>
                  </tr></thead>
                  <tbody>{rows_html}</tbody>
                </table>""", unsafe_allow_html=True)

                st.markdown("")
                st.markdown('<div class="section-label">Download Results</div>', unsafe_allow_html=True)

                col_dl1, col_dl2, col_dl3 = st.columns(3)
                clean_results = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]

                with col_dl1:
                    st.download_button("⬇  Output (JSON)",
                                       data=json.dumps(clean_results, indent=2, ensure_ascii=False),
                                       file_name="output.json", mime="application/json",
                                       use_container_width=True)

                if has_expected:
                    eval_report = {
                        "hit_rate_at_3":    metrics["hit3"],
                        "mrr_at_5":         metrics["mrr5"],
                        "avg_latency_secs": metrics["avg_lat"],
                        "total_queries":    metrics["total"],
                        "hits":             metrics["hits"],
                        "pass_hit3":        metrics["hit3"] > 80,
                        "pass_mrr5":        metrics["mrr5"] > 0.7,
                        "pass_latency":     metrics["avg_lat"] < 5,
                    }
                    with col_dl2:
                        st.download_button("📊  Report (JSON)",
                                           data=json.dumps(eval_report, indent=2),
                                           file_name="eval_report.json", mime="application/json",
                                           use_container_width=True)

                with col_dl3:
                    csv_data = "ID,Query,Rank1,Rank2,Rank3,Expected,Status,Latency_s\n"
                    for r in results:
                        exp = r.get("expected_standards", [])
                        ret = r.get("retrieved_standards", [])
                        hit = "HIT" if exp and any(_match(e, rv) for e in exp for rv in ret[:3]) else "MISS"
                        csv_data += f"{r['id']},\"{r['query']}\",{ret[0] if len(ret)>0 else '—'},{ret[1] if len(ret)>1 else '—'},{ret[2] if len(ret)>2 else '—'},{';'.join(exp) if exp else '—'},{hit},{r.get('latency_seconds', 0):.3f}\n"

                    st.download_button("📈  Report (CSV)",
                                       data=csv_data,
                                       file_name="eval_report.csv", mime="text/csv",
                                       use_container_width=True)

            except Exception as e:
                st.error(f"🚨 Pipeline Error: {e}")
                import traceback; st.code(traceback.format_exc(), language="python")


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════���══════════════════════════════════════════════════════════════
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;font-family:'Space Mono',monospace;
  font-size:0.63rem;color:#4a6fa5;padding:1rem 0;letter-spacing:0.14em;
  text-transform:uppercase;background:linear-gradient(90deg, rgba(0,212,255,0.05), transparent, rgba(0,212,255,0.05));
  border-radius:8px;margin-top:2rem;">
  ✨ BIS Standards Engine — Ultra Premium Edition ✨<br>
  <span style="margin-top:0.5rem;display:block;color:#1a3a5c;">
  Bureau of Indian Standards · SP 21 Building Materials
  </span>
</div>
""", unsafe_allow_html=True)