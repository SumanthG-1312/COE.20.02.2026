"""
✈️ Flight Ticket Price Predictor — Animated Edition
Run: streamlit run app.py
pip install streamlit scikit-learn pandas numpy openpyxl
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings, os
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# ALL CSS + ANIMATIONS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ══════════════════════════════════════════
   FONTS & ROOT VARIABLES
══════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;600;700&family=Exo+2:ital,wght@0,100..900;1,400&family=Share+Tech+Mono&display=swap');

:root {
  --sky-deep:    #020b18;
  --sky-mid:     #041832;
  --sky-dawn:    #0a2a52;
  --horizon:     #0e4d8c;
  --cloud-white: rgba(255,255,255,0.82);
  --gold:        #f5c842;
  --gold-light:  #ffe98a;
  --teal:        #00d4ff;
  --teal-dim:    #007da6;
  --panel-bg:    rgba(4,24,50,0.85);
  --panel-border:rgba(0,212,255,0.18);
  --text-bright: #e8f4ff;
  --text-dim:    #8ab4d4;
  --font-head:   'Rajdhani', sans-serif;
  --font-body:   'Exo 2', sans-serif;
  --font-mono:   'Share Tech Mono', monospace;
}

/* ══════════════════════════════════════════
   GLOBAL RESET & BASE
══════════════════════════════════════════ */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
  background: var(--sky-deep) !important;
  color: var(--text-bright);
  font-family: var(--font-body);
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #020d1f 0%, #041428 100%) !important;
  border-right: 1px solid var(--panel-border);
}
[data-testid="stSidebar"] * { color: var(--text-bright) !important; }

/* ══════════════════════════════════════════
   ANIMATED SKY CANVAS  (pseudo-layers)
══════════════════════════════════════════ */
/* Starfield layer — injected behind everything */
#sky-canvas {
  position: fixed; inset: 0; z-index: 0;
  pointer-events: none; overflow: hidden;
}
#sky-canvas .star {
  position: absolute; border-radius: 50%;
  background: white;
  animation: twinkle var(--t, 3s) ease-in-out infinite alternate;
}
@keyframes twinkle {
  from { opacity: 0.1; transform: scale(0.8); }
  to   { opacity: 0.9; transform: scale(1.2); }
}

/* Atmospheric glow bands */
#sky-canvas .aurora {
  position: absolute; left: -20%; width: 140%; height: 30%;
  border-radius: 50%;
  filter: blur(80px); opacity: 0.08;
  animation: aurora-drift 18s ease-in-out infinite alternate;
}
.aurora-1 { background: radial-gradient(ellipse, #0066ff 0%, transparent 70%); top: 10%; }
.aurora-2 { background: radial-gradient(ellipse, #00d4ff 0%, transparent 70%); top: 30%; animation-delay: -6s; opacity: 0.06; }
.aurora-3 { background: radial-gradient(ellipse, #4400cc 0%, transparent 70%); top: 50%; animation-delay: -12s; opacity: 0.05; }
@keyframes aurora-drift {
  from { transform: translateX(-5%) scaleY(1); }
  to   { transform: translateX(5%) scaleY(1.3); }
}

/* ── Cloud layers ── */
.cloud-layer { position: fixed; inset: 0; z-index: 1; pointer-events: none; overflow: hidden; }
.cloud {
  position: absolute;
  background: radial-gradient(ellipse at center, rgba(255,255,255,0.18) 0%, transparent 70%);
  border-radius: 50%;
  filter: blur(18px);
}
.cloud-a {
  width: 320px; height: 90px; top: 12%;
  animation: cloud-drift-a 55s linear infinite;
}
.cloud-b {
  width: 500px; height: 120px; top: 22%;
  animation: cloud-drift-b 80s linear infinite;
  opacity: 0.6;
}
.cloud-c {
  width: 220px; height: 70px; top: 38%;
  animation: cloud-drift-a 42s linear infinite reverse;
  opacity: 0.4;
}
.cloud-d {
  width: 600px; height: 130px; top: 8%;
  animation: cloud-drift-b 95s linear infinite;
  opacity: 0.3;
}
@keyframes cloud-drift-a {
  from { transform: translateX(-400px); }
  to   { transform: translateX(110vw);  }
}
@keyframes cloud-drift-b {
  from { transform: translateX(110vw);  }
  to   { transform: translateX(-600px); }
}

/* ── Airplane silhouettes ── */
.plane-layer { position: fixed; inset: 0; z-index: 2; pointer-events: none; overflow: hidden; }
.plane {
  position: absolute;
  font-size: 28px;
  filter: drop-shadow(0 0 12px rgba(0,212,255,0.8));
  animation: plane-fly var(--fly-dur, 22s) linear infinite;
  animation-delay: var(--fly-delay, 0s);
  opacity: 0;
}
.plane::after {
  content: '';
  position: absolute;
  right: 100%; top: 50%; transform: translateY(-50%);
  height: 2px; width: 0;
  background: linear-gradient(90deg, transparent, rgba(0,212,255,0.6));
  animation: contrail-grow var(--fly-dur, 22s) linear infinite;
  animation-delay: var(--fly-delay, 0s);
}

.plane-1 { top: 8%;  --fly-dur: 20s; --fly-delay: 0s;    font-size: 22px; }
.plane-2 { top: 18%; --fly-dur: 28s; --fly-delay: -8s;   font-size: 32px; }
.plane-3 { top: 31%; --fly-dur: 16s; --fly-delay: -14s;  font-size: 18px; }
.plane-4 { top: 5%;  --fly-dur: 35s; --fly-delay: -22s;  font-size: 14px; opacity: 0; }
.plane-5 { top: 25%; --fly-dur: 24s; --fly-delay: -5s;   font-size: 26px; transform: scaleX(-1); }

@keyframes plane-fly {
  0%   { transform: translateX(-120px);              opacity: 0; }
  5%   { opacity: 1; }
  95%  { opacity: 1; }
  100% { transform: translateX(calc(100vw + 200px)); opacity: 0; }
}
.plane-5 {
  animation-name: plane-fly-rtl;
}
@keyframes plane-fly-rtl {
  0%   { transform: scaleX(-1) translateX(-120px); opacity: 0; }
  5%   { opacity: 1; }
  95%  { opacity: 1; }
  100% { transform: scaleX(-1) translateX(calc(100vw + 200px)); opacity: 0; }
}
@keyframes contrail-grow {
  0%   { width: 0; opacity: 0; }
  10%  { opacity: 1; }
  60%  { width: 180px; }
  90%  { width: 220px; opacity: 0.3; }
  100% { width: 0; opacity: 0; }
}

/* ── Radar sweep ── */
.radar-wrap {
  position: fixed; bottom: 30px; right: 30px;
  width: 120px; height: 120px; z-index: 3;
  pointer-events: none; opacity: 0.35;
}
.radar-ring {
  position: absolute; inset: 0; border-radius: 50%;
  border: 1px solid var(--teal);
}
.radar-ring:nth-child(2) { inset: 20%; opacity: 0.7; }
.radar-ring:nth-child(3) { inset: 40%; opacity: 0.5; }
.radar-cross {
  position: absolute; inset: 0;
  background:
    linear-gradient(var(--teal), var(--teal)) center/1px 100% no-repeat,
    linear-gradient(var(--teal), var(--teal)) center/100% 1px no-repeat;
  opacity: 0.4;
}
.radar-sweep {
  position: absolute; inset: 0; border-radius: 50%; overflow: hidden;
}
.radar-sweep::after {
  content: '';
  position: absolute; top: 50%; left: 50%;
  width: 50%; height: 50%;
  background: conic-gradient(from 0deg, transparent 80%, rgba(0,212,255,0.7) 100%);
  transform-origin: 0% 100%;
  animation: radar-spin 4s linear infinite;
}
.radar-blip {
  position: absolute;
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--teal);
  box-shadow: 0 0 8px var(--teal);
  animation: blip-pulse 4s ease-out infinite;
}
.blip-1 { top: 30%; left: 60%; animation-delay: 0s; }
.blip-2 { top: 65%; left: 25%; animation-delay: 1.5s; }
.blip-3 { top: 20%; left: 35%; animation-delay: 2.8s; }
@keyframes radar-spin  { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
@keyframes blip-pulse  {
  0%,100% { opacity: 0; transform: scale(0.5); }
  50%     { opacity: 1; transform: scale(1.4); }
}

/* ── Runway lights (bottom edge) ── */
.runway-strip {
  position: fixed; bottom: 0; left: 0; right: 0;
  height: 4px; z-index: 2;
  background: repeating-linear-gradient(
    90deg,
    var(--gold) 0px,
    var(--gold) 20px,
    transparent 20px,
    transparent 60px
  );
  animation: runway-pulse 1.8s ease-in-out infinite alternate;
  pointer-events: none;
}
@keyframes runway-pulse {
  from { opacity: 0.3; }
  to   { opacity: 0.8; }
}

/* ══════════════════════════════════════════
   MAIN CONTENT WRAPPER (above bg layers)
══════════════════════════════════════════ */
[data-testid="stMainBlockContainer"] {
  position: relative; z-index: 10;
}

/* ══════════════════════════════════════════
   HERO BANNER
══════════════════════════════════════════ */
.hero-banner {
  position: relative;
  background: linear-gradient(135deg, rgba(4,24,50,0.95) 0%, rgba(2,11,24,0.98) 100%);
  border: 1px solid var(--panel-border);
  border-radius: 16px;
  padding: 40px 48px 32px;
  margin-bottom: 28px;
  overflow: hidden;
  box-shadow: 0 8px 48px rgba(0,0,0,0.6), inset 0 1px 0 rgba(0,212,255,0.12);
}
.hero-banner::before {
  content: '';
  position: absolute; inset: 0;
  background:
    radial-gradient(ellipse 60% 80% at 80% 50%, rgba(0,90,180,0.12) 0%, transparent 60%),
    repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(0,212,255,0.04) 40px),
    repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(0,212,255,0.04) 40px);
  pointer-events: none;
}
.hero-title {
  font-family: var(--font-head);
  font-size: 3.2rem; font-weight: 700; letter-spacing: 3px;
  background: linear-gradient(90deg, #fff 0%, var(--teal) 50%, var(--gold) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
  animation: title-shimmer 4s ease-in-out infinite alternate;
  text-transform: uppercase;
  margin: 0;
}
@keyframes title-shimmer {
  from { filter: brightness(1); }
  to   { filter: brightness(1.25) drop-shadow(0 0 20px rgba(0,212,255,0.5)); }
}
.hero-sub {
  font-family: var(--font-body); font-size: 1.05rem;
  color: var(--text-dim); margin-top: 8px; letter-spacing: 1px;
}
.hero-plane-anim {
  position: absolute; right: 48px; top: 50%; transform: translateY(-50%);
  font-size: 88px; opacity: 0.15;
  animation: hero-plane-hover 3s ease-in-out infinite alternate;
  filter: drop-shadow(0 0 30px rgba(0,212,255,0.6));
}
@keyframes hero-plane-hover {
  from { transform: translateY(-55%) rotate(-4deg); opacity: 0.12; }
  to   { transform: translateY(-45%) rotate(4deg);  opacity: 0.22; }
}
/* Animated dashed route line */
.hero-route {
  position: absolute; bottom: 18px; left: 48px;
  display: flex; align-items: center; gap: 8px;
  font-family: var(--font-mono); font-size: 0.75rem;
  color: var(--teal); letter-spacing: 2px; opacity: 0.6;
}
.route-dash {
  width: 80px; height: 1px;
  background: repeating-linear-gradient(90deg, var(--teal) 0px, var(--teal) 8px, transparent 8px, transparent 14px);
  animation: route-march 1s linear infinite;
  background-size: 14px 1px;
}
@keyframes route-march {
  from { background-position: 0; }
  to   { background-position: 14px; }
}

/* ══════════════════════════════════════════
   METRIC CARDS
══════════════════════════════════════════ */
.metric-card {
  background: linear-gradient(135deg, rgba(4,24,50,0.9) 0%, rgba(2,15,35,0.95) 100%);
  border: 1px solid rgba(0,212,255,0.2);
  border-radius: 12px; padding: 18px 20px; text-align: center;
  position: relative; overflow: hidden;
  transition: border-color 0.3s, box-shadow 0.3s;
  animation: card-float 5s ease-in-out infinite alternate;
}
.metric-card:nth-child(2) { animation-delay: -1s; }
.metric-card:nth-child(3) { animation-delay: -2s; }
.metric-card:nth-child(4) { animation-delay: -3s; }
.metric-card::after {
  content: '';
  position: absolute; top: 0; left: -100%; width: 60%; height: 100%;
  background: linear-gradient(90deg, transparent, rgba(0,212,255,0.08), transparent);
  animation: card-scan 4s ease-in-out infinite;
}
@keyframes card-scan {
  0%   { left: -100%; }
  100% { left: 200%; }
}
@keyframes card-float {
  from { transform: translateY(0); }
  to   { transform: translateY(-4px); }
}
.metric-label {
  font-family: var(--font-mono); font-size: 0.65rem;
  color: var(--teal); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 6px;
}
.metric-value {
  font-family: var(--font-head); font-size: 1.6rem; font-weight: 700;
  color: var(--gold); letter-spacing: 1px;
}

/* ══════════════════════════════════════════
   FORM PANEL
══════════════════════════════════════════ */
.form-header {
  font-family: var(--font-head); font-size: 1.1rem; font-weight: 600;
  color: var(--teal); letter-spacing: 3px; text-transform: uppercase;
  margin-bottom: 16px; display: flex; align-items: center; gap: 10px;
}
.form-header::after {
  content: '';
  flex: 1; height: 1px;
  background: linear-gradient(90deg, rgba(0,212,255,0.4), transparent);
}

[data-testid="stForm"] {
  background: linear-gradient(135deg, rgba(4,24,50,0.88) 0%, rgba(2,11,24,0.95) 100%) !important;
  border: 1px solid rgba(0,212,255,0.15) !important;
  border-radius: 16px !important; padding: 28px !important;
  box-shadow: 0 4px 32px rgba(0,0,0,0.5) !important;
  position: relative; overflow: hidden;
}
[data-testid="stForm"]::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, var(--teal), var(--gold), var(--teal), transparent);
  animation: top-scan 3s ease-in-out infinite;
}
@keyframes top-scan {
  0%,100% { opacity: 0.4; }
  50%     { opacity: 1; }
}

/* Streamlit widget overrides */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input,
[data-testid="stSlider"] {
  background: rgba(0,20,45,0.8) !important;
  border: 1px solid rgba(0,212,255,0.25) !important;
  border-radius: 8px !important;
  color: var(--text-bright) !important;
}
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label,
[data-testid="stSlider"] label,
[data-testid="stDateInput"] label {
  color: var(--text-dim) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.78rem !important; letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
}

/* Submit button */
[data-testid="stFormSubmitButton"] button {
  background: linear-gradient(90deg, #0055aa, #0099cc, #0055aa) !important;
  background-size: 200% 100% !important;
  border: 1px solid var(--teal) !important;
  border-radius: 10px !important;
  color: white !important; font-family: var(--font-head) !important;
  font-size: 1.1rem !important; font-weight: 700 !important; letter-spacing: 3px !important;
  text-transform: uppercase !important;
  padding: 14px 32px !important;
  animation: btn-gradient 3s ease-in-out infinite;
  transition: all 0.3s !important;
  box-shadow: 0 4px 20px rgba(0,212,255,0.3) !important;
}
[data-testid="stFormSubmitButton"] button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 32px rgba(0,212,255,0.5) !important;
}
@keyframes btn-gradient {
  0%   { background-position: 0%; }
  50%  { background-position: 100%; }
  100% { background-position: 0%; }
}

/* ══════════════════════════════════════════
   TAKEOFF ANIMATION (on submit)
══════════════════════════════════════════ */
.takeoff-wrapper {
  position: relative; overflow: hidden; border-radius: 16px;
  margin: 24px 0;
}
.takeoff-plane {
  display: block; text-align: center;
  font-size: 64px;
  animation: takeoff-launch 1.6s cubic-bezier(0.25, 0.46, 0.45, 0.94) forwards;
  filter: drop-shadow(0 0 24px rgba(0,212,255,0.9));
}
@keyframes takeoff-launch {
  0%   { transform: translateX(-60vw) translateY(0) rotate(-8deg); opacity: 0; }
  30%  { opacity: 1; }
  70%  { transform: translateX(0) translateY(-20px) rotate(8deg); opacity: 1; }
  100% { transform: translateX(60vw) translateY(-80px) rotate(15deg); opacity: 0; }
}
.contrail-trail {
  position: absolute; bottom: 40%; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, rgba(0,212,255,0.6), rgba(255,255,255,0.3), transparent);
  animation: contrail-fade 2s ease-out forwards;
}
@keyframes contrail-fade {
  0%   { opacity: 0; transform: scaleX(0); }
  40%  { opacity: 1; transform: scaleX(1); }
  100% { opacity: 0; transform: scaleX(1); }
}

/* ══════════════════════════════════════════
   BOARDING PASS RESULT CARD
══════════════════════════════════════════ */
.boarding-pass {
  background: linear-gradient(135deg, #020d1f 0%, #041428 100%);
  border: 1px solid rgba(0,212,255,0.3);
  border-radius: 16px; overflow: hidden;
  box-shadow: 0 12px 60px rgba(0,0,0,0.7), 0 0 0 1px rgba(0,212,255,0.1);
  animation: bp-enter 0.8s cubic-bezier(0.34, 1.56, 0.64, 1) both;
  position: relative;
}
@keyframes bp-enter {
  from { transform: translateY(30px) scale(0.95); opacity: 0; }
  to   { transform: translateY(0) scale(1);       opacity: 1; }
}
.boarding-pass::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, var(--teal), var(--gold), var(--teal));
  background-size: 200% 100%;
  animation: bp-shimmer 2s linear infinite;
}
@keyframes bp-shimmer {
  from { background-position: -100%; }
  to   { background-position: 200%; }
}
.boarding-pass::after {
  /* Perforation dots */
  content: '';
  position: absolute; top: 0; bottom: 0; right: 42%;
  width: 1px;
  background: repeating-linear-gradient(
    180deg,
    transparent 0, transparent 8px,
    rgba(0,212,255,0.3) 8px, rgba(0,212,255,0.3) 14px
  );
}
.bp-main {
  display: grid; grid-template-columns: 1fr auto;
  padding: 28px 32px 28px 32px;
  gap: 20px;
}
.bp-route {
  display: flex; align-items: center; gap: 16px;
  margin-bottom: 20px;
}
.bp-city {
  font-family: var(--font-head); font-size: 3rem; font-weight: 700;
  color: white; letter-spacing: 2px; line-height: 1;
}
.bp-city-name {
  font-family: var(--font-mono); font-size: 0.65rem;
  color: var(--teal); letter-spacing: 2px; text-transform: uppercase; margin-top: 4px;
}
.bp-arrow-wrap {
  display: flex; flex-direction: column; align-items: center; gap: 4px; flex: 1;
}
.bp-plane-icon {
  font-size: 28px;
  filter: drop-shadow(0 0 12px rgba(0,212,255,0.8));
  animation: bp-plane-fly 3s ease-in-out infinite;
}
@keyframes bp-plane-fly {
  0%,100% { transform: translateX(-6px) rotate(-5deg); }
  50%     { transform: translateX(6px)  rotate(5deg);  }
}
.bp-route-line {
  width: 100%; height: 1px;
  background: linear-gradient(90deg, var(--teal), var(--gold), var(--teal));
  background-size: 200%;
  animation: bp-shimmer 2s linear infinite;
}
.bp-info-grid {
  display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px;
}
.bp-info-item {}
.bp-info-label {
  font-family: var(--font-mono); font-size: 0.6rem; color: var(--teal);
  letter-spacing: 2px; text-transform: uppercase; margin-bottom: 4px;
}
.bp-info-value {
  font-family: var(--font-head); font-size: 1rem; font-weight: 600;
  color: var(--text-bright);
}
.bp-stub {
  width: 42%;padding: 28px 20px;
  background: rgba(0,212,255,0.04);
  display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px;
  border-left: none; /* handled by ::after perf line */
}
.bp-price-big {
  font-family: var(--font-head); font-size: 2.8rem; font-weight: 700;
  color: var(--gold);
  text-shadow: 0 0 30px rgba(245,200,66,0.5);
  animation: price-glow 2s ease-in-out infinite alternate;
  letter-spacing: 2px; text-align: center;
}
@keyframes price-glow {
  from { text-shadow: 0 0 20px rgba(245,200,66,0.3); }
  to   { text-shadow: 0 0 40px rgba(245,200,66,0.8), 0 0 80px rgba(245,200,66,0.3); }
}
.bp-price-label {
  font-family: var(--font-mono); font-size: 0.65rem; color: var(--teal);
  letter-spacing: 3px; text-transform: uppercase;
}
.bp-range {
  font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-dim);
  text-align: center; line-height: 1.5;
}
.bp-barcode {
  display: flex; gap: 2px; align-items: center; height: 36px; margin-top: 4px;
}
.barcode-bar {
  background: var(--teal); border-radius: 1px;
  animation: barcode-pulse 1.5s ease-in-out infinite alternate;
  opacity: 0.7;
}
@keyframes barcode-pulse {
  from { opacity: 0.4; }
  to   { opacity: 0.9; }
}

/* ══════════════════════════════════════════
   BOOKING SUMMARY TABLE
══════════════════════════════════════════ */
.summary-panel {
  background: linear-gradient(135deg, rgba(4,24,50,0.88) 0%, rgba(2,11,24,0.95) 100%);
  border: 1px solid rgba(0,212,255,0.18); border-radius: 16px;
  padding: 24px 28px;
  animation: bp-enter 0.9s cubic-bezier(0.34, 1.56, 0.64, 1) both;
  animation-delay: 0.15s;
}
.summary-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 9px 0; border-bottom: 1px solid rgba(0,212,255,0.07);
  animation: row-in 0.4s ease both;
}
.summary-row:last-child { border-bottom: none; }
.summary-key {
  font-family: var(--font-mono); font-size: 0.72rem;
  color: var(--teal); letter-spacing: 1.5px; text-transform: uppercase;
}
.summary-val {
  font-family: var(--font-body); font-size: 0.88rem;
  color: var(--text-bright); text-align: right;
}
@keyframes row-in {
  from { transform: translateX(12px); opacity: 0; }
  to   { transform: translateX(0);    opacity: 1; }
}

/* ══════════════════════════════════════════
   FEATURE IMPORTANCE BARS
══════════════════════════════════════════ */
.feat-row {
  display: flex; align-items: center; gap: 10px; margin-bottom: 8px;
}
.feat-name {
  font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-dim);
  min-width: 220px; max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.feat-bar-wrap {
  flex: 1; height: 8px; background: rgba(0,212,255,0.08); border-radius: 4px; overflow: hidden;
}
.feat-bar {
  height: 100%; border-radius: 4px;
  background: linear-gradient(90deg, var(--teal), var(--gold));
  animation: bar-fill 1s cubic-bezier(0.25,1,0.5,1) both;
  transform-origin: left;
}
@keyframes bar-fill {
  from { transform: scaleX(0); }
  to   { transform: scaleX(1); }
}
.feat-score {
  font-family: var(--font-mono); font-size: 0.65rem; color: var(--gold);
  min-width: 48px; text-align: right;
}

/* ══════════════════════════════════════════
   SIDEBAR EXTRAS
══════════════════════════════════════════ */
.sidebar-logo {
  text-align: center; padding: 12px 0 20px;
}
.sidebar-logo-plane {
  font-size: 52px;
  display: inline-block;
  animation: logo-orbit 6s ease-in-out infinite;
  filter: drop-shadow(0 0 16px rgba(0,212,255,0.7));
}
@keyframes logo-orbit {
  0%,100% { transform: rotate(-12deg) scale(1);   }
  50%     { transform: rotate(12deg)  scale(1.08); }
}
.sidebar-badge {
  display: inline-block; padding: 3px 10px;
  background: rgba(0,212,255,0.12); border: 1px solid rgba(0,212,255,0.3);
  border-radius: 20px; font-family: var(--font-mono); font-size: 0.65rem;
  color: var(--teal); letter-spacing: 2px; text-transform: uppercase;
  animation: badge-blink 2s ease-in-out infinite;
}
@keyframes badge-blink {
  0%,100% { border-color: rgba(0,212,255,0.3); }
  50%     { border-color: rgba(0,212,255,0.8); }
}

/* ══════════════════════════════════════════
   EXPANDER
══════════════════════════════════════════ */
[data-testid="stExpander"] {
  background: rgba(4,24,50,0.7) !important;
  border: 1px solid rgba(0,212,255,0.15) !important;
  border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
  color: var(--teal) !important;
  font-family: var(--font-mono) !important; font-size: 0.8rem !important;
  letter-spacing: 1.5px !important; text-transform: uppercase !important;
}

/* ══════════════════════════════════════════
   FOOTER
══════════════════════════════════════════ */
.footer {
  text-align: center; padding: 24px 0 12px;
  font-family: var(--font-mono); font-size: 0.65rem;
  color: rgba(138,180,212,0.4); letter-spacing: 2px; text-transform: uppercase;
}
.footer span { color: rgba(0,212,255,0.5); }
</style>

<!-- ── Sky canvas: stars + auroras ── -->
<div id="sky-canvas">
  <div class="aurora aurora-1"></div>
  <div class="aurora aurora-2"></div>
  <div class="aurora aurora-3"></div>
</div>

<!-- ── Cloud layers ── -->
<div class="cloud-layer">
  <div class="cloud cloud-a" style="left:-200px;"></div>
  <div class="cloud cloud-b" style="right:-300px;"></div>
  <div class="cloud cloud-c" style="left:30%;"></div>
  <div class="cloud cloud-d" style="left:-400px;"></div>
</div>

<!-- ── Flying planes ── -->
<div class="plane-layer">
  <div class="plane plane-1">✈</div>
  <div class="plane plane-2">✈</div>
  <div class="plane plane-3">✈</div>
  <div class="plane plane-4">✈</div>
  <div class="plane plane-5">✈</div>
</div>

<!-- ── Radar widget ── -->
<div class="radar-wrap">
  <div class="radar-ring"></div>
  <div class="radar-ring"></div>
  <div class="radar-ring"></div>
  <div class="radar-cross"></div>
  <div class="radar-sweep"></div>
  <div class="radar-blip blip-1"></div>
  <div class="radar-blip blip-2"></div>
  <div class="radar-blip blip-3"></div>
</div>

<!-- ── Runway strip ── -->
<div class="runway-strip"></div>

<!-- ── Starfield (generated via inline script) ── -->
<script>
(function(){
  const canvas = document.getElementById('sky-canvas');
  if(!canvas) return;
  for(let i=0;i<120;i++){
    const s = document.createElement('div');
    s.className='star';
    const size = Math.random()*2.5+0.5;
    s.style.cssText = `
      width:${size}px; height:${size}px;
      left:${Math.random()*100}%;
      top:${Math.random()*60}%;
      --t:${(Math.random()*3+1.5).toFixed(1)}s;
      animation-delay:${(Math.random()*4).toFixed(1)}s;
      opacity:${(Math.random()*0.5+0.1).toFixed(2)};
    `;
    canvas.appendChild(s);
  }
})();
</script>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PREPROCESSING HELPERS
# ─────────────────────────────────────────────
def parse_duration(duration_str):
    if pd.isna(duration_str): return 0
    duration_str = str(duration_str).strip()
    hours = minutes = 0
    if 'h' in duration_str:
        parts = duration_str.split('h')
        hours = int(parts[0].strip()) if parts[0].strip().isdigit() else 0
        rem = parts[1].strip() if len(parts) > 1 else ''
        if 'm' in rem:
            m = rem.replace('m','').strip()
            minutes = int(m) if m.isdigit() else 0
    elif 'm' in duration_str:
        m = duration_str.replace('m','').strip()
        minutes = int(m) if m.isdigit() else 0
    return hours * 60 + minutes

def stops_to_num(val):
    return {'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4}.get(str(val).strip().lower(),0)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); df.dropna(inplace=True)
    df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], dayfirst=True, errors='coerce')
    df['Journey_Day']   = df['Date_of_Journey'].dt.day
    df['Journey_Month'] = df['Date_of_Journey'].dt.month
    df.drop('Date_of_Journey', axis=1, inplace=True)
    df['Dep_Time']  = df['Dep_Time'].str.strip()
    df['Dep_Hour']  = df['Dep_Time'].str.split(':').str[0].astype(int, errors='ignore')
    df['Dep_Min']   = df['Dep_Time'].str.split(':').str[1].astype(int, errors='ignore')
    df.drop('Dep_Time', axis=1, inplace=True)
    df['Arrival_Time'] = df['Arrival_Time'].str.split(' ').str[0].str.strip()
    df['Arr_Hour']     = df['Arrival_Time'].str.split(':').str[0].astype(int, errors='ignore')
    df['Arr_Min']      = df['Arrival_Time'].str.split(':').str[1].astype(int, errors='ignore')
    df.drop('Arrival_Time', axis=1, inplace=True)
    df['Duration_Min'] = df['Duration'].apply(parse_duration)
    df.drop('Duration', axis=1, inplace=True)
    df['Stops'] = df['Total_Stops'].apply(stops_to_num)
    df.drop('Total_Stops', axis=1, inplace=True)
    df.drop('Route', axis=1, inplace=True)
    cat_cols = ['Airline','Source','Destination','Additional_Info']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    return df

# ─────────────────────────────────────────────
# MODEL TRAINING (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_and_train(file_path: str):
    df_raw  = pd.read_excel(file_path)
    df_proc = preprocess(df_raw)
    X = df_proc.drop('Price', axis=1); y = df_proc['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=5,
                                   n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X.columns.tolist(), df_raw, r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <div class="sidebar-logo-plane">✈</div><br>
      <span class="sidebar-badge">AI-Powered</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### 📂 Dataset")
    DEFAULT_PATH = "Data_Train.xlsx"
    uploaded = st.file_uploader("Upload Data_Train.xlsx", type=["xlsx"])
    if uploaded:
        tmp_path = "/tmp/Data_Train_upload.xlsx"
        with open(tmp_path, "wb") as f: f.write(uploaded.read())
        DATA_PATH = tmp_path
    elif os.path.exists(DEFAULT_PATH):
        DATA_PATH = DEFAULT_PATH
    else:
        DATA_PATH = None
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("Predicts **Indian domestic flight prices** using a Random Forest model "
                "trained on ~10,000 flights (March–June 2019).")
    st.markdown("**Features used:**")
    for feat in ["Airline", "Source & Destination", "Journey Date",
                 "Departure Time", "Flight Duration", "Stops", "Additional Info"]:
        st.markdown(f"• {feat}")

# ─────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-plane-anim">✈</div>
  <h1 class="hero-title">✈ Flight Price Predictor</h1>
  <p class="hero-sub">AI-powered fare estimation · Indian Domestic Flights · Random Forest Model</p>
  <div class="hero-route">
    <span>DEL</span>
    <div class="route-dash"></div>
    <span>✈</span>
    <div class="route-dash"></div>
    <span>BOM</span>
    &nbsp;&nbsp;|&nbsp;&nbsp;
    <span>HYD</span>
    <div class="route-dash"></div>
    <span>✈</span>
    <div class="route-dash"></div>
    <span>COK</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA GUARD
# ─────────────────────────────────────────────
if DATA_PATH is None:
    st.warning("⚠️ Upload **Data_Train.xlsx** via the sidebar to activate the model.")
    st.stop()

with st.spinner("🔄 Training model — first run takes ~10 seconds…"):
    try:
        model, feature_cols, df_raw, r2, mae = load_and_train(DATA_PATH)
    except Exception as e:
        st.error(f"❌ Error: {e}"); st.stop()

# ─────────────────────────────────────────────
# METRIC STRIP
# ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, label, val in [
    (c1, "Model",            "Random Forest"),
    (c2, "R² Score",         f"{r2:.3f}"),
    (c3, "Mean Abs. Error",  f"₹{mae:,.0f}"),
    (c4, "Training Samples", f"{len(df_raw):,}"),
]:
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{val}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FORM
# ─────────────────────────────────────────────
airlines     = sorted(df_raw['Airline'].dropna().unique())
sources      = sorted(df_raw['Source'].dropna().unique())
destinations = sorted(df_raw['Destination'].dropna().unique())
add_infos    = sorted(df_raw['Additional_Info'].dropna().unique())
stop_options = ['non-stop','1 stop','2 stops','3 stops','4 stops']

st.markdown('<div class="form-header">🔍 Enter Flight Details</div>', unsafe_allow_html=True)

with st.form("prediction_form"):
    c1, c2, c3 = st.columns([1.5,1,1])
    with c1: airline  = st.selectbox("✈️ Airline",         airlines,     index=list(airlines).index("IndiGo") if "IndiGo" in airlines else 0)
    with c2: source   = st.selectbox("🛫 Source (From)",   sources,      index=list(sources).index("Delhi") if "Delhi" in sources else 0)
    with c3:
        dest_opts   = [d for d in destinations if d != source]
        destination = st.selectbox("🛬 Destination (To)", dest_opts, index=dest_opts.index("Cochin") if "Cochin" in dest_opts else 0)

    c4, c5, c6 = st.columns(3)
    with c4:
        journey_date = st.date_input("📅 Date of Journey",
            value=pd.Timestamp("2019-05-15"),
            min_value=pd.Timestamp("2019-01-01"),
            max_value=pd.Timestamp("2019-12-31"))
    with c5:
        dep_hour = st.slider("🕐 Departure Hour", 0, 23, 10)
        dep_min  = st.selectbox("Departure Minutes", [0,5,10,15,20,25,30,35,40,45,50,55])
    with c6:
        arr_hour = st.slider("🕑 Arrival Hour", 0, 23, 13)
        arr_min  = st.selectbox("Arrival Minutes", [0,5,10,15,20,25,30,35,40,45,50,55])

    c7, c8, c9 = st.columns(3)
    with c7:
        dur_hours = st.number_input("⏱️ Duration (Hours)", 0, 24, 2)
        dur_mins  = st.number_input("Duration (Minutes)",  0, 59, 30, step=5)
    with c8:
        total_stops = st.selectbox("🔁 Number of Stops", stop_options)
    with c9:
        additional_info = st.selectbox("ℹ️ Additional Info", add_infos,
            index=list(add_infos).index("No info") if "No info" in add_infos else 0)

    submitted = st.form_submit_button("🛫  PREDICT TICKET PRICE  ✈", use_container_width=True)

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
if submitted:
    duration_total = dur_hours * 60 + dur_mins
    input_dict = {
        'Journey_Day': journey_date.day, 'Journey_Month': journey_date.month,
        'Dep_Hour': dep_hour, 'Dep_Min': dep_min,
        'Arr_Hour': arr_hour, 'Arr_Min': arr_min,
        'Duration_Min': duration_total, 'Stops': stops_to_num(total_stops),
    }
    for col in feature_cols:
        if col not in input_dict: input_dict[col] = 0
    for prefix, value in [('Airline_',airline),('Source_',source),('Destination_',destination),('Additional_Info_',additional_info)]:
        col_name = f"{prefix}{value}"
        if col_name in input_dict: input_dict[col_name] = 1

    input_df   = pd.DataFrame([input_dict])[feature_cols]
    prediction = model.predict(input_df)[0]
    low, high  = prediction * 0.90, prediction * 1.10

    # ── Takeoff animation ──
    st.markdown("""
    <div class="takeoff-wrapper" style="height:80px;display:flex;align-items:center;justify-content:center;">
      <div class="takeoff-plane">✈️</div>
      <div class="contrail-trail"></div>
    </div>""", unsafe_allow_html=True)

    # ── Boarding pass ──
    # Build barcode bars (decorative, seeded by price)
    np.random.seed(int(prediction))
    bars_html = ""
    for _ in range(40):
        w = np.random.randint(1, 5)
        h = np.random.randint(20, 36)
        d = np.random.randint(0, 800)
        bars_html += f'<div class="barcode-bar" style="width:{w}px;height:{h}px;animation-delay:{d}ms;"></div>'

    st.markdown(f"""
    <div class="boarding-pass">
      <div class="bp-main">
        <!-- Left: main info -->
        <div>
          <div class="bp-route">
            <div>
              <div class="bp-city">{source[:3].upper()}</div>
              <div class="bp-city-name">{source}</div>
            </div>
            <div class="bp-arrow-wrap">
              <div class="bp-plane-icon">✈</div>
              <div class="bp-route-line"></div>
            </div>
            <div>
              <div class="bp-city">{destination[:3].upper()}</div>
              <div class="bp-city-name">{destination}</div>
            </div>
          </div>
          <div class="bp-info-grid">
            <div class="bp-info-item">
              <div class="bp-info-label">Airline</div>
              <div class="bp-info-value">{airline}</div>
            </div>
            <div class="bp-info-item">
              <div class="bp-info-label">Date</div>
              <div class="bp-info-value">{journey_date.strftime('%d %b %Y')}</div>
            </div>
            <div class="bp-info-item">
              <div class="bp-info-label">Departure</div>
              <div class="bp-info-value">{dep_hour:02d}:{dep_min:02d}</div>
            </div>
            <div class="bp-info-item">
              <div class="bp-info-label">Arrival</div>
              <div class="bp-info-value">{arr_hour:02d}:{arr_min:02d}</div>
            </div>
            <div class="bp-info-item">
              <div class="bp-info-label">Duration</div>
              <div class="bp-info-value">{dur_hours}h {dur_mins}m</div>
            </div>
            <div class="bp-info-item">
              <div class="bp-info-label">Stops</div>
              <div class="bp-info-value">{total_stops}</div>
            </div>
          </div>
        </div>
        <!-- Right stub: price + barcode -->
        <div class="bp-stub">
          <div class="bp-price-label">Estimated Fare</div>
          <div class="bp-price-big">₹{prediction:,.0f}</div>
          <div class="bp-range">
            Likely range<br>
            ₹{low:,.0f} – ₹{high:,.0f}
          </div>
          <div class="bp-barcode">{bars_html}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Summary panel ──
    details = [
        ("✈️ Airline", airline),
        ("🛫 Route",   f"{source} → {destination}"),
        ("📅 Date",    journey_date.strftime("%d %b %Y")),
        ("🕐 Departs", f"{dep_hour:02d}:{dep_min:02d}"),
        ("🕑 Arrives", f"{arr_hour:02d}:{arr_min:02d}"),
        ("⏱️ Duration", f"{dur_hours}h {dur_mins}m"),
        ("🔁 Stops",   total_stops),
        ("💰 Estimate", f"₹{prediction:,.0f}"),
        ("📊 Range",   f"₹{low:,.0f} – ₹{high:,.0f}"),
    ]
    rows_html = "".join(
        f'<div class="summary-row" style="animation-delay:{i*0.05:.2f}s">'
        f'<span class="summary-key">{k}</span>'
        f'<span class="summary-val">{v}</span>'
        f'</div>'
        for i,(k,v) in enumerate(details)
    )
    st.markdown(f"""
    <div class="summary-panel">
      <div style="font-family:var(--font-mono);font-size:.75rem;color:var(--teal);
                  letter-spacing:2px;text-transform:uppercase;margin-bottom:12px;">
        📋 Booking Summary
      </div>
      {rows_html}
    </div>""", unsafe_allow_html=True)

    # ── Feature importance expander ──
    with st.expander("📊 Top 15 Feature Importances — What drives the price?"):
        imp_df = (
            pd.DataFrame({'Feature': feature_cols, 'Importance': model.feature_importances_})
            .sort_values('Importance', ascending=False).head(15)
        )
        max_imp = imp_df['Importance'].max()
        for i, row in enumerate(imp_df.itertuples()):
            pct = row.Importance / max_imp
            pretty = (row.Feature
                .replace('Airline_','✈️ ').replace('Source_','🛫 ')
                .replace('Destination_','🛬 ').replace('Additional_Info_','ℹ️ ')
                .replace('_',' '))
            delay = i * 0.06
            st.markdown(f"""
            <div class="feat-row" style="animation-delay:{delay:.2f}s">
              <span class="feat-name">{pretty}</span>
              <div class="feat-bar-wrap">
                <div class="feat-bar" style="width:{pct*100:.1f}%;animation-delay:{delay:.2f}s"></div>
              </div>
              <span class="feat-score">{row.Importance:.4f}</span>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
  ✈ Flight Price Predictor &nbsp;·&nbsp;
  <span>Random Forest</span> &nbsp;·&nbsp;
  Indian Domestic Flights 2019 &nbsp;·&nbsp;
  <span>Powered by Scikit-Learn + Streamlit</span>
</div>""", unsafe_allow_html=True)