<p align="center">
  <img src="assets/logo.png" alt="Dark Trace AI Logo" width="220"/>
</p>

# Dark Trace AI  
**AI-Powered Detection of Suspicious Chat & Transaction Patterns**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) 
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/) 
[![Repo](https://img.shields.io/badge/GitHub-darktraceai-blue?logo=github)](https://github.com/your-username/DarkTraceAI)

---

## 📱 Overview
**Dark Trace AI** is a full-stack research & prototype system that detects and triages suspicious activity across text chats and transaction metadata. The system combines:

- curated **slang & emoji dictionaries** (textual + emoji-coded signals),  
- a **rules engine** (high-precision red flags from lexica and weight/price tokens),  
- **ML models** (tabular + hybrid features: lexical, behavioral, graph), and  
- a lightweight **analyst dashboard** for explanation and triage.

It is built for lawful research and authorized investigative workflows (see **Ethics & Compliance**).

---

## 🔖 Table of contents
- [Overview](#overview)  
- [✨ Features](#-features)  
- [🛠️ Tech stack](#-tech-stack)  
- [🚀 Getting started](#-getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
  - [Configuration](#configuration)  
  - [Quick demo (local)](#quick-demo-local)  
- [📁 Project structure](#-project-structure)  
- [📂 Datasets](#-datasets)  
- [⚙️ How it works (pipeline)](#%EF%B8%8F-how-it-works-pipeline)  
- [🧪 Training & evaluation](#-training--evaluation)  
- [🧭 Serving & API](#-serving--api)  
- [🎛️ UI / Dashboard features](#%EF%B8%8F-ui--dashboard-features)  
- [🔐 Ethics, privacy & limitations](#-ethics-privacy--limitations)  
- [🛣️ Roadmap](#-roadmap)  
- [📄 License](#-license)  
- [🙏 Acknowledgements & contact](#-acknowledgements--contact)

> Click any item above to jump to that section.

---

## ✨ Features
- Hybrid detection: **rules + ML** (XGBoost/LightGBM) for fast triage.  
- **Slang & Emoji dictionaries** (text and emoji mapping CSVs included).  
- **Behavioral metadata analysis**: frequency, session activity, group churn.  
- **Network/graph analysis**: user graph, centrality, community detection (NetworkX / Neo4j optional).  
- **Explainability**: SHAP explanations + rule reasons returned with each alert.  
- REST API to score messages and bulk CSV scoring.  
- React / Streamlit-based dashboard with alert list, detail view, and network visualization.  
- Dockerized for easy demo packaging.

---

## 🛠️ Tech stack
- **Language:** Python 3.10+  
- **NLP / ML:** HuggingFace Transformers (optional), scikit-learn, XGBoost, LightGBM, SHAP  
- **Graph:** NetworkX, (optional Neo4j)  
- **Backend:** FastAPI (model serving)  
- **Frontend:** React (Vite) and/or Streamlit for quick demos  
- **Data:** Pandas, Parquet for feature artifacts  
- **Dev / Deploy:** Docker, docker-compose, GitHub Actions (CI)  

---

## 🚀 Getting started

### Prerequisites
- Python 3.10+  
- Git  
- Docker & Docker Compose (optional, for containerized run)  
- (Optional) Node.js & npm for React UI

### Installation (local dev)
```bash
# clone
git clone https://github.com/your-username/DarkTraceAI.git
cd DarkTraceAI

# python env
python -m venv .venv
source .venv/bin/activate     # macOS / Linux
# .venv\Scripts\activate      # Windows

pip install -r requirements.txt
