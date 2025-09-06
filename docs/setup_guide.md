# Setup Guide

## Quick Start

1. Install Python 3.9+
2. Clone repository
3. Install dependencies: `pip install -e .`
4. Add OpenAI API key to `.env`
5. Add PDFs to `data/` folder
6. Process PDFs: `python -m psychmet_chatbot.cli process`
7. Run web app: `streamlit run src/psychmet_chatbot/app.py`

## Detailed Setup...