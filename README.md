# 📊 StatQuickie: Your Fast Lane to Data Insights!

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-brightgreen?logo=streamlit)](https://statquickie.streamlit.app/)
[![Made with Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-red?logo=streamlit)](https://streamlit.io/)

Say hello to **StatQuickie** — the playful, no-nonsense stats explorer that helps you *understand your data* in record time.  
Upload a file, sip your coffee, and boom — insight! ☕📈

👉 **Try it live:** [https://statquickie.streamlit.app](https://statquickie.streamlit.app)

---

## 🚀 What’s Inside?

- 🗂 Upload CSV or Excel files
- 🔍 Auto-detects numbers, categories, and dates
- 🧠 Layman-friendly summaries (e.g., “tightly clustered” vs “widely spread”)
- 📊 Visualize with Plotly + Matplotlib (histograms, KDE, ECDFs, etc.)
- 🧪 Run t-tests, fit regression lines, get R² and MSE instantly
- 🎛 Interactive UI with Streamlit — no code needed!

---

## 🛠 Installation

### Option 1: One-Click macOS Installer

```bash
bash installer-macos-universal.sh
```

This will:
- Detect your Mac architecture (Intel or Apple Silicon)
- Install Miniforge (if needed)
- Create the `statquickie` environment
- Add Desktop shortcut to launch the app【93†source】

---

### Option 2: One-Click Windows Installer

```powershell
Right-click → Run with PowerShell → installer-windows.ps1
```

This will:
- Detect Anaconda/Miniconda installation
- Create or update `statquickie` conda environment using `__environment__.yml`
- Create a launcher script and desktop shortcut
- Generate an uninstaller for cleanup

> 💡 **Note:** Ensure Conda is installed before running.

---

### Option 3: Manual Setup

```bash
git clone https://github.com/your-username/statquickie.git
cd statquickie
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔧 Dependencies

From `requirements.txt`:

- `streamlit`, `pandas`, `numpy`, `matplotlib`, `plotly`, `openpyxl`, `xlrd`
- `scikit-learn`, `scipy`, `lightgbm`【95†source】

---

## 🎯 Why Use StatQuickie?

Because not everyone has time to write Python scripts or decipher p-values.  
StatQuickie lets you:
- Get the story behind the numbers
- Show off visual insights in seconds
- Wow your colleagues (or your future self)

Whether you're a data newbie or seasoned analyst, StatQuickie makes stats feel less... staticky.

---

## 🤝 Contribute

Pull requests welcome! Open an issue, suggest features, or drop by with a virtual high-five ✋  
Let’s make stats less scary, together.

---

## 📄 License

MIT License — do what you want, just don’t blame us if your boss loves it too much.

---

## 🙌 Acknowledgements

Thanks to [OpenAI's ChatGPT](https://chat.openai.com/) for helping brainstorm, draft, and polish this README — and making documentation (and stats) a lot more fun.
