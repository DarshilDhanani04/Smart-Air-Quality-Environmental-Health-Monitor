# 🌫️ Smart Air Quality & Environmental Health Monitor

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B?logo=streamlit)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-green)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq-AI-orange)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **GTU Python for Data Science Project** — A Streamlit dashboard that analyses urban air pollution data using Pandas, NumPy, SciPy, Matplotlib and Seaborn, with an AI health advisory layer powered by LangChain and Groq.

---

## 📸 Features

| Tab | What it does | Syllabus Topics |
|-----|-------------|-----------------|
| 📊 **Data & Statistics** | Cleans CSV (missing sensor readings, duplicates). Shows metric cards and a table of mean, median, mode, std dev, variance, IQR, skewness and kurtosis | Pandas, NumPy, SciPy |
| 📈 **Pollution Visualization** | Bar chart of most polluted cities, time-series trend, histogram and correlation heatmap | Matplotlib, Seaborn |
| 📝 **AI Health Advisory** | A LangChain PromptTemplate passes computed statistics to a Groq LLM, which writes an official health advisory | LangChain, Groq |
| 💬 **Chat with Sensor Data** | A LangChain tool-calling agent writes pandas code, runs it on the data, and answers in plain English | Agentic AI |

---

## 🚀 How to Run the Project (Step-by-Step CMD Commands)

Follow these steps exactly in **Command Prompt (CMD)** or **PowerShell**.

---

### ✅ Prerequisites

Make sure the following are installed on your system before starting:

| Tool | Minimum Version | Download |
|------|----------------|---------|
| Python | 3.10+ | https://www.python.org/downloads/ |
| Git | Any | https://git-scm.com/downloads |
| pip | Bundled with Python | — |

To verify your installations, open CMD and run:

```cmd
python --version
pip --version
git --version
```

---

### Step 1 — Clone the Repository

Open **Command Prompt (CMD)** and run:

```cmd
git clone https://github.com/DarshilDhanani04/Smart-Air-Quality-Environmental-Health-Monitor.git
```

Then move into the project folder:

```cmd
cd Smart-Air-Quality-Environmental-Health-Monitor
```

---

### Step 2 — Create a Virtual Environment

```cmd
python -m venv .venv
```

> This creates an isolated Python environment inside a `.venv` folder so dependencies do not conflict with your system Python.

---

### Step 3 — Activate the Virtual Environment

**On Windows (Command Prompt):**
```cmd
.venv\Scripts\activate
```

**On Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
source .venv/bin/activate
```

> ✅ After activation you will see `(.venv)` at the beginning of your terminal prompt, e.g. `(.venv) C:\...>`

---

### Step 4 — Install All Dependencies

```cmd
pip install -r requirements.txt
```

> This automatically installs: `streamlit`, `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `tabulate`, `langchain-core`, `langchain-groq`, `groq`

---

### Step 5 — Run the Application

```cmd
streamlit run app.py
```

The app will automatically open in your default browser at:

```
http://localhost:8501
```

> If the browser does not open automatically, manually copy-paste `http://localhost:8501` into your browser.

---

### Step 6 — Get a Free Groq API Key (for AI features)

The AI Health Advisory and Chat tabs require a free Groq API key:

1. Go to → https://console.groq.com/keys
2. Sign up or log in with your Google / GitHub account
3. Click **"Create API Key"** and copy the key
4. Paste the key into the **sidebar** inside the running app

> The Statistics and Visualization tabs work **without** an API key.

---

### Step 7 — Load Your Dataset

Once the app is running, you have two options:

- **Upload a CSV file** using the file uploader in the sidebar, **OR**
- Check **"Use built-in sample data"** to explore the app instantly without any file

---

### 🛑 Stopping the App

To stop the Streamlit server, press **`Ctrl + C`** in your terminal.

To deactivate the virtual environment when you are done:

```cmd
deactivate
```

---

## 📂 Recommended Dataset

**Kaggle:** [Air Quality Data in India (2015–2020)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)

- File to download: **`city_day.csv`**
- Expected columns: `City`, `Date`, `PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, `O3`, `AQI`

The app also auto-detects common column variants such as `location`, `rspm`, `pm2_5`, and `sampling_date`.

---

## 🏗️ Project Structure

```
Smart-Air-Quality-Environmental-Health-Monitor/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
├── LICENSE             # MIT License
└── .gitignore          # Files excluded from version control
```

---

## 🧹 Data Cleaning Approach

- Rows with **unparseable dates** are dropped
- Rows where **every** pollutant reading is missing (dead sensor) are dropped
- Remaining gaps are filled with the **city-wise median**, falling back to the overall median
- **Duplicate rows** are removed

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥ 1.40 | Web dashboard framework |
| `pandas` | ≥ 2.2 | Data loading, cleaning, filtering |
| `numpy` | ≥ 1.26 | Numerical computations |
| `scipy` | ≥ 1.13 | Statistical analysis (IQR, skewness, kurtosis) |
| `matplotlib` | ≥ 3.8 | Chart rendering (bar, histogram, time-series) |
| `seaborn` | ≥ 0.13 | Statistical data visualisation |
| `tabulate` | ≥ 0.9 | Table formatting |
| `langchain-core` | ≥ 0.3 | LangChain core library |
| `langchain-groq` | ≥ 0.2 | Groq LLM integration for LangChain |
| `groq` | ≥ 0.30 | Groq Python SDK |

---

## ⚠️ Important Notes

- The **chat agent** executes Python code generated by the LLM. Run this app **only on your own machine** (not on a shared/public server without sandboxing).
- The **primary pollutant** is determined by the highest ratio of mean concentration to its Indian NAAQS 24-hour limit.
- A **Groq API key is required** for the AI tabs. The Statistics and Visualization tabs work without it.

---

## ❓ Troubleshooting

| Problem | Solution |
|---------|---------|
| `streamlit: command not found` | Make sure the virtual environment is **activated** (Step 3) |
| `ModuleNotFoundError` | Re-run `pip install -r requirements.txt` inside the activated venv |
| Browser does not open | Manually open `http://localhost:8501` in your browser |
| PowerShell script error | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` then retry Step 3 |
| Groq API errors | Double-check your API key is pasted correctly in the sidebar |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` to use a different port |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Darshil Dhanani** — GTU Python for Data Science Project
