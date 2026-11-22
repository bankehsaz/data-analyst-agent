# Data Analyst Agent

An AI-powered data analysis assistant that allows users to upload CSV datasets and analyze them using natural language.

The application uses a local LLM through Ollama and Python tools to perform data analysis and generate visualizations.

## Features

- Upload CSV datasets
- Preview dataset information
- Ask questions about data using natural language
- Generate descriptive statistics
- Analyze missing values and duplicate records
- Calculate correlations between numerical columns
- Generate visualizations:
  - Histograms
  - Box plots
  - Scatter plots
  - Bar charts
  - Line charts
  - Correlation heatmaps
- Local LLM integration using Ollama
- Tool Calling for executing Python functions

## Technologies Used

- Python
- Pandas
- Matplotlib
- Streamlit
- Ollama
- Qwen2.5 (0.5B)

## Project Structure

```text
data-analyst-agent/
│
├── agent.py
├── tools.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd data-analyst-agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Install Ollama and download the model

Install Ollama from:

https://ollama.com/

Download the required model:

```bash
ollama pull qwen2.5:0.5b
```

Make sure Ollama is running before starting the application.

### 6. Run the application

```bash
streamlit run app.py
```

Open the local URL provided by Streamlit in your browser.

## Usage

1. Upload a CSV dataset.
2. Review the dataset preview and information.
3. Enter a question about your data.
4. Click **Analyze Data**.
5. Review the analysis results and generated charts.

## Project Purpose

This project was developed as a practical learning project to explore AI Agents, LLM integration, Tool Calling, Python-based data analysis, and interactive data applications.

## Limitations and Future Improvements

- This project currently uses Qwen2.5 (0.5B), a lightweight local language model selected to run on hardware with limited computational resources.
- Due to its small size, the model may occasionally struggle with tool selection, argument generation, and multi-step reasoning.
- Using more capable language models is recommended for improved tool-calling accuracy and more reliable analytical workflows.

Future improvements may include:
- Integrating larger and more capable LLMs.
- Improving tool selection and argument validation.
- Enhancing multi-step reasoning and agent workflows.
- Adding support for more data formats.