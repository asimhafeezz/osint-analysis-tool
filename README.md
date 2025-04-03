# OSINT Company Analysis Tool

An AI-powered multi-agent tool for performing deep Open-Source Intelligence (OSINT) analysis on any company using advanced LLMs, web search, and structured research agents.

## Overview

The tool takes a company name as input and performs the following core OSINT tasks:

* **Gathers Fundamental Information:** Identifies the company's website, a brief overview, main social media profiles, recent news, and general public sentiment.
* **Investigates Technical and Legal Aspects:** Looks into domain registration details and any readily available information about website security, regulatory filings, or significant legal issues.
* **Generates a Concise Report:** Compiles the findings into a single, easy-to-understand report displayed in the application.

## Technologies Used

* **Python:** The primary programming language.
* **Streamlit:** For creating the interactive web user interface.
* **CrewAI:** A framework for orchestrating autonomous AI agents to work collaboratively.
* **Langchain:** Provides the foundation for working with Language Models and tools.
* **Ollama:** Used as the local Language Model (specifically `llama3.1`).
* **Serper API:** For performing web searches to gather information.
* **dotenv:** For managing API keys and environment variables.

## Prerequisites

Before running the application, ensure you have the following:

* **Python 3.7+** installed on your system.
* **pip** (Python package installer) installed.
* **Ollama** installed and running locally. You need to have the `llama3.1` model available in Ollama. You can download it by running `ollama pull llama3.1` in your terminal.
* **A Serper API Key:** You can obtain a free API key from [https://serper.dev/](https://serper.dev/).

## Setup

1.  **Clone the repository (if applicable) or create a new project directory.**
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On macOS/Linux
    .venv\Scripts\activate  # On Windows
    ```
3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```
    (If you don't have a `requirements.txt` file, create one with the following content and then run the command above):
    ```
    streamlit
    crewai
    langchain
    langchain-community
    python-dotenv
    requests
    ```
4.  **Create a `.env` file in the root of your project directory.**
5.  **Add your Serper API key to the `.env` file:**
    ```
    SERPER_API_KEY=YOUR_SERPER_API_KEY
    ```
    Replace `YOUR_SERPER_API_KEY` with your actual API key.

## Running the Application

1.  **Ensure your virtual environment is activated:**
    ```bash
    source .venv/bin/activate  # On macOS/Linux
    .venv\Scripts\activate  # On Windows
    ```
2.  **Navigate to the root of your project directory in the terminal.**
3.  **Run the Streamlit application:**
    ```bash
    streamlit run your_script_name.py
    ```
    Replace `your_script_name.py` with the name of the Python file containing the code (e.g., `osint_app.py`).

4.  **The application will open in your web browser.** Enter the name of the company you want to analyze in the provided text input field and press Enter.

5.  **The application will display a spinner while the analysis is in progress.** Once complete, the OSINT analysis report will be shown on the screen.

## Understanding the Output

The report will contain a concise summary of the publicly available information gathered by the AI agents, including:

* The company's official website URL.
* A brief overview of the company.
* Links to its main social media profiles (if found).
* A summary of recent news articles.
* A general overview of public perception.
* Domain registration details.
* A brief note on website security (SSL certificate).
* Any identified significant regulatory or legal information (if readily available).

Potential areas for future development include:

* Adding more specialized OSINT agents to investigate specific aspects in greater depth.
* Integrating more diverse OSINT tools and APIs.
* Improving the analysis and interpretation of the gathered information.
* Adding features for exporting the report in different formats.
* Enhancing the user interface and providing more control over the analysis process.
