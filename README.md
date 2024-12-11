# README: How to Run the Application

### Authors: Thomas Kudey, Chiron Martini, Ryan Corvi

This guide explains how to set up and run the Shiny-based application for analyzing stock data and articles.

---

## Prerequisites

### Ensure You Have the Following Installed:

1. **Python 3.9 or Later**
   - **Check if Python is installed**:  
     ```bash
     python3 --version
     ```
   - **Install Python**:  
     Download and install it from [Python's official website](https://www.python.org/).

2. **pip (Python Package Manager)**
   - **Check if pip is installed**:  
     ```bash
     pip --version
     ```
   - **Install pip**:  
     ```bash
     python3 -m ensurepip --upgrade
     ```

3. **Git**
   - **Check if Git is installed**:  
     ```bash
     git --version
     ```
   - **Install Git**:  
     Download and install it from [Git's official website](https://git-scm.com/).

4. **Google Chrome**
   - Download and install Chrome from [Google's Chrome website](https://www.google.com/chrome/).

5. **ChromeDriver**
   - Ensure the ChromeDriver version matches your Chrome browser version.
   - **Download ChromeDriver**:  
     From [ChromeDriver's official website](https://sites.google.com/chromium.org/driver/).
   - **Place ChromeDriver in your system's PATH**:  
     ```bash
     sudo mv chromedriver /usr/local/bin/
     ```

6. **Git LFS (Git Large File Storage)**
   - **Install Git LFS**:  
     ```bash
     git lfs install
     ```
   - **Verify installation**:  
     ```bash
     git lfs version
     ```

## Steps to Run the Program

### 1. Clone the Repository
Clone the Git repository to your local machine:
```bash
git clone <repository_url>
cd <repository_name>
```

### 2. Pull Large CSV Files Using Git LFS

Ensure the large CSV files tracked by Git LFS are downloaded:

```
git lfs pull
```

### 3. Install Python Dependencies

Install all required Python dependencies listed in the requirements.txt file:

```
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a key.env file in the project root directory and add your OpenAI API key:

```
OPENAI_API_KEY=<your_openai_api_key>
```

## Run the Program

To start the Shiny application with live reloading, use the following command:
```
shiny run --reload app.py
```
The application will launch and display the dashboard. Open your browser to the URL displayed in the terminal (usually http://127.0.0.1:8000).

Application Features
	1.	Input Field:
	•	Enter a stock ticker (e.g., AAPL for Apple, NVDA for NVIDIA) to analyze.
	2.	Stock Data and Articles:
	•	Fetch and display real-time stock data using Yahoo Finance and Selenium.
	•	Retrieve and process related news articles.
	3.	Analysis:
	•	Predict the t+1 stock price using a pretrained XGBoost model.
	•	Extract insights (sentiment, mentions, urgency) from articles using OpenAI’s GPT model.

Troubleshooting

Missing or Corrupted CSV File
	•	Ensure you pulled the correct large files using Git LFS (git lfs pull).
	•	Confirm the data/prepared_for_model.csv file exists and is in the correct directory.

Selenium Issues
	•	Ensure ChromeDriver is installed, and its version matches your Chrome browser.
	•	If ChromeDriver isn’t in your PATH, provide its location explicitly in the input_ticker_obj.py file.

OpenAI API Issues
	•	Verify your API key is set correctly in key.env.
	•	Ensure you have sufficient quota and permissions for the OpenAI API.

Customization

You can modify various aspects of the application:
	•	UI Colors and Layout:
	•	Update the CSS styles in app.py to change the look and feel.
	•	Pretrained Model:
	•	Replace trained_xgb_model.json with a custom XGBoost model.
	•	API Integration:
	•	Enhance the OpenAI GPT prompts in input_ticker_obj.py to generate different insights.

For any issues, feel free to raise a ticket in the repository. Enjoy using the application!

