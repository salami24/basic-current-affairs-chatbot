Basic Current Affairs Chatbot

A simple, interactive Python chatbot that answers questions about United Nations member countries using a question-and-answer dataset and natural language similarity matching.

The application provides a web interface built with Streamlit, allowing users to enter questions and receive relevant answers from the available dataset.

Table of Contents

Overview
Features
Technologies Used
Project Structure
Prerequisites
Installation and Setup
Running the Application
How It Works
Dataset Format
Troubleshooting
Future Improvements
Contributing
License
Overview

The Basic Current Affairs Chatbot is designed to help users retrieve basic information about countries under the United Nations umbrella.

Instead of searching through a dataset manually, users can enter a question through a simple web interface. The application compares the question with questions in its dataset and returns the answer associated with the closest match.

This project demonstrates how Python, machine learning libraries, text similarity, and Streamlit can be combined to build a basic question-answering application.

Note: The current implementation retrieves answers from its existing dataset. It is not a live news service, and its answers are limited by the information available in that dataset.

Features

Interactive web interface: Ask questions using a Streamlit application.
Question similarity matching: Identify dataset questions that are semantically similar to user input.
Automated answer retrieval: Display the answer associated with the selected question.
CSV dataset support: Use structured question-and-answer data.
Alternative implementation: Includes a second application that uses a saved vectorizer and serialized dataset.
Beginner-friendly project: Provides a practical introduction to Python-based chatbot development.
🛠️ Technologies Used
Technology	Purpose
Python	Main programming language
Streamlit	Web application interface
Pandas	Dataset loading and processing
Sentence Transformers	Semantic text embeddings in app.py
Scikit-learn	Text similarity calculations in currentaffairs.py
PyTorch	Backend dependency used by the Sentence Transformers stack

Project Structure

The repository contains the following files:

text
basic-current-affairs-chatbot/
│
├── app.py
├── currentaffairs.py
├── UN_Countries_QA_History.csv
├── chatbot_dataset.pkl
├── currentaffairs_chatbot_vectorizer.pkl
├── requirements.txt
└── README.md

File descriptions
app.py: Main semantic-search chatbot implementation. Loads the CSV dataset, generates question embeddings, and returns the answer associated with the most similar question.
currentaffairs.py: Alternative Streamlit application that uses a previously saved vectorizer and dataset.
UN_Countries_QA_History.csv: Source dataset containing questions and corresponding answers.
chatbot_dataset.pkl: Serialized dataset used by the alternative implementation.
currentaffairs_chatbot_vectorizer.pkl: Serialized vectorizer used to transform questions into numerical features.
requirements.txt: Lists the dependencies required by the project.
README.md: Project documentation and setup instructions.

Prerequisites

Before installing the project, ensure you have:

Python installed on your computer. Python 3.10 or 3.11 is a reasonable starting point, although compatibility should be verified against the installed dependencies.
Git installed on your computer.
An internet connection for downloading dependencies and, on the first run of app.py, potentially downloading the Sentence Transformers model.

A terminal or command prompt.

Check your Python installation:

bash
python --version


Check your Git installation:

bash
git --version


If your system uses python3 instead of python, use python3 in the commands below.

Installation and Setup

Follow these steps to run the project locally.

Step 1: Clone the repository

Open your terminal and execute:

bash
git clone https://github.com/salami24/basic-current-affairs-chatbot.git

Step 2: Navigate to the project directory
bash
cd basic-current-affairs-chatbot

Step 3: Create a virtual environment

A virtual environment keeps this project's Python packages separate from other projects.

Windows:

bash
python -m venv venv
venv\Scripts\activate


macOS or Linux:

bash
python3 -m venv venv
source venv/bin/activate


After activation, your terminal should indicate that the virtual environment is active.

Step 4: Upgrade pip
bash
python -m pip install --upgrade pip

Step 5: Install the dependencies

Install the packages listed in the project's requirements file:

bash
pip install -r requirements.txt


The dependency list includes Streamlit, Pandas, Scikit-learn, Sentence Transformers, and PyTorch.

Installation may take several minutes, depending on your internet connection and operating system.

Step 6: Verify the project files

Confirm that the following files are present in the project directory:

text
app.py
UN_Countries_QA_History.csv
requirements.txt


If you intend to run the alternative application, also verify that these files exist:

text
currentaffairs.py
chatbot_dataset.pkl
currentaffairs_chatbot_vectorizer.pkl

Running the Application

The repository contains two application scripts. Start with app.py for the semantic-search implementation.

Option 1: Run the main chatbot

Execute:

bash
streamlit run app.py


Streamlit should start a local web server and display a URL in your terminal, commonly:

text
http://localhost:8501


Open that address in your web browser if it does not open automatically.

How to use it:

Wait for the application to load.
Locate the question input field.
Enter a question about a country or a topic covered by the dataset.
Review the answer displayed by the application.
Check the similarity confidence score shown beneath the answer.

The first launch may take longer because the Sentence Transformers model may need to be downloaded and initialized.

Option 2: Run the alternative chatbot

To launch the implementation that uses the saved vectorizer and dataset, execute:

bash
streamlit run currentaffairs.py


Open the local URL provided by Streamlit.

Enter a question in the input field and review the returned answer.

This implementation depends on the two existing pickle files. It does not use the CSV-loading and Sentence Transformer workflow implemented in app.py.

Security note: Only load pickle files from sources you trust. Pickle files can execute malicious code when deserialized.

Stop the application

When you want to stop the server, return to the terminal where Streamlit is running and press:

text
Ctrl + C

How It Works

Main implementation: app.py

The main application follows this workflow:

Load the dataset: Pandas reads UN_Countries_QA_History.csv.
Load the language model: Sentence Transformers initializes the all-MiniLM-L6-v2 model.
Generate question embeddings: The application converts dataset questions into numerical vectors.
Accept user input: Streamlit displays a text field for the user's question.
Process the question: The model converts the submitted question into an embedding.
Calculate similarity: Cosine similarity compares the submitted question with the dataset questions.
Select the best match: The application identifies the question with the highest similarity score.
Display the answer: The corresponding answer is shown along with a confidence-like similarity score.
Alternative implementation: currentaffairs.py

The alternative implementation follows a similar retrieval process, but it uses a saved vectorizer rather than generating semantic embeddings with Sentence Transformers.

Load the serialized vectorizer and dataset.
Transform the dataset questions into numerical features.
Transform the user's question using the same vectorizer.
Calculate cosine similarity between the question and dataset.
Retrieve and display the answer associated with the highest-scoring match.

The two implementations use different text-representation methods, so their matching results may differ.

Dataset Format

The main application expects the CSV file to contain two columns named exactly:

Question
Answer

Example:

Question	Answer
What is the capital of France?	Paris
What is the capital of Nigeria?	Abuja
Where is Japan located?	Japan is an island country in East Asia.

These examples illustrate the expected format; they are not a claim about the current contents of the repository's dataset.

When creating or updating the dataset:

Use a question column and a corresponding answer column.
Keep the column names consistent with the code.
Save the file in CSV format.
Place the file in the project directory or update the file path in the application.
Restart the application after changing the dataset.

The alternative implementation expects its saved dataset to contain Question and Answer columns as well.

Troubleshooting

1. ModuleNotFoundError

Cause: One or more Python packages may be missing.

Solution:

Activate your virtual environment and run:

bash
pip install -r requirements.txt

2. CSV file not found

Cause: The CSV file is missing or the application is being launched from an unexpected working directory.

Solution:

Confirm that UN_Countries_QA_History.csv exists in the project directory, then run:

bash
cd basic-current-affairs-chatbot
streamlit run app.py

3. Missing pickle files

Cause: The alternative application cannot locate its saved vectorizer or dataset.

Solution:

Confirm that both chatbot_dataset.pkl and currentaffairs_chatbot_vectorizer.pkl are present in the expected directory.

Only use trusted pickle files.

4. Model download or loading errors

Cause: The Sentence Transformers model may not have downloaded successfully, or the installed dependencies may be incompatible.

Solution:

Check your internet connection.
Review the error message in the terminal.
Confirm that the virtual environment is active.
Verify the compatibility of your Python, PyTorch, and Sentence Transformers versions.
5. Unexpected or incorrect answers

Cause: The chatbot retrieves an answer from its existing dataset, so an incomplete dataset or a weak similarity match can produce an irrelevant response.

Solution:

Try rephrasing the question.
Ensure the dataset contains relevant questions and accurate answers.
Review the similarity score when using app.py.
Consider implementing a minimum similarity threshold to reject weak matches.
6. Streamlit command not found

Cause: Streamlit may not be installed in the active Python environment.

Solution:

bash
python -m pip install streamlit
python -m streamlit run app.py

Future Improvements

Possible enhancements for the project include:

Add a minimum similarity threshold to handle questions outside the dataset.
Expand the dataset with verified information about UN member states.
Include country profiles, capitals, official languages, currencies, and population statistics.
Improve error handling and validate dataset columns at startup.
Add automated tests for data loading and answer retrieval.
Improve the user interface with chat history and clearer response formatting.
Add a verified, regularly updated source for genuinely current news.
Deploy the application to a suitable hosting platform.

Contributing

Contributions, suggestions, and bug reports are welcome.

To contribute:

Fork this repository.
Create a new branch for your changes.
Implement and test your changes locally.
Commit your changes with a descriptive message.
Push your branch to your fork.
Open a pull request describing your changes.

For bugs or feature requests, use the repository's GitHub Issues section.

License

No license information has been confirmed for this project. Before distributing or reusing the code, add an appropriate license file and update this section to reflect the chosen license.

Author

GitHub: @salami24

Support

If you find this project useful, consider starring the repository and sharing suggestions for future improvements.

Repository: https://github.com/salami24/basic-current-affairs-chatbot
