🍔 FastFood Chatbot (Transformer + Gradio)

This project is a chatbot for fast-food ordering and queries, built using Transformers and fine-tuned on a synthetic dataset.
It provides an easy-to-use Gradio interface for user interaction.

📂 Project Structure

FastFood-Chatbot/

│── app.py                          # Gradio app for chatbot interface

│── Final.ipynb                     # Jupyter Notebook (training + testing)

│── chatbot_combined_dataset.csv    # Synthetic dataset used for fine-tuning

│── Test_Data_Sample.csv            # Evaluation dataset

│── Guide.txt                       # Project notes/instructions

│── README.md                       # Documentation

│── model/                          # (Not uploaded due to large size)


⚡ Features

Fine-tuned Transformer model for fast-food domain conversations.

Supports queries like menu items, order details, offers, and basic small talk.

Gradio-powered interface for real-time chatbot interaction.

Training notebook (Final.ipynb) contains fine-tuning process with synthetic dataset.

🚀 How to Run

Since the fine-tuned model is very large (>100 MB), it is not uploaded to GitHub.
Instead, you can run the notebook to download and fine-tune the model automatically.

1️⃣ Clone the repository

git clone https://github.com/your-username/FastFood-Chatbot.git

cd FastFood-Chatbot

2️⃣ Install dependencies

pip install -r requirements.txt

3️⃣ Run the Jupyter Notebook

jupyter notebook Final.ipynb


Executes training and evaluation pipeline.

Downloads a pre-trained Transformer model from Hugging Face automatically.

4️⃣ Run the Gradio App

python app.py

This will open a Gradio interface in your browser, where you can chat with the FastFood Bot.

📌 Notes

Model folder is not included because of GitHub’s 100MB limit.

The code will automatically pull the base Transformer model from Hugging Face.

For reproducibility, you can upload your fine-tuned model to Hugging Face Hub and load it in app.py.
