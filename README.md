🥦 Vegetable Classification Web App
A simple web application built with Flask and TensorFlow to classify images of vegetables using different pretrained deep learning models.

🚀 Features
Upload an image of a vegetable

Choose from 5 different pretrained models:
- baseline model (MobileNet)
- Add 64 neuron hidden layer with 5 epoch
- Add 128 neuron hidden layer with 5 epoch
- Add 64 neuron hidden layer with 10 epoch
- Add 128 neuron hidden layer with 10 epoch

🧠 Model Classes
The model supports classification of the following vegetables:
["Bean", "Bitter_Gourd", "Bottle_Gourd", "Brinjal", "Broccoli",
 "Cabbage", "Capsicum", "Carrot", "Cauliflower", "Cucumber",
 "Papaya", "Potato", "Pumpkin", "Radish", "Tomato"]

⚙️ Installation & Running
1. Clone the repository
2. Create virtual environment
  - python3 -m venv venv
  - source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies
  - pip install -r requirements.txt
4. Run the app
  - python app.py
5. Open in browser
  - Go to http://127.0.0.1:5000/ to use the app. 
