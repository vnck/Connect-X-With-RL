# Connect-X with Reinforcement Learning
Final Project for 50.021 Artificial Intelligence.

A web application that allows you to play the classical game of connect 4 against reinforcement learning models, or watch the  models compete against each other.

Agents Available: 
- Human
- Alpha Zero (Monte Carlo Tree Search)
- Alpha Zero (Greedy)
- Negamax
- Random

## Frontend
A react application.

### Install Requirements
```sh
cd frontend
yarn install
```

### Running the React Application
```sh
cd frontend
yarn start
```

## Backend
A flask application.

### Install Requirements
```sh
cd backend
pip install -r requirements.txt
```

### Running the Flask Application
```sh
cd backend
python app.py
```

## Models
We implemented a Dense Deep Q Learning (DQN) model, as well as a Alpha0 model in our application. The model architectures and training pipelines can be viewed in their respective notebooks.

We also have notebooks exploring Double DQN and Convolutional DQN, but these were not added into our final application.

---
Project Members: Gary, Ivan, Joshua, Zhi Yao