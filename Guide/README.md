# This file has the explanation to all the codes in the project .

## The main directories in repo are :
  - [data](https://github.com/nikhilsp242/Mental-Health-Prediction/tree/main/data) 
  - [examples](https://github.com/nikhilsp242/Mental-Health-Prediction/tree/main/examples)
      * The files in the "examples" folder are configuration files. These configurations define settings for different aspects of a machine learning or deep learning project, such as preprocessing parameters, model architectures, and training configurations. Configuration files are commonly used to easily customize and manage the various parameters and settings in a project, making it more modular and adaptable.
  - [src](https://github.com/nikhilsp242/Mental-Health-Prediction/tree/main/src)

## Architecture diagrams :
### 1.

![image](https://github.com/nikhilsp242/Mental-Health-Prediction/assets/112267674/b543f21d-1baf-48a9-9991-4416742049e9)


This architecture diagram outlines a common structure used in natural language processing (NLP) tasks, particularly for tasks like sentiment analysis or text classification. Let's break down each component:

1. **Input IDs:**
   - These are typically integer values representing the unique identifier assigned to each word or token in a given input text sequence. Each word/token has a corresponding ID in a vocabulary.

2. **Embedding Layer:**
   - The Embedding Layer takes the input IDs and converts them into dense vectors of fixed size, often referred to as embeddings. Each unique ID is associated with a unique vector representation.
   - The Embedding Layer is responsible for capturing semantic relationships between words, placing similar words closer together in the embedding space.

3. **LSTM Cell (n layers = 2):**
   - LSTM stands for Long Short-Term Memory, which is a type of recurrent neural network (RNN) cell. The "n layers = 2" indicates that there are two stacked LSTM layers.
   - LSTM cells are designed to capture dependencies and patterns in sequential data, making them suitable for processing sequences like sentences.
   - The LSTM layers process the sequence of word embeddings and update their internal states at each step.

4. **Representations:**
   - The output from the LSTM layers represents the contextual information and dependencies learned from the input sequence. Each time step in the sequence produces a corresponding representation.

5. **Output Layer:**
   - The output from the LSTM layers is fed into an additional layer, often a dense layer, which produces raw output scores or logits.
   - This layer helps transform the learned representations into a format suitable for the final classification.

6. **Sigmoid Activation:**
   - The raw output scores (logits) are passed through a Sigmoid activation function. This function squashes the values between 0 and 1, producing probabilities.
   - This step is common in binary classification problems where you're predicting probabilities for two classes (e.g., positive and negative sentiment).

7. **Probabilities (probs):**
   - The final output consists of probabilities representing the likelihood of the input sequence belonging to a particular class. In binary classification, there are two probabilities, often summing up to 1.

In summary, this architecture leverages an Embedding Layer to convert input IDs into dense vectors, stacked LSTM layers to capture sequential dependencies, and a final layer with a Sigmoid activation for binary classification. This structure is commonly used for tasks where understanding the sequential nature of the input data is crucial.


### 2.
![image](https://github.com/nikhilsp242/Mental-Health-Prediction/assets/112267674/0b4db900-494f-4e2b-901a-a4e5c437d671)


This diagram outlines the operations within an LSTM (Long Short-Term Memory) cell, which is a type of recurrent neural network (RNN) architecture. The LSTM cell is designed to capture and remember long-term dependencies in sequential data. Let's break down the components and operations:

1. **Input (Xₜ):**
   - \(Xₜ\) represents the input at the current time step (\(t\)) in the sequence. This could be the word embedding or input representation for the current word/token.

2. **Previous Hidden State (hₜ₋₁):**
   - \(hₜ₋₁\) is the hidden state from the previous time step (\(t-1\)). It carries information from the past into the current time step and acts as the "memory" of the LSTM cell.

3. **Memory Cell:**
   - The "memory" or "cell state" is represented by the variable \(Sₜ₋₁\) (memory from the previous time step). This serves as a long-term memory component.

4. **LSTM Operations:**
   - **Forget Gate:**
     - The "Acts as forget gate" block represents the forget gate of the LSTM. It takes both the previous hidden state (\(hₜ₋₁\)) and the current input (\(Xₜ\)) as input.
     - The forget gate decides what information from the previous memory state (\(Sₜ₋₁\)) to forget and what to remember. This is done through element-wise multiplication and addition.

   - **Input Gate:**
     - The "Acts as an input gate" block represents the input gate of the LSTM. It determines which information from the current input (\(Xₜ\)) should be added to the memory cell.
     - This gate involves element-wise multiplication and addition.

   - **Memory Update:**
     - The operations involving the forget gate, previous memory state, and input gate result in an updated memory state (\(Sₜ\)). This is updated information considering both the past memory (\(Sₜ₋₁\)) and the relevant information from the current input (\(Xₜ\)).

   - **Output Gate:**
     - The output gate determines what information from the updated memory state (\(Sₜ\)) should be passed as the hidden state (\(hₜ\)) to the next time step.
     - The output gate involves element-wise multiplication and addition.

5. **Output (hₜ):**
   - \(hₜ\) represents the hidden state at the current time step (\(t\)). It carries information to the next time step and serves as the output of the LSTM cell.

In summary, an LSTM cell utilizes a combination of forget gates, input gates, and output gates to selectively update its memory state and hidden state. This architecture allows LSTMs to capture long-term dependencies in sequential data, making them effective for tasks such as natural language processing where understanding context over longer sequences is important.

