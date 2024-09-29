
# Language Feedback Prediction with RoBERTa

## Project Overview

This project aims to predict various language feedback scores (such as `cohesion`, `syntax`, `vocabulary`, etc.) based on input texts from a language test. Initially, the project used the BERT model, but later switched to the RoBERTa model to improve accuracy. The project uses a multi-regression approach to predict several language features simultaneously.

### Models and Techniques Used:
1. **BERT and RoBERTa:** Initially, BERT was used, but due to low accuracy, RoBERTa, which generally performs better on natural language tasks, was adopted.
2. **Multi-Regression:** The model predicts multiple target values simultaneously (cohesion, syntax, vocabulary, etc.).
3. **Fine-tuning:** The model is fine-tuned using language feedback data for several epochs, with advanced techniques like warm-up scheduling to optimize learning.

## Installation and Setup

### Prerequisites:
To run the project, first install the following dependencies:

```bash
pip install torch transformers scikit-learn pandas numpy matplotlib
```

### Running the Project:

1. Place your data files in the `input` folder.
2. Execute the Jupyter Notebook or Python script to train the model and predict language feedback scores.

To run the script:

```bash
python train.py
```

Alternatively, open the Jupyter Notebook and follow the training steps.

## Project Structure

- `train.py`: Main script for training the model.
- `dataset.py`: Classes for handling datasets and DataLoaders.
- `models.py`: Models used in the project (BERT and RoBERTa).
- `utils.py`: Helper functions like time formatting.
- `submission.csv`: Final output with the predicted scores.
- `README.md`: This file, containing the project description.

## Model Training

The model is trained for 5 epochs using the `AdamW` optimizer with a learning rate of 2e-5. A learning rate scheduler (`get_linear_schedule_with_warmup`) is used to gradually decrease the learning rate over time.

### Training Details:
- **Data**: Training data is read from a CSV file and tokenized using the RoBERTa tokenizer.
- **Model**: The `RoBERTa` model is used for multi-regression, outputting six scores for each language feature.
- **Loss Function**: The `MSE` (Mean Squared Error) loss function is used to compute the training error.

## Output

Once training is complete, the model is evaluated on the test dataset. The predicted scores are saved in a `submission.csv` file, which contains the predicted values for each language feature.

## Improvement Suggestions

- **Advanced Models**: For better accuracy, you can experiment with larger models like `RoBERTa-large` or `DeBERTa`.
- **Fine-tuning**: Hyperparameters such as the learning rate and number of epochs can be adjusted to better fit the data and available computational resources.

## Contributing

If you’re interested in improving this project, feel free to submit a Pull Request or open an Issue with your suggestions and feedback.


