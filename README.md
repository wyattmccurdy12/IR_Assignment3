# Ranking and Re-Ranking System

This project implements a ranking and re-ranking system using bi-encoder and cross-encoder models. Both pretrained and finetuned models are utilized.

## Project Structure

- `main.ipynb`: Jupyter notebook for interactive development and experimentation.
- `main.py`: Main script for running the ranking and re-ranking system.
- `README.md`: Project documentation.

## Models

### Bi-Encoder

The bi-encoder model is used for initial ranking. It encodes queries and documents independently.

### Cross-Encoder

The cross-encoder model is used for re-ranking. It takes pairs of queries and documents and encodes them together for more accurate scoring.

## Usage

### Running the Script

To run the main script, use the following command:

```sh
python main.py
```