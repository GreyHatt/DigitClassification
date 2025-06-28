# Digit Classification Project

This project demonstrates how to recognize hand-written digits using scikit-learn. It utilizes a support vector classifier to train on the digits dataset and visualize the results.

## Project Structure

```
DigitClassification
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── plot_digits_classification.py
└── README.md
```

## Requirements

The project requires the following Python packages:

- scikit-learn
- matplotlib

These dependencies are listed in the `requirements.txt` file.

## Docker Setup

To run this project using Docker, follow these steps:

1. **Build the Docker image**:
   Navigate to the project directory and run the following command:
   ```
   docker build -t digit-classification .
   ```

2. **Run the Docker container**:
   You can run the container using Docker Compose with the following command:
   ```
   docker-compose up
   ```

This will start the application and execute the `plot_digits_classification.py` script.