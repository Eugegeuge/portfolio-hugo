# Math Solver 🧮📷

> **Real-time Handwritten Equation Solver using Computer Vision & Hybrid AI.**

**Math Solver** is an academic computer vision project developed to solve handwritten mathematical equations and operations in real-time. Inspired by industry-standard tools like Photomath, this project bridges the gap between analog writing and digital solving using a custom hybrid AI architecture.

## 🚀 Features

*   **Equation & Operation Solving**: Distinguishes between standard arithmetic (e.g., `3+5`) and linear equations (e.g., `2x+4=0`).
*   **Hybrid Recognition System**:
    *   **CNN (Convolutional Neural Network)**: Trained on MNIST for high-accuracy digit recognition (0-9).
    *   **SVM (Support Vector Machine)**: Optimized for mathematical symbols (+, -, *, /, x).
*   **Intelligent Segmentation**: Uses **HSV Color Space** to route characters to the correct model (Black Ink ➡️ Digits, Red Ink ➡️ Symbols).

## 📂 Project Structure

*   `src/`: Core logic and application source code.
    *   `digitRecognition.py`: **Main Application**. Run this to start the solver.
    *   `symbols.py`: SVM symbol recognition logic.
    *   `ecuacion.py`: Equation parsing and solving.
*   `training/`: Scripts used to train the CNN and SVM models.
*   `docs/`: Project documentation and memory (`Memoria.pdf`).

## 🛠️ Installation & Usage

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/eugegeuge/mathsolver.git
    cd mathsolver
    ```

2.  **Set up Virtual Environment (Optional but Recommended)**:
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application**:
    ```bash
    python src/digitRecognition.py
    ```

## 👥 Authors

*   **Hugo Sevilla**
*   **Juan Diego Serrato**
*   **Hugo López**
*   **Gabriel Segovia**

---
*Built for the Computer Vision course at the University of Alicante.*
