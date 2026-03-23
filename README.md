# Sign Language Digit Classification: From CNN to Transfer Learning

This project explores the classification of sign language digits (0-9) using Deep Learning. It demonstrates the evolutionary process of model performance, comparing a baseline CNN built from scratch with a Transfer Learning approach using VGG16. This project was developed as **a mid-term project for the Department of AI at Korea National Open University (KNOU) in April 2025.**

## 📖 Project Overview
* **Goal**: Classify hand gestures representing digits 1-10.
* **Framework**: TensorFlow 2.x, Keras, Matplotlib.
* **Key Achievement**: Improved accuracy from **10% (Baseline)** to **67% (Transfer Learning)**, achieving a **6x performance boost**.

---

## 📂 Repository Structure
* **[project_cnn/](./project_cnn/)**: Implementation of the custom CNN architecture (Baseline).
* **[project_vgg16/](./project_vgg16/)**: Implementation of Transfer Learning and Fine-tuning using VGG16.
* **[images/](./images/)**: Contains training logs and performance visualizations.

> **Note on Dataset**: The dataset used in this project is not included in this repository as it was provided for academic purposes (KNOU Coursework).

---

## 📊 Training Results & Analysis

The following chart illustrates the training and validation performance of the VGG16 Transfer Learning model. 

![Training Results](./images/training_results.png)

### **Performance Breakdown**
* **Baseline CNN**: Achieved a modest **10% accuracy**. The simple architecture was insufficient for the complexity of sign language gestures.
* **Transfer Learning (VGG16)**: Transitioning to VGG16 with pre-trained ImageNet weights immediately boosted accuracy.
* **Fine-Tuning**: By unfreezing the final convolutional blocks and applying a low learning rate ($10^{-5}$), the model reached a final accuracy of **67%**.

### **Technical Insights**
* **Overcoming Data Scarcity**: With a limited dataset of ~1,000 images, Transfer Learning proved essential for effective feature extraction.
* **Regularization**: Integration of **Global Average Pooling** and **Dropout (0.5)** was critical in managing overfitting and stabilizing the validation loss.
* **Optimization**: Utilizing the **Adam optimizer** with a dynamic learning rate strategy ensured steady convergence during the fine-tuning phase.

---
**Acknowledgment:** Technical documentation, English terminology refactoring, and code optimization for this project were supported by **Google Gemini**.
