# Sign Language Digit Classification: From CNN to Transfer Learning

> **Project Goal:** To improve the accuracy of sign language digit recognition using VGG16 Transfer Learning.

---

## 📖 Project Overview

This project develops a deep learning system for sign language digit classification to improve accessibility for the hearing impaired. By transitioning from a baseline CNN to a **VGG16-based Transfer Learning** architecture, I achieved a **6x performance increase**, raising classification accuracy from **10% to 67%**. This demonstrates a viable strategy for building reliable AI tools even with limited datasets (approx. 1,000 images).

---

## 🛠️ Tech Stack
* **Language:** Python
* **Machine Learning:** TensorFlow, Keras, VGG16(Transfer Learning)
* **Libraries:** Scikit-learn, NumPy, Matplotlib
---

## 📈 Model Analysis & Insights

### **1️⃣ Two-Stage Training & Fine-Tuning Strategy**

To achieve stable and high-performance learning, I implemented a two-stage training strategy:

* **Phase 1 (Epoch 1-16): Feature Extraction**
    * **Strategy:** Froze the VGG16 backbone to stabilize top-layer weights.
    * **Result:** The best model weights were captured at **Epoch 11**.
* **Phase 2 (Epoch 16-38): Fine-Tuning**
    * **Strategy:** Unfroze the final convolutional blocks with a low learning rate ($10^{-5}$) for deep optimization.
    * **Result:** The best model weights were captured at **Epoch 33**.

*![Training Results](../images/01_training_results.png)*

* **Final Selection:** The best-performing weights from **Epoch 33** were selected as the final model to ensure **maximum generalization**, reaching a final accuracy of **67%**.

### **2️⃣ Technical Insights & Optimization**
* **Handling Data Scarcity:** Beyond Transfer Learning, I applied **Data Augmentation** (rotation, translation, contrast) to artificially expand the 1,000-image dataset, which was crucial for model generalization.
* **Overfitting Control:** Strategic use of **Global Average Pooling** and **Dropout** prevented the model from memorizing the limited training samples, ensuring stable validation performance.
* **Optimization:** The **Adam Optimizer** with a dynamic learning rate allowed for steady convergence without destroying pre-trained features during the fine-tuning phase.

---

* **Project Origin**: This project was developed as a mid-term project for the Department of AI at Korea National Open University (KNOU) in April 2025.
* **Note on Dataset**: The dataset used in this project is not included in this repository as it was provided for academic purposes (KNOU Coursework).
* **Acknowledgment:** Code optimization, English terminology refactoring, and Technical documentation for this project were supported by **Google Gemini**.
