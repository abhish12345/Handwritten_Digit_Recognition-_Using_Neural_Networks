# 🧠 MNIST Classification Project  

This project involves classifying handwritten digits from the MNIST dataset using various deep learning models. The goal was to experiment with different architectures and hyperparameter tuning to achieve high accuracy.  

## 🚀 Project Overview  

### Models Implemented:  
1. **Simple CNN**  
2. **Advanced MLP**  
3. **LeNet-5**  
4. **MLP with Hyperparameter Tuning**  

### Achieved Accuracies:  
- **LeNet-5:** **98.96%**  
- **Simple CNN:** **92.4%**  
- **Advanced CNN:** **97.67%**  

---

## 🛠️ Techniques and Methodologies  

### 1. **Simple CNN**  
- Built a basic Convolutional Neural Network with a few convolutional and pooling layers.  
- Achieved **92.4% accuracy** with minimal tuning.  

### 2. **Advanced MLP**  
- Added deeper fully connected layers.  
- Used techniques like Dropout and Batch Normalization for regularization.  

### 3. **LeNet-5 Architecture**  
- Implemented the classic **LeNet-5** architecture for MNIST classification.  
- Tuned hyperparameters like learning rate, batch size, and optimizer.  
- Achieved the highest accuracy of **98.96%**.  

### 4. **MLP with Hyperparameter Tuning**  
- Experimented with hyperparameters:  
  - Learning rates.  
  - Number of hidden layers.  
  - Units per layer.  
- Improved performance significantly with **custom tuning strategies**.  

---

## 📊 Dataset Details  
- **Dataset Used:** [MNIST](http://yann.lecun.com/exdb/mnist/)  
- **Training Samples:** 60,000  
- **Test Samples:** 10,000  
- **Classes:** 10 (Digits: 0–9)  

---

## ⚙️ Tools and Libraries Used  
- **Framework:** PyTorch  
- **Libraries:**  
  - NumPy  
  - Matplotlib (for visualizations)  
  - Scikit-learn (for metrics and data preprocessing)  

---

## ⚙️ Installation and Usage  

1. **Clone the repository:**  
   ```bash  
   git clone https://github.com/your-username/mnist-classification.git  
   cd mnist-classification  
   ```  

2. **Install dependencies:**  
   ```bash  
   pip install -r requirements.txt  
   ```  

3. **Run the project:**  
   - Training scripts for each model are available in the `scripts/` directory.  
   ```bash  
   python train_lenet5.py  
   python train_simple_cnn.py  
   python train_advanced_mlp.py  
   ```  

4. **Visualize results:**  
   Run the evaluation script to generate performance metrics and confusion matrices.  
   ```bash  
   python evaluate.py  
   ```  


---

✨ **Happy Coding!**  
