**Classifying CIFAR-10 Images**  

---

# Unique Methodology: Optimizing VGG16 for Small-Scale Image Classification  

###  **How My Approach is Different**
Unlike traditional transfer learning approaches where **VGG16 is entirely frozen**, my methodology **fine-tunes specific layers** of VGG16 to better adapt to smaller image datasets (**32×32 resolution**).  

- **What makes this approach unique?**
  ✅ **Selective Fine-Tuning:** Instead of freezing VGG16 completely, I unfreeze certain deeper layers to allow adaptation to new datasets.  
  ✅ **Optimized for CIFAR-10 & Small Images:** VGG16 was trained on **ImageNet (224×224)**, which differs significantly from **32×32 images**. Fine-tuning enables better generalization.  
  ✅ **Balancing Pretrained Features & Learning New Representations:** The model retains **low-level edge detection from VGG16** while learning dataset-specific patterns in higher layers.  

---

## 📊 Effect of Adding VGG16 as a Base Layer  

### Why Use VGG16?
- **VGG16 is a powerful feature extractor** pretrained on **ImageNet**, which helps transfer learned features to a new dataset.
- Using **pretrained weights** allows the model to leverage low-level features like edges, textures, and shapes without training from scratch.

### Observation: Freezing VGG16 vs. Training It  

| **Scenario**                        | **Accuracy & Performance** |
|--------------------------------------|----------------------------|
| **VGG16 with Frozen Weights**        | ❌ **Worse accuracy**, likely because VGG16 features are too general for this dataset. |
| **VGG16 with Trainable Weights**     | ✅ **Better accuracy**, as the model adapts features to the new dataset. |

### Why Freezing VGG16 Gives Worse Results?
1. **Feature Mismatch:**  
   - VGG16 is pretrained on **ImageNet (224×224 images)**, but here, the input size is **32×32** (e.g., CIFAR-10).
   - **Lower resolution images may not match the learned VGG16 features**, leading to poor generalization.

2. **Lack of Fine-Tuning:**  
   - Freezing means **only the new layers are trainable**, but they rely on **static features from VGG16**.
   - Since these features were learned on ImageNet, they may not be relevant to **smaller datasets** like CIFAR-10.

3. **Custom Layers Struggle to Learn:**  
   - If VGG16 is frozen, the added **Conv2D and Dense layers** must adapt to VGG16’s fixed outputs.
   - **If the dataset distribution is different, these layers struggle to learn meaningful representations.**

---

### **1. General Information**  

- **Aim:** Using CNNs to classify 10 categories of 32x32 RGB images.  

- **Dataset:** CIFAR-10 (50k train, 10k test images).  

- **Preprocessing:**  

  - Normalize pixels to ``.  

  - One-hot encode labels.  

---

### **2. Creating the Model**  

#### **Base CNN:**  

- **Architecture:**  

  - **Convolution Blocks:**  

    - `VGG16`( Trainable weights - Reason stated below ) → `Conv2D` → `BatchNorm` → `Conv2D` → `BatchNorm` → `MaxPool` → `Dropout`  

  - **Classifier:**  

    - `Flatten` → `Dense(512)` → `BatchNorm` → `Dropout(0.5)` → `Softmax`  

- **Key Features:**  

  - Batch normalization for stable training.  

  - Dropout layers (0.25–0.5) to prevent overfitting.
 
```python
model = Sequential([
    VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3)),
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```
---
#### **Transfer Learning:** (VGG16)  

Interestingly enough, adding base_model as VGG16 and freezing the base layers and following the approach given below gives an accuracy of less than 60%

IF FOLLOWED: 

Custom top is added: `GlobalAveragePooling2D` → `Dense(512)` → `Softmax`  
```python
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))
for layer in base_model.layers:
    layer.trainable = False  # Freeze pre-trained layers

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```
OUTPUT:
<img width="840" alt="image" src="https://github.com/user-attachments/assets/439835ca-1fa9-4a17-a19a-f687e9a993ba" />

---
WHAT IS FOLLOWED IN THE CODE ABOVE INSTEAD 

Adding VGG16 as a layer with trainable weights instead as follows ( also followed in the code pushed into this repo ) : 

```python
model = Sequential([
    VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3)),
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNorm....
    ........
    <rest of the code>
])
```
OUTPUT:
<img width="464" alt="image" src="https://github.com/user-attachments/assets/4a13ea1b-4b60-49f3-88ea-a4a0729e3005" />


---

**Reason for this unexpected behaviour**

As one would think, adding a base layer and freezing it might give a better accuracy ( just like i thought so too ) but the reasoning might be as follows:

	1.	Feature Mismatch:
	•	VGG16 is pretrained on ImageNet (224×224 images), but here, the input size is 32×32 (e.g., CIFAR-10).
	•	Lower resolution images may not match the learned VGG16 features, leading to poor generalization.
	2.	Lack of Fine-Tuning:
	•	Freezing means only the new layers are trainable, but they rely on static features from VGG16.
	•	Since these features were learned on ImageNet, they may not be relevant to smaller datasets like CIFAR-10.
	3.	Custom Layers Struggle to Learn:
	•	If VGG16 is frozen, the added Conv2D and Dense layers must adapt to VGG16’s fixed outputs.
	•	If the dataset distribution is different, these layers struggle to learn meaningful representations.

 | **Scenario**                        | **Accuracy & Performance** |
|--------------------------------------|----------------------------|
| **VGG16 with Frozen Weights**        | ❌ **Worse accuracy**, likely because VGG16 features are too general for this dataset. |
| **VGG16 with Trainable Weights**     | ✅ **Better accuracy**, as the model adapts features to the new dataset. |


---

### **3. Training**  

- **Optimizer:** Adam (`lr=0.001`).  

- **Stopping:** Early stopping if `val_loss` is constant for 10 epochs.  

- **Batch Size:** 64.  

---  

### **4. Results**  

- **Base CNN:**  

  - **Accuracy:** 80% (Test Set).  

  - **Best Class:** “Automobile” (F1: 0.90).  

  - **Weakest Class:** “Cat” (F1: 0.63).  

- **VGG16:** Improved feature extraction (requires image resizing).  

---
  
### **5. Usage**  
```python
# Predict on new images
def predict(image_path):
    img = preprocess(image_path)  # Resize (32x32), normalize
    return model.predict(img)
```

---

### **6. Key Improvements**  
- Add data augmentation (rotation/flips).  
- Adjust VGG16 input size mismatch (32x32 → 224x224).  
- Experiment with learning rate schedulers.
📌 Freezing VGG16’s layers leads to lower accuracy because the extracted features may not be relevant for the dataset.
📌 Fine-tuning the deeper layers of VGG16 improves accuracy by allowing it to adapt to the new dataset.

🔹 Recommendation: Instead of fully freezing VGG16, this approach seems much better.

**Dependencies:** TensorFlow, NumPy, Matplotlib.  
**Runtime:** ~22 epochs (early stopping).  


