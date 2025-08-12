# 🍣 Sushi Classifier App

_A sushi companion for the introverted foodie._

Have you ever wanted to learn about different types of sushi but felt too shy to ask the chef? This project is for you.

We're building an image classifier to identify different kinds of sushi—dish and fish—because the ones that are currently out there just don’t cut it.

---

## 🛠️ Project Milestones

### ~~Basics~~ ✅

- **Build a basic classifier using PyTorch** ✅ -> Used Resnet18
  - Experiment with different architectures
  - Research and document findings
  - Limit to two types of sashimi - salmon and otoro

- **Data Collection** ✅
  - Manually curate and label a dataset
  - Train the initial model

- **Add a Frontend** ✅ -> Used Streamlit
  - Let users upload or take pictures
  - Optionally incorporate user-submitted images into the dataset

---

### Desirable Features
- **Create a custom classifier** 

- **Expand the Dataset** ✅ -> Used Google API
  - Include more types of sashimi
  - Automate data gathering & processing
  - 
- **Go Beyond Sashimi** ✅ -> Classifies different types of sushi 

- **Extend to Images with Multiple Difference Pieces of Sashimi** ❌ -> Unable to label bounding boxes
  - Reflect in frontend, attaching labels to each 'group' of fish

- **Add Benchmarking**
  - Measure and improve speed and accuracy
  - Compare model performance over time

---

### Above and Beyond

- Implement unit tests for critical components
- Extend to other types of sushi: nigiri, chirashi, poke, etc.
  - Include descriptors (taste, texture, etc.)
  - Add location-based sushi recommendations (Automated _Beli_?)

---
