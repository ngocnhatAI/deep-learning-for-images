## 📊 Dataset Descriptions

### 1. 👩‍🦰👨 Gender Classification Dataset

This dataset consists of **10,000 face images**, categorized for gender classification tasks. The images are organized into three main folders:

* **Train**: 8,000 labeled images

  * `Female/`: 4,000 images of female faces
  * `Male/`: 4,000 images of male faces

* **Val**: 1,000 labeled images

  * `Female/`: 500 images
  * `Male/`: 500 images

* **Test**: 1,000 **unlabeled** face images

  > Used for public and private evaluation.

---

### 2. 🧠 Medical Image Segmentation Dataset

This dataset contains **medical images and their corresponding segmentation masks**, used for training models in semantic segmentation tasks.

* **Train**:

  * `Image/`: 1,087 `.jpg` images
  * `Mask/`: 1,087 `.png` segmentation masks

    > Each mask corresponds to an image in the `Image/` folder.

* **Test**:

  * 192 `.jpg` images **without masks**

  > Used for public and private testing.

> Participants are encouraged to create custom validation splits from the training set as needed.

---

### 3. 🍎🍌🍍 Object Detection Dataset

This dataset is designed for multi-class object detection with bounding boxes and classification labels.

* **Train**:

  * `images/`: Training images
  * `labels/`: Annotation files with:

    * Object class (Apple: 1, Grapes: 2, Pineapple: 3, Orange: 4, Banana: 5, Watermelon: 6)
    * Bounding box coordinates

* **Test**:

  * Contains images **without labels**

  > Used for evaluating object detection performance.