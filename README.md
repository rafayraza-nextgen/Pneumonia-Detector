# CNN Chest X-Ray Pneumonia Detection

This is a Deep Learning project. It uses Convolutional Neural Networks (CNNs) to look at chest X-ray images and group them into two types:
- **Normal**
- **Pneumonia**

This project works with two kinds of data:
- 🧪 **Synthetic dataset:** Fake data for quick testing and fixing errors.
- 🏥 **Real dataset:** Real chest X-ray images from Kaggle.

It also compares two training methods (**Adam vs SGD optimizers**) and gives you clear charts to see how well the model works.

---

## 🚀 Features

- Image sorting using CNNs.
- Fake data maker for fast testing.
- Real data loader (Kaggle Chest X-ray dataset).
- Data augmentation (making small changes to images) so the model learns better.
- Compares Adam and SGD optimizers.
- Shows a Confusion Matrix and a clear report of the results.
- Tracks how well the model does over time (weekly checks).
- Shows training charts.
- Shows examples of the model's guesses.
- Includes a picture of the CNN design.

---

## 🧠 Model Design

The CNN is built with these parts:
- Conv2D (32 filters)
- Conv2D (64 filters)
- Conv2D (128 filters)
- Batch Normalization (to keep data balanced)
- MaxPooling layers (to shrink data size)
- Fully Connected layer (256 neurons)
- Dropout (0.5) (to stop the model from memorizing too much)
- Output layer (Sigmoid activation to make the final guess)

---

## 📂 Project Folders

```text
cnn-xray/
│
├── cnn_synthetic.py  # Code for the fake dataset
├── cnn_real.py       # Code for the real Kaggle dataset
├── outputs/          # Saved charts and results
└── README.md         # This file

📊 The Data
🧪 Synthetic Dataset
Automatically makes images that look like X-rays.

You do not need to download anything.

Great for testing if the code works.

🏥 Real Dataset (Kaggle)
You can download the real data here:
Kaggle Chest X-Ray Dataset

Your folders should look like this after downloading:

chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/

⚙️ How to Install
Open your terminal and run these commands:

git clone [https://github.com/your-username/cnn-xray.git](https://github.com/your-username/cnn-xray.git)
cd cnn-xray
pip install -r requirements.txt

▶️ How to Run the Code
For the Fake (Synthetic) Version:

python cnn_synthetic.py

For the Real Dataset Version:
First, make sure your code points to the right folder:
load_real_dataset("chest_xray")

Then run:

python cnn_real.py

📈 What You Get (Outputs)
All charts and results are saved in the outputs/ folder.
This includes:

Charts showing training accuracy and loss.

Confusion Matrix (for both Adam and SGD).

A chart showing weekly performance.

Pictures of the model's test guesses.

A drawing of the model's design.

A summary text file.

📊 Summary of Results
Adam optimizer: Learns faster and makes better guesses.

SGD optimizer: Learns slower but is very steady.

Changing the images slightly (data augmentation) helps the model a lot.

The CNN is very good at finding pneumonia in the X-rays.

📉 How We Score the Model
Accuracy

Precision

Recall

F1-score

Confusion Matrix

Weekly performance tracking

🔬 Main Takeaways
Adding variety to the images stops the model from just memorizing them.

The Adam training method works best for these images.

The model successfully learns the cloudy patterns that mean pneumonia is there.

The fake dataset is very helpful for fixing bugs early.

📌 What to Add Next (Future Ideas)
Add "Grad-CAM" to show exactly where the model is looking in the image.

Try using models that are already trained (like ResNet or EfficientNet).

Put the model on a website using Flask or FastAPI.

Make the model smaller so it runs on phones (TensorFlow Lite).

Handle the problem of having unequal amounts of normal and sick images.

👨‍💻 Author
Made for learning about Deep Learning and to show as a portfolio project.

📜 License
This project is open-source. It is free to use for learning and school projects.

⭐ Support
If you like this project or find it helpful, please give it a ⭐ on GitHub!
