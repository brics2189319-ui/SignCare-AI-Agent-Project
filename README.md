# SignCare AI

SignCare AI is a computer-vision prototype that recognizes hand signs from a webcam and helps convert detected alphabet signs into text. It is designed as an accessibility and communication project, with a possible future application in healthcare communication.

> **Important:** SignCare AI is an academic prototype. It is not a medical device, diagnostic system, emergency service, or substitute for a qualified interpreter or healthcare professional.

## Project Overview

People with hearing or speech disabilities may face communication barriers when interacting with others, including in healthcare environments. SignCare AI explores whether webcam-based hand landmark detection and machine learning can provide a simple, local communication aid.

The current system recognizes alphabet signs from one detected hand. The desktop interface can build words and sentences, provide word suggestions, speak text aloud, translate selected supported words, and save local history and favorites.

## Features

- Live webcam-based hand tracking using MediaPipe Hands
- Alphabet-sign prediction using a trained Random Forest classifier
- Wrist-relative landmark normalization for more position-independent features
- Confidence display for model predictions
- Temporal stabilization to reduce repeated or unstable predictions
- Word and sentence builder
- Word suggestions from the built-in dictionary
- Text-to-speech output using `pyttsx3`
- Translation for the words currently defined in `translator_utils.py`
- Save and view local translation history
- Save, view, and clear favorite words or sentences
- ASL alphabet reference within the desktop interface
- Separate experimental clinical dashboard in `inference.py`

## Demonstration Workflow

1. Start the application.
2. Select **Start Camera**.
3. Place one hand clearly in front of the webcam.
4. Show an alphabet sign until the prediction becomes stable.
5. Use the sentence controls to add spaces, delete letters, or clear the sentence.
6. Use **Speak**, **Translate**, **Save History**, or **Add Favorite** as needed.
7. Select **Stop Camera** before closing the application.

## Technology Stack

- Python
- OpenCV for camera capture and image processing
- MediaPipe Hands for 21-point hand landmark detection
- scikit-learn Random Forest for classification
- pandas and NumPy for dataset processing
- CustomTkinter for the desktop interface
- pyttsx3 for text-to-speech
- Pillow for displaying camera frames in the interface
- deep-translator for supported word translation

## System Architecture

```text
Webcam frame
    |
    v
OpenCV capture and RGB conversion
    |
    v
MediaPipe Hands: detect 21 hand landmarks
    |
    v
Wrist-relative x, y, z normalization
    |
    v
Random Forest model: predict alphabet label and confidence
    |
    v
Prediction stabilization
    |
    v
Word/sentence builder -> suggestions, speech, translation, history
```

## Repository Structure

```text
.
├── main.py                    # Main CustomTkinter desktop application
├── predictor.py               # MediaPipe preprocessing and model inference
├── train_model.py             # Train and save the Random Forest model
├── collect_data.py            # Collect normalized hand-landmark samples
├── clean_data.py              # Remove selected labels from the CSV dataset
├── inference.py               # Experimental clinical dashboard
├── translator_utils.py        # Supported Hindi and Gujarati translations
├── word_dictionary.py         # Word suggestion dictionary
├── requirements.txt           # Python dependencies
├── test_predictor.py          # Webcam prediction smoke test
├── Data/
│   └── alphabet_data.csv      # Landmark features and alphabet labels
├── model/
│   └── sign_language_model.p  # Trained model serialized with pickle
├── history/                   # Locally saved history files
├── favorites/                 # Locally saved favorites
└── assets/                    # Application assets
```

## Installation

### Requirements

- Windows, macOS, or Linux
- Python 3.10 or a compatible Python version
- Working webcam
- Working microphone and speakers for speech features

### Setup

From the project directory, create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Application

The main desktop application is started with:

```bash
python main.py
```

The existing trained model must be available at:

```text
model/sign_language_model.p
```

The application uses webcam index `0` by default. Close other applications that may be using the webcam if camera access fails.

## Dataset and Model Method

The dataset contains 63 numerical features for each sample: 21 hand landmarks multiplied by three coordinates (`x`, `y`, and `z`). The coordinates are made relative to the wrist landmark before being stored. This reduces the effect of the hand's position in the camera frame.

`collect_data.py` records samples for a selected sign. Each captured hand sample is stored along with its label, and a mirrored version is also stored to increase variation in the training data.

`train_model.py` performs the following steps:

1. Loads `Data/alphabet_data.csv`.
2. Separates the feature columns from the `label` column.
3. Splits the data into 80% training and 20% testing data using stratification.
4. Trains a `RandomForestClassifier` with 100 trees.
5. Calculates accuracy on the held-out test split.
6. Saves the trained model to `model/sign_language_model.p`.

To collect additional samples:

```bash
python collect_data.py
```

To retrain the model:

```bash
python train_model.py
```

Before retraining, review the dataset labels and sample balance. `clean_data.py` currently demonstrates removal of a selected label and should be edited carefully before use.

## Evaluation

Record the result printed by `train_model.py` and replace the placeholders below with measured values from your own experiment.

| Metric            |         Result |
| ----------------- | -------------: |
| Test split        |            20% |
| Number of classes |  [enter value] |
| Total samples     |  [enter value] |
| Test accuracy     | [enter value]% |
| Evaluation date   |   [enter date] |

Accuracy alone does not describe real-world performance. A stronger evaluation should also report a confusion matrix, per-class precision, recall, F1-score, and performance under different lighting, backgrounds, hand orientations, and users.

## Limitations

- The current model is focused on alphabet-style static signs rather than full sign-language grammar or continuous sign-language sentences.
- Recognition quality can change with lighting, camera angle, background, hand position, occlusion, and signing speed.
- The system is configured for one detected hand.
- Word construction depends on individual letter predictions and may require manual correction.
- Translation coverage is limited to the entries defined in `translator_utils.py`.
- The clinical dashboard is an experimental demonstration and must not be used for medical triage or treatment decisions.
- The measured accuracy from a random train/test split may not represent performance on new users.
- Local history and favorites are stored as plain files and should not contain real patient information.

## Research Scope

This project can be evaluated as a human-computer interaction and applied machine-learning prototype. Possible research questions include:

- How does wrist-relative normalization affect recognition across different users?
- How much does temporal stabilization improve prediction reliability?
- How does performance change across lighting conditions and camera distances?
- How well does the model generalize to users who were not included in data collection?
- Does text-to-speech reduce communication time compared with manually typing a message?

For a fair experiment, use a fixed test protocol, keep test users separate from training users, report the class distribution, and repeat measurements across multiple conditions.

## Future Improvements

- Add a larger and more diverse multi-user dataset.
- Add automated train/validation/test evaluation and confusion-matrix reports.
- Support continuous gestures, motion-based signs, and sign-language grammar.
- Improve sentence correction and language modeling.
- Add accessibility controls such as adjustable text size and keyboard navigation.
- Add an explicit consent and privacy workflow for data collection.
- Replace plain-text storage with safer configurable storage when personal data is involved.
- Package the application for easier installation.
- Add automated tests that do not require a physical webcam.

## Testing

`test_predictor.py` is a webcam smoke test. Run it with:

```bash
python test_predictor.py
```

Press `q` to close the test window. This test requires a working camera and displays the detected label and confidence on the video feed.

## Responsible Use

Use only data collected with the participant's permission. Do not record, publish, or store identifiable video or medical information without informed consent. Predictions should always be confirmed by the user, especially in a healthcare setting.

## Author and Academic Details

- **Project:** SignCare AI
- **Project type:** Pahel 2.0 research and evaluation project
- **Student name:** [enter your name]
- **Course/department:** [enter your course or department]
- **Institution:** Gyanmanjari Innovative University
- **Supervisor:** [enter supervisor name]
- **Version/date:** [enter version and date]
