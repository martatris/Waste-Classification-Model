# Waste Classification Model (Biodegradable vs Non-Biodegradable)

This project trains a machine learning model to classify waste into **biodegradable** (`b`) or **non-biodegradable** (`n`) categories using images.

## Dataset Structure

The original dataset was split into multiple folders (`TRAIN.1`, `TRAIN.2`, `TRAIN.3`, `TRAIN.4`, and `TEST`).  
All training folders were merged into a single dataset folder for easier model training.

Final merged structure:
```
dataset_merged/
   train/
      b/   (biodegradable images)
      n/   (non-biodegradable images)
   test/
      b/
      n/
```

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- scikit-learn

Install dependencies:
```
pip install tensorflow scikit-learn numpy
```

## Training the Model

Run the training script:
```
python train_model.py
```

This will:
- Load the dataset
- Train using EfficientNet with additional layers
- Fine-tune the model
- Evaluate accuracy using the test dataset

The trained model will be saved as:
```
waste_classifier_final.h5
```

## Making Predictions

To classify a new image:
```
python predict.py <image_path>
```

It will output either:
- `Biodegradable`
- `Non-Biodegradable`

## Tips
- Ensure your images are clear and show only the waste object.
- More varied training images improve accuracy.
