# Histopathologic Cancer Detection

This project uses Convolutional Neural Networks (CNNs) to detect cancer in histopathologic image patches. The goal is to classify whether an image patch contains cancerous tissue or not. 

This was done as part of a **peer-graded assignment** for the **Introduction to Deep Learning** course and is also submitted to the [Kaggle competition](https://www.kaggle.com/competitions/histopathologic-cancer-detection).

## Files

**Included in this repository:**
- `cnn_cancer_detection_kaggle_mini_project.ipynb`: Jupyter notebook with EDA, model building, training, evaluation, and final submission generation.
- `cnn_cancer_detection_kaggle_mini_project.pdf`: PDF version of the full notebook run including EDA, model training, evaluation, and final results — ideal for quick review without setting up the environment.

**Not included due to file size constraints:**
- `train_labels.csv`: CSV file with training image labels.
- `train/`: Folder containing original `.tif` image patches used for training.
- `dataset/train/0` and `dataset/train/1`: Reorganized folders for training (automatically created by code).
- `test/`: Folder with test image patches.
- `submission.csv`: File with model predictions for Kaggle submission.

>  **To run the notebook successfully**, please download the dataset directly from the [Kaggle Histopathologic Cancer Detection competition page](https://www.kaggle.com/competitions/histopathologic-cancer-detection) and follow the instructions in the notebook to prepare the data.

## Models
Three CNN models were trained and compared:
1. **Baseline CNN**: 3 convolutional layers, dropout = 0.5, learning rate = 0.001
2. **Variant 1**: Same as baseline, but with dropout = 0.3
3. **Variant 2**: Deeper CNN with an extra conv layer, batch normalization, and a lower learning rate of 0.0001

## Results
The models were evaluated using validation accuracy and Kaggle's private leaderboard score. The deeper CNN model performed the best, achieving the highest validation accuracy and the best Kaggle score (0.8674). The baseline model performed moderately, while the dropout variant failed to learn effectively. Full results, accuracy plots, and comparison tables are available in the notebook.

## Requirements
- Python 3.10
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Seaborn
- PIL (Pillow)

Install requirements with:
```bash
pip install tensorflow pandas numpy matplotlib seaborn pillow
