The implementation of the automatic sleep staging study
[**A Two-Stage Learning Convolutional Neural Network for Sleep Stage Classification Using a Filterbank and Single Feature**](https://ieeexplore.ieee.org/document/9791443)

- Run load_DREAMS_subject_dataset, load_Sleep_EDF.py or load_Sleep_EDFx.py load dataset and correscponding hypnograms.
- Run feature_ext_filterbank.py to extract features for each epoch of the EEG signal using filterbank.
- Run 2stage_learning_CNN.py for the classification of the extacted features.
