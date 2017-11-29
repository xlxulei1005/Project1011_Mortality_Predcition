# Project1011_Mortality_Predcition
The project repository for Motality prediction for Intersive Care Unit patients

**Team Member:** Sheng Liu, Haichao Wu, Nan Wu and Lei Xu

## Data Description
All of our processed data can be found at our [Google Drive](https://drive.google.com/drive/u/1/folders/1EJVIHULMXFmasnBwDTGb-j-mA5duN7ck)
* cleaned_notes.pickle : 
   * Structure: A python dictionary with three terms: 'CHARTTIME', 'SUBJECT_ID' and 'DOCUMENTS'. And the value is a list of the relevant 
   information in the same order. To be clear, 'CHARTTIME' is the excat time when the notes was created (type: pandas.tslib.Timestamp). 
   'SUBJECT_ID' is the unique ID for the subject (type: numpy.int64). DOCUMENTS is the entification replaced, multi-space replaced lower cases 
   notes (type: string)
    
* patient_timesheet_final.pickle : 
   * Structure: A python dictionary with all valid SUBJECT_ID as keys. The values are also dictionaries with following terms: 
      * 'ADMITTIME': 
      * 'CHARTTIME': an array of time for all the notes created for the subject
      * 'DEATHTIME': If the subject is recovered, the value will be NaT
      * 'DISCHTIME': DISCHTIME and DEATHTIME are equal in our sample.
      * 'CHARTTIME_valid': all notes with CHARTTIME before DISCHTIME
      * 'Stay_interval': DISCHTIME - ADMITTIME. How long the subject stayed in ICU no matter what the reason for discharge(death or recover)
      * 'CHARTTIME_todeath': DISCHTIME - CHARTTIME. The time length from the note created to the subject be discharged (death or recover)
        This will be one of the lable (Time length) for our model
      * 'CHARTTIME_interval': CHARTTIME - ADMITTIME. The time length between the the note created and the subject be admitted. Data for model trained on different time periods should be sampled according to this variable. 
      

* summarized_data/tokenized data/voc_100_new/train_15m.npy (and all other files like train/val/test_15m/6h/24h/all.npy):
   * Structure: A python dictionary with "DATA", "MORTALITY_LABEL", "SUBJECT_ID", "NOTE_ID"
      * 'DATA': Array of 3-dimensional lists that use word index to represent patient notes. For example, data[i][j][k][l] is a integer represents word l from sentence k, from note j, from patient i.
      * 'SUBJECT_ID'
      * 'MORTALITY_LABEL': If the subject has DEATHTIME, the value is 1, else it is 0.
      * 'NOTE_ID': array of list. Each list represents the all notes index from this patient.
      * 'TIME_TO_DEATH': array of list . Each list represents the DISCHTIME - CHARTTIME (measured in minutes) for each note of single patient. 



## Recurrent Hierarchical Attention Mechanism
The structure of our model is:
![picture](/graph_explanation/RNN-NLP_model.png)

