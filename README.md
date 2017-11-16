# Project1011_Mortality_Predcition
The project repo for Natural Language Processing

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
      
   

* all_data.npy :
   * Structure: Array of 3-dimensional lists that use word index to represent patient notes. For example, all_data[i][j][k][l] is a integer represents word l from sentence k, from note j, from patient i.
      * 1st dimension (array): patient
      * 2nd dimension (list): note
      * 3rd dimension (list): sentence
      * 4rd dimension (list): word

## Recurrent Hierarchical Model
The structure of our model is:

![alt text](https://github.com/xlxulei1005/Project1011_Mortality_Predcition/blob/master/HAN.pdf)

