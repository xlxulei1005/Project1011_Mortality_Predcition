def data_split(patient_list, downsampling_rate = 0.3):
    # Argments:
    #       patient_list : in format [[patient_id, label]] where death label = 1 otherwise label = 0
    #       downsampling_rate : the percentage of negative sample in the final training data
    # Output:
    #        train_data, validation data and test_data
    while True:
        random.shuffle(patient_list)
        patient_list = np.array(patient_list)
        num_patients = len(patient_list)
        train_data = patient_list[:int(0.6*num_patients)]
        val_data   = patient_list[int(0.6*num_patients):int(0.8*num_patients)]
        test_data  = patient_list[int(0.8*num_patients):num_patients]
        #downsampling
        if sum(train_data[:,1]) > 0:
            break
        else:
            random.shuffle(patient_list)
    
    if sum(train_data[:,1])/len(train_data[:,1]) <= 0.3:

        downsampling_size = int(sum(train_data[:,1])*(1 - downsampling_rate)/downsampling_rate)

        train_data_survive = train_data[train_data[:,1] != 1][:downsampling_size]
        train_data_dead = train_data[train_data[:,1] == 1]
        print(train_data_survive)
        print(train_data_dead)
        train_data = np.vstack((train_data_survive,train_data_dead))
        #random.shuffle(train_data)
        print('The percentage of negative sample after downsampling is {:.1%}'.format(sum(train_data[:,1])/len(train_data[:,1])))
        return train_data[:,0], val_data[:,0], test_data[:,0]

    else:
        print('The percentage of negative sample after downsampling is {:.1%}'.format(sum(train_data[:,1])/len(train_data[:,1])))
        return train_data[:,0], val_data[:,0], test_data[:,0]