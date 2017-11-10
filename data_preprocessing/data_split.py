def data_split(patient_list):
    # Format of patien_list :
    # [[patient_id, label]], death label = 1 otherwise label = 0
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
    
    if sum(train_data[:,1])/len(train_data[:,1]) <= 0.3:

        downsampling_size = int(sum(train_data[:,1])*7.0/3.0)

        train_data_survive = train_data[train_data[:,1] != 1][:downsampling_size]
        train_data_dead = train_data[train_data[:,1] == 1]
        print(train_data_survive)
        print(train_data_dead)
        train_data = np.vstack((train_data_survive,train_data_dead))
        #random.shuffle(train_data)
        print('The percentage of negative sample after downsampling is {:.1%}'.format(sum(train_data[:,1])/len(train_data[:,1])))
        return train_data, val_data, test_data

    else:
        print('The percentage of negative sample after downsampling is {:.1%}'.format(sum(train_data[:,1])/len(train_data[:,1])))
        return train_data, val_data, test_data