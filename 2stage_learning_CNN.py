


def two_stage_model(X, y, Random_state, num_cls):
    
    
    '''
    implementation of two-stream, two-stage learning CNN
    Parameters
    -----------
    signal: 2D array (Input data)
    
    y: 1D array (input labels)
    
    Random_state: int (random state for test and train data split)
    
    num_cls: int (2-5 for the two stage - five-stage classification)
    
    Returns
    ---------
    trained Model
    '''
    
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle=True,
                                                        random_state=Random_state)

    x_train = np.reshape(x_train, (x_train.shape[0],  x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0],  x_test.shape[1], 1))
    
    
    input1 = Input(shape = (49, 1))
    layer = Conv1D(32, (5) , activation = 'relu', padding = 'same', name='conv11')(input1) 
    layer = BatchNormalization()(layer)
    layer = MaxPooling1D(pool_size = (3))(layer) 
    layer = Dropout(0.3)(layer)
    layer = Conv1D(64, (2), activation = 'relu',   name='conv12')(layer) 
    layer = MaxPooling1D(pool_size = (2))(layer) 
    layer = Conv1D(128, (2), activation = 'relu', name='conv13')(layer) 
    layer = Dropout(0.3)(layer)
    stream1 = Flatten()(layer)
    
    
    input2 = Input(shape = (49, 1))
    layer2 = Conv1D(32, (5) , activation = 'relu', padding = 'same', name='conv21')(input2) 
    bbatch = BatchNormalization()(layer2)
    layer2 = MaxPooling1D(pool_size = (3))(layer2) 
    layer2 = Dropout(0.3)(layer2)
    layer2 = Conv1D(64, (2), activation = 'relu',  name='conv22')(layer2) 
    layer2 = MaxPooling1D(pool_size = (2))(layer2) 
    layer2 = Conv1D(128, (2), activation = 'relu', name='conv23')(layer2) 
    layer2 = Dropout(0.3)(layer2)
    stream2 = Flatten()(layer2)
    
    fused = concatenate([stream1,stream2])
    
    dense1 = Dense(units = 200, activation = 'relu',name='dense1')(fused) 
    dropd = Dropout(0.5)(dense1)
    dense2 = Dense(units = 200, activation = 'relu',name='dense2')(dropd) 
    dropd2 = Dropout(0.5)(dense2)
    dense3 = Dense(units =num_cls, activation = 'softmax',name='dense3')(dropd2)
    two_stage_cnn = Model(inputs = [input1,input2], outputs = dense3)
    opt = Adam(lr=0.0003)
    two_stage_cnn.compile(optimizer = opt, loss = "sparse_categorical_crossentropy", metrics = ['accuracy'])
    history1 = two_stage_cnn.fit([x_train[:,:49,:],x_train[:,49:,:]], y_train,
                            validation_data=([x_test[:,:49,:],x_test[:,49:,:]], y_test), batch_size=256, epochs=100, verbose = 1)
    two_stage_cnn.save("two_stage_cnn.h5")
    
    
    
    input1 = Input(shape = (49, 1))
    layer = Conv1D(32, (5) , activation = 'relu', padding = 'same',trainable=False, name='conv11')(input1) 
    layer = BatchNormalization()(layer)
    layer = MaxPooling1D(pool_size = (3))(layer) 
    layer = Dropout(0.3)(layer)
    layer = Conv1D(64, (2), activation = 'relu',  trainable=False, name='conv12')(layer) 
    layer = MaxPooling1D(pool_size = (2))(layer) 
    layer = Conv1D(128, (2), activation = 'relu',trainable=False, name='conv13')(layer) 
    layer = Dropout(0.3)(layer)
    stream1 = Flatten()(layer)
    
    
    input2 = Input(shape = (49, 1))
    layer2 = Conv1D(32, (5) , activation = 'relu', padding = 'same',trainable=False, name='conv21')(input2) 
    bbatch = BatchNormalization()(layer2)
    layer2 = MaxPooling1D(pool_size = (3))(layer2) 
    layer2 = Dropout(0.3)(layer2)
    layer2 = Conv1D(64, (2), activation = 'relu', trainable=False, name='conv22')(layer2) 
    layer2 = MaxPooling1D(pool_size = (2))(layer2) 
    layer2 = Conv1D(128, (2), activation = 'relu', trainable=False,name='conv23')(layer2) 
    layer2 = Dropout(0.3)(layer2)
    stream2 = Flatten()(layer2)
    

    fused = concatenate([stream1,stream2])
    
    dense1 = Dense(units = 200, activation = 'relu',name='dense1')(fused) 
    dropd = Dropout(0.5)(dense1)
    dense2 = Dense(units = 200, activation = 'relu',name='dense2')(dropd) 
    dropd2 = Dropout(0.5)(dense2)
    
    dense3 = Dense(units =num_cls, activation = 'softmax',name='dense3')(dropd2)
    two_stage_cnn = Model(inputs = [input1,input2], outputs = dense3)
    opt = Adam(lr=0.0003)
    two_stage_cnn.compile(optimizer = opt, loss = "sparse_categorical_crossentropy", metrics = ['accuracy'])
    
    two_stage_cnn.load_weights('two_stage_cnn.h5', by_name=True)
    
    
    history2 = two_stage_cnn.fit([x_train[:,:49,:],x_train[:,49:,:]], y_train,
                            validation_data=([x_test[:,:49,:],x_test[:,49:,:]], y_test), batch_size=256, epochs=100, verbose = 1)

    two_stage_cnn.save('edf2cls_'+str(Random_state)+'.h5')

    return two_stage_cnn





if __name__ == "__main__":
    from load_SleepEDF_dataset import *
    from feature_ext_filterank import *
    m5cls= two_stage_model(X, H, 0, 5)
   







# run for the different number of classes
# H2 = H.copy()
# H2[H2==2] = 1
# H2[H2==3] = 1
# H2[H2==4] = 1
# m2cls= two_stage_model(X, H2, 3, 2)


# H3 = H.copy()
# H3[H3==2]= 1
# H3[H3==3]= 1
# H3[H3==4] = 2
# m3cls= two_stage_model(X, H3, 3, 3)

# H4 = H.copy()
# H4[H4==3]= 2
# H4[H4 == 4] = 3
# m4cls= two_stage_model(X, H4, 3, 4)

