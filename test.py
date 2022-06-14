#from utils_rgbd import LoadData
import tensorflow as tf
import numpy as np
import sys
import os


def Predict(model, testingData):
    # predict and format output to use with sklearn
    predict = model.predict(testingData)
    # predict = np.argmax(predict, axis=1)

    return predict


def Test(name, mode):
    if mode=="rgb":
        from utils_rgb import LoadData
    elif mode=="depth":
        from utils_depth import LoadData
    elif mode=="rgbd":
        from utils_rgbd import LoadData
    print("Loading Test Data")
    testingData, testingLabels = LoadData("test")

    print("Loading model")
    model = tf.keras.models.load_model(name+".h5")
    print("Making predictions on test data")

    prediction = Predict(model, testingData)

    model.evaluate(testingData, testingLabels)

    f = open('results/prediction.txt', 'w')

    f.write("ground truth --- prediction\n")

    for i in range(prediction.shape[0]):
        f.write("{}             {}\n".format(
            testingLabels[i], prediction[i][0]))

    # prediction for entire video starts from here -----


if __name__ == '__main__':
    #saved_model = "checkpoint"
    mode = sys.argv[1]
    directory = "threemodels"
    saved_model = directory+"/"+mode
    print("testing on {} mode".format(mode))
    Test(saved_model, mode)
