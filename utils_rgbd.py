import numpy as np
import cv2
import matplotlib.pyplot as plt


def LoadData(dataType):
    #directory = 'dataset_small'
    directory = 'dataset'
    total_frames = np.load(directory+'/frames.npy')[25000:50000,...,0:4]
    #rgb_frames = total_frames[...,0:3]
    depth_frames = total_frames[...,3]
    depth_frames = np.expand_dims(total_frames[...,3], -1)
    depth_frames = np.concatenate((depth_frames,depth_frames,depth_frames),axis=-1)
    #print("rgb_frame shape", rgb_frames.shape)
    #print("depth_frame shape", depth_frames.shape)
    #final_frames = total_frames
    #depth_frames = total_frames
    final_frames = np.concatenate((total_frames[...,0:3], depth_frames), axis=-2)
    counts = np.load(directory+'/count.npy')[25000:50000]

    print("final frames shape", final_frames.shape)

    train_size, valid_size = int(
        depth_frames.shape[0] * 0.6), int(depth_frames.shape[0] * 0.2)

    if dataType == "train":
        #train_x = final_frames[:train_size]
        train_y = counts[: train_size]

        #valid_x = final_frames[train_size: train_size + valid_size]
        valid_y = counts[train_size: train_size + valid_size]
        #train_x = np.array(train_x)
        train_y = np.array(train_y)
        #valid_x = np.array(valid_x)
        valid_y = np.array(valid_y)
        return np.array(final_frames[:train_size]), train_y, np.array(final_frames[train_size: train_size + valid_size]), valid_y

    else:
        test_x = final_frames[train_size + valid_size:]
        test_y = counts[train_size + valid_size:]
        test_x = np.array(test_x)
        test_y = np.array(test_y)

        return test_x, test_y


def DrawGraph(train, valid):
    # removing first element because usually that is a very big number
    #del train[0]
    #del valid[0]
    print("train loss", train)
    print("valid loss", valid)

    epochs = list(range(1, len(train)+1))

    plt.plot(epochs, train, label="Train")

    plt.plot(epochs, valid, label="Valid")

    # naming the x axis
    plt.xlabel('Epochs')
    # naming the y axis
    plt.ylabel('Loss')
    # giving a title to my graph
    plt.title('Train and validation loss')

    # show a legend on the plot
    plt.legend()

    # function to show the plot
    # plt.show()
    # plt.savefig('figures/loss-{}.png'.format(exp_name))
    plt.savefig('results/bn_loss_vgg16_bsize32_lr_0.0001.png')
    plt.clf()
