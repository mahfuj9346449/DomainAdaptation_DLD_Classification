import os
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('agg')

sourceDomainPath = '../SourceDomain/'
targetDomainPath = '../TargetDomain/'

experimentalPath = '../experiment_data/'

pulmonary_category = {0: 'CON',
                      1: 'M-GGO',
                      2: 'HCM',
                      3: 'EMP',
                      4: 'NOD',
                      5: 'NOR'}


def save2file(message, checkpointPath, model_name):
    if not os.path.isdir(checkpointPath):
        os.makedirs(checkpointPath)
    logfile = open(checkpointPath + model_name + '.txt', 'a+')
    print(message)
    print(message, file=logfile)
    logfile.close()


def plotAccuracy(x, y1, y2, figName, line1Name, line2Name, savePath):
    plt.figure(figsize=(20.48, 10.24))
    plt.plot(x, y1, linewidth=1.0, linestyle='-', label=line1Name)
    plt.plot(x, y2, linewidth=1.0, color='red', linestyle='--', label=line2Name)
    plt.title('Accuracy')
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(savePath + figName + '_accuracy.png')
    plt.close()


def plotLoss(x, y1, y2, figName, line1Name, line2Name, savePath):
    plt.figure(figsize=(20.48, 10.24))
    plt.plot(x, y1, linewidth=1.0, linestyle='-', label=line1Name)
    plt.plot(x, y2, linewidth=1.0, color='red', linestyle='--', label=line2Name)
    plt.title('Loss')
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(savePath + figName + '_loss.png')
    plt.close()


def normalizeInput(inputData, mode):
    if mode == 'Paired':
        inputData[0] = inputData[0] / 127.5 - 1.0
    elif mode == 'Unpaired':
        inputData = inputData / 127.5 - 1.0
    else:
        print('Error in Normalize Input')
        exit(0)
    print('Normalization Finish')
    return inputData


def getPartialDataSet(dataSet, keep_percentage):
    image_label_dic = {0: [],
                       1: [],
                       2: [],
                       3: [],
                       4: [],
                       5: []}

    chosenImage, choseLabel = [], []

    for singleImage, singleLabel in zip(dataSet[0], dataSet[1]):
        c = int(np.argmax(singleLabel))
        image_label_dic[c].append([singleImage, singleLabel])

    for i in range(6):
        np.random.shuffle(image_label_dic[i])
        chosenGroup = image_label_dic[i][: int(len(image_label_dic[i]) * keep_percentage)]
        for j in range(len(chosenGroup)):
            chosenImage.append(chosenGroup[j][0])
            choseLabel.append(chosenGroup[j][1])

    print('Partial Data Process Finish, Total Get {} Images / Labels'.format(len(chosenImage)))

    return [np.array(chosenImage), np.array(choseLabel)]
