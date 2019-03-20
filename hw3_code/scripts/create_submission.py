import sys
import numpy as np

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print ("Usage: {0} stage".format(sys.argv[0]))
        print ("stage -- EF/LF/DF/K")
        print ("Multiclass -- True/False")
        exit(1)

    stage = sys.argv[1]
    multiclass = str2bool(sys.argv[2])

    predict = []

    if multiclass:
        file = '../output/P001_' + stage + '.lst'
        with open(file, 'r') as f:
            for line in f.readlines():
                data = line[1:6].split(' ')
                if data[0] == '1':
                    predict.append(1)
                elif data[1] == '1':
                    predict.append(2)
                elif data[2] == '1':
                    predict.append(3)
                else:
                    predict.append(0)
    else:
        temp = np.zeros((1699, 3))
        for i in range(3):
            file = '../output/P00' + str(i+1) + '_' + stage + '.lst'
            with open(file, 'r') as f:
                for j, line in enumerate(f.readlines()):
                    data = int(line.replace('\n',''))
                    if data == 1:
                        temp[j][i] = i+1
        predict = list(np.max(temp.astype(int), axis=1))

    names = []
    with open('../../all_test_fake.lst', 'r') as f:
        for line in f.readlines():
            name = line.replace('\n','').split(' ')[0]
            names.append(name)

    with open('../output/submission.csv', 'w') as f:
        f.write("VideoID,label\n")
        for name, result in zip(names, predict):
            item = name + ',' + str(result)
            f.write("%s\n" % item)
