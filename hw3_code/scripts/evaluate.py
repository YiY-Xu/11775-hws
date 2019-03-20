import numpy as np

if __name__ == '__main__':
    names = []
    bases = []
    labels = []
    with open('../baseline.csv', 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue
            name, label = line.replace('\n','').split(',')
            names.append(name)
            bases.append(int(label))
    bases = np.array(bases)

    with open('../output/submission.csv', 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue
            _, label = line.replace('\n','').split(',')
            labels.append(int(label))

    labels = np.array(labels)

    length = (labels[labels>0]).shape[0]

    negatives = 0
    for i in range(len(names)):
        if labels[i] > 0 and labels[i]!=bases[i]:
            negatives += 1

    print("difference from baseline: " + str(float(negatives)/length))
    print(float((labels[labels==1]).shape[0])/len(names), float((labels[labels==2]).shape[0])/len(names), float((labels[labels==3]).shape[0])/len(names))

