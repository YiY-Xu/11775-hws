import torch
from torch.nn import Module


class MLP(Module):
    def __init__(self, layers, outputsize):
        super().__init__()
        self.hidden = torch.nn.ModuleList()
        for i in range(len(layers)-1):
            self.hidden.append(torch.nn.Linear(layers[i], layers[i+1]))
            self.hidden.append(torch.nn.BatchNorm1d(num_features=layers[i+1]))
            self.hidden.append(torch.nn.ReLU())

        self.last = torch.nn.Linear(layers[-1], outputsize)
        print(self)

    def forward(self, x):
        for layer in self.hidden:
            print(layer)
            print("---------------")
            print(x.size())
            x = layer(x)
            print(x.size())
        x = self.last(x)
        return x

    def to_Variable(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)

    def train_MLP(model, train_X, train_y, learning_rate, epochs, param_file):
        loss_fn = torch.nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_X = torch.from_numpy(train_X).float()
        train_y = torch.from_numpy(train_y).long()
        dataset = torch.utils.data.TensorDataset(train_X, train_y)
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=3, 
                                                 shuffle=True, 
                                                 drop_last=True)
        if torch.cuda.is_available():
            model = model.cuda()
            loss_fn = loss_fn.cuda()
            torch.cuda.empty_cache()

        for epoch in range(epochs):
            losses = []
            model.train()
            for (input_val, label) in dataloader:
                optim.zero_grad()
                print(input_val.size())
                pred = model(to_Variable(input_val))
                print(label)
                loss = loss_fn(pred, to_Variable(label.long()))
                loss.backward()
                losses.append(loss.data.cpu().numpy())
                optim.step()
            loss_train = np.mean(losses)
            print(loss_train)
        torch.save(model.state_dict(), param_file)

    def train_SVM(model, train_X, train_y):
        pass


    def test_MLP(model, test, test_filenames, param_file):
        model.load_state_dict(param_file)
        model.eval()
        pred = list(model(to_Variable(test)))
        with open('submission.csv', 'w') as f:
            for name, label in zip(test_filenames, pred):
                result = name + ', ' + str(label)
                f.write("%s\n" % result)