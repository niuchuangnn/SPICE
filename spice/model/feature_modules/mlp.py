import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, num_neurons, drop_out=-1, last_activation=None, return_extra_index=[], batch_norm=False):
        super(MLP, self).__init__()
        num_layer = len(num_neurons) - 1
        for i in range(num_layer):
            layer_name = "lin{}".format(i+1)
            layer = nn.Linear(num_neurons[i], num_neurons[i+1])
            self.add_module(layer_name, layer)

            if batch_norm:
                layer_name = "bn{}".format(i+1)
                layer = nn.BatchNorm1d(num_neurons[i+1])
                self.add_module(layer_name, layer)

        self.num_layer = num_layer
        self.drop_out = drop_out
        self.last_activation = last_activation
        self.return_extra_index = return_extra_index
        self.batch_norm = batch_norm

    def forward(self, x):
        num_layer = self.num_layer
        outs_extra = []
        for i in range(num_layer):
            layer_name = "lin{}".format(i+1)
            layer = self.__getattr__(layer_name)
            x = layer(x)

            if self.batch_norm:
                bn_name = "bn{}".format(i+1)
                bn = self.__getattr__(bn_name)
                x = bn(x)

            if i < num_layer - 1:
                if self.drop_out >= 0:
                    x = F.dropout(x, p=self.drop_out, training=self.training)
                x = F.relu(x, inplace=True)

            if (i+1) in self.return_extra_index:
                outs_extra.append(x)

        if self.last_activation == "relu":
            x = F.relu(x, inplace=True)
        elif self.last_activation == "sigmoid":
            x = torch.sigmoid(x)
        elif self.last_activation == "exp_norm":
            x = torch.exp(x - x.max(dim=1)[0].unsqueeze(1))
        elif self.last_activation == "tanh":
            x = torch.tanh(x)
        elif self.last_activation == "softmax":
            x = torch.softmax(x, dim=1)
        elif self.last_activation is None:
            pass
        else:
            assert TypeError

        if len(outs_extra) > 0:
            return [x] + outs_extra
        else:
            return x