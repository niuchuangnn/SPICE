import torch
import torch.nn as nn


class ExpNorm(nn.Module):
    def __init__(self):
        super(ExpNorm, self).__init__()

    def forward(self, x):
        return torch.exp(x - torch.nn.MaxPool2d(int(x.shape[-1]))(x))


class ConvNet(nn.Module):
    def __init__(self, input_channel, conv_layers, kernels, strides, pads, num_block, fc_layers=[], output_paddings=None,
                 batch_norm=True, transpose=False, return_pool_idx=True, last_fc_activation=None, conv_fea_size=[7,7],
                 use_ave_pool=False, fc_input_neurons=None, last_conv_activation="relu", use_last_conv_bn=True,
                 output_feas=False, output_feas_pool=False):
        super(ConvNet, self).__init__()
        if transpose:
            block_id = num_block
            last_block_ids = list(range(2, num_block+1))
        else:
            block_id = 1
            last_block_ids = list(range(1, num_block))
        in_channel = input_channel
        num_layers = len(conv_layers)
        self.encoder = nn.Sequential()
        for l in range(num_layers):
            sub_layers = conv_layers[l]
            sub_kernels = kernels[l]
            sub_strides = strides[l]
            sub_pads = pads[l]
            if isinstance(sub_layers, list):
                assert isinstance(sub_kernels, list) and len(sub_layers) == len(sub_kernels)
                assert isinstance(sub_strides, list) and len(sub_layers) == len(sub_strides)
                assert isinstance(sub_pads, list) and len(sub_layers) == len(sub_pads)
                for i in range(len(sub_layers)):
                    out_channel = sub_layers[i]
                    kernel_size = sub_kernels[i]
                    stride = sub_strides[i]
                    pad = sub_pads[i]
                    sub_layer_id = i+1

                    if transpose:
                        output_padding = 0
                        if output_paddings is not None:
                            output_padding = output_paddings[l][i]
                        sub_layer_id = len(sub_layers) - i
                        layer = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, pad, output_padding=output_padding)
                        layer_name = "deconv{}_{}".format(block_id, sub_layer_id)
                    else:
                        layer = nn.Conv2d(in_channel, out_channel, kernel_size, stride, pad)
                        layer_name = "conv{}_{}".format(block_id, sub_layer_id)

                    # self.add_module(layer_name, layer)
                    self.encoder.add_module(layer_name, layer)

                    if batch_norm:
                        if not use_last_conv_bn and block_id not in last_block_ids and i >= len(sub_layers)-1:
                            pass
                        else:
                            bn_name = "bn{}_{}".format(block_id, sub_layer_id)
                            bn_layer = nn.BatchNorm2d(out_channel)
                            # self.add_module(bn_name, bn_layer)
                            self.encoder.add_module(bn_name, bn_layer)

                    if block_id in last_block_ids or i < len(sub_layers)-1:
                        relu_name = "relu{}_{}".format(block_id, sub_layer_id)
                        self.encoder.add_module(relu_name, nn.ReLU(inplace=True))
                    else:
                        if last_conv_activation == "sigmoid":
                            act_name = "sigmoid_score"
                            self.encoder.add_module(act_name, nn.Sigmoid())
                        elif last_conv_activation == "relu":
                            relu_name = "relu{}".format(block_id)
                            self.encoder.add_module(relu_name, nn.ReLU(inplace=True))

                        elif last_conv_activation == "exp_norm":
                            exp_name = "exp_norm"
                            self.encoder.add_module(exp_name, ExpNorm())
                        elif last_conv_activation == "tanh":
                            tanh_name = "tanh{}".format(block_id)
                            self.encoder.add_module(tanh_name, nn.Tanh())

                        elif last_conv_activation is None:
                            pass
                        else:
                            raise TypeError

                    in_channel = out_channel

            elif isinstance(sub_layers, str):
                assert isinstance(sub_kernels, int)
                assert isinstance(sub_strides, int)
                assert isinstance(sub_pads, int)
                if sub_layers == "max_pooling":
                    layer_name = "pool{}".format(block_id)
                    layer = nn.MaxPool2d(sub_kernels, sub_strides, sub_pads, return_indices=return_pool_idx)
                    self.encoder.add_module(layer_name, layer)
                elif sub_layers == "max_unpooling":
                    layer_name = "unpool{}".format(block_id)
                    layer = nn.MaxUnpool2d(sub_kernels, sub_strides, sub_pads)
                    self.encoder.add_module(layer_name, layer)
                else:
                    raise TypeError

                if transpose:
                    block_id -= 1
                else:
                    block_id += 1

            else:
                raise TypeError

        num_fc_layers = len(fc_layers)
        self.encoder_fc = None
        if num_fc_layers > 0:
            self.encoder_fc = nn.Sequential()
            assert fc_input_neurons is not None

            for l in range(num_fc_layers):
                fc_output_neurons = fc_layers[l]
                layer_name = "fc{}".format(l+1)
                layer = nn.Linear(fc_input_neurons, fc_output_neurons)
                self.encoder_fc.add_module(layer_name, layer)
                if batch_norm:
                    bn_name = "bn_fc{}".format(l+1)
                    bn = nn.BatchNorm1d(fc_output_neurons)
                    self.encoder_fc.add_module(bn_name, bn)

                fc_input_neurons = fc_output_neurons

                if l == num_fc_layers - 1:
                    if last_fc_activation is None:
                        pass
                    elif last_fc_activation == "softmax":
                        layer = nn.Softmax(dim=1)
                        self.encoder_fc.add_module("fc_{}".format(last_fc_activation), layer)
                    elif last_fc_activation == "tanh":
                        layer = nn.Tanh()
                        self.encoder_fc.add_module("fc_{}".format(last_fc_activation), layer)
                    elif last_fc_activation == "sigmoid":
                        layer = nn.Sigmoid()
                        self.encoder_fc.add_module("fc_{}".format(last_fc_activation), layer)
                    elif last_fc_activation == "relu":
                        layer = nn.ReLU(inplace=True)
                        self.encoder_fc.add_module("fc_{}".format(last_fc_activation), layer)
                    else:
                        raise TypeError

                else:
                    self.encoder_fc.add_module("relu_fc{}".format(l+1), nn.ReLU(inplace=True))

        if use_ave_pool:
            self.ave_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.ave_pool = None

        self.transpose = transpose
        self.return_pool_idx = return_pool_idx
        self.conv_fea_size = conv_fea_size
        self.output_feas = output_feas
        self.output_feas_pool = output_feas_pool

    def forward(self, x):
        conv_fea_size = self.conv_fea_size
        pool_idx = dict()
        feas = []
        feas_pool = dict()
        if not self.transpose:
            for name, layer in self.encoder._modules.items():

                if self.return_pool_idx and "pool" in name:
                    x, idx = layer(x)
                    pool_idx["{}_idx".format(name)] = idx
                    if self.output_feas_pool:
                        feas_pool[name] = x
                else:
                    x = layer(x)

            if self.output_feas:
                feas.append(x)

            if self.encoder_fc is not None:
                if self.ave_pool is not None:
                    x = self.ave_pool(x)
                x = torch.flatten(x, start_dim=1)

                x = self.encoder_fc(x)
        else:
            if isinstance(x, tuple):
                unpool_idx = x[1]
                x = x[0]
            else:
                unpool_idx = []

            if self.encoder_fc is not None:
                x = self.encoder_fc(x)
                assert isinstance(conv_fea_size, list)
                if self.ave_pool is not None:
                    x = x.reshape([x.shape[0], x.shape[1], 1, 1]).repeat(1, 1, conv_fea_size[0], conv_fea_size[1]) \
                        / (conv_fea_size[0] * conv_fea_size[1])
                else:
                    x = x.reshape([x.shape[0], -1, conv_fea_size[0], conv_fea_size[1]])

            pool_id = len(unpool_idx)
            for name, layer in self.encoder._modules.items():
                if "unpool" in name:
                    if self.output_feas_pool:
                        feas_pool[name] = x

                    name_idx = "pool{}_idx".format(pool_id)
                    x = layer(x, unpool_idx[name_idx])

                    pool_id -= 1
                else:
                    x = layer(x)

        if len(pool_idx) > 0:
            outs = (x, pool_idx)
        else:
            outs = x

        if self.output_feas:
            outs = [outs] + feas

        if self.output_feas_pool:
            if isinstance(outs, list):
                outs.append(feas_pool)
            else:
                outs = [outs] + [feas_pool]

        return outs