import torch


def load_model_weights(model, weights_file, model_type, head_id=0):
    print("=> Initializing model '{}'".format(weights_file))
    pre_model = torch.load(weights_file, map_location="cpu")
    if model_type == "simclr":
        # rename simclr pre-trained keys
        state_dict = pre_model
        for k in list(state_dict.keys()):
            # Initialize the feature module with simclr.
            if k.startswith('backbone.'):
                # remove prefix
                state_dict["module.feature_module.{}".format(k[len('backbone.'):])] = state_dict[k]

            del state_dict[k]
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    elif model_type == "moco":
        # rename moco pre-trained keys
        print("=> Initializing feature model '{}'".format(weights_file))
        state_dict = pre_model['state_dict']
        for k in list(state_dict.keys()):
            # Initialize the feature module with encoder_q of moco.
            if k.startswith('module.encoder_q') and (not k.startswith('module.encoder_q.mlp') and not k.startswith('module.encoder_q.fc')):
                # remove prefix
                state_dict["module.feature_module.{}".format(k[len('module.encoder_q.'):])] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)

    elif model_type == "moco_select":
        # rename moco pre-trained keys
        print("=> Initializing feature model '{}'".format(weights_file))
        state_dict = pre_model['state_dict']

        if head_id == 0:
            msg = model.load_state_dict(state_dict, strict=False)
            print(msg)
            return 0

        for k in list(state_dict.keys()):
            # print(k)

            if k.startswith('module.head'):
                if k.startswith('module.head.head_{}'.format(head_id)):

                    state_dict['module.head.head_0.{}'.format(k[len('module.head.head_{}.'.format(head_id))::])] = state_dict[k]

                del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=True)
        print(msg)

    elif model_type == "moco_all":
        # rename moco pre-trained keys
        print("=> Initializing feature model '{}'".format(weights_file))
        state_dict = pre_model['state_dict']
        for k in list(state_dict.keys()):
            # Initialize the feature module with encoder_q of moco.
            if k.startswith('module.encoder_q'):
                # remove prefix
                state_dict["module.feature_module.{}".format(k[len('module.encoder_q.'):])] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)

    elif model_type == "simclr_sim":

        # rename simclr pre-trained keys
        state_dict = pre_model
        for k in list(state_dict.keys()):
            # Initialize the feature module with simcrl_sim.
            if k.startswith('backbone.'):
                # remove prefix
                state_dict["module.{}".format(k[len('backbone.'):])] = state_dict[k]

            if k.startswith('contrastive_head.'):
                state_dict["module.mlp.{}".format(k[len('contrastive_head.'):])] = state_dict[k]

            del state_dict[k]

        model.load_state_dict(state_dict, strict=True)

    elif model_type == "simclr_sim_feature":

        # rename simclr pre-trained keys
        state_dict = pre_model
        for k in list(state_dict.keys()):
            # Initialize the feature module with simcrl_sim.
            if k.startswith('backbone.'):
                # remove prefix
                state_dict["module.{}".format(k[len('backbone.'):])] = state_dict[k]

            if k.startswith('contrastive_head.'):
                state_dict["module.mlp.{}".format(k[len('contrastive_head.'):])] = state_dict[k]

            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)

    elif model_type == "moco_sim":
        # rename moco pre-trained keys
        state_dict = pre_model['state_dict']
        for k in list(state_dict.keys()):
            # Initialize the feature module with encoder_q of moco.
            if k.startswith('module.encoder_q'):
                # remove prefix
                state_dict["module.{}".format(k[len('module.encoder_q.'):])] = state_dict[k]

            # delete renamed or unused k
            del state_dict[k]

        model.load_state_dict(state_dict, strict=True)

    elif model_type == "moco_sim_feature":
        # rename moco pre-trained keys
        state_dict = pre_model['state_dict']
        for k in list(state_dict.keys()):
            # Initialize the feature module with encoder_q of moco.
            if k.startswith('module.encoder_q'):
                # remove prefix
                state_dict["module.{}".format(k[len('module.encoder_q.'):])] = state_dict[k]
                # state_dict["{}".format(k[len('module.encoder_q.'):])] = state_dict[k]

            # delete renamed or unused k
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)

    elif model_type == "moco_embedding":
        # rename moco pre-trained keys
        state_dict = pre_model['state_dict']
        for k in list(state_dict.keys()):
            # Initialize the feature module with encoder_q of moco.
            if k.startswith('module.encoder_q'):
                # remove prefix
                # state_dict["module.{}".format(k[len('module.encoder_q.'):])] = state_dict[k]
                state_dict["{}".format(k[len('module.encoder_q.'):])] = state_dict[k]

            # delete renamed or unused k
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)

    elif model_type == "moco_sim1":
        # rename moco pre-trained keys
        state_dict = pre_model['state_dict']
        for k in list(state_dict.keys()):
            # Initialize the feature module with encoder_q of moco.
            if k.startswith('module.encoder_q'):
                # remove prefix
                state_dict["{}".format(k[len('module.encoder_q.'):])] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        model.load_state_dict(state_dict, strict=True)

    else:
        raise TypeError
