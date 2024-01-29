import pickle
import torch
import utils

def trans_path():
    for phase in ['train', 'val', 'test']:
        
        with open(f'./materials/mini_imagenet_480_{phase}.pickle', 'rb') as f:
            pack = pickle.load(f)
            paths = pack['data']
                        
            paths_new = ['./materials/mini_imagenet_480/' + path[-21:] for path in paths]

            pack['data'] = paths_new

        with open(f'./materials/mini_imagenet_480_{phase}.pickle', 'wb') as f:
            pickle.dump(pack, f)

def trans_ckpt(load_path, save_path, model_name='vit-small'):
        
    model = torch.load(load_path)
    state_dict = model['teacher']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

    ckpt = {}
    ckpt['model'] = 'meta-baseline'
    ckpt['model_args'] = {'encoder': model_name, 'encoder_args': {}}
    ckpt['model_sd'] = state_dict
    torch.save(ckpt, save_path)
    
