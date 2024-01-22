import os
import requests
import torchkeras
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from liga.model import LIGAModel
from segment_anything import SamPredictor, sam_model_registry
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from tqdm import tqdm
from utils.utils import StepRunner
from utils.dataset import RefCOCO, collate_fn
from accelerate import Accelerator, DistributedDataParallelKwargs
torchkeras.KerasModel.StepRunner = StepRunner

CONFIG_PATH = 'configs/train.yaml'
def parse_args():
    return OmegaConf.load(CONFIG_PATH)

def main(args):
    model_args = args.model
    train_args = args.train
    ckpt = torch.load(train_args.save_path)
    liga_model = LIGAModel.from_pretrained(model_args.clip_model, **model_args)
    liga_model.load_state_dict(ckpt)
    hf_model_save_path = train_args.save_path[:-3] + '-hf'
    liga_model.save_pretrained(hf_model_save_path)
    print('hf model saved!')

if __name__ == '__main__':
    main(parse_args())