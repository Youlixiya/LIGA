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
    # print(args)
    
    model_args = args.model
    train_args = args.train
    sam = sam_model_registry["vit_h"](checkpoint=model_args.sam_path).cuda()
    del sam.image_encoder
    del sam.mask_decoder
    prompt_encoder = sam.prompt_encoder
    torch_dtype = torch.float16 if train_args.torch_dtype == 'half' else torch.float32
    liga_model = LIGAModel.from_pretrained(model_args.clip_model, torch_dtype=torch_dtype, **model_args).cuda()
    for p in liga_model.vision_model.parameters():
        p.requires_grad = False
    for p in liga_model.dino_model.parameters(): 
        p.requires_grad = False
    # for p in liga_model.box_decoder.parameters():
    #     p.requires_grad = False
    for p in liga_model.visual_projection.parameters():
        p.requires_grad = False
    for p in liga_model.text_projection.parameters():
        p.requires_grad = False
    loss_fn = nn.MSELoss() if train_args.loss_fn == 'mse' else nn.L1Loss()
    OPTIMIZER = torch.optim.Adam if train_args.optimizer == 'adam' else torch.optim.SGD
    # optimizer = OPTIMIZER(box_decoder.parameters(), lr=train_args.lr)
    train_dataset = RefCOCO(train_args.data_root, dataset=train_args.dataset, splitBy=train_args.splitBy, split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=train_args.batch_size, shuffle=True, collate_fn=partial(collate_fn, prompt_encoder=prompt_encoder))
    val_dataset = RefCOCO(train_args.data_root, dataset=train_args.dataset, splitBy=train_args.splitBy, split='val')
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=partial(collate_fn, prompt_encoder=prompt_encoder))
    # for name, parameter in liga_model.clip_projector.named_parameters():
    print(liga_model.object_embedding.requires_grad, liga_model.object_embedding.is_leaf)

    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    model = torchkeras.KerasModel(liga_model,
      loss_fn = loss_fn,
      optimizer= OPTIMIZER(list(liga_model.text_model.parameters()) + \
                           list(liga_model.object_projector.parameters()) +\
                           list(liga_model.dino_projector.parameters()) +\
                           list(liga_model.box_encoder.parameters()) +\
                           list(liga_model.box_decoder.parameters()) +\
                           [liga_model.object_embedding] +\
                           list(liga_model.clip_projector.parameters()),lr=train_args.lr),
      metrics_dict = {},
    )
    dfhistory=model.fit(
                    # num_processes=4,
                    train_data=train_dataloader, 
                    val_data=val_dataloader, 
                    epochs=train_args.epochs, 
                    patience=5, 
                    ckpt_path=train_args.save_path,
                    # mixed_precision='fp16',
                    monitor="val_iou",
                    mode="max",
                    plot=True,
                    accelerator=accelerator
                   )
    # image = Image.open('../TinyLLava/images/bee.png').convert('RGB')
    # images = [image, image]
    # texts = ['sadsadas', 'sadasdasd']
    # boxes = torch.tensor([[100, 200, 300, 400],
    #                      [200, 300, 400, 500]]).cuda()
    # boxes_embeddings = prompt_encoder(points=None, boxes=boxes, masks=None)[0]
    # shape = (image.size[1], image.size[0])
    # ori_shapes = torch.LongTensor([[shape[0], shape[1]],
    #                               [shape[0], shape[1]]])
    # input = {
    #     'images':images,
    #     'texts': texts, 
    #     'boxes': boxes,
    #     'boxes_embeddings': boxes_embeddings,
    #     'ori_shapes': ori_shapes,
    # }
    
    # print(input)
    # print(liga_model(**input))
    
    # os.makedirs(train_args.save_path, exist_ok=True)
    # liga_model.save_pretrained(train_args.save_path)
    # print(LIGAModel.from_pretrained(train_args.save_path, torch_dtype=torch.float16))
    # model_args = args.model
    # train_args = args.train
    # sam = sam_model_registry["vit_h"](checkpoint=model_args.sam_path).cuda()
    # del sam.image_encoder
    # del sam.mask_decoder
    # sam.prompt_encoder
    # decoder = nn.Sequential(
    #     nn.Linear(model_args.object_embedding_dim, model_args.object_embedding_dim),
    #     nn.SiLU(inplace=True),
    #     nn.Linear(model_args.object_embedding_dim, 4),
    #     nn.Sigmoid()
    # ).cuda()
    # loss_fn = nn.MSELoss() if train_args.loss_fn == 'mse' else nn.L1Loss()
    # OPTIMIZER = torch.optim.Adam if train_args.optimizer == 'adam' else torch.optim.SGD
    # optimizer = OPTIMIZER(decoder.parameters(), lr=train_args.lr)
    # for _ in tqdm(range(train_args.steps)):
    #     boxes = torch.randint(0, train_args.image_size, (train_args.batch_size, 4)).cuda()
    #     box_embeddings = sam.prompt_encoder(points=None, boxes=boxes, masks=None)[0]
    #     pred_boxes = decoder(box_embeddings.reshape(-1, model_args.object_embedding_dim))
    #     loss = loss_fn(pred_boxes, boxes/train_args.image_size)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    # torch.save(decoder.state_dict(), train_args.save_path)
    # print('decoder has saved!')
    # print('test start!')
    # box = torch.randint(0, train_args.image_size, (1, 4)).cuda()
    # box_embedding = sam.prompt_encoder(points=None, boxes=box, masks=None)[0]
    # pred_box = decoder(box_embedding.reshape(-1, model_args.object_embedding_dim))
    # pred_box *= train_args.image_size
    # print(f'pred_box:{pred_box}')
    # print(f'pred_box:{box}')
    

if __name__ == '__main__':
    main(parse_args())