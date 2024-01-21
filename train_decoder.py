from omegaconf import OmegaConf
from tqdm import tqdm
import torch
from torch import nn
from segment_anything import SamPredictor, sam_model_registry


CONFIG_PATH = 'configs/train_decoder.yaml'
def parse_args():
    return OmegaConf.load(CONFIG_PATH)

def main(args):
    # print(args)
    model_args = args.model
    train_args = args.train
    sam = sam_model_registry["vit_h"](checkpoint=model_args.sam_path).cuda()
    del sam.image_encoder
    del sam.mask_decoder
    sam.prompt_encoder
    box_decoder = nn.Sequential(
        nn.Linear(model_args.object_embedding_dim, model_args.object_embedding_dim),
        nn.SiLU(inplace=True),
        nn.Linear(model_args.object_embedding_dim, 4),
        nn.Sigmoid()
    ).cuda()
    loss_fn = nn.MSELoss() if train_args.loss_fn == 'mse' else nn.L1Loss()
    OPTIMIZER = torch.optim.Adam if train_args.optimizer == 'adam' else torch.optim.SGD
    optimizer = OPTIMIZER(box_decoder.parameters(), lr=train_args.lr)
    for _ in tqdm(range(train_args.steps)):
        boxes = torch.randint(0, train_args.image_size, (train_args.batch_size, 4)).cuda()
        box_embeddings = sam.prompt_encoder(points=None, boxes=boxes, masks=None)[0]
        pred_boxes = box_decoder(box_embeddings.reshape(-1, model_args.object_embedding_dim))
        loss = loss_fn(pred_boxes, boxes/train_args.image_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(box_decoder.state_dict(), train_args.save_path)
    print('decoder has saved!')
    print('test start!')
    box = torch.randint(0, train_args.image_size, (1, 4)).cuda()
    box_embedding = sam.prompt_encoder(points=None, boxes=box, masks=None)[0]
    pred_box = box_decoder(box_embedding.reshape(-1, model_args.object_embedding_dim))
    pred_box *= train_args.image_size
    print(f'pred_box:{pred_box}')
    print(f'pred_box:{box}')
    

if __name__ == '__main__':
    main(parse_args())