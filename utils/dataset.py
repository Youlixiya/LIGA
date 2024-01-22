from torch.utils.data import Dataset
import random
from PIL import Image
import numpy as np
from .refer import REFER
import torch

class RefCOCO(Dataset):
    def __init__(self, data_root, dataset='refcocog', splitBy='umd', split='train') -> None:
        super().__init__()
        # if dataset == 'refclef':
        #     if splitBy == 'unc':
        #         splits = ['train', 'val', 'testA', 'testB', 'testC']
        #     else:
        #         splits = ['train', 'val', 'test']
        # elif dataset == 'refcoco':
        #     splits = ['train', 'val', 'test']
        # elif dataset == 'refcoco+':
        #     splits = ['train', 'val', 'test']
        # elif dataset == 'refcocog':
        #     splits = ['train', 'val', 'test']  # we don't have test split for refcocog right now.
        self.refer = REFER(data_root, dataset, splitBy)
        self.ref_ids = self.refer.getRefIds(split=split)
    
    def __len__(self):
        return len(self.ref_ids)
    
    def __getitem__(self, index):
        image = self.refer.getRefImage(self.ref_ids[index])
        box = np.array(self.refer.getRefBox(self.ref_ids[index]), dtype=np.int64)
        box[2] = box[2] + box[0]
        box[3] = box[3]
        sentences = [sentence['sent'] for sentence in self.refer.Refs[index]['sentences']]
        sentence = sentences[random.randint(0, len(sentences)-1)]
        # ref_ids = serefer.getRefIds(split=split)
        return image, box, sentence

def collate_fn(
    batch,
    prompt_encoder,
    image_size=1024
):
    images = []
    boxes = []
    # resize_boxes = []
    boxes_embeddings = []
    sentences = []
    ori_shapes = []
    # cnt = 0
    # offset_list = [0]
    for (
        image,
        box,
        sentence,
    ) in batch:
        
        
        image_shape = image.shape[:2]
        ori_shapes.append(image_shape)
        # scale_h = image_size/ image_shape[0]
        # scale_w = image_size/ image_shape[1]
        box = box.astype(np.float32)
        box[[0, 2]] /= image_shape[1]
        box[[1, 3]] /= image_shape[0]
        boxes.append(box)
        # box = box.copy().astype(np.float32)
        # box[[0, 2]] *= scale_w
        # box[[1, 3]] *= scale_h
        with torch.no_grad():
            boxes_embeddings.append(prompt_encoder(points=None, boxes=torch.from_numpy(box[None]), masks=None)[0])
        
        images.append(Image.fromarray(image))
        sentences.append(sentence)
        # cnt += len(sentences)
        # offset_list.append(cnt)
        
        return {
            'images': images,
            'texts': sentences,
            'boxes': torch.from_numpy(np.stack(boxes)),
            # 'boxes_embeddings': torch.cat(boxes_embeddings),
            'ori_shapes': torch.LongTensor(ori_shapes)
            }
