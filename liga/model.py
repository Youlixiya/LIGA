from typing import Optional, Tuple, Union
from PIL.Image import Image
from dataclasses import dataclass
from torchvision.ops.boxes import box_area
from transformers import AutoImageProcessor, AutoBackbone, CLIPModel, CLIPProcessor, CLIPConfig
from transformers.models.llama.modeling_llama import LlamaFlashAttention2 as LIGAFlashAttention2
from transformers.models.llama.modeling_llama import LlamaSdpaAttention as LIGASdpaAttention
from transformers.models.llama.modeling_llama import LlamaRMSNorm as LIGARMSNorm
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask

from segment_anything import SamPredictor, sam_model_registry


import torch
from torch import nn
# try:
#     from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
#     USE_FLASH_ATTN2 = Fasle
#     # ATTENTION_MODULE=LIGAFlashAttention2
# except:
    # ATTENTION_MODULE=LIGASdpaAttention
USE_FLASH_ATTN2 = False
# print(f'use_flash_attn2:{USE_FLASH_ATTN2}')
@dataclass
class LIGAOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    # hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # attentions: Optional[Tuple[torch.FloatTensor]] = None
    loss: torch.FloatTensor = None
    iou: torch.FloatTensor = None
    giou: torch.FloatTensor = None
    boxes: torch.FloatTensor = None
    embedding_mse_loss: torch.FloatTensor = None
    boxes_mse_loss: torch.FloatTensor = None
    giou_loss: torch.FloatTensor = None
    reconstruction_loss: torch.FloatTensor = None
    object_embeddings: torch.FloatTensor = None

class LIGAMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
    
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj

class LIGAAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.attention_dropout = config.attention_dropout
        self.q_proj0 = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj0 = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj0 = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.q_proj1 = nn.Linear(self.head_dim, self.head_dim, bias=config.attention_bias)
        self.k_proj1 = nn.Linear(self.head_dim, self.head_dim, bias=config.attention_bias)
        self.v_proj1 = nn.Linear(self.head_dim, self.head_dim, bias=config.attention_bias)
        self.task_proj =  nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
    def forward(
        self,
        vision_hidden_states: torch.Tensor,
        text_hidden_states: torch.Tensor,
        task_hidden_states: torch.Tensor,
    ) -> torch.Tensor:

        bsz, vision_len, _ = vision_hidden_states.shape
        text_len = text_hidden_states.shape[1]
        task_len = task_hidden_states.shape[1]
        total_len = vision_len + text_len + task_len
        hidden_states = torch.cat([vision_hidden_states, text_hidden_states, task_hidden_states], dim=1)
        q0 = self.q_proj0(hidden_states).reshape(bsz, total_len, self.num_heads, self.head_dim)
        k0 = self.k_proj0(hidden_states).reshape(bsz, total_len, self.num_heads, self.head_dim)
        v0 = self.v_proj0(hidden_states).reshape(bsz, total_len, self.num_heads, self.head_dim)
        # q1 = self.q_proj(hidden_states).reshape(bsz, total_len, self.num_heads, self.head_dim)
        # k1 = self.k_proj(hidden_states).reshape(bsz, total_len, self.num_heads, self.head_dim)
        # v1 = self.v_proj(hidden_states).reshape(bsz, total_len, self.num_heads, self.head_dim)
        # task = self.task_proj(task_hidden_states).reshape(bsz, task_len, self.num_heads, self.head_dim)
        # task_q = q[:, [0], :, :]
        vison_q = q0[:, :vision_len, :, :]
        text_q = q0[:, vision_len:vision_len+text_len, :, :]
        task_q = q0[:, vision_len+text_len:, :, :]
        vison_k = k0[:, :vision_len, :, :]
        text_k = k0[:, vision_len:vision_len+text_len, :, :]
        task_k = k0[:, vision_len+text_len:, :, :]
        vison_v = v0[:, :vision_len, :, :]
        text_v = v0[:, vision_len:vision_len+text_len, :, :]
        task_v = q0[:, vision_len+text_len:, :, :]
        if USE_FLASH_ATTN2:
            vison2text_attn_out = flash_attn_func(vison_q, text_k, text_v, dropout_p=self.attention_dropout, softmax_scale=None, causal=False)
        else:
            vison2text_attn_out = self.attn_fun(vison_q, text_k, text_v)
        if USE_FLASH_ATTN2:
            text2vision_attn_out = flash_attn_func(text_q, vison_k, vison_v, dropout_p=self.attention_dropout, softmax_scale=None, causal=False)
        else:
            text2vision_attn_out = self.attn_fun(text_q, vison_k, vison_v)
        
        hidden_states = torch.cat([vison2text_attn_out, text2vision_attn_out], dim=1)
        q1 = torch.cat([task_q, self.q_proj1(hidden_states)], dim=1)
        k1 = torch.cat([task_k, self.k_proj1(hidden_states)], dim=1)
        v1 = torch.cat([task_v, self.v_proj1(hidden_states)], dim=1)
        if USE_FLASH_ATTN2:
            attn_out = flash_attn_func(q1, k1, v1, dropout_p=self.attention_dropout, softmax_scale=None, causal=False)
        else:
            attn_out = self.attn_fun(q1, k1, v1)
        
        hidden_states = self.o_proj(attn_out.reshape(bsz, total_len, self.hidden_size))
        # return hidden_states[:, :vision_len, :], hidden_states[:, vision_len:vision_len+text_len, :], hidden_states[:, vision_len+text_len:, :]
        return hidden_states
    
    def attn_fun(self, q, k, v):
        b, q_l, n, d = q.shape
        k_l = k.shape[1]
        v_l = v.shape[1]
        q = q.permute(0, 2, 1, 3).reshape(b*n, q_l, d)
        k = k.permute(0, 2, 1, 3).reshape(b*n, k_l, d)
        v = v.permute(0, 2, 1, 3).reshape(b*n, v_l, d)
        attn = torch.bmm(q, k.transpose(-1, -2))
        attn_score = (attn / d).softmax(-1)
        return torch.bmm(attn_score, v).reshape(b, n, q_l, d).permute(0, 2, 1, 3)

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    # assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou, iou - (area - union) / area


class LIGAEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LIGAAttention(config=config)

        self.mlp = LIGAMLP(config)
        self.input_layernorm = LIGARMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LIGARMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        vision_hidden_states: torch.Tensor,
        text_hidden_states: torch.Tensor,
        task_hidden_states: torch.Tensor,
        # attention_mask: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        # past_key_value: Optional[Tuple[torch.Tensor]] = None,
        # output_attentions: Optional[bool] = False,
        # use_cache: Optional[bool] = False,
        # **kwargs,
    ) -> torch.FloatTensor:
        

        hidden_states = torch.cat([vision_hidden_states, text_hidden_states, task_hidden_states], dim=1)
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        vision_len = vision_hidden_states.shape[1]
        text_len = text_hidden_states.shape[1]
        task_len = task_hidden_states.shape[1]

        hidden_states = self.self_attn(
            vision_hidden_states=hidden_states[:, :vision_len, :],
            text_hidden_states=hidden_states[:, vision_len: vision_len+text_len, :],
            task_hidden_states=hidden_states[:, vision_len+text_len:, :],
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # outputs = (hidden_states,)
        
        # if output_attentions:
        #     outputs += (self_attn_weights,)

        # if use_cache:
        #     outputs += (present_key_value,)

        return hidden_states[:, :vision_len, :], hidden_states[:, vision_len:vision_len+text_len, :], hidden_states[:, vision_len+text_len:, :]

class NextChatConfig(CLIPConfig):
    model_type = "liga"
    dino_model = 'facebook/dinov2-base'
    clip_model = "openai/clip-vit-base-patch14"
    object_embedding_dim = 512
    image_size = 1024
    sam_path = 'checkpoints/sam/sam_vit_h_4b8939.pth'
    decoder_path = 'checkpoints/decoder.pt'
    fusion_encoder_layers = 4
    hidden_size = 512
    intermediate_size = 2048
    num_hidden_layers = 4

class LIGAModel(CLIPModel):
    
    # config_class = NextChatConfig
    
    def __init__(self, config, **kwargs):
        
        # if not hasattr(config, 'dino_model'):
        super().__init__(config)
        for key, value in kwargs.items():
            setattr(config, key, value)
        self.config = config
        self.clip_vison_hidden_size = config.vision_config.hidden_size
        self.clip_text_hidden_size = config.text_config.hidden_size
        self.hidden_size = self.config.hidden_size
        self.object_embedding = nn.Parameter(torch.randn(1, self.hidden_size).requires_grad_(True))
        self.clip_processor = CLIPProcessor.from_pretrained(self.config.clip_model)
        # self.clip_model = CLIPModel.from_pretrained(config.clip_model)
        self.dino_processor = AutoImageProcessor.from_pretrained(self.config.dino_model)
        self.dino_model = AutoBackbone.from_pretrained(self.config.dino_model, out_features=["stage2", "stage5", "stage8", "stage11"])
        # self.encoder_layers = nn.ModuleList(
        #     [LIGAEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        # )
        self.fusion_encoder = nn.ModuleList(
            [LIGAEncoderLayer(self.config) for _ in range(self.config.num_hidden_layers)]
        )
        self.object_projector = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.config.object_embedding_dim)
        )
        self.box_encoder = nn.Sequential(
            nn.Linear(4, self.config.object_embedding_dim),
            nn.SiLU(),
            nn.Linear(self.config.object_embedding_dim, self.config.object_embedding_dim),
            # nn.Sigmoid()
        )
        self.box_decoder = nn.Sequential(
            nn.Linear(self.config.object_embedding_dim, self.config.object_embedding_dim),
            nn.SiLU(),
            nn.Linear(self.config.object_embedding_dim, 4),
            nn.Sigmoid()
        )
        self.dino_projector = nn.Sequential(
            nn.Linear(self.dino_model.config.hidden_size, self.dino_model.config.hidden_size),
            nn.SiLU(),
            nn.Linear(self.dino_model.config.hidden_size, self.hidden_size)
        )
        self.clip_vision_projector = nn.Sequential(
            nn.Linear(self.clip_vison_hidden_size, self.clip_vison_hidden_size),
            nn.SiLU(),
            nn.Linear(self.clip_vison_hidden_size, self.hidden_size)
        )
        self.clip_text_projector = nn.Sequential(
            nn.Linear(self.clip_text_hidden_size, self.clip_text_hidden_size),
            nn.SiLU(),
            nn.Linear(self.clip_text_hidden_size, self.hidden_size)
        )
        self.loss_fn = nn.MSELoss()
        self.post_init()
        # self.load_decoder()
        
        
    
    @property
    def dtype(self):
        return self.text_projection.weight.data.dtype
    
    def forward(self, images, texts, boxes=None, ori_shapes=None):
        dino_input = self.dino_processor(images, return_tensors="pt").to(device=self.device, dtype=self.dtype)
        dino_embeddings = self.dino_model(**dino_input).feature_maps[-1]
        b, c = dino_embeddings.shape[:2]
        dino_embeddings = self.dino_projector(dino_embeddings.reshape(b, c, -1).permute(0, 2, 1))
        clip_input = self.clip_processor(text=texts, images=images, return_tensors="pt", padding=True)
        # clip_input_new = {}
        clip_input.pixel_values = clip_input.pixel_values.to(device=self.device, dtype=self.dtype)
        for key, value in clip_input.items():
            if key == 'pixel_values':
                clip_input[key] = value.to(device=self.device, dtype=self.dtype)
            else:
                clip_input[key] = value.to(device=self.device)
        clip_vision_embeddings = self.clip_vision_projector(self.vision_model(clip_input.pop('pixel_values')).last_hidden_state[:, 1:, :])
        clip_text_embeddings = self.clip_text_projector(self.text_model(**clip_input).last_hidden_state[:, 1:, :])
        
        vision_embeddings = torch.cat([clip_vision_embeddings, dino_embeddings], dim=1)
        task_embeddings = self.object_embedding[None].repeat(b, 1, 1)
        # print(vision_embeddings.shape)
        # print(clip_text_embeddings.shape)
        # print(task_embeddings.shape)
        return self.forward_fusuion_encoder(vision_embeddings, clip_text_embeddings, task_embeddings, boxes, ori_shapes)
        
        
        # print(vision_embeddings.shape)
        # return self.forward_text_transformer(**clip_input,
        #                                      boxes=boxes,
        #                                     #  boxes_embeddings=boxes_embeddings,
        #                                      ori_shapes=ori_shapes,
        #                                      vision_embeddings=vision_embeddings)
    
    def forward_fusuion_encoder(
        self,
        vision_embeddings,
        text_embeddings,
        task_embeddings,
        # attention_mask = None,
        boxes = None,
        # boxes_embeddings = None,
        ori_shapes = None,
        # return_dict = None,
    ):  
        for encoder in self.fusion_encoder:
            vision_embeddings, text_embeddings, task_embeddings = encoder(vision_embeddings, text_embeddings, task_embeddings)

        last_hidden_state = torch.cat([task_embeddings, vision_embeddings, text_embeddings],dim=1)
        object_embeddings = self.object_projector(task_embeddings[:, 0])
        pred_boxes = self.box_decoder(object_embeddings)
        if boxes is not None:
            boxes = boxes.to(device=self.device, dtype=self.dtype)
            boxes_embeddings = self.box_encoder(boxes)
            embedding_mse_loss = self.loss_fn(object_embeddings, boxes_embeddings)
            boxes_mse_loss = self.loss_fn(pred_boxes, boxes)
            reconstruction_loss = self.loss_fn(boxes, self.box_decoder(boxes_embeddings))
            iou, giou = generalized_box_iou(pred_boxes, boxes)
            iou = torch.diag(iou)
            giou = torch.diag(giou)
            giou_loss = (1 - giou).mean(0)
            iou = iou.mean(0)
            giou = giou.mean(0)
            # pred_boxes = self.boxes_decoder(object_embeddings.detach(), ori_shapes.to(self.device))
            loss = embedding_mse_loss + boxes_mse_loss + giou_loss + reconstruction_loss
        else:
            iou = None
            giou = None
            embedding_mse_loss = None
            boxes_mse_loss = None
            giou_loss = None
            reconstruction_loss = None
            loss = None
        
        if ori_shapes is not None:
            final_boxes = pred_boxes.clone()
            final_boxes[:, [0, 2]] *= ori_shapes[:, 1]
            final_boxes[:, [1, 3]] *= ori_shapes[:, 0]

        # if self.eos_token_id == 2:
        #     # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
        #     # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
        #     # ------------------------------------------------------------
        #     # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        #     # take features from the eot embedding (eot_token is the highest number in each sequence)
        #     # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        #     pooled_output = last_hidden_state[
        #         torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
        #         input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        #     ]
        # else:
        #     # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
        #     pooled_output = last_hidden_state[
        #         torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
        #         # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
        #         (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id)
        #         .int()
        #         .argmax(dim=-1),
        #     ]
        # if not return_dict:
        #     return (last_hidden_state, loss, iou, object_embeddings) + encoder_outputs[1:]

        return LIGAOutput(
            last_hidden_state=last_hidden_state,
            iou=iou,
            giou=giou,
            embedding_mse_loss=embedding_mse_loss,
            boxes_mse_loss=boxes_mse_loss,
            giou_loss=giou_loss,
            reconstruction_loss=reconstruction_loss,
            loss=loss,
            boxes=final_boxes,
            object_embeddings=object_embeddings,
        )
    
    # def forward_text_transformer(
    #     self,
    #     input_ids = None,
    #     vision_embeddings = None,
    #     attention_mask = None,
    #     position_ids = None,
    #     output_attentions = None,
    #     output_hidden_states = None,
    #     boxes = None,
    #     # boxes_embeddings = None,
    #     ori_shapes = None,
    #     return_dict = None,
    # ):
    #     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    #     output_hidden_states = (
    #         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    #     )
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    #     if input_ids is None:
    #         raise ValueError("You have to specify input_ids")

    #     input_shape = input_ids.shape
    #     # print(input_shape)
    #     b = input_shape[0]
    #     # input_shape[1] += 1
    #     vision_embeddings_num = vision_embeddings.shape[1]
    #     # input_ids = input_ids.view(-1, input_shape[-1])
    #     # print(input_ids)
        

    #     hidden_states = self.text_model.embeddings(input_ids=input_ids, position_ids=position_ids)
    #     # input_shape[1] += vision_embeddings_num
    #     # print(vision_embeddings.shape)
    #     # print(hidden_states.shape)
    #     # print(self.object_embedding[None].repeat(b, 1, 1).shape)
    #     hidden_states = torch.cat([vision_embeddings, hidden_states, self.object_embedding[None].repeat(b, 1, 1)], dim=1)

    #     # CLIP's text model uses causal mask, prepare it here.
    #     # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
    #     causal_attention_mask_text = _create_4d_causal_attention_mask(
    #         (input_shape[0], input_shape[1]), hidden_states.dtype, device=hidden_states.device
    #     )
    #     causal_attention_mask = torch.zeros((input_shape[0], 1, vision_embeddings_num+input_shape[1]+1, vision_embeddings_num+input_shape[1]+1), dtype=hidden_states.dtype, device=hidden_states.device)
    #     causal_attention_mask[:, :, :input_shape[1], :input_shape[1]] = causal_attention_mask_text
    #     # causal_attention_mask[:, :, :vision_embeddings_num, :vision_embeddings_num] = 1
    #     # expand attention_mask
    #     attention_mask = None
    #     # if attention_mask is not None:
            
    #     #     # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    #     #     attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
            
    #     encoder_outputs = self.text_model.encoder(
    #         inputs_embeds=hidden_states,
    #         attention_mask=attention_mask,
    #         causal_attention_mask=causal_attention_mask,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #     )

    #     last_hidden_state = encoder_outputs[0]
    #     last_hidden_state = self.text_model.final_layer_norm(last_hidden_state)
    #     object_embeddings = self.object_projector(last_hidden_state[:, -1])
        
    #     # if boxes_embeddings is not None:
    #     #     loss = self.loss_fn(object_embeddings, boxes_embeddings.reshape(b, -1).to(device=self.device, dtype=self.dtype))
        
    #     # else:
    #     #     loss = None
    #     pred_boxes = self.box_decoder(object_embeddings)
    #     if boxes is not None:
    #         boxes = boxes.to(device=self.device, dtype=self.dtype)
    #         boxes_embeddings = self.box_encoder(boxes)
    #         embedding_mse_loss = self.loss_fn(object_embeddings, boxes_embeddings)
    #         boxes_mse_loss = self.loss_fn(pred_boxes, boxes)
    #         iou, giou = calculate_giou(pred_boxes, boxes)
    #         giou_loss = (1 - giou).mean(0)
    #         iou = iou.mean(0)
    #         giou = giou.mean(0)
    #         pred_boxes = self.boxes_decoder(object_embeddings.detach(), ori_shapes.to(self.device))
    #         loss = embedding_mse_loss + boxes_mse_loss + giou_loss
    #     else:
    #         iou = None
    #         giou = None
    #         embedding_mse_loss = None
    #         boxes_mse_loss = None
    #         giou_loss = None
    #         loss = None
        
    #     if ori_shapes is not None:
    #         pred_boxes[:, [0, 2]] *= ori_shapes[:, 1]
    #         pred_boxes[:, [1, 3]] *= ori_shapes[:, 0]

    #     # if self.eos_token_id == 2:
    #     #     # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
    #     #     # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
    #     #     # ------------------------------------------------------------
    #     #     # text_embeds.shape = [batch_size, sequence_length, transformer.width]
    #     #     # take features from the eot embedding (eot_token is the highest number in each sequence)
    #     #     # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
    #     #     pooled_output = last_hidden_state[
    #     #         torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
    #     #         input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
    #     #     ]
    #     # else:
    #     #     # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
    #     #     pooled_output = last_hidden_state[
    #     #         torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
    #     #         # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
    #     #         (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id)
    #     #         .int()
    #     #         .argmax(dim=-1),
    #     #     ]
    #     if not return_dict:
    #         return (last_hidden_state, loss, iou, object_embeddings) + encoder_outputs[1:]

    #     return LIGAOutput(
    #         last_hidden_state=last_hidden_state,
    #         iou=iou,
    #         giou=giou,
    #         embedding_mse_loss=embedding_mse_loss,
    #         boxes_mse_loss=boxes_mse_loss,
    #         giou_loss=giou_loss,
    #         loss=loss,
    #         boxes=pred_boxes,
    #         object_embeddings=object_embeddings,
    #         hidden_states=encoder_outputs.hidden_states,
    #         attentions=encoder_outputs.attentions,
    #     )
    # @torch.no_grad()
    def boxes_decoder(self, boxes_embedding, ori_shapes):
        boxes = self.box_decoder(boxes_embedding.reshape(-1, self.config.object_embedding_dim))
        boxes[:, [0, 2]] *= ori_shapes[:, 1]
        boxes[:, [1, 3]] *= ori_shapes[:, 0]
        return boxes.to(dtype=torch.long)
    
    def set_sam(self):
        self.sam = sam_model_registry["vit_h"](checkpoint=self.config.sam_path)
        self.sam_predictor = SamPredictor(self.sam)
    
    def get_sam_box_embedding(self, boxes, ori_shapes):
        scale_h = ori_shapes[:, 0] / self.config.image_size
        scale_w = ori_shapes[:, 1] / self.config.image_size
        boxes[:, [0, 2]] *= scale_w
        boxes[:, [1, 3]] *= scale_h
        return self.sam_predictor.get_box_embedding(boxes)
    
    def load_decoder(self):
        decoder_ckpt = torch.load(self.config.decoder_path)
        self.box_decoder.load_state_dict(decoder_ckpt)
        

        
# if __name__ == '__main__':
    # print(_create_4d_causal_attention_mask((5, 5), torch.float32, 'cuda').shape)
    # print(torch.tensor([[100, 200, 300, 400]).reshape(1, 4).shape)
    # print(calculate_iou(torch.tensor([[100, 200, 300, 400],
    #                      [200, 300, 400, 500]]),
    #       torch.tensor([[110, 200, 300, 400],
    #                      [200, 300, 400, 500]])))