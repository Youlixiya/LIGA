from typing import Optional, Tuple, Union
from PIL.Image import Image
from dataclasses import dataclass
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
try:
    import flash_attn
    # use_flash_attn2 = True
    ATTENTION_MODULE=LIGAFlashAttention2
except:
    ATTENTION_MODULE=LIGASdpaAttention
@dataclass
class LIGAOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    loss: torch.FloatTensor = None
    iou: torch.FloatTensor = None
    giou: torch.FloatTensor = None
    boxes: torch.FloatTensor = None
    embedding_mse_loss: torch.FloatTensor = None
    boxes_mse_loss: torch.FloatTensor = None
    giou_loss: torch.FloatTensor = None
    object_embeddings: torch.FloatTensor = None

# class CLIPPreTrainedModel(PreTrainedModel):
#     """
#     An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
#     models.
#     """

#     config_class = CLIPConfig
#     base_model_prefix = "clip"
#     supports_gradient_checkpointing = True

#     def _init_weights(self, module):
#         """Initialize the weights"""
#         factor = self.config.initializer_factor
#         if isinstance(module, CLIPTextEmbeddings):
#             module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
#             module.position_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
#         elif isinstance(module, CLIPVisionEmbeddings):
#             factor = self.config.initializer_factor
#             nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
#             nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
#             nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
#         elif isinstance(module, CLIPAttention):
#             factor = self.config.initializer_factor
#             in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
#             out_proj_std = (module.embed_dim**-0.5) * factor
#             nn.init.normal_(module.q_proj.weight, std=in_proj_std)
#             nn.init.normal_(module.k_proj.weight, std=in_proj_std)
#             nn.init.normal_(module.v_proj.weight, std=in_proj_std)
#             nn.init.normal_(module.out_proj.weight, std=out_proj_std)
#         elif isinstance(module, CLIPMLP):
#             factor = self.config.initializer_factor
#             in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
#             fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
#         nn.init.normal_(module.fc1.weight, std=fc_std)
#             nn.init.normal_(module.fc2.weight, std=in_proj_std)
#         elif isinstance(module, CLIPModel):
#             nn.init.normal_(
#                 module.text_projection.weight,
#                 std=module.text_embed_dim**-0.5 * self.config.initializer_factor,
#             )
#             nn.init.normal_(
#                 module.visual_projection.weight,
#                 std=module.vision_embed_dim**-0.5 * self.config.initializer_factor,
#             )
#         elif isinstance(module, CLIPVisionModelWithProjection):
#             nn.init.normal_(
#                 module.visual_projection.weight,
#                 std=self.config.hidden_size**-0.5 * self.config.initializer_factor,
#             )
#         elif isinstance(module, CLIPTextModelWithProjection):
#             nn.init.normal_(
#                 module.text_projection.weight,
#                 std=self.config.hidden_size**-0.5 * self.config.initializer_factor,
#             )

#         if isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#         if isinstance(module, nn.Linear) and module.bias is not None:
#             module.bias.data.zero_()
# class LIGAMetaModel:
    
#     def __init__(self, config):
#         super(LIGAMetaModel, self).__init__(config)
        
        # self.config = config
        # if not hasattr(config, 'dino_model'):
        #     self.config
#         pass

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

def calculate_iou(boxes1, boxes2):
    intersection_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    intersection_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    intersection_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    intersection_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

    # 计算相交区域的面积
    intersection_area = torch.clamp(intersection_x2 - intersection_x1 + 1, min=0) * torch.clamp(intersection_y2 - intersection_y1 + 1, min=0)

    # 计算两组边界框的面积
    boxes1_area = (boxes1[:, 2] - boxes1[:, 0] + 1) * (boxes1[:, 3] - boxes1[:, 1] + 1)
    boxes2_area = (boxes2[:, 2] - boxes2[:, 0] + 1) * (boxes2[:, 3] - boxes2[:, 1] + 1)

    # 计算并集区域的面积
    union_area = boxes1_area + boxes2_area - intersection_area

    # 计算 IoU
    iou = intersection_area / union_area

    return iou

def calculate_giou(boxes1, boxes2):

    # iou = calculate_iou(boxes1, boxes2)
    
    intersection_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    intersection_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    intersection_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    intersection_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

    intersection_area = torch.clamp(intersection_x2 - intersection_x1 + 1, min=0) * torch.clamp(intersection_y2 - intersection_y1 + 1, min=0)

    boxes1_area = (boxes1[:, 2] - boxes1[:, 0] + 1) * (boxes1[:, 3] - boxes1[:, 1] + 1)
    boxes2_area = (boxes2[:, 2] - boxes2[:, 0] + 1) * (boxes2[:, 3] - boxes2[:, 1] + 1)

    union_area = boxes1_area + boxes2_area - intersection_area

    iou = intersection_area / union_area

    # 计算最小包围框的面积
    convex_hull_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    convex_hull_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    convex_hull_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    convex_hull_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
    convex_hull_area = (convex_hull_x2 - convex_hull_x1 + 1) * (convex_hull_y2 - convex_hull_y1 + 1)

    # 计算并集区域的面积
    union_area = boxes1_area + boxes2_area - iou * (convex_hull_area - union_area)

    # 计算 GIoU
    giou = iou - (convex_hull_area - union_area) / convex_hull_area

    return iou, giou


class LIGAEncoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int=None):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = ATTENTION_MODULE(config=config, layer_idx=layer_idx)

        self.mlp = LIGAMLP(config)
        self.input_layernorm = LIGARMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LIGARMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class NextChatConfig(CLIPConfig):
    model_type = "liga"
    dino_model = 'facebook/dinov2-base'
    clip_model = "openai/clip-vit-base-patch14"
    object_embedding_dim = 512
    image_size = 1024
    sam_path = 'checkpoints/sam/sam_vit_h_4b8939.pth'
    decoder_path = 'checkpoints/decoder.pt'
    fusion_encoder_layers = 4

class LIGAModel(CLIPModel):
    
    config_class = NextChatConfig
    
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        if not hasattr(config, 'dino_model'):
            self.config.dino_model = kwargs['dino_model']
            self.config.clip_model = kwargs['clip_model']
            self.config.object_embedding_dim = kwargs['object_embedding_dim']
            self.config.clip_model = kwargs['clip_model']
            self.config.image_size = kwargs['image_size']
            self.config.sam_path = kwargs['sam_path']
            self.config.decoder_path = kwargs['decoder_path']
        self.clip_vison_hidden_size = config.vision_config.hidden_size
        self.clip_text_hidden_size = config.text_config.hidden_size
        self.object_embedding = nn.Parameter(torch.randn(1, self.clip_text_hidden_size).requires_grad_(True))
        self.clip_processor = CLIPProcessor.from_pretrained(self.config.clip_model)
        # self.clip_model = CLIPModel.from_pretrained(config.clip_model)
        self.dino_processor = AutoImageProcessor.from_pretrained(self.config.dino_model)
        self.dino_model = AutoBackbone.from_pretrained(self.config.dino_model, out_features=["stage2", "stage5", "stage8", "stage11"])
        # self.encoder_layers = nn.ModuleList(
        #     [LIGAEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        # )
        self.object_projector = nn.Sequential(
            nn.Linear(self.clip_text_hidden_size, self.clip_text_hidden_size),
            nn.SiLU(inplace=True),
            nn.Linear(self.clip_text_hidden_size, self.config.object_embedding_dim)
        )
        self.box_encoder = nn.Sequential(
            nn.Linear(4, self.config.object_embedding_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.config.object_embedding_dim, self.config.object_embedding_dim),
            # nn.Sigmoid()
        )
        self.box_decoder = nn.Sequential(
            nn.Linear(self.config.object_embedding_dim, self.config.object_embedding_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.config.object_embedding_dim, 4),
            nn.Sigmoid()
        )
        self.dino_projector = nn.Sequential(
            nn.Linear(self.dino_model.config.hidden_size, self.dino_model.config.hidden_size),
            nn.SiLU(inplace=True),
            nn.Linear(self.dino_model.config.hidden_size, self.clip_text_hidden_size)
        )
        self.clip_projector = nn.Sequential(
            nn.Linear(self.clip_vison_hidden_size, self.clip_vison_hidden_size),
            nn.SiLU(inplace=True),
            nn.Linear(self.clip_vison_hidden_size, self.clip_text_hidden_size)
        )
        self.loss_fn = nn.MSELoss()
        # self.load_decoder()
        self.post_init()
    
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
        clip_embeddings = self.clip_projector(self.vision_model(clip_input.pop('pixel_values')).last_hidden_state[:, 1:, :])
        vision_embeddings = torch.cat([clip_embeddings, dino_embeddings], dim=1)
        # print(vision_embeddings.shape)
        return self.forward_text_transformer(**clip_input,
                                             boxes=boxes,
                                            #  boxes_embeddings=boxes_embeddings,
                                             ori_shapes=ori_shapes,
                                             vision_embeddings=vision_embeddings)
    
    def forward_text_transformer(
        self,
        input_ids = None,
        vision_embeddings = None,
        attention_mask = None,
        position_ids = None,
        output_attentions = None,
        output_hidden_states = None,
        boxes = None,
        # boxes_embeddings = None,
        ori_shapes = None,
        return_dict = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.shape
        # print(input_shape)
        b = input_shape[0]
        # input_shape[1] += 1
        vision_embeddings_num = vision_embeddings.shape[1]
        # input_ids = input_ids.view(-1, input_shape[-1])
        # print(input_ids)
        

        hidden_states = self.text_model.embeddings(input_ids=input_ids, position_ids=position_ids)
        # input_shape[1] += vision_embeddings_num
        # print(vision_embeddings.shape)
        # print(hidden_states.shape)
        # print(self.object_embedding[None].repeat(b, 1, 1).shape)
        hidden_states = torch.cat([vision_embeddings, hidden_states, self.object_embedding[None].repeat(b, 1, 1)], dim=1)

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask_text = _create_4d_causal_attention_mask(
            (input_shape[0], input_shape[1]), hidden_states.dtype, device=hidden_states.device
        )
        causal_attention_mask = torch.zeros((input_shape[0], 1, vision_embeddings_num+input_shape[1]+1, vision_embeddings_num+input_shape[1]+1), dtype=hidden_states.dtype, device=hidden_states.device)
        causal_attention_mask[:, :, :input_shape[1], :input_shape[1]] = causal_attention_mask_text
        # causal_attention_mask[:, :, :vision_embeddings_num, :vision_embeddings_num] = 1
        # expand attention_mask
        attention_mask = None
        # if attention_mask is not None:
            
        #     # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        #     attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
            
        encoder_outputs = self.text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.text_model.final_layer_norm(last_hidden_state)
        object_embeddings = self.object_projector(last_hidden_state[:, -1])
        
        # if boxes_embeddings is not None:
        #     loss = self.loss_fn(object_embeddings, boxes_embeddings.reshape(b, -1).to(device=self.device, dtype=self.dtype))
        
        # else:
        #     loss = None
        pred_boxes = self.box_decoder(object_embeddings)
        if boxes is not None:
            boxes = boxes.to(device=self.device, dtype=self.dtype)
            boxes_embeddings = self.box_encoder(boxes)
            embedding_mse_loss = self.loss_fn(object_embeddings, boxes_embeddings)
            boxes_mse_loss = self.loss_fn(pred_boxes, boxes)
            iou, giou = calculate_giou(pred_boxes, boxes)
            giou_loss = (1 - giou).mean(0)
            iou = iou.mean(0)
            giou = giou.mean(0)
            pred_boxes = self.boxes_decoder(object_embeddings.detach(), ori_shapes.to(self.device))
            loss = embedding_mse_loss + boxes_mse_loss + giou_loss
        else:
            iou = None
            giou = None
            embedding_mse_loss = None
            boxes_mse_loss = None
            giou_loss = None
            loss = None
        
        if ori_shapes is not None:
            pred_boxes[:, [0, 2]] *= ori_shapes[:, 1]
            pred_boxes[:, [1, 3]] *= ori_shapes[:, 0]

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
        if not return_dict:
            return (last_hidden_state, loss, iou, object_embeddings) + encoder_outputs[1:]

        return LIGAOutput(
            last_hidden_state=last_hidden_state,
            iou=iou,
            giou=giou,
            embedding_mse_loss=embedding_mse_loss,
            boxes_mse_loss=boxes_mse_loss,
            giou_loss=giou_loss,
            loss=loss,
            boxes=pred_boxes,
            object_embeddings=object_embeddings,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
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