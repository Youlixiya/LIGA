import torchkeras
from accelerate import Accelerator

class StepRunner:
    def __init__(self, net, loss_fn=None, accelerator=None, stage = "train", metrics_dict = None, 
                 optimizer = None, lr_scheduler = None
                 ):
        self.net,self.loss_fn,self.metrics_dict,self.stage = net,loss_fn,metrics_dict,stage
        self.optimizer,self.lr_scheduler = optimizer,lr_scheduler
        self.accelerator = accelerator if accelerator is not None else Accelerator()
        if self.stage=='train':
            self.net.train() 
        else:
            self.net.eval()
    
    def __call__(self, batch):
        
        
        #loss
        preds = self.net(**batch)
        # preds_log_softmax = F.log_softmax(preds, dim=-1)
        embedding_mse_loss = preds.embedding_mse_loss
        boxes_mse_loss = preds.boxes_mse_loss
        giou_loss = preds.giou_loss
        reconstruction_loss = preds.reconstruction_loss
        loss = preds.loss
        iou = preds.iou
        giou = preds.giou
            

        #backward()
        if self.optimizer is not None and self.stage=="train":
            self.accelerator.backward(loss)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
        all_embedding_mse_loss = self.accelerator.gather(embedding_mse_loss).mean()
        all_boxes_mse_loss = self.accelerator.gather(boxes_mse_loss).mean()
        all_giou_loss = self.accelerator.gather(giou_loss).mean()
        all_reconstruction_loss = self.accelerator.gather(reconstruction_loss).mean()
        all_loss = self.accelerator.gather(loss).mean()
        all_iou = self.accelerator.gather(iou).mean()
        all_giou = self.accelerator.gather(giou).mean()
        
        #losses （or plain metric that can be averaged）
        step_losses = {self.stage+"_embedding_mse_loss":all_embedding_mse_loss.item(),
                       self.stage+"_boxes_mse_loss":all_boxes_mse_loss.item(),
                       self.stage+"_giou_loss":all_giou_loss.item(),
                       self.stage+"_all_reconstruction_loss":all_reconstruction_loss.item(),
                       self.stage+"_loss":all_loss.item()}
        
        #metrics (stateful metric)
        step_metrics = {self.stage+'_iou':all_iou.item(),
                        self.stage+'_giou':all_giou.item()}
        if self.stage=="train":
            if self.optimizer is not None:
                step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            else:
                step_metrics['lr'] = 0.0
        return step_losses,step_metrics
torchkeras.KerasModel.StepRunner = StepRunner
