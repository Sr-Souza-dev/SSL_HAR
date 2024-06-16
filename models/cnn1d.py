import torch
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler 
from utils.enums import ModelTypes, Sets, Datas
from torchmetrics.functional import accuracy
from utils.checkpoints import saveBestModel, loadBestModel

input_linear_size = 236
models_path = "best_models/"

# Seletor de features (backbone)
class Backbone(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv1 = nn.Conv1d(in_channels=6,  out_channels=19, kernel_size=7)
        self.conv2 = nn.Conv1d(in_channels=19, out_channels=37, kernel_size=5)
        self.conv3 = nn.Conv1d(in_channels=37, out_channels=59, kernel_size=3)

    def require_grad(self, state = True):
        for param in self.conv1.parameters():
            param.requires_grad = state
        for param in self.conv2.parameters():
            param.requires_grad = state
        for param in self.conv3.parameters():
            param.requires_grad = state

    def forward(self, x):
        x = F.leaky_relu(input=self.conv1(x), negative_slope=0.01)
        x = self.pool(x)
        x = F.leaky_relu(input=self.conv2(x), negative_slope=0.01)
        x = self.pool(x)
        x = F.leaky_relu(input=self.conv3(x), negative_slope=0.01)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        return x

# Cabeça de predição para a tarefa de pretexto
class ProjectionHead(L.LightningModule):
    def __init__(self, num_classes=6):
        super().__init__()
        self.linear1 = nn.Linear(input_linear_size, num_classes)
    
    def forward(self, x):
        x = F.leaky_relu(input=self.linear1(x), negative_slope=0.01)
        return x

# Cabeça de predição para a tarefa de downstream
class PredictionHead(L.LightningModule):
    def __init__(self, num_classes=6):
        super().__init__()
        self.linear1 = nn.Linear(input_linear_size, num_classes)
        # self.linear1 = nn.Linear(input_linear_size, 40)
        # self.linear2 = nn.Linear(40, num_classes)
    
    def forward(self, x):
        x = F.leaky_relu(input=self.linear1(x), negative_slope=0.01)
        # x = F.leaky_relu(input=self.linear2(x), negative_slope=0.01)
        return x

# Monta um modelo completo a partir das entradas
class CNN1d(L.LightningModule):
    def __init__(
        self, 
        type = ModelTypes.PRETEXT.value, 
        data_label = Datas.MOBIT.value,
        require_grad = True,
        num_classes=6,
        ref = './'

    ):
        super().__init__()
        self.num_classes = num_classes
        self.data_label = data_label
        self.type_task = type
        self.ref = ref
        self.criterion = nn.CrossEntropyLoss()

        if type == ModelTypes.PRETEXT.value:
            self.backbone = Backbone()
            self.pred_head = ProjectionHead(self.num_classes)

        elif type == ModelTypes.DOWNSTREAM.value:
            self.backbone = Backbone()
            self.pred_head = PredictionHead(self.num_classes)
            self.load_backbone(require_grad=require_grad)
        
        else:
            print("Opção invalida! Não foi possível gerar um modelo.")
            return
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.pred_head(x)
        return x
    
    def _common_step(self, batch: torch.Tensor, set:Sets):
        x, y = batch
        pred = self.forward(x)
        loss = self.criterion(pred, y)
    
        class_pred = torch.argmax(pred, dim=1)
        accurary = accuracy(class_pred, y, num_classes=self.num_classes, task='multiclass')

        self.log(f"{set}_loss", loss, on_epoch=True, prog_bar=True)
        self.log(f"{set}_acc", accurary, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch: torch.Tensor):
        return self._common_step(batch=batch, set=Sets.TRAIN.value)
    def test_step(self, batch: torch.Tensor):
        return self._common_step(batch=batch, set=Sets.TEST.value)
    def validation_step(self, batch: torch.Tensor):
        return self._common_step(batch=batch, set=Sets.VALIDATION.value)
    
    def require_grads_backbone (self, state=True):
        self.backbone.require_grad(state=state)

    def load_backbone(self, device='cpu', require_grad = True):
        try:
            self.backbone.load_state_dict(
                loadBestModel(
                    device=device,
                    path=f"{self.ref}{models_path}", 
                    file_name=f"backbone_{self.data_label}"
                )
            )
            # self.backbone = loadBestModel(
            #     device=device,
            #     path=f"{self.ref}{models_path}", 
            #     file_name=f"backbone_{self.data_label}"
            # )
            self.backbone.require_grad(require_grad)
            self.backbone.requires_grad_(require_grad)
        except:
            print("Erro ao carregar modelo de backbone! (Modelo padrão gerado)")
    
    def save_backbone(self, num_epoch, accuracy, batch_size):
        saveBestModel(
            accuracy=accuracy, 
            batch_size=batch_size, 
            epoch=num_epoch, 
            model=self.backbone.state_dict(), 
            path=f"{self.ref}{models_path}", 
            file_name=f"backbone_{self.data_label}"
        )

    def save_full_model(self, num_epoch, accuracy, batch_size):
        saveBestModel(
            accuracy=accuracy, 
            batch_size=batch_size, 
            epoch=num_epoch, 
            model=self.backbone.state_dict(), 
            path=f"{self.ref}{models_path}", 
            file_name=f"model_{self.data_label}_{self.type_task}"
        )

    def configure_optimizers(self, step_size=100, gamma=0.5, learning_rate=0.1) -> tuple[torch.optim.Optimizer, lr_scheduler.LRScheduler]:
        params = list(self.backbone.parameters()) + list(self.pred_head.parameters())
        optimizer = torch.optim.SGD(params, lr=learning_rate)
        step_lr_scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)  
        return [optimizer], [step_lr_scheduler]
    
    def configure_head_optimizers(self, step_size=100, gamma=0.5, learning_rate=0.1) -> tuple[torch.optim.Optimizer, lr_scheduler.LRScheduler]:
        optimizer = torch.optim.SGD(list(self.pred_head.parameters()), lr=learning_rate)
        step_lr_scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)  
        return optimizer, step_lr_scheduler
    
    def configure_backbone_optimizers(self, step_size=100, gamma=0.5, learning_rate=0.1) -> tuple[torch.optim.Optimizer, lr_scheduler.LRScheduler]:
        optimizer = torch.optim.SGD(self.backbone.parameters(), lr=learning_rate)
        step_lr_scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)  
        return optimizer, step_lr_scheduler