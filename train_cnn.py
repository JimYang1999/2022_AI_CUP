#TEAM_1970

from locale import normalize
import torch
from torch import nn
import torchvision
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor , InterpolationMode
import matplotlib.pyplot as plt
import os
from pathlib import Path
import random
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
from timeit import default_timer as timer

device = "cuda" if torch.cuda.is_available() else "cpu"
save_name = 'model'
image_path = Path("./AI_CUP/")
train_dir = image_path / "train"
test_dir = image_path / "test"

BATCH_SIZE = 16
NUM_WORKERS = os.cpu_count()
NUM_CLASS=33
NUM_EPOCHS = 200
LEARNINGRATE=0.01
model = torchvision.models.efficientnet_b7(pretrained=True , progress=True)
#model = torch.load('./model.pth').to(device) #若要繼續訓練，可使用這一行程式碼
model.classifier[-1]= nn.Linear(model.classifier[-1].in_features,NUM_CLASS)
model.to(device)
random.seed(42)
torch.manual_seed(42) 
torch.cuda.manual_seed(42)

mean = [0.485,0.456,0.406]
std = [0.229,0.224,0.225]
size = (320,320)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size),
    transforms.TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.RandomErasing(p=0.1)
])
test_transform = transforms.Compose([
    transforms.Resize(size=(320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean,std)])

train_data = datasets.ImageFolder(root=train_dir,
                                  transform=train_transform,
                                  target_transform=None)

test_data = datasets.ImageFolder(root=test_dir, 
                                 transform=test_transform,
                                 target_transform=None)

train_dataloader = DataLoader(dataset=train_data, 
                              batch_size=BATCH_SIZE, 
                              num_workers=NUM_WORKERS, 
                              shuffle=True,
                              pin_memory=True)

test_dataloader = DataLoader(dataset=test_data, 
                             batch_size=BATCH_SIZE, 
                             num_workers=NUM_WORKERS, 
                             shuffle=False,
                             pin_memory=True)

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer):
    model.train()
    train_loss, train_acc = 0, 0
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module):
    model.eval()
    test_loss, test_acc = 0, 0
    test_pred_list=[]
    ans_list = []
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            test_pred_list.append(test_pred_logits)
            ans_list.append(y)
        ans_list, test_pred_list = torch.cat(ans_list) , torch.cat(test_pred_list)
        pred_class = torch.nn.functional.softmax(test_pred_list, dim=-1)
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    scheduler.step(test_loss)
    return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):

    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }
    max_acc=0
    for epoch in tqdm(range(epochs)):   
        train_loss, train_acc = train_step(model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn)
        if epoch %20==0 and epoch >20:
            torch.save(model, f'./{save_name}/{save_name}_{epoch+1}.pth')
            print(f'torch.save : {epoch} , abs(train_loss-test_loss) = {abs(train_loss-test_loss)}')
        if test_acc > max_acc :
            print(f'org max_acc {max_acc} --> {test_acc} in epochs {epoch+1}')
            max_acc = test_acc
            torch.save(model , f'./{save_name}/{save_name}_max_acc.pth')
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f} | "
            f"lr : {optimizer.state_dict()['param_groups'][0]['lr']}"
        )
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    return results

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNINGRATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
start_time = timer()
history = train(model=model, 
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn, 
                        epochs=NUM_EPOCHS)
end_time = timer()

print(f"Total training time: {end_time-start_time:.3f} seconds")

history.keys()
torch.save(model, f'./{save_name}/{save_name}.pth')