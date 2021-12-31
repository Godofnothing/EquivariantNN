import torch

from torch import nn


def train(model : nn.Module, dataloaders, criterion, optimizer, num_epochs: int, scheduler = None, device='cpu'):
    history = {
        "train" : {"loss" : [], "acc" : []},
        "val"   : {"loss" : [], "acc" : []}
    }
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)
        
        for phase in ['train', 'val']:
            with torch.set_grad_enabled(phase == 'train'):
                
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                    
                running_loss = 0
                running_correct = 0
                
                for images, labels in dataloaders[phase]:
                    images, labels = images.to(device), labels.to(device)
                    # get model output
                    logits = model(images)
                    # compute loss
                    loss = criterion(logits, labels)
                    
                    if phase == 'train':
                        # make gradient step  
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                    # get predictions from logits
                    preds = torch.argmax(logits, dim=1)          
                    # statistics
                    running_loss += loss.item() * images.size(0)
                    running_correct += torch.sum(preds == labels).item()
                    
                total_samples = len(dataloaders[phase].dataset)
                    
                epoch_loss = running_loss / total_samples
                epoch_acc  = running_correct / total_samples
                
            print(f"{phase:>5} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            history[phase]["loss"].append(epoch_loss)
            history[phase]["acc"].append(epoch_acc)
            
            if scheduler:
                scheduler.step()
                
    return history