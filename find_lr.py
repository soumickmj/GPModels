"""# Function to find the optimum learning rate
"""

"""
Ref: https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html Posted on Tue 20 March 2018 in Basics
"""

import math
import torch

def find_lr(train_gen, optimizer, model, criterion, device, batch, aug, num_class, init_value = 1e-8, final_value=10., beta = 0.98):    #, beta = 0.98):
    num = len(train_gen)-1
    print("no of epochs",num)
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    for data in train_gen:
        batch_num += 1
        #As before, get the loss for this mini-batch of inputs/outputs
        labels, inputs = data
        labels, inputs = labels, inputs 
        # set label as cuda if device is cuda
        labels, inputs  = labels.to(device), inputs.to(device)
        optimizer.zero_grad()
        outputs = model(inputs.view(-1, 1, 512, 512))
        loss = criterion(outputs.float(), torch.argmax(labels.view(batch*aug, num_class), dim=1).long())
        #Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) *loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        #Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        #Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        #Do the SGD step
        loss.backward()
        optimizer.step()
        #Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    return log_lrs, losses