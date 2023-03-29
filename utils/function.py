import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
from globals_mod import settings
from torchvision.utils import save_image
import sys
import random
import logging

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
flat_shape = [784]
cond_shape=10
logging.basicConfig(filename="log_traces/logfilename.log", level=logging.INFO)
# Hard coding values for testing purpose
flat_shape = [784]
cond_shape=10

torch.manual_seed(0)
label_flipping_indices = [[7,5], [4,2]]
# label_flipping_indices = [[7,5]]

def train(model, train_dataloader, config, label_flipping=False, train_cvae=True, device=DEVICE, args=None):
    """Train the network on the training set."""
    
    if label_flipping:
        log_label_flipping(label_flipping_indices)

    if train_cvae:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
    else:
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
        model.classifier.train()

    for epoch in range(config["local_epochs"]):
        train_loss = 0
        classif_accuracy = 0
        for batch, (images, labels) in enumerate(train_dataloader):
            images = images.to(device) #[64, 1, 28, 28]
            labels = labels.to(device)
            
            if label_flipping:
                labels = label_flip(labels, label_flipping_indices)

            if train_cvae:
                # 1. Forward pass
                mu, logvar, recon_batch, c_out = model((images, labels))
                flat_data = images.view(-1, flat_shape[0]).to(device)                            
                y_onehot = F.one_hot(labels, cond_shape).to(device)
                inp = torch.cat((flat_data, y_onehot), 1)

                # 2. Calculate loss
                loss, C_loss, BCE, KLD = loss_fn(recon_batch, inp, mu, logvar, c_out, y_onehot)
                train_loss += loss.item()
                classif_accuracy += accuracy_fn(labels, torch.argmax(c_out, dim=1))
            else:
                c_out = model(images)                        
                y_onehot = F.one_hot(labels, cond_shape).to(device)


            # 3. Zero grad
            optimizer.zero_grad()

            # 4. Backprop
            loss.backward()

            # 5. Step
            optimizer.step()

            if batch % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tBCE:{:.4f}\tKLD:{:.4f}\tC_loss:{:.4f}'.format(
                    epoch,
                    batch * len(images),
                    len(train_dataloader.dataset),
                    100. * batch / len(train_dataloader),
                    loss.item() / len(images), BCE.item() / len(images), KLD.item() / len(images), C_loss.item() / len(images)))
                logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tBCE:{:.4f}\tKLD:{:.4f}\tC_loss:{:.4f}'.format(
                    epoch,
                    batch * len(images),
                    len(train_dataloader.dataset),
                    100. * batch / len(train_dataloader),
                    loss.item() / len(images), BCE.item() / len(images), KLD.item() / len(images), C_loss.item() / len(images)))
        print('====> Epoch: {} Average loss: {:.4f}\tClassifier Accuracy: {:.4f}'.format(
            epoch, train_loss / len(train_dataloader.dataset), classif_accuracy/len(train_dataloader)))
            
        logging.info('====> Epoch: {} Average loss: {:.4f}\tClassifier Accuracy: {:.4f}'.format(
            epoch, train_loss / len(train_dataloader.dataset), classif_accuracy/len(train_dataloader)))

    if config["log_img"]:
        generate_and_save_images(model)


def train_standard_classifier(model, train_dataloader, config, label_flipping=False, device=DEVICE, args=None):
    """Train the network on the training set."""
    
    if label_flipping:
        log_label_flipping(label_flipping_indices)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(config["local_epochs"]):
        train_loss = 0
        classif_accuracy = 0
        for batch, (images, labels) in enumerate(train_dataloader):
            images = images.to(device) #[64, 1, 28, 28]
            labels = labels.to(device)
            
            if label_flipping:
                labels = label_flip(labels, label_flipping_indices)

            # 1. Forward pass
            c_out = model(images)                        
            y_onehot = F.one_hot(labels, cond_shape).to(device)

            # 2. Calculate loss
            loss = loss_fn_standard_classifier(c_out, y_onehot)
            train_loss += loss.item()
            classif_accuracy += accuracy_fn(labels, torch.argmax(c_out, dim=1))

            # 3. Zero grad
            optimizer.zero_grad()

            # 4. Backprop
            loss.backward()

            # 5. Step
            optimizer.step()

            if batch % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch * len(images),
                    len(train_dataloader.dataset),
                    100. * batch / len(train_dataloader),
                    loss.item() / len(images)))
                logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch * len(images),
                    len(train_dataloader.dataset),
                    100. * batch / len(train_dataloader),
                    loss.item() / len(images)))
        print('====> Epoch: {} Average loss: {:.4f}\tClassifier Accuracy: {:.4f}'.format(
            epoch, train_loss / len(train_dataloader.dataset), classif_accuracy/len(train_dataloader)))
            
        logging.info('====> Epoch: {} Average loss: {:.4f}\tClassifier Accuracy: {:.4f}'.format(
            epoch, train_loss / len(train_dataloader.dataset), classif_accuracy/len(train_dataloader)))


def train_regression(model, train_dataloader, config, label_flipping=False, device=DEVICE, args=None):
    """Train the network on the training set."""
    
    if label_flipping:
        log_label_flipping(label_flipping_indices)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_func = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(config["local_epochs"]):
        train_loss = 0
        classif_accuracy = 0
        for batch, (images, labels) in enumerate(train_dataloader):
            images = images.to(device) #[64, 1, 28, 28]
            labels = labels.to(device)
            
            if label_flipping:
                labels = label_flip(labels, label_flipping_indices)

            # 1. Forward pass
            log_probs = model((images, labels))
            
            # 2. Calculate loss
            loss = loss_func(log_probs, labels)
            train_loss += loss.item()
            classif_accuracy += accuracy_fn(labels, torch.argmax(log_probs, dim=1))

            # 3. Zero grad
            optimizer.zero_grad()

            # 4. Backprop
            loss.backward()

            # 5. Step
            optimizer.step()

            if batch % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch * len(images),
                    len(train_dataloader.dataset),
                    100. * batch / len(train_dataloader),
                    loss.item() / len(images)))
                logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch * len(images),
                    len(train_dataloader.dataset),
                    100. * batch / len(train_dataloader),
                    loss.item() / len(images)))
        print('====> Epoch: {} Average loss: {:.4f}\tClassifier Accuracy: {:.4f}'.format(
            epoch, train_loss / len(train_dataloader.dataset), classif_accuracy/len(train_dataloader)))
            
        logging.info('====> Epoch: {} Average loss: {:.4f}\tClassifier Accuracy: {:.4f}'.format(
            epoch, train_loss / len(train_dataloader.dataset), classif_accuracy/len(train_dataloader)))


def train_cvae_regression(model, train_dataloader, config, label_flipping=False, device=DEVICE, args=None):
    """Train the network on the training set."""
    
    if label_flipping:
        log_label_flipping(label_flipping_indices)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_func = nn.CrossEntropyLoss

    model.train()
    for epoch in range(config["local_epochs"]):
        train_loss = 0
        classif_accuracy = 0
        for batch, (images, labels) in enumerate(train_dataloader):
            images = images.to(device) #[64, 1, 28, 28]
            labels = labels.to(device)
            
            if label_flipping:
                labels = label_flip(labels, label_flipping_indices)

            # 1. Forward pass
            mu, logvar, recon_batch, c_out = model((images, labels))
            flat_data = images.view(-1, flat_shape[0]).to(device)                            
            y_onehot = F.one_hot(labels, cond_shape).to(device)
            inp = torch.cat((flat_data, y_onehot), 1)

            # 2. Calculate loss
            loss, C_loss, BCE, KLD = loss_fn(recon_batch, flat_data, mu, logvar, c_out, y_onehot, loss=loss_func)
            train_loss += loss.item()
            classif_accuracy += accuracy_fn(labels, torch.argmax(c_out, dim=1))

            # 3. Zero grad
            optimizer.zero_grad()

            # 4. Backprop
            loss.backward()

            # 5. Step
            optimizer.step()

            if batch % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tBCE:{:.4f}\tKLD:{:.4f}\tC_loss:{:.4f}'.format(
                    epoch,
                    batch * len(images),
                    len(train_dataloader.dataset),
                    100. * batch / len(train_dataloader),
                    loss.item() / len(images), BCE.item() / len(images), KLD.item() / len(images), C_loss.item() / len(images)))
                logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tBCE:{:.4f}\tKLD:{:.4f}\tC_loss:{:.4f}'.format(
                    epoch,
                    batch * len(images),
                    len(train_dataloader.dataset),
                    100. * batch / len(train_dataloader),
                    loss.item() / len(images), BCE.item() / len(images), KLD.item() / len(images), C_loss.item() / len(images)))
        print('====> Epoch: {} Average loss: {:.4f}\tClassifier Accuracy: {:.4f}'.format(
            epoch, train_loss / len(train_dataloader.dataset), classif_accuracy/len(train_dataloader)))
            
        logging.info('====> Epoch: {} Average loss: {:.4f}\tClassifier Accuracy: {:.4f}'.format(
            epoch, train_loss / len(train_dataloader.dataset), classif_accuracy/len(train_dataloader)))

    if config["log_img"]:
        generate_and_save_images(model)


def test(model, test_dataloader, device=DEVICE):
    #Sets the module in evaluation mode
    model.eval()
    test_loss = 0
    c_test_loss = 0
    classif_accuracy = 0
    with torch.inference_mode():
        for i, (X, y) in enumerate(test_dataloader):
            X = X.to(device)
            y = y.to(device)
            # 1. Forward pass
            mu, logvar, recon_batch, c_out = model((X, y))

            flat_data = X.view(-1, flat_shape[0]).to(device)
            y_onehot = F.one_hot(y, cond_shape).to(device)
            inp = torch.cat((flat_data, y_onehot), 1)

            # 2. Loss
            tot_loss, C_loss, BCE, KLD = loss_fn(recon_batch, inp, mu, logvar, c_out, y_onehot)
            test_loss += tot_loss.item()
            c_test_loss += C_loss.item()
            classif_accuracy += accuracy_fn(y, torch.argmax(c_out, dim=1))


    test_loss /= len(test_dataloader.dataset)
    c_test_loss /= len(test_dataloader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss, c_test_loss, classif_accuracy/len(test_dataloader)


def test_standard_classifier(model, test_dataloader, device=DEVICE):
    #Sets the module in evaluation mode
    model.eval()
    test_loss = 0
    classif_accuracy = 0
    with torch.inference_mode():
        for i, (X, y) in enumerate(test_dataloader):
            X = X.to(device)
            y = y.to(device)
            # 1. Forward pass
            c_out = model(X)

            flat_data = X.view(-1, flat_shape[0]).to(device)
            y_onehot = F.one_hot(y, cond_shape).to(device)

            # 2. Loss
            loss = loss_fn_standard_classifier(c_out, y_onehot)
            test_loss += loss.item()
            classif_accuracy += accuracy_fn(y, torch.argmax(c_out, dim=1))


    test_loss /= len(test_dataloader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss, classif_accuracy/len(test_dataloader)


def test_regression(model, test_dataloader, device=DEVICE):
    #Sets the module in evaluation mode
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    test_loss = 0
    classif_accuracy = 0
    with torch.inference_mode():
        for i, (X, y) in enumerate(test_dataloader):
            X = X.to(device)
            y = y.to(device)
            # 1. Forward pass
            log_probs = model((X,y))

            # 2. Loss
            loss = loss_fn(log_probs, y)
            test_loss += loss.item()
            classif_accuracy += accuracy_fn(y, torch.argmax(log_probs, dim=1))


    test_loss /= len(test_dataloader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss, classif_accuracy/len(test_dataloader)


def test_cvae_regression(model, test_dataloader, device=DEVICE):
    #Sets the module in evaluation mode
    model.eval()
    test_loss = 0
    c_test_loss = 0
    classif_accuracy = 0
    with torch.inference_mode():
        for i, (X, y) in enumerate(test_dataloader):
            X = X.to(device)
            y = y.to(device)
            # 1. Forward pass
            mu, logvar, recon_batch, c_out = model((X, y))

            flat_data = X.view(-1, flat_shape[0]).to(device)
            y_onehot = F.one_hot(y, cond_shape).to(device)
            inp = torch.cat((flat_data, y_onehot), 1)

            # 2. Loss
            tot_loss, C_loss, BCE, KLD = loss_fn(recon_batch, flat_data, mu, logvar, c_out, y_onehot, loss=nn.CrossEntropyLoss)
            test_loss += tot_loss.item()
            c_test_loss += C_loss.item()
            classif_accuracy += accuracy_fn(y, torch.argmax(c_out, dim=1))


    test_loss /= len(test_dataloader.dataset)
    c_test_loss /= len(test_dataloader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss, c_test_loss, classif_accuracy/len(test_dataloader)


def loss_fn(recon, x, mu, logvar, c_out, y_onehot, loss=torch.nn.BCELoss, device=DEVICE):
    y_onehot1 = y_onehot.type(torch.FloatTensor).to(device)
    classif_loss = loss()(c_out, y_onehot1)
    BCE = F.binary_cross_entropy(recon, x, reduction='sum')
    KLD = -0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
    return classif_loss+BCE+KLD, classif_loss, BCE, KLD


def loss_fn_standard_classifier(c_out, y_onehot, device=DEVICE):
    y_onehot1 = y_onehot.type(torch.FloatTensor).to(device)
    classif_loss = torch.nn.BCELoss()(c_out, y_onehot1)
    return classif_loss


def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """

    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def log_label_flipping(indices):
    print(f'Çlasses swapped: ')
    logging.info(f'Çlasses swapped: ')
    for label_flip in label_flipping_indices:
        print(f'{label_flip[0]} and {label_flip[1]}')
        logging.info(f'{label_flip[0]} and {label_flip[1]}')


def label_flip(labels, indices):
    for label_flip in indices:
        for idx in range(len(labels)):
            if labels[idx] == label_flip[0]:
                labels[idx] = label_flip[1]
            elif labels[idx] == label_flip[1]:
                labels[idx] = label_flip[0]
    return labels


def generate_and_save_images(model):
    log_img_dir = 'fl_logs/img/client_generation'

    # Image generation
    model.eval()
    for i in range(2):
        sample = torch.randn(64, 20).to(DEVICE)
        c = np.zeros(shape=(sample.shape[0],))
        label = i + 1
        c[:] = label
        c = torch.FloatTensor(c)
        c = c.to(torch.int64)
        c = c.to(DEVICE)
        c = F.one_hot(c, cond_shape)
        model.eval()
        with torch.inference_mode():
            sample = model.decoder((sample, c)).to(DEVICE)
            sample = sample[:, 0:sample.shape[1]-10]
            saving_path = f'{log_img_dir}/round-{config["current_round"]}'
            os.makedirs(saving_path, exist_ok=True)
            save_image(sample, f'{saving_path}/client-{args.num}-label-{label}.png')