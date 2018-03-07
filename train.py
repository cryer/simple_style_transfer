from __future__ import division
from torch.autograd import Variable
from torch.backends import cudnn
from torchvision import transforms
import torch
import torchvision
from load_image import *
from model import VGGNet
from config import Config



def train(**kwargs):
    cfg = Config()
    for k, v in kwargs.items():
        setattr(cfg, k, v)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    content = load_image(cfg.content, transform, max_size=cfg.max_size)
    style = load_image(cfg.style, transform, shape=[content.size(3), content.size(2)])

    target = Variable(content.clone(), requires_grad=True)
    optimizer = torch.optim.Adam([target], lr=cfg.lr, betas=[0.5, 0.999])

    vgg = VGGNet()
    if cfg.use_gpu:
        vgg.cuda()

    for step in range(cfg.total_step):
        target_features = vgg(target)
        content_features = vgg(Variable(content))
        style_features = vgg(Variable(style))

        style_loss = 0
        content_loss = 0
        for f1, f2, f3 in zip(target_features, content_features, style_features):
            # Compute content loss
            content_loss += torch.mean((f1 - f2) ** 2)
            _, c, h, w = f1.size()
            f1 = f1.view(c, h * w)
            f3 = f3.view(c, h * w)
            # Compute gram matrix
            f1 = torch.mm(f1, f1.t())
            f3 = torch.mm(f3, f3.t())
            style_loss += torch.mean((f1 - f3) ** 2) / (c * h * w)

        loss = content_loss + cfg.style_weight * style_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % cfg.log_step == 0:
            print('Step [%d/%d], Content Loss: %.4f, Style Loss: %.4f'
                  % (step + 1, cfg.total_step, content_loss.data[0], style_loss.data[0]))

        if (step + 1) % cfg.sample_step == 0:
            denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
            img = target.clone().cpu().squeeze()
            img = denorm(img.data).clamp_(0, 1)
            torchvision.utils.save_image(img, 'output-%d.png' % (step + 1))

if __name__ == "__main__":
    import fire
    fire.Fire()