import math

import torch
from ..attack import Attack
from ..utils import *


class MDCSMI(Attack):
    """
    MI-FGSM Attack with MDCS

    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1.0, targeted=False, random_start=False,
                norm='linfty', loss='crossentropy', device=None, attack='MDCSMI', mdcs_gamma=1.8, **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device, **kwargs)
        self.alpha = epsilon/epoch
        self.epoch = epoch
        self.decay = decay
        self.mdcs_gamma = mdcs_gamma
        self.loss = self.loss_function(loss)
        self.losses = []


    def forward(self, data, label, **kwargs):
        if self.targeted:
            assert len(label) == 2
            label = label[1] # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)
        d_t = torch.ones_like(data).to(self.device)
        momentum = torch.zeros_like(data).to(self.device)
        self.losses = []
        for _ in range(self.epoch):
            # Obtain the output
            logits = self.get_logits(self.transform(data+delta, momentum=momentum))

            # Calculate the loss
            loss = self.get_loss(logits, label)
            self.losses.append(loss.item())


            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            abs_momentum = torch.abs(momentum)
            term_clip = self.mdcs_gamma / (abs_momentum + 1e-30)
            d_t = torch.min(term_clip, d_t)

            # Update adversarial perturbation
            delta = self.update_delta_nosign(delta, data, momentum, self.alpha, d_t)

        return delta.detach()


    def update_delta_nosign(self, delta, data, grad, alpha, dt, **kwargs):
        if self.norm == 'linfty':
            delta = torch.clamp((delta + alpha*dt*grad) / torch.sqrt(dt), -self.epsilon, self.epsilon)
        else:
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1, 1)
            scaled_grad = grad / (grad_norm + 1e-30)
            delta = (delta + scaled_grad * alpha).view(delta.size(0), -1).renorm(p=2, dim=0, maxnorm=self.epsilon).view_as(delta)
        delta = clamp(delta, img_min-data, img_max-data)
        return delta




   