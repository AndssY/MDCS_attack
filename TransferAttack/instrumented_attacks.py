import torch
import numpy as np
import math
from transferattack.gradient.mifgsm import MIFGSM
from transferattack.gradient.mef import MEF
from transferattack.gradient.mdcsmi import MDCSMI
from transferattack.gradient.mdcsmef import MDCSMEF
from transferattack.gradient.ifgsm import IFGSM
from transferattack.gradient.mdcsifgsm import MDCSIFGSM
from transferattack.input_transformation.ops import OPS
from transferattack.input_transformation.mdcsops import MDCSOPS
from transferattack.utils import *

class InstrumentedMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dt_history = []
        self.momentum_history = []
        
    def log_dt_momentum(self, d_t, momentum):
        if d_t is not None:
            self.dt_history.append(d_t.mean().item() if isinstance(d_t, torch.Tensor) else float(d_t))
        if momentum is not None:
            if isinstance(momentum, torch.Tensor):
                self.momentum_history.append(momentum.abs().mean().item())
            else:
                self.momentum_history.append(float(np.abs(momentum).mean() if isinstance(momentum, np.ndarray) else abs(momentum)))

class InstrumentedIFGSM(InstrumentedMixin, IFGSM):
    def forward(self, data, label, **kwargs):
        if self.targeted:
            assert len(label) == 2
            label = label[1]
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        delta = self.init_delta(data)
        momentum = 0
        self.dt_history = []
        self.momentum_history = []
        
        for _ in range(self.epoch):
            logits = self.get_logits(self.transform(data+delta))
            loss = self.get_loss(logits, label)
            grad = self.get_grad(loss, delta)
            momentum = self.get_momentum(grad, momentum) 
            
            delta = self.update_delta(delta, data, momentum, self.alpha)
            self.log_dt_momentum(1.0, momentum)
            
        return delta.detach()

class InstrumentedMDCSIFGSM(InstrumentedMixin, MDCSIFGSM):
    def forward(self, data, label, **kwargs):
        return InstrumentedMDCSMI.forward(self, data, label, **kwargs)

class InstrumentedMIFGSM(InstrumentedMixin, MIFGSM):
    def forward(self, data, label, **kwargs):
        if self.targeted:
            assert len(label) == 2
            label = label[1]
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        delta = self.init_delta(data)
        momentum = 0
        self.dt_history = []
        self.momentum_history = []
        
        for _ in range(self.epoch):
            logits = self.get_logits(self.transform(data+delta, momentum=momentum))
            loss = self.get_loss(logits, label)
            grad = self.get_grad(loss, delta)
            momentum = self.get_momentum(grad, momentum)
            
            delta = self.update_delta(delta, data, momentum, self.alpha)
            
            self.log_dt_momentum(1.0, momentum)
            
        return delta.detach()

class InstrumentedMEF(InstrumentedMixin, MEF):
    def forward(self, data, label, **kwargs):
        if self.targeted:
            assert len(label) == 2
            label = label[1]
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        delta = self.init_delta(data)
        momentum = 0
        self.dt_history = []
        self.momentum_history = []
        
        b, c, h, w = data.shape
        grad_pgia = torch.zeros([self.num_neighbor, b, c, h, w]).to(self.device)
        
        for _ in range(self.epoch):
            sample_delta = self.get_conditional_sampled_points(delta, grad_pgia)
            gradient = self.get_points_gradient(data, sample_delta, label)
            grad_pgia = ((gradient / torch.mean(torch.abs(gradient), (2, 3, 4), keepdim=True)).detach() - self.inner_decay * grad_pgia)
            momentum = self.get_momentum(gradient.sum(0), momentum)

            delta = self.update_delta(delta, data, momentum, self.alpha)
            
            self.log_dt_momentum(1.0, momentum)

        return delta.detach()

class InstrumentedOPS(InstrumentedMixin, OPS):
    def forward(self, data, label, **kwargs):
        if self.targeted:
             assert len(label) == 2
             label = label[1]
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        delta = self.init_delta(data)
        if self.using_sampling:
            self.init_eps_list(delta)

        momentum, averaged_gradient = 0, 0
        self.dt_history = []
        self.momentum_history = []
        
        for _ in range(self.epoch):
            averaged_gradient = self.get_averaged_gradient(data, delta, label)
            momentum = self.get_momentum(averaged_gradient, momentum)
            
            delta = self.update_delta(delta, data, momentum, self.alpha)
            
            self.log_dt_momentum(1.0, momentum)

        return delta.detach()

class InstrumentedMDCSMI(InstrumentedMixin, MDCSMI):
    def forward(self, data, label, **kwargs):
        if self.targeted:
             assert len(label) == 2
             label = label[1]
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        delta = self.init_delta(data)
        d_t = torch.ones_like(data).to(self.device)
        momentum = torch.zeros_like(data).to(self.device)
        self.losses = []
        self.dt_history = []
        self.momentum_history = []
        
        for _ in range(self.epoch):
            logits = self.get_logits(self.transform(data+delta, momentum=momentum))
            loss = self.get_loss(logits, label)
            self.losses.append(loss.item())
            grad = self.get_grad(loss, delta)
            momentum = self.get_momentum(grad, momentum)
            
            abs_momentum = torch.abs(momentum)
            term_clip = self.mdcs_gamma / (abs_momentum + 1e-30)
            d_t = torch.min(term_clip, d_t)
            
            delta = self.update_delta_nosign(delta, data, momentum, self.alpha, d_t)
            
            self.log_dt_momentum(d_t, momentum)

        return delta.detach()

class InstrumentedMDCSMEF(InstrumentedMixin, MDCSMEF):
    def forward(self, data, label, **kwargs):
        if self.targeted:
            assert len(label) == 2
            label = label[1]
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        delta = self.init_delta(data)
        momentum = 0
        d_t = torch.ones_like(data).to(self.device)
        self.dt_history = []
        self.momentum_history = []

        b, c, h, w = data.shape
        grad_pgia = torch.zeros([self.num_neighbor, b, c, h, w]).to(self.device)
        
        for _ in range(self.epoch):
            sample_delta = self.get_conditional_sampled_points(delta, grad_pgia)
            gradient = self.get_points_gradient(data, sample_delta, label)
            grad_pgia = ((gradient / torch.mean(torch.abs(gradient), (2, 3, 4), keepdim=True)).detach() - self.inner_decay * grad_pgia)
            momentum = self.get_momentum(gradient.sum(0), momentum)

            d_t = self.get_dt(momentum, d_t)
            
            delta = self.update_delta_nosign(delta, data, momentum, self.alpha, d_t)
            
            self.log_dt_momentum(d_t, momentum)

        return delta.detach()

class InstrumentedMDCSOPS(InstrumentedMixin, MDCSOPS):
    def forward(self, data, label, **kwargs):
        if self.targeted:
            assert len(label) == 2
            label = label[1]
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        delta = self.init_delta(data)
        if self.using_sampling:
            self.init_eps_list(delta)

        momentum, averaged_gradient = 0, 0
        d_t = torch.ones_like(data).to(self.device)
        self.dt_history = []
        self.momentum_history = []
        
        for _ in range(self.epoch):
            averaged_gradient = self.get_averaged_gradient(data, delta, label)
            momentum = self.get_momentum(averaged_gradient, momentum)
            
            d_t = self.get_dt(momentum, d_t)
            
            delta = self.update_delta_nosign(delta, data, momentum, self.alpha, d_t)
            
            self.log_dt_momentum(d_t, momentum)
            
        return delta.detach()
