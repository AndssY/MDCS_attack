import torch
from ..utils import *
from ..attack import Attack
import torch.nn as nn

class MDCSMEF(Attack):
    """
    MDCS-MEF Attack (MEF with MDCS)
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, num_neighbor=20, gamma=2., kesai=0.15, epoch=10, inner_decay=0.9, decay=0.5, targeted=False, 
                random_start=False, norm='linfty', loss='crossentropy_no_reduction', device=None, attack='MDCSMEF', mdcs_gamma=1.8, **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.kesai = kesai * epsilon
        self.gamma = gamma * epsilon
        self.epoch = epoch
        self.inner_decay = inner_decay
        self.decay = decay
        self.num_neighbor = num_neighbor
        self.mdcs_gamma = mdcs_gamma
        print(f"{alpha:.6f}")

    
    def loss_function(self, loss):
        """
        Get the loss function
        """
        if loss == 'crossentropy':
            return nn.CrossEntropyLoss()
        elif loss == 'crossentropy_no_reduction':
            return nn.CrossEntropyLoss(reduction='none')
        else:
            raise Exception("Unsupported loss {}".format(loss))
            
    def get_conditional_sampled_points(self, delta, grad_pgia):
        """
        Neighborhood conditional sampling
        """
        sample_delta = self.transform(delta + torch.zeros_like(grad_pgia).uniform_(-self.gamma, self.gamma))
        sample_delta = self.transform(sample_delta + self.kesai * grad_pgia)
        return sample_delta
        
    def get_points_gradient(self, data, delta, label, **kwargs):
        """
        Calculate the gradients of the sampled points
        """
        b, c, h, w = data.shape
        loss_list = torch.zeros([self.num_neighbor, b]).to(self.device)
        grad_list = torch.zeros([self.num_neighbor, b, c, h, w]).to(self.device)
        for i in range(self.num_neighbor):

            # Get the conditional sampled points x_min
            x_min = self.transform(data + delta[i])

            # Calculate the output of the x_min
            logits = self.get_logits(x_min)

            # Calculate the loss of the x_min
            loss_list[i] = self.get_loss(logits, label)

            # Calculate the gradient of the x_min
            grad_list[i] = self.get_grad(loss_list[i].mean(), x_min)
        
        # Calculate the gradient of the loss function
        grad = (1/self.num_neighbor) * grad_list

        return grad

    def forward(self, data, label, **kwargs):
        """
        The attack procedure for MEF

        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1] # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        momentum = 0
        d_t = torch.ones_like(data).to(self.device)

        b, c, h, w = data.shape
        grad_pgia = torch.zeros([self.num_neighbor, b, c, h, w]).to(self.device)
        for _ in range(self.epoch):

            # Neighborhood conditional sampling
            sample_delta = self.get_conditional_sampled_points(delta, grad_pgia)

            # Calculate the gradient of each point
            gradient = self.get_points_gradient(data, sample_delta, label)

            # Update gradient for previous gradient inversion approximation
            grad_pgia = ((gradient / torch.mean(torch.abs(gradient), (2, 3, 4), keepdim=True)).detach() - self.inner_decay * grad_pgia)

            # Calculate the momentum
            momentum = self.get_momentum(gradient.sum(0), momentum)

            # Update adversarial perturbation
            # delta = self.update_delta(delta, data, momentum, self.alpha)

            d_t = self.get_dt(momentum, d_t)
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


    def get_dt(self, momentum, d_t, **kwargs):
        """
        The d_t calculation
        """
        abs_momentum = torch.abs(momentum)
        term_clip = self.mdcs_gamma / (abs_momentum + 1e-30)
        d_t = torch.min(term_clip, d_t)

        return d_t


