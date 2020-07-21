import torch
import numpy as np
import scipy.stats as stats
from torch import nn, optim
from tqdm import tqdm
from matplotlib import rc
import matplotlib.pyplot as plt

rc('text', usetex=True)

mu1, v1 = -1., 0.0625
mu2, v2 = 2., 2.
omega = 0.67

sigma1, sigma2 = np.sqrt(v1), np.sqrt(v2)
x = np.linspace(-3., 7, 1000)


def sample_real_data(num_samples):
    data_1 = np.random.normal(loc=mu1, scale=sigma1, size=int(num_samples * (1 - omega)))
    data_2 = np.random.normal(loc=mu2, scale=sigma2, size=num_samples - data_1.shape[0])
    return torch.tensor(np.concatenate([data_1, data_2], 0)).type(torch.FloatTensor)


def sample_fake_data(mu, sigma, num_samples):
    return torch.tensor(np.random.normal(loc=mu, scale=sigma, size=num_samples)).type(torch.FloatTensor)


class MLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, X):
        X = X.view(X.shape[0], -1)
        return self.main(X).squeeze()


def get_divergence_function(divergence):
    fn, grad_exp, conjugate_grad_exp = None, None, None
    if divergence == 'KL':
        def fn(x):
            return x * torch.log(x)

        def grad_exp(x):
            return torch.tensor(1.) + x

        def conjugate_grad_exp(x):
            return torch.exp(x.clamp(max=3.))

    elif divergence == 'Reverse-KL':
        def fn(x):
            return - torch.log(x)

        def grad_exp(x):
            return - torch.tensor(1.) / torch.exp(x)

        def conjugate_grad_exp(x):
            return - torch.tensor(1.) + x
    elif divergence == 'Jensen-Shannon':
        def fn(x):
            return - (x + 1) * torch.log(0.5 * (x + 1)) + x * torch.log(x)

        def grad_exp(x):
            return torch.log(torch.tensor(2.)) + x - torch.log(1. + torch.exp(x))

        def conjugate_grad_exp(x):
            return -torch.log(torch.tensor(2.)) + torch.log(1. + torch.exp(x))
    elif divergence == 'Squared-Hellinger':
        def fn(x):
            return (torch.sqrt(x) - 1) ** 2

        def grad_exp(x):
            return torch.tensor(1.) - torch.tensor(1.) / torch.exp(0.5 * x)

        def conjugate_grad_exp(x):
            return torch.exp(0.5 * x) - 1
    elif divergence == 'Alpha':
        global alpha

        def fn(x):
            return 1. / (alpha * (alpha - 1)) * (x ** alpha - 1 - alpha * (x - 1))

        def grad_exp(x):
            return 1. / (alpha - 1) * (torch.exp((alpha - 1) * x) - 1)

        def conjugate_grad_exp(x):
            return 1. / alpha * torch.exp(alpha * x) - 1. / alpha
    elif divergence == 'Neyman-X2':
        def fn(x):
            return (1 - x) ** 2 / x

        def grad_exp(x):
            return 1. - torch.exp(-2. * x)

        def conjugate_grad_exp(x):
            return 2. - 2. * torch.exp(-x)
    elif divergence == 'Pearson-X2':
        def fn(x):
            return (x - 1) ** 2

        def grad_exp(x):
            return 2. * torch.exp(x.clamp(max=6.)) - 2

        def conjugate_grad_exp(x):
            return torch.exp((2. * x).clamp(max=6.)) - 1

    return fn, grad_exp, conjugate_grad_exp


plt.figure()
plt.plot(x, (1 - omega) * stats.norm.pdf(x, mu1, sigma1) + omega * stats.norm.pdf(x, mu2, sigma2), linewidth=2, color='blue', label='Real Data')
'''
Direct Minimization of KL
'''
mu = torch.tensor(0., requires_grad=True).type(torch.FloatTensor)
sigma = torch.tensor(1., requires_grad=True).type(torch.FloatTensor)
optim_e = optim.SGD([mu, sigma], lr=1e-3)
for i in range(50000):
    real_data = sample_real_data(1000)
    log_likelihood = - 0.5 * ((real_data - mu) / sigma) ** 2 - torch.log(sigma)
    loss = - log_likelihood.mean()
    optim_e.zero_grad()
    loss.backward()
    optim_e.step()

plt.plot(x, stats.norm.pdf(x, mu.item(), sigma.item()), color='orange', linestyle='-', linewidth=2, label='KL Optimal')
print('=' * 5 + 'KL Optimal' + '=' * 5)
print('mu', mu.item(), 'sigma', sigma.item())

'''
Contrastive Divergence
'''
learning_rate = 1e-3
fn, grad_exp, conjugate_grad_exp = get_divergence_function('KL')
mu = torch.tensor(0., requires_grad=True).type(torch.FloatTensor)
sigma = torch.tensor(1., requires_grad=True).type(torch.FloatTensor)
optim_e = optim.SGD([mu, sigma], lr=1e-3)
for i in range(20000):
    real_data = sample_real_data(1000)
    fake_data = sample_fake_data(mu.item(), sigma.item(), 1000)
    real_energy = 0.5 * ((real_data - mu) / sigma) ** 2
    fake_energy = 0.5 * ((fake_data - mu) / sigma) ** 2
    loss = real_energy.mean() - fake_energy.mean()
    optim_e.zero_grad()
    loss.backward()
    optim_e.step()

plt.plot(x, stats.norm.pdf(x, mu.item(), sigma.item()), color='r', linestyle=':', linewidth=2, label='Contrastive Divergence')
print('=' * 5 + 'Contrastive Divergence' + '=' * 5)
print('mu', mu.item(), 'sigma', sigma.item())
plt.legend()
plt.savefig('/atlas/u/lantaoyu/projects/ebm-pytorch/samples/gaussian_ebm/KL-CD.pdf')

'''
f-EBM
'''

divergence = 'Jensen-Shannon'
alpha = -1
learning_rate = 1e-3
fn, grad_exp, conjugate_grad_exp = get_divergence_function(divergence)
plt.figure()
plt.plot(x, (1 - omega) * stats.norm.pdf(x, mu1, sigma1) + omega * stats.norm.pdf(x, mu2, sigma2), linewidth=2,
         color='blue', label='Real Data')

mu = torch.tensor(0., requires_grad=True).type(torch.FloatTensor)
sigma = torch.tensor(1., requires_grad=True).type(torch.FloatTensor)
optim_e = optim.SGD([mu, sigma], lr=1e-3)
for i in tqdm(range(30000)):
    noise = torch.randn(1000)
    reparam_x = noise * sigma + mu
    p_over_q = ((1 - omega) / sigma1 * torch.exp(- 0.5 * ((reparam_x - mu1) / sigma1) ** 2) +
                omega / sigma2 * torch.exp(- 0.5 * ((reparam_x - mu2) / sigma2) ** 2)) / \
               (1. / sigma * torch.exp(- 0.5 * ((reparam_x - mu) / sigma) ** 2))
    loss = fn(p_over_q).mean()
    optim_e.zero_grad()
    loss.backward()
    optim_e.step()
    # if i % 1000 == 0:
    #     print(i, mu.item(), sigma.item())
plt.plot(x, stats.norm.pdf(x, mu.item(), sigma.item()), color='orange', linestyle='-', linewidth=2, label='%s Optimal' % divergence)
# plt.plot(x, stats.norm.pdf(x, mu.item(), sigma.item()), color='orange', linestyle='-', linewidth=2, label='Alpha-Div (-1) Optimal')
print('=' * 5 + '%s Optimal' % divergence + '=' * 5)
print('mu', mu.item(), 'sigma', sigma.item())

mu = torch.tensor(0., requires_grad=True).type(torch.FloatTensor)
sigma = torch.tensor(1., requires_grad=True).type(torch.FloatTensor)
optim_e = optim.SGD([mu, sigma], lr=learning_rate)
model_f = MLP()
optim_f = optim.SGD(model_f.parameters(), lr=learning_rate)
for i in tqdm(range(30000)):
    real_data = sample_real_data(1000)
    fake_data = sample_fake_data(mu.item(), sigma.item(), 1000)
    real_energy = 0.5 * ((real_data - mu) / sigma) ** 2
    fake_energy = 0.5 * ((fake_data - mu) / sigma) ** 2
    real_f = model_f(real_data)
    fake_f = model_f(fake_data)

    loss_f = -(grad_exp(real_f + real_energy) - conjugate_grad_exp(fake_f + fake_energy)).mean()
    optim_f.zero_grad()
    loss_f.backward()
    optim_f.step()

    real_energy = 0.5 * ((real_data - mu) / sigma) ** 2
    fake_energy = 0.5 * ((fake_data - mu) / sigma) ** 2
    real_f = model_f(real_data)
    fake_f = model_f(fake_data)

    loss_e = torch.mean(grad_exp(real_f + real_energy)) + \
             torch.mean(fake_energy * conjugate_grad_exp(fake_f + fake_energy).detach()) - \
             torch.mean(fake_energy) * torch.mean(conjugate_grad_exp(fake_f + fake_energy)).detach() - \
             torch.mean(conjugate_grad_exp(fake_f + fake_energy))

    optim_e.zero_grad()
    loss_e.backward()
    optim_e.step()
    if i % 1000 == 0:
        print(i, mu.item(), sigma.item())
print('=' * 5 + '%s f-EBM' % divergence + '=' * 5)
print('mu', mu.item(), 'sigma', sigma.item())
torch.save(model_f.state_dict(), '/atlas/u/lantaoyu/projects/ebm-pytorch/samples/gaussian_ebm/model_%s.pt' % divergence)

plt.plot(x, stats.norm.pdf(x, mu.item(), sigma.item()), color='r', linestyle=':', linewidth=2, label='%s f-EBM' % divergence)
# plt.plot(x, stats.norm.pdf(x, mu.item(), sigma.item()), color='r', linestyle=':', linewidth=2, label='Alpha-Div (-1) f-EBM')
plt.legend()
if divergence is 'Alpha':
    output = '/atlas/u/lantaoyu/projects/ebm-pytorch/samples/gaussian_ebm/%s.pdf' % (divergence + '_' + str(alpha))
else:
    output = '/atlas/u/lantaoyu/projects/ebm-pytorch/samples/gaussian_ebm/%s.pdf' % divergence
plt.savefig(output)

# Plot density ratio
plt.figure()
x_tensor = torch.tensor(x).type(torch.FloatTensor)
p_over_q = ((1 - omega) / sigma1 * torch.exp(- 0.5 * ((x_tensor - mu1) / sigma1) ** 2) +
            omega / sigma2 * torch.exp(- 0.5 * ((x_tensor - mu2) / sigma2) ** 2)) / \
               (1. / sigma * torch.exp(- 0.5 * ((x_tensor - mu) / sigma) ** 2))
log_p_over_q = torch.log(p_over_q).detach().numpy()
x_f = model_f(x_tensor)
x_e = 0.5 * ((x_tensor - mu) / sigma) ** 2
x_f_e = (x_f + x_e).detach().numpy()
plt.plot(x, log_p_over_q, color='orange', linestyle='-', linewidth=2, label=r'$\log(p(x)/q_\theta(x))$')
plt.plot(x, x_f_e, color='r', linestyle=':', linewidth=2, label=r'$H_\omega(x) + E_\theta(x)$')
plt.legend()
plt.savefig('/atlas/u/lantaoyu/projects/ebm-pytorch/samples/gaussian_ebm/%s_density_ratio.pdf' % divergence)


'''
=====KL Optimal=====
mu 1.0106583833694458 sigma 1.8289546966552734
=====Contrastive Divergence=====
mu 1.0120408535003662 sigma 1.8290705680847168
=====KL f-EBM=====
mu 1.0153669118881226 sigma 1.8302433490753174
=====Reverse KL Optimal=====
mu 1.5845422744750977 sigma 1.631060242652893
=====Reverse KL f-EBM=====
mu 1.5852361917495728 sigma 1.6345337629318237
=====Jensen-Shannon Optimal=====
mu 1.3032277822494507 sigma 1.7671679258346558
=====Jensen-Shannon f-EBM=====
mu 1.3166966438293457 sigma 1.7604091167449951
=====Squared-Hellinger Optimal=====
mu 1.3202418088912964 sigma 1.7308987379074097
=====Squared-Hellinger f-EBM=====
mu 1.3227406740188599 sigma 1.7471095323562622
=====Alpha -0.5 Optimal=====
mu 1.7464290857315063 sigma 1.555694818496704
=====Alpha -0.5 f-EBM=====
mu 1.7433207035064697 sigma 1.5520931482315063
=====Alpha -1 Optimal=====
mu 1.8292300701141357 sigma 1.5184484720230103
=====Alpha -1 f-EBM=====
mu 1.819793939590454 sigma 1.5251394510269165
=====Alpha 0.9 Optimal=====
mu 1.0705689191818237 sigma 1.8109134435653687
=====Alpha 0.9 f-EBM=====
mu 1.0723780393600464 sigma 1.8185255527496338
=====Neyman-X2 Optimal=====
mu 1.8303701877593994 sigma 1.5150835514068604
=====Neyman-X2 f-EBM=====
mu 1.8267630338668823 sigma 1.515981674194336
=====Pearson-X2 Optimal=====
mu 0.5758056044578552 sigma 1.9217272996902466
=====Pearson-X2 f-EBM=====
mu 0.5656395554542542 sigma 1.9346171617507935
'''
