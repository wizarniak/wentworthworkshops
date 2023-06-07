# %%
import torch as t
import torchvision
import torch.nn as nn
from torch.autograd import grad
from scipy.sparse.linalg import LinearOperator, eigsh
import numpy as np
import torch.nn.functional as F

# %%
MIDDLE_LAYER_NEURONS = 128
MODEL_PATH = './results/model10.pth'
NETWORK_PARAMS = 784*MIDDLE_LAYER_NEURONS + MIDDLE_LAYER_NEURONS + MIDDLE_LAYER_NEURONS*10 + 10
# %%
class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.fc1 = nn.Linear(784,MIDDLE_LAYER_NEURONS)
        self.fc2 = nn.Linear(MIDDLE_LAYER_NEURONS, 10)

    def forward(self, x):
        x = t.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return x

network = DenseNet()
network.load_state_dict(t.load(MODEL_PATH))
criterion = nn.MSELoss()

test_loader = t.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.1307,), (0.3081,))
    ])),
    batch_size=128, shuffle=True
)

network.eval()

images, labels = next(iter(test_loader))

params = [p for p in network.parameters()]
param_shape = sum(p.numel() for p in params)
grads_per_param = t.zeros(param_shape, 10)

outputs = network(images)
for class_idx in range(outputs.size(1)):
    grads = grad(outputs[:,class_idx].sum(), params, retain_graph=True)

    grads_per_param[:,class_idx] = t.cat([g.view(-1) for g in grads])

# %%
def mv(x):
    x = t.from_numpy(x.astype(np.float32))
    first_multiply = t.einsum('ij,j->i',grads_per_param.T,x)
    second_multiply = t.einsum('ij,j->i',grads_per_param, first_multiply)
    return 2*second_multiply
A = LinearOperator((NETWORK_PARAMS,NETWORK_PARAMS), matvec=mv)
eigenvalues, eigenvectors = eigsh(A, 100, which='LM')

# %%
# not real log volume, we leave out the constant terms
eigenvalues = t.from_numpy(eigenvalues.astype(np.float32))
log_det_hessian = t.sum(t.log(t.abs(eigenvalues)))
log_volume = -0.5 * log_det_hessian
print(log_volume)


# %%
