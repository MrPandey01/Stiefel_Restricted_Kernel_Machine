from utils import create_dirs, convert_to_imshow_format
from stiefel_rkm_model import *
import logging
import argparse
import time
import matplotlib.pyplot as plt
from datetime import datetime

# Model Settings =================================================================================================
parser = argparse.ArgumentParser(description='St-RKM Model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset_name', type=str, default='mnist',
                    help='Dataset name: mnist/fashion-mnist/svhn/dsprites/3dshapes/cars3d')
parser.add_argument('--h_dim', type=int, default=10, help='Dim of latent vector')
parser.add_argument('--capacity', type=int, default=64, help='Conv_filters of network')
parser.add_argument('--mb_size', type=int, default=256, help='Mini-batch size')
parser.add_argument('--x_fdim1', type=int, default=256, help='Input x_fdim1')
parser.add_argument('--x_fdim2', type=int, default=50, help='Input x_fdim2')
parser.add_argument('--c_accu', type=float, default=1, help='Input weight on recons_error')
parser.add_argument('--noise_level', type=float, default=1e-3, help='Noise-level')
parser.add_argument('--loss', type=str, default='deterministic', help='loss type: deterministic/noisyU/splitloss')

# Training Settings =============================
parser.add_argument('--lr', type=float, default=2e-4, help='Input learning rate for ADAM optimizer')
parser.add_argument('--lrg', type=float, default=1e-4, help='Input learning rate for Cayley_ADAM optimizer')
parser.add_argument('--max_epochs', type=int, default=1000, help='Input max_epoch')
parser.add_argument('--proc', type=str, default='cuda', help='device type: cuda or cpu')
parser.add_argument('--workers', type=int, default=16, help='Number of workers for dataloader')
parser.add_argument('--shuffle', type=bool, default=True, help='shuffle dataset: True/False')

opt = parser.parse_args()
# ==================================================================================================================

device = torch.device(opt.proc)

if torch.cuda.is_available():
    torch.cuda.empty_cache()

ct = time.strftime("%Y%m%d-%H%M")
dirs = create_dirs(name=opt.dataset_name, ct=ct)
dirs.create()

# noinspection PyArgumentList
logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    handlers=[logging.FileHandler('log/{}/{}_{}.log'.format(opt.dataset_name, opt.dataset_name, ct)),
                              logging.StreamHandler()])

""" Load Training Data """
xtrain, ipVec_dim, nChannels = get_dataloader(args=opt)

""" Visualize some training data """
# perm1 = torch.randperm(len(xtrain.dataset))
# it = 0
# fig, ax = plt.subplots(5, 5)
# for i in range(5):
#     for j in range(5):
#         ax[i, j].imshow(convert_to_imshow_format(xtrain.dataset[perm1[it]][0].numpy()))
#         it+=1
# plt.suptitle('Ground Truth Data')
# plt.setp(ax, xticks=[], yticks=[])
# plt.show()

ngpus = torch.cuda.device_count()

rkm = RKM_Stiefel(ipVec_dim=ipVec_dim, args=opt, nChannels=nChannels, ngpus=ngpus).to(device)
logging.info(rkm)
logging.info(opt)
logging.info('\nN: {}, mb_size: {}'.format(len(xtrain.dataset), opt.mb_size))
logging.info('We are using {} GPU(s)!'.format(ngpus))

# Accumulate trainable parameters in 2 groups:
# 1. Manifold_params 2. Network params
param_g, param_e1 = param_state(rkm)

optimizer1 = stiefel_opti(param_g, opt.lrg)
optimizer2 = torch.optim.Adam(param_e1, lr=opt.lr, weight_decay=0)

# Train =========================================================================================================
start = datetime.now()
Loss_stk = np.empty(shape=[0, 3])
cost, l_cost = np.inf, np.inf  # Initialize cost
is_best = False
t = 1
while cost > 1e-10 and t <= opt.max_epochs:  # run epochs until convergence or cut-off
    avg_loss, avg_f1, avg_f2 = 0, 0, 0

    for _, sample_batched in enumerate(tqdm(xtrain, desc="Epoch {}/{}".format(t, opt.max_epochs))):
        loss, f1, f2 = rkm(sample_batched[0].to(device))

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
        optimizer1.step()

        avg_loss += loss.item()
        avg_f1 += f1.item()
        avg_f2 += f2.item()
    cost = avg_loss

    # Remember lowest cost and save checkpoint
    is_best = cost < l_cost
    l_cost = min(cost, l_cost)
    dirs.save_checkpoint({
        'epochs': t,
        'rkm_state_dict': rkm.state_dict(),
        'optimizer1': optimizer1.state_dict(),
        'optimizer2': optimizer2.state_dict(),
        'Loss_stk': Loss_stk,
    }, is_best)

    logging.info('Epoch {}/{}, Loss: [{}], Kpca: [{}], Recon: [{}]'.format(t, opt.max_epochs, cost, avg_f1, avg_f2))
    Loss_stk = np.append(Loss_stk, [[cost, avg_f1, avg_f2]], axis=0)
    t += 1

logging.info('Finished Training. Lowest cost: {}'
             '\nLoading best checkpoint [{}] & computing sub-space...'.format(l_cost, dirs.dircp))
# ==================================================================================================================

# Load Checkpoint
sd_mdl = torch.load('cp/{}/{}'.format(opt.dataset_name, dirs.dircp))
rkm.load_state_dict(sd_mdl['rkm_state_dict'])

h, U = final_compute(model=rkm, args=opt, ct=ct)
logging.info("\nTraining complete in: " + str(datetime.now() - start))

# Save Model and Tensors ======================================================================================
torch.save({'rkm': rkm,
            'rkm_state_dict': rkm.state_dict(),
            'optimizer1': optimizer1.state_dict(),
            'optimizer2': optimizer2.state_dict(),
            'Loss_stk': Loss_stk,
            'opt': opt,
            'h': h, 'U': U}, 'out/{}/{}'.format(opt.dataset_name, dirs.dirout))
logging.info('\nSaved File: {}'.format(dirs.dirout))
