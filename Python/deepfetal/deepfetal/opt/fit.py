import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torch
import torch.optim
import copy
import numpy as np
from ..opt.cost import *
import os.path


class ImageFit(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        return self.x, self.y


class ExperimentStruct:
    def __init__(self, experiment_pars):
        # Network / Fitting structure / Data / Optimizer
        self.net, self.fit, self.data, self.opt = None, None, None, None
        # ASSIGNMENTS
        if not isinstance(experiment_pars, dict):
            experiment_pars = experiment_pars.__dict__
        for x in dir(self):
            if any(y == x for y in experiment_pars):
                setattr(self, x, experiment_pars[x])

        # print(self.__dict__)

class DataStruct:
    def __init__(self, data_pars):
        # Input / Train / Output / Ground truth / Loss train / Loss ground truth
        self.x, self.y, self.y_hat, self.loss_func, self.loss_func_ast = None, None, None, None, None
        # ASSIGNMENTS
        if not isinstance(data_pars, dict):
            data_pars = data_pars.__dict__
        for x in dir(self):
            if any(y == x for y in data_pars):
                setattr(self, x, data_pars[x])

        # print(self.__dict__)


class FitStruct:
    def __init__(self, fit_pars):
        # DEFAULTS
        # Data type: gpu or not
        # print('Number of devices:')
        # print(torch.cuda.device_count())
        # print(torch.cuda.is_available())
        if torch.cuda.device_count() > 0:
            self.type = torch.cuda.FloatTensor
        else:
            self.type = torch.FloatTensor
        # Number of epochs / Learning rate / Add noise to the input / Decay factor for random input
        self.epochs, self.lr, self.random_input_sigma, self.random_input_decay_factor = 1, 0.001, 0, 1
        # Frequency to show information / Factor of decrease to pick up best network / Assumed noise free observation
        self.verbose_frequency, self.factor_best, self.y_ast = 50, float('inf'), None
        # Batch size / Use data loader (for mini-batches) / Type of encoding / Struct of encoding
        self.batch_size, self.use_data_loader, self.typ_enc, self.str_enc = 32, False, 'unit', None
        # File name input / File name output / Use profiling / Detect anomaly
        self.file_name_in, self.file_name_ou, self.use_prof, self.detect_anomaly = None, None, False, False
        # Show memory usage / Separable forward pass / Epoch to start estimation of detail coefficients
        self.show_memory, self.sep_pass, self.epoch_detail = False, False, 0
        # Use autocast
        self.use_autocast = False

        # ASSIGNMENTS
        if not isinstance(fit_pars, dict):
            fit_pars = fit_pars.__dict__
        for x in dir(self):
            if any(y == x for y in fit_pars):
                setattr(self, x, fit_pars[x])

        # TYPE CONVERSIONS
        self.epochs = np.ceil(self.epochs).astype(int)
        self.batch_size = np.ceil(self.batch_size).astype(int).item()  # .item() converts from numpy to py, required by
                                                                       # DataLoader

        # ENCODING STRUCT
        self.str_enc = EncodingStruct(self.str_enc)

        # print(self.__dict__)


def fit(x, y, net, fit_struct=None):
    fs = FitStruct(fit_struct)

    rseed = 3  # Deterministic
    # rseed = random.randint(1, 2147483647)  # Random
    torch.manual_seed(rseed)  # Very little difference between deterministic and random
    file_name_in = fs.file_name_in
    file_name_ou = fs.file_name_ou

    if os.path.isfile(file_name_in):
        checkpoint = torch.load(fs.file_name_in)
        net.load_state_dict(checkpoint['net'])
        net.eval()
        fs = checkpoint['fit']
        x = checkpoint['data'].x
        y = checkpoint['data'].y
        loss_func = checkpoint['data'].loss_func
        loss_func_ast = checkpoint['data'].loss_func_ast
    else:
        net = net.type(fs.type)
        x = Variable(torch.from_numpy(np.array(x))).type(fs.type)
        y = Variable(torch.from_numpy(np.array(y))).type(fs.type)

        x, y = matchdims(x, y)
        if fs.y_ast is not None:
            fs.y_ast = Variable(torch.from_numpy(np.array(fs.y_ast))).type(fs.type)
            x, fs.y_ast = matchdims(x, fs.y_ast)
        if fs.use_data_loader:
            data_loader = DataLoader(ImageFit(x, y), batch_size=fs.batch_size, pin_memory=True)
            x, y = next(iter(data_loader))

        loss_func = EncodeMSELoss(buildEncoding(fs.typ_enc), fs.str_enc)
        loss_func_ast = EncodeMSELoss()

    x_orig = x.data.clone()
    noise = x.data.clone()

    p = [x for x in net.parameters()]
    #print('Number of parameters of the network: %d' % (len(p)))
    #print('Target data size: ', y.shape)

    opt = torch.optim.Adam(p, lr=fs.lr) #betas default (0.9, 0.999)
    #opt = torch.optim.SparseAdam(p, lr=fs.lr) #betas default (0.9, 0.999)
    if os.path.isfile(file_name_in):
        opt.load_state_dict(checkpoint['opt'])

    mse_wrt_y = np.zeros(fs.epochs)
    mse_wrt_yast = None
    if fs.y_ast is not None:
        mse_wrt_yast = np.zeros(fs.epochs)
        mse_noise = loss_func_ast(y, fs.y_ast)

    best_net = copy.deepcopy(net)
    best_mse = float('inf')

    if fs.show_memory:
        print('Memory before training: %d' % torch.cuda.memory_allocated(None))
    if fs.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
    #if fs.use_prof:
    #    torch.autograd.profiler.enable_profiler(use_cuda=True)
    #with torch.autograd.profiler.profile(use_cuda=True) as prof:
    if fs.use_autocast:
        scaler = torch.cuda.amp.GradScaler()

    for i in range(fs.epochs):
        net_prev = copy.deepcopy(net)

        if fs.random_input_sigma > 0:
            fs.random_input_sigma *= fs.random_input_decay_factor
            x = Variable(x_orig + (noise.normal_() * fs.random_input_sigma))
        else:
            x = Variable(x_orig)

        lim_batch = x.size()[0]
        #permutation = torch.randperm(lim_batch)

        loss_acc = 0
        for j in range(0, lim_batch, fs.batch_size):
            # if fs.net_type == 1:  # Siren
            #    ind = permutation[j:j + fs.batch_size]
            #    x_batch, y_batch = x[ind, :], y[ind, :]
            # We should only use batch from here on

            def closure():
                opt.zero_grad()
                if fs.sep_pass:
                    perm_wav = torch.randperm(net.NparUnits)
                    for k in range(net.NparUnits):
                        if i >= fs.epoch_detail or k == 0 or k == perm_wav[0]:
                            y_hat = net(x.type(fs.type), k)
                            loss = loss_func(y_hat, y)
                            loss.backward()
                else:
                    if fs.use_autocast:
                        with torch.cuda.amp.autocast():
                            y_hat = net(x.type(fs.type))
                            loss = loss_func(y_hat, y)
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                    else:
                        y_hat = net(x.type(fs.type))
                        loss = loss_func(y_hat, y)
                        loss.backward()
                        opt.step()
                mse_wrt_y[i] += loss.data.cpu().numpy()

                with torch.no_grad():
                    #if sep_pass:
                    #    y_hat = net(x.type(fs.type))
                    #    loss = loss_func(y_hat, y)

                    if fs.y_ast is not None:
                        loss_ast = loss_func_ast(Variable(y_hat), fs.y_ast)
                        mse_wrt_yast[i] += loss_ast.data.cpu().numpy()

                    if fs.verbose_frequency < 1e6 and i % fs.verbose_frequency == 0 and j+fs.batch_size >= lim_batch:
                        loss_orig = loss_func(net(Variable(x_orig).type(fs.type)), y)
                        print('Iter %05d.   Losses:  Train %05.2f  Original %05.2f'
                              % (i, lossTodB(loss), lossTodB(loss_orig)),
                              end='')
                        if fs.y_ast is not None:
                            loss_ast_orig = loss_func_ast(net(Variable(x_orig).type(fs.type)), fs.y_ast)
                            print('  True %05.2f  True Original %05.2f  Noise %05.2f'
                                    % (lossTodB(loss_ast.data), lossTodB(loss_ast_orig), lossTodB(mse_noise)), end='')
                        print()
                return loss
            #loss = opt.step(closure) # In this case we don't need the step() inside
            loss = closure()
            loss_acc += loss.data

        if best_mse > fs.factor_best * loss_acc:
            best_mse = loss_acc
            best_net = copy.deepcopy(net_prev)

    #if fs.use_prof:
        #prof = torch.autograd.profiler.disable_profiler()
    #print(prof.key_averages().table(sort_by="cuda_time_total"))
    if fs.detect_anomaly:
        torch.autograd.set_detect_anomaly(False)
    if fs.show_memory:
        print('Memory after training: %d' % torch.cuda.memory_allocated(None))

    with torch.no_grad():
        x = x_orig
        loss_acc = loss_func(net(Variable(x).type(fs.type)), y).data
        if loss_acc < best_mse:
            best_net = copy.deepcopy(net)
        y_hat = best_net(Variable(x).type(fs.type))

    if file_name_ou:
        ds = {'x': x, 'y': y, 'yhat': y_hat, 'loss_func': loss_func, 'loss_func_ast': loss_func_ast}
        ds = DataStruct(ds)

        exper = {'net': best_net.state_dict(), 'data': ds, 'fit': fs, 'opt': opt.state_dict()}
        # exper = ExperimentStruct(exper)
        torch.save(exper, file_name_ou)
        print('Model written to file %s' % file_name_ou)

    return [best_net, y_hat.data.cpu().numpy(), x.data.cpu().numpy(), mse_wrt_y, mse_wrt_yast]  # Network / Estimation / Input / MSE train / MSE true
