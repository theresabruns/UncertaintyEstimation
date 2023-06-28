from src.base_net import *
import torch.nn.functional as F
import torch.nn as nn
from scipy.stats import entropy
from src.Deep_Ensemble.resnet import ResNet34, ResNet50
from src.Deep_Ensemble.densenet import DenseNet121


class Ens_net(BaseNet):
    eps = 1e-6

    def __init__(self, lr=1e-3, channels_in=3, side_in=28, cuda=True, device='cuda:0', classes=10, 
                batch_size=128, weight_decay=0, n_hid=200, data='usecase', model='linear'):
        super(Ens_net, self).__init__()
        cprint('y', ' Creating Net! ')
        self.lr = lr
        self.schedule = None  # [] #[50,200,400,600]
        self.cuda = cuda
        self.channels_in = channels_in
        self.weight_decay = weight_decay
        self.classes = classes
        self.n_hid = n_hid
        self.batch_size = batch_size
        self.side_in = side_in
        self.create_net(data, model)
        self.create_opt()
        self.epoch = 0
        self.test = False
        self.device = device

    def create_net(self, data='usecase', model='linear'):
        torch.manual_seed(42)
        if self.cuda:
            torch.cuda.manual_seed(42)

        if model == 'resnet34':
            self.model = ResNet34(num_classes=self.classes, input_size=self.side_in)
        elif model == 'resnet50':
            self.model = ResNet50(num_classes=self.classes, input_size=self.side_in)
        elif model == 'vgg16':
            #self.model = ConvNet(in_channels=self.channels_in, out_channels=self.classes, input_size=self.side_in)
            self.model = VGG16(in_channels=self.channels_in, out_channels=self.classes, input_size=self.side_in)
        elif model == 'linear':
            self.model = Linear_2L(in_channels=self.channels_in, out_channels=self.classes, input_size=self.side_in, n_hid=self.n_hid)
        elif model == 'densenet':
            self.model = DenseNet121(num_classes=self.classes, input_size=self.side_in)

        if self.cuda:
            self.model.cuda()

        print('    Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))

    def create_opt(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=self.weight_decay)

    def fit(self, x, y):
        x, y = x.to(self.device), y.to(self.device)

        self.optimizer.zero_grad()

        out = self.model(x)
        loss = F.cross_entropy(out, y, reduction='sum')

        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
        self.optimizer.step()

        # out: (batch_size, out_channels, out_caps_dims)
        pred = out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return loss.data, err

    def eval(self, x, y, train=False):
        x, y = x.to(self.device), y.to(self.device)

        out = self.model(x)

        loss = F.cross_entropy(out, y, reduction='sum')

        probs = F.softmax(out, dim=1).data.cpu()

        pred = out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return loss.data, err, probs

    # alternative version
    # used for previous Ensemble prediction version (with completely separately trained x-Ensembles)
    def sample_eval(self, x, y, modeldir, model, seed, data):
        x, y = x.to(self.device), y.to(self.device)

        model_files = []
        for filename in os.listdir(modeldir):
            if filename.endswith('.pt'):
                model_files.append(filename)

        self.filtered_files = [s for s in model_files if 'seed'+str(seed) in s and model in s and data in s]
        logits = x.data.new(len(self.filtered_files), x.shape[0], self.model.out_channels)
        entropies = x.data.new(len(self.filtered_files), x.shape[0], 1)

        for i, file in enumerate(self.filtered_files):
            # Load the stored model's parameters into the net shell
            self.load(os.path.join(modeldir,file))
            self.model = self.model.to(self.device)

            logit = self.model.forward(x)
            logits[i] = logit

            prob = F.softmax(logit, dim=1).data
            single_entropy = entropy(prob.detach().cpu().numpy().T)
            single_entropy = torch.from_numpy(single_entropy)
            entropies[i] = single_entropy
        
        mean_logit = logits.mean(dim=0, keepdim=False)
        mean_entropy = entropies.mean(dim=0, keepdim=False)
        probs = F.softmax(mean_logit, dim=1).data
        pred = probs.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()
        acc = (len(self.filtered_files) - err)/len(self.filtered_files)

        return err, probs, acc, mean_entropy

    def get_sample_scores(self, x, y, num_classes, modeldir, model, seed, data, members):
        x, y = x.to(self.device), y.to(self.device)
        batch_size = x.shape[0]

        model_files = []
        self.filtered_files = []
        for filename in os.listdir(modeldir):
            if filename.endswith('.pt'):
                model_files.append(filename)

        if members == 1:
            self.filtered_files = [s for s in model_files if 'seed'+str(seed) in s and model in s and data in s and 'model0' in s]
        else:
            self.filtered_files = [s for s in model_files if 'seed'+str(seed) in s and model in s and data in s]
        if not len(self.filtered_files) == members:
            print("NUMBER OF LOADED MODELS DIFFERENT TO MEMBER COUNT!")
        logits = x.data.new(len(self.filtered_files), batch_size, self.model.out_channels)
        entropies = x.data.new(len(self.filtered_files), batch_size)

        for i, file in enumerate(self.filtered_files):
            # Load the stored model's parameters into the net shell
            self.load(os.path.join(modeldir,file))
            self.model = self.model.to(self.device)

            logit = self.model.forward(x)
            logits[i] = logit
            prob = F.softmax(logit, dim=1).data
            single_entropy = entropy(prob.detach().cpu().numpy().T)
            single_entropy = torch.from_numpy(single_entropy)
            entropies[i] = single_entropy
        
        mean_logit = logits.mean(dim=0, keepdim=False)
        mean_entropy = entropies.mean(dim=0, keepdim=False)

        probs = F.softmax(mean_logit, dim=1).data # (batch, classes)
        # classification error
        # get the indexes of the max log-probability
        pred = probs.max(dim=1, keepdim=False)[1] # (batch, )
        batch_error = pred.ne(y.data).sum() #int
        error = batch_error / batch_size #int
        # NLL score
        batch_nll = F.nll_loss(probs, y)
        nll = -batch_nll / batch_size
        # Brier score
        one_hot = np.zeros((batch_size, num_classes))
        for i in range(y.shape[0]):
            one_hot[i][y[i]] = 1
        probs_arr = probs.detach().cpu().numpy()
        diff = np.power((one_hot - probs_arr), 2)
        batch_brier = np.sum(diff, axis=1) / num_classes # sum up over classes -> (batch, )
        brier = np.sum(batch_brier) / batch_size
        return error, nll, brier, probs, mean_entropy

    def get_weight_samples(self):
        weight_vec = []

        state_dict = self.model.state_dict()

        for key in state_dict.keys():

            if 'weight' in key:
                weight_mtx = state_dict[key].cpu().data
                for weight in weight_mtx.view(-1):
                    weight_vec.append(weight)

        return np.array(weight_vec)

#------------------------------------------------------------------------------------------------------------------------------------
class Linear_2L(nn.Module):
    def __init__(self, in_channels, out_channels, input_size, n_hid):
        super(Linear_2L, self).__init__()

        input_dim = in_channels * input_size * input_size
        self.input_dim = input_dim
        self.out_channels = out_channels

        self.fc1 = nn.Linear(input_dim, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, n_hid)
        self.fc4 = nn.Linear(n_hid, n_hid)
        self.fc5 = nn.Linear(n_hid, out_channels)
        self.bn = nn.BatchNorm1d(n_hid) 
        self.act = nn.ReLU(inplace=True)


    def forward(self, x):
        x = x.view(-1, self.input_dim)  # view(batch_size, input_dim)
        # -----------------
        x = self.fc1(x)
        x = self.bn(x)
        x = self.act(x)
        # -----------------
        x = self.fc2(x)
        x = self.bn(x)
        x = self.act(x)
        # -----------------
        #x = self.fc3(x)
        #x = self.bn(x)
        #x = self.act(x)
        # -----------------
        #x = self.fc4(x)
        #x = self.bn(x)
        #x = self.act(x)
        # -----------------
        y = self.fc5(x)

        return y

#---------------------------------------------------------------------------------------------------------------
class VGG16(nn.Module):
    def __init__(self, in_channels, out_channels, input_size):
        super(VGG16, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.input_size = input_size
        if input_size >= 100 and input_size < 200:
            flat_size = int(input_size/32)
            pool_size = 2
            fclayer = flat_size * flat_size * 512
            fclayer2 = 4096
            fclayer3 = 4096
        elif input_size >= 200:
            flat_size = int(input_size/256)
            pool_size = 4
            fclayer = flat_size * flat_size * 512
            fclayer2 = 64
            fclayer3 = 32
        elif input_size > 32 and input_size < 100:
            flat_size = int(input_size/8)
            pool_size = 2
            fclayer = flat_size * flat_size * 32
            fclayer2 = 512
            fclayer3 = 512
        else:
            flat_size = int(input_size/4)
            pool_size = 1
            fclayer = flat_size * flat_size * 32
            fclayer2 = 4096
            fclayer3 = 4096
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = pool_size, stride = pool_size))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = pool_size, stride = pool_size))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = pool_size, stride = pool_size))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fclayer, fclayer2),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fclayer2, fclayer3),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(fclayer3, self.out_channels))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

#---------------------------------------------------------------------------------------------------------------
class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, input_size):
        super(ConvNet, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.input_size = input_size

        self.conv1 = nn.Conv2d(in_channels, in_channels*2, (3, 3))
        self.conv2 = nn.Conv2d(in_channels*2, in_channels*4, (3, 3))
        img_size = (input_size-4) // 2
        flat_size = (in_channels*4)*img_size*img_size
        self.fc1 = nn.Linear(flat_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
