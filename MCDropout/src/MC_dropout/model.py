from src.base_net import *
import torch.nn.functional as F
import torch.nn as nn
from scipy.stats import entropy
from src.MC_dropout.resnet import ResNet34, ResNet50
from src.MC_dropout.densenet import DenseNet121


def MC_dropout(act_vec, p=0.5, mask=True, inplace=True):
    return F.dropout(act_vec, p=p, training=mask, inplace=inplace)

class MC_drop_net(BaseNet):
    eps = 1e-6

    def __init__(self, lr=1e-3, channels_in=3, side_in=28, cuda=True, device='cuda:0', 
                classes=3, batch_size=128, weight_decay=0, n_hid=600, pdrop=0.5, 
                data='usecase', model='linear'):
        super(MC_drop_net, self).__init__()
        cprint('y', ' Creating Net! \t Pdrop = '+str(pdrop))
        self.lr = lr
        self.schedule = None  # [] #[50,200,400,600]
        self.cuda = cuda
        self.channels_in = channels_in
        self.weight_decay = weight_decay
        self.classes = classes
        self.n_hid = n_hid
        self.pdrop = pdrop
        self.batch_size = batch_size
        self.side_in = side_in
        self.create_net(data, model)
        self.create_opt()
        self.epoch = 0
        self.device = device

        self.test = False

    def create_net(self, data='usecase', model='linear'):
        torch.manual_seed(42)
        if self.cuda:
            torch.cuda.manual_seed(42)

        if model == 'resnet34':
            self.model = ResNet34(num_classes=self.classes, input_size=self.side_in, pdrop=self.pdrop)
        elif model == 'resnet50':
            self.model = ResNet50(num_classes=self.classes, input_size=self.side_in, pdrop=self.pdrop)
        elif model == 'vgg16':
            #self.model = ConvNet(in_channels=self.channels_in, out_channels=self.classes, input_size=self.side_in, pdrop=self.pdrop)
            self.model = VGG16(in_channels=self.channels_in, out_channels=self.classes, input_size=self.side_in, pdrop=self.pdrop)
        elif model == 'linear':
            self.model = Linear_2L(in_channels=self.channels_in, out_channels=self.classes, input_size=self.side_in, n_hid=self.n_hid, pdrop=self.pdrop)
        elif model == 'densenet':
            self.model = DenseNet121(num_classes=self.classes, input_size=self.side_in, pdrop=self.pdrop)

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

    def sample_eval(self, x, y, num_classes, Nsamples):
        x, y = x.to(self.device), y.to(self.device)
        batch_size = x.shape[0]
        out = self.model.sample_predict(x, Nsamples) # (Nsamples, batch_size, num_classes)
        mean_out = out.mean(dim=0, keepdim=False) # (batch_size, num_classes)
        probs = F.softmax(mean_out, dim=1).data

        # mean entropy for MI
        single_probs = F.softmax(out, dim=2).data
        single_entropies = entropy(single_probs.detach().cpu().numpy(), axis=2)
        single_entropies = torch.from_numpy(single_entropies)
        mean_entropy = single_entropies.mean(dim=0, keepdim=False) #out: 1 mean entropy value for given data instance

        # classification error
        # get the indexes of the max log-probability
        pred = probs.max(dim=1, keepdim=False)[1]  # (batch, )
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

    def all_sample_eval(self, x, y, Nsamples):
        x, y = x.to(self.device), y.to(self.device)

        out = self.model.sample_predict(x, Nsamples)

        prob_out = F.softmax(out, dim=2)
        prob_out = prob_out.data

        return prob_out

    def get_weight_samples(self):
        weight_vec = []

        state_dict = self.model.state_dict()

        for key in state_dict.keys():

            if 'weight' in key:
                weight_mtx = state_dict[key].cpu().data
                for weight in weight_mtx.view(-1):
                    weight_vec.append(weight)

        return np.array(weight_vec)

#---------------------------------------------------------------------------------------------------------------------
class Linear_2L(nn.Module):
    def __init__(self, in_channels, out_channels=3, input_size=256, n_hid=100, pdrop=0.5):
        super(Linear_2L, self).__init__()

        self.pdrop = pdrop

        input_dim = in_channels * input_size * input_size
        self.input_dim = input_dim
        self.output_dim = out_channels
        print("Output dim: ", self.output_dim)
        print("Out_Channels: ", out_channels)

        self.fc1 = nn.Linear(input_dim, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, n_hid)
        self.fc4 = nn.Linear(n_hid, n_hid)
        self.fc5 = nn.Linear(n_hid, out_channels)
        print("Last layer: ", self.fc5)
        self.bn = nn.BatchNorm1d(n_hid) 
        self.act = nn.ReLU(inplace=True)


    def forward(self, x, sample=True):
        mask = self.training or sample  # if training or sampling, mc dropout will apply random binary mask
        # Otherwise, for regular test set evaluation, we can just scale activations

        x = x.view(-1, self.input_dim)  # view(batch_size, input_dim)
        # -----------------
        x = self.fc1(x)
        x = MC_dropout(x, p=self.pdrop, mask=mask)
        x = self.bn(x)
        x = self.act(x)
        # -----------------
        x = self.fc2(x)
        x = MC_dropout(x, p=self.pdrop, mask=mask)
        x = self.bn(x)
        x = self.act(x)
        # -----------------
        #x = self.fc3(x)
        #x = MC_dropout(x, p=self.pdrop, mask=mask)
        #x = self.bn(x)
        #x = self.act(x)
        # -----------------
        #x = self.fc4(x)
        #x = MC_dropout(x, p=self.pdrop, mask=mask)
        #x = self.bn(x)
        #x = self.act(x)
        # -----------------
        y = self.fc5(x)
        return y

    def sample_predict(self, x, Nsamples):
        # Just copies type from x, initializes new vector
        predictions = x.data.new(Nsamples, x.shape[0], self.output_dim)
        for i in range(Nsamples):
            y = self.forward(x, sample=True)
            predictions[i] = y
        return predictions # (Nsamples, batch_size, num_classes)

#-------------------------------------------------------------------------------------------------------------------

class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, input_size, pdrop=0.5):
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
        self.pdrop = pdrop

    def forward(self, x, sample=True):
        mask = self.training or sample  # if training or sampling, mc dropout will apply random binary mask

        x = self.conv1(x)
        x = F.relu(x)
        x = MC_dropout(x, p=self.pdrop, mask=mask, inplace=False)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = MC_dropout(x, p=self.pdrop, mask=mask)
        x = F.relu(x)
        x = self.fc2(x)
        x = MC_dropout(x, p=self.pdrop, mask=mask)
        x = F.relu(x)
        x = self.fc3(x)
        return x
    
    def sample_predict(self, x, Nsamples):
        # Just copies type from x, initializes new vector
        logits = x.data.new(Nsamples, x.shape[0], self.out_channels)

        for i in range(Nsamples):
            y = self.forward(x, sample=True)
            logits[i] = y

        return logits

#-------------------------------------------------------------------------------------------------------------------

class VGG16(nn.Module):
    def __init__(self, in_channels, out_channels=3, input_size=256, pdrop=0.5):
        super(VGG16, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.input_size = input_size
        self.pdrop = pdrop
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
        else:
            flat_size = int(input_size/32)
            pool_size = 2
            fclayer = flat_size * flat_size * 512
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
        
    #add MCDropout after each block that does NOT end with a MaxPool    
    def forward(self, x, sample=True):
        mask = self.training or sample  # if training or sampling, mc dropout will apply random binary mask
        # Otherwise, for regular test set evaluation, we can just scale activations

        out = self.layer1(x)
        out = MC_dropout(out, p=self.pdrop, mask=mask, inplace=False)
        out = self.layer2(out)
        out = self.layer3(out)
        out = MC_dropout(out, p=self.pdrop, mask=mask, inplace=False)
        out = self.layer4(out)
        out = self.layer5(out)
        #out = MC_dropout(out, p=self.pdrop, mask=mask, inplace=False)
        out = self.layer6(out)
        out = MC_dropout(out, p=self.pdrop, mask=mask, inplace=False)
        out = self.layer7(out)
        out = self.layer8(out)
        #out = MC_dropout(out, p=self.pdrop, mask=mask, inplace=False)
        out = self.layer9(out)
        out = MC_dropout(out, p=self.pdrop, mask=mask, inplace=False)
        out = self.layer10(out)
        out = self.layer11(out)
        #out = MC_dropout(out, p=self.pdrop, mask=mask, inplace=False)
        out = self.layer12(out)
        out = MC_dropout(out, p=self.pdrop, mask=mask, inplace=False)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = MC_dropout(out, p=self.pdrop, mask=mask, inplace=False)
        out = self.fc1(out)
        #out = MC_dropout(out, p=self.pdrop, mask=mask, inplace=False)
        out = self.fc2(out)
        return out

    def sample_predict(self, x, Nsamples):
        # Just copies type from x, initializes new vector
        logits = x.data.new(Nsamples, x.shape[0], self.out_channels)

        for i in range(Nsamples):
            y = self.forward(x, sample=True)
            logits[i] = y

        return logits