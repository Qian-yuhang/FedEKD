import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def calculate_beta(epsilon, m, num_classes, num_users):      # calculate beta
    beta = (np.exp(epsilon / ((num_users - 1) * m)) - 1) / (np.exp(epsilon / ((num_users - 1) * m)) -1 + num_classes)
    return beta

# pl：softmax output of local model
# beta：correctness or RR
# num_classes：num of Classification categories
def LDP(pl, beta, num_classes, args):
    size = len(pl)
    is_all_zeros = torch.all(pl == 0, dim = 1)
    max_indices = torch.argmax(pl, dim = 1)
    x = F.one_hot(max_indices, num_classes=pl.size(1))
    max_indices = torch.where(is_all_zeros, torch.tensor([999] * pl.size(0), dtype = max_indices.dtype), max_indices)
    
    # Draw Bernoulli random variable for each local prediction
    x_c = torch.rand(size)
    x_c = (x_c < beta).unsqueeze(1).int()
    for idx in range(len(max_indices)):
        if(max_indices[idx] == 999):
            x_c[idx][0] = 1
    # Generate random category label for each local prediction 
    n_c = torch.randint(0, num_classes, (size,))
    n_c = F.one_hot(n_c, num_classes)

    # # Perturb
    pl = x_c * x + (1 - x_c) * n_c
    return pl

class KnowledgeBuffer():        # diatillation knowledge cache
    def __init__(self,size):
        self.size = size
        self.index = []
        self.predits = []

    def push(self,index,predits):
        # Store aggergated knowledge in the current round
        self.index.append(index)
        self.predits.append(predits)
        
        # Make sure the buffer contains at most ``size'' pieces of aggergated kwnoeldge
        self.index = self.index[-self.size:]
        self.predits = self.predits[-self.size:]
        
    def fetch(self,):
        # Read knowledge in the buffer
        return self.index,self.predits


def importance_sampling(m_num, net_glob, public_images, args):     # importance_sampling
    net_glob.eval().to(args.device)
    net_glob.eval()
    data_loader = DataLoader(public_images, batch_size = 512)
    coff_list = []
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        with torch.no_grad():
            coff = net_glob(data)    
        coff_list.append(coff.to(args.device))
    coff_all = torch.cat(coff_list, dim = 0)
    coff_soft = F.softmax(coff_all, dim = 1)
    coff_log = torch.log(coff_soft + 1e-08)
    entropy_information = (-coff_soft*coff_log).sum(dim = 1)
    # Sampling probability
    prob = F.softmax(entropy_information, dim=0)
    
    # Sampling knolwedge transfer data
    # selections = torch.multinomial(prob, m_num, replacement = False)

    sorted_indices = torch.argsort(prob, descending = True)
    selections = sorted_indices[:m_num]
    
    return selections

# predictions of local model on data x
def predict(net_glob, public_images, uid, args):
    net_glob.eval().to(args.device)
    data_loader = DataLoader(public_images, batch_size = 10)
    coff_list = []
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args.device), target.to(args.device)
        with torch.no_grad():
            coff = net_glob(data)
        if(idx == uid * args.m_num or idx == uid * args.m_num + 1):
            coff = torch.zeros_like(coff)      
        coff_list.append(coff.cpu())
        # y_pred = coff.data.max(1, keepdim=True)[1]
    coff_all = torch.cat(coff_list, dim = 0)

    coff_soft = F.softmax(coff_all, dim = 1)
    coff_log = torch.log(coff_soft + 1e-08)
    entropy_information = (-coff_soft*coff_log).sum(dim = 1)

    return coff_all, entropy_information                # return output + information entropy

def storage(knowledge_transfer_data, aggregated_predictions, knowledge_buffer):

    knowledge_transfer_data = knowledge_transfer_data
    aggregated_predictions = aggregated_predictions.numpy()

    for x in range(len(knowledge_transfer_data)):
        knowledge_buffer.push(knowledge_transfer_data[x],aggregated_predictions[x])
    Fine_tuning_knowledge_transfer_data,Fine_tuning_aggregated_predictions = knowledge_buffer.fetch()
    
    Fine_tuning_knowledge_transfer_data = torch.tensor(Fine_tuning_knowledge_transfer_data)
    Fine_tuning_aggregated_predictions = np.array(Fine_tuning_aggregated_predictions)
    Fine_tuning_aggregated_predictions = torch.tensor(Fine_tuning_aggregated_predictions)

    return Fine_tuning_knowledge_transfer_data, Fine_tuning_aggregated_predictions