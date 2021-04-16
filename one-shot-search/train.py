import os 
import argparse
import torch
from corrupter import BernCorrupter
from read_data import DataLoader
from utils import logger_init, plot_config, gen_struct
from select_gpu import select_gpu
from base_model import BaseModel

from hyperopt_master.hyperopt import fmin, tpe, hp, STATUS_OK, Trials, partial

# TODO

parser = argparse.ArgumentParser(description="Parser for Knowledge Graph Embedding")
parser.add_argument('--task_dir', type=str, default='../KG_Data/FB15K237', help='the directory to dataset')
parser.add_argument('--model', type=str, default='random', help='model type')
parser.add_argument('--save', type=bool, default=False, help='whether save model')
parser.add_argument('--load', type=bool, default=False, help='whether load from pretrain model')
parser.add_argument('--optim', type=str, default='adagrad', help='optimization method')
parser.add_argument('--lamb', type=float, default=0.4, help='set weight decay value')
parser.add_argument('--decay_rate', type=float, default=1.0, help='set weight decay value')
parser.add_argument('--n_dim', type=int, default=256, help='set embedding dimension')
parser.add_argument('--n_sample', type=int, default=25, help='number of negative samples')
parser.add_argument('--cmpl', type=bool, default=False, help='whether use complex value or not')
parser.add_argument('--gpu', type=str, default='1', help='set gpu #')
parser.add_argument('--parrel', type=int, default=1, help='set gpu #')
parser.add_argument('--lr', type=float, default=0.7, help='set learning rate')
parser.add_argument('--n_epoch', type=int, default=4000, help='number of training epochs')
parser.add_argument('--n_batch', type=int, default=4096, help='number of training batches')
parser.add_argument('--epoch_per_test', type=int, default=400, help='frequency of testing')
parser.add_argument('--test_batch_size', type=int, default=50, help='test batch size')
parser.add_argument('--filter', type=bool, default=True, help='whether do filter in testing')
parser.add_argument('--out_file_info', type=str, default='_tune', help='extra string for the output file name')
parser.add_argument('--log_to_file', type=bool, default=False, help='log to file')
parser.add_argument('--log_dir', type=str, default='./log', help='log save dir')

parser.add_argument('--log_prefix', type=str, default='', help='log prefix')

args = parser.parse_args()

dataset = args.task_dir.split('/')[-1]
torch.set_num_threads(6)
#directory = os.path.join('results', args.model)
directory = 'results'
if not os.path.exists(directory):
    os.makedirs(directory)
os.environ["OMP_NUM_THREADS"] = "4"   
os.environ["MKL_NUM_THREADS"] = "4"   
args.out_dir = directory
args.perf_file = os.path.join(directory, dataset + args.out_file_info + '.txt')
print('output file name:', args.perf_file)

# TODO: select which gpu to use
logger_init(args)

task_dir = args.task_dir

loader = DataLoader(task_dir)
n_ent, n_rel = loader.graph_size()

train_data = loader.load_data('train')
valid_data = loader.load_data('valid')
test_data  = loader.load_data('test')
print("Number of train:{}, valid:{}, test:{}.".format(len(train_data[0]), len(valid_data[0]), len(test_data[0])))
n_train = len(train_data[0])
#args.lamb = args.lamb * args.n_batch/n_train

heads, tails = loader.heads_tails()

train_data = [torch.LongTensor(vec) for vec in train_data]
valid_data = [torch.LongTensor(vec) for vec in valid_data]
test_data  = [torch.LongTensor(vec) for vec in test_data]

corrupter = None

def run_kge(params, struct=[0,1,2,3]):
    tester_val = lambda: model.test_link(valid_data, n_ent, heads, tails, args.filter)
    tester_tst = lambda: model.test_link(test_data, n_ent, heads, tails, args.filter)
    torch.cuda.set_device(select_gpu())
    
    args.lr = params["lr"]
    args.lamb = 10**params["lamb"]
    args.decay_rate = params["decay_rate"]
    args.n_batch = params["n_batch"]
    args.n_dim = params["n_dim"]
    plot_config(args)

    model = BaseModel(n_ent, n_rel, args, struct)
    best_mrr, best_str = model.train(train_data, corrupter, tester_val, tester_tst)
    with open(args.perf_file, 'a') as f:
        print('structure:', struct)
        print('write best mrr', best_str)
        for s in struct:
            f.write(str(s)+' ')
        #struc_str = ' '.join(struct) + '\t\t'
        f.write('\t\tbest_performance: ' + best_str + '\n')

    return best_mrr


if __name__ == '__main__':
    space4kge = {
            "lr": hp.uniform("lr", 0, 1),
            "lamb": hp.uniform("lamb", -5, 0),
            "decay_rate": hp.uniform("decay_rate", 0.99, 1.0),
            "n_batch": hp.choice("n_batch", [128, 256, 512, 1024]),
            "n_dim": hp.choice("n_dim", [64]),
    }

    def f(params):
        mrr = run_kge(params, [2,3,0,1])
        return {'loss': -mrr, 'status': STATUS_OK}
    
    trials = Trials()
    best = fmin(f, space4kge, algo=partial(tpe.suggest, n_startup_jobs=25), max_evals=200, trials=trials)
    print('best performance:', best)




