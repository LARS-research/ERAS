import os 
import argparse
import torch
from read_data import DataLoader
from utils import logger_init, plot_config, gen_struct, default_search_hyper, record, read_struct_scu
from select_gpu import select_gpu
from base_model import BaseModel


from hyperopt_master.hyperopt import fmin, tpe, hp, STATUS_OK, Trials, partial


"""
Build Default Arguments
"""
def register_default_args():
    parser = argparse.ArgumentParser(description="Parser for Knowledge Graph Embedding")
    parser.add_argument('--task_dir', type=str, default='/export/data/sdiaa/KG_Data/umls', help='the directory to dataset')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--model', type=str, default='random', help='model type')
    parser.add_argument('--save', type=bool, default=False, help='whether save model')
    parser.add_argument('--load', type=bool, default=False, help='whether load from pretrain model')
    parser.add_argument('--n_sample', type=int, default=25, help='number of negative samples')
    parser.add_argument('--classification', type=bool, default=False, help='number of negative samples')
    parser.add_argument('--cmpl', type=bool, default=False, help='whether use complex value or not')
    parser.add_argument('--parrel', type=int, default=1, help='set gpu #')
              
    # epoch and batch
    parser.add_argument('--n_batch', type=int, default=4096, help='number of training batches')
    parser.add_argument('--n_oas_epoch', type=int, default=5, help='')
    parser.add_argument('--n_stand_epoch', type=int, default=5, help='')
    
    # hyper-parameters related to embeddings
    parser.add_argument('--optim', type=str, default='adagrad', help='optimizer for embedding')
    parser.add_argument('--lr', type=float, default=0.7, help='set learning rate')
    parser.add_argument('--lamb', type=float, default=0.4, help='set weight decay value')
    parser.add_argument('--decay_rate', type=float, default=1.0, help='set weight decay value')
    parser.add_argument('--n_dim', type=int, default=256, help='set embedding dimension')

    # hyper-parameters related to embeddings
    parser.add_argument('--controller_optim', type=str, default='adam', help='optimizer for controller')
    parser.add_argument('--n_controller_epoch', type=int, default=20, help='step for controller parameters')
    parser.add_argument('--n_derive_sample', type=int, default=2, help='')
       
    parser.add_argument('--epoch_per_test', type=int, default=10, help='frequency of testing')
    parser.add_argument('--filter', type=bool, default=True, help='whether do filter in testing')
    parser.add_argument('--out_file_info', type=str, default='_tune', help='extra string for the output file name')
    parser.add_argument('--log_to_file', type=bool, default=False, help='log to file')
    parser.add_argument('--log_dir', type=str, default='./log', help='log save dir')
    parser.add_argument('--log_prefix', type=str, default='', help='log prefix')

    return parser

"""
main function
"""
def main(args, struct, rela, reward, params=None):

    dataset = args.dataset
    m, n = args.m, args.n
    cluster_way = args.clu
    trial = args.trial
    
    # set number of threads in pytorch
    torch.set_num_threads(6)
    
    # select which gpu to use
    logger_init(args)
    if args.GPU:
        torch.cuda.set_device(args.gpu) 
    
    # load data
    task_dir = args.task_dir
    loader = DataLoader(task_dir)
    n_ent, n_rel = loader.graph_size()
    train_data = loader.load_data('train')
    valid_data = loader.load_data('valid')
    test_data  = loader.load_data('test')
    print("Number of train:{}, valid:{}, test:{}.".format(len(train_data[0]), len(valid_data[0]), len(test_data[0])))
    
    heads, tails = loader.heads_tails()
    train_data = [torch.LongTensor(vec) for vec in train_data]
    valid_data = [torch.LongTensor(vec) for vec in valid_data]
    test_data  = [torch.LongTensor(vec) for vec in test_data]
    
    # the default settings for correspdonding dataset
    args, rela_cluster = default_search_hyper(args)
       
    if args.classification:
        valid_trip_pos, valid_trip_neg = loader.load_triplets('valid')
        test_trip_pos,  test_trip_neg  = loader.load_triplets('test')
        valid_trip_pos = [torch.LongTensor(vec).cuda() for vec in  valid_trip_pos]
        valid_trip_neg = [torch.LongTensor(vec).cuda() for vec in  valid_trip_neg]
        test_trip_pos = [torch.LongTensor(vec).cuda() for vec in  test_trip_pos]
        test_trip_neg = [torch.LongTensor(vec).cuda() for vec in  test_trip_neg]
    else:
        tester_trip_class = None
    
    if tunning:
        if cluster_way == "pde":
            file_path = "tune_pde" + "_" + str(m) + "_" + str(n)
        elif cluster_way == "scu":
            file_path = "tune_scu"  + "_" + str(m) + "_" + str(n)
        elif cluster_way == "one_clu":
            file_path = "tune_one"  + "_" + str(m) + "_" + str(n)
    else:
        if cluster_way == "pde":
            file_path = "std_pde" + "_" + str(m) + "_" + str(n)
        elif cluster_way == "scu":
            file_path = "std_scu"  + "_" + str(m) + "_" + str(n)
        elif cluster_way == "one_clu":
            file_path = "std_one"  + "_" + str(m) + "_" + str(n)
            
    directory = os.path.join("results", dataset, file_path, str(trial))
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    os.environ["OMP_NUM_THREADS"] = "4"   
    os.environ["MKL_NUM_THREADS"] = "4"   
    
    args.out_dir = directory
    
    args.perf_file = os.path.join(directory, dataset + '_oas_scu_' + str(m) + "_" + str(n)  + '.txt')

    print('output file name:', args.perf_file)
    
    def tester_val(struct, test, randint):
        return model.test_link(struct=struct, test = test, test_data=valid_data, randint = randint, n_ent=n_ent, heads=heads, tails=tails, filt=args.filter)
    
    def tester_tst(struct, test, randint):
        return model.test_link(struct=struct, test = test, test_data=test_data, randint = randint, n_ent=n_ent, heads=heads, tails=tails, filt=args.filter)
    
    
    if args.classification:
        tester_trip_class = lambda: model.test_trip_class(valid_trip_pos, valid_trip_neg, test_trip_pos, test_trip_neg)
    else:
        tester_trip_class = None
    
    if tunning:
        args.lr = params["lr"]
        args.lamb = 10**params["lamb"]
        args.decay_rate = params["decay_rate"]
        args.n_batch = params["n_batch"]
        args.n_dim = params["n_dim"]

    plot_config(args)
    
    model = BaseModel(n_ent, n_rel, args, rela_cluster, tester_val, tester_tst, tester_trip_class)
    best_mrr = model.train_stand(train_data, valid_data, struct, rela, reward)

    return best_mrr

if __name__ == '__main__':
    
    

    parser = register_default_args()
    
    parser.add_argument('--n', type=int, default=4, help='number of groups')
    parser.add_argument('--m', type=int, default=4, help='number of cluster')    # please note that args.n_dim must can be divided by m
    parser.add_argument('--clu', type=str, default="scu", help='scu or pde')
    
    parser.add_argument('--dataset', type=str, default="umls", help='')
    parser.add_argument('--GPU', type=bool, default=False, help='')
    parser.add_argument('--gpu', type=int, default=1, help='set gpu #')                        
    parser.add_argument('--trial', type=int, default=101, help='')
    args = parser.parse_args()    
    args.task_dir = "../KG_Data/" + args.dataset
    
    structPath = "struct/umls/umls_oas_topK_4_4.txt"
    
    reward_rela_struct_list = read_struct_scu(structPath)
    
    tunning = False
    if tunning:
        
        space4kge = {
        "lr": hp.uniform("lr", 0, 1),
        "lamb": hp.uniform("lamb", -5, 0),
        "decay_rate": hp.uniform("decay_rate", 0.99, 1.0),
        "n_batch": hp.choice("n_batch", [128, 256, 512, 1024]),
        "n_dim": hp.choice("n_dim", [256, 512, 1024]),
        }
             
        reward, rela, struct = reward_rela_struct_list[0][0], reward_rela_struct_list[0][1], reward_rela_struct_list[0][2]
        
        def f(params):
            mrr = main(args, struct, rela, reward, params)
            return {'loss': -mrr, 'status': STATUS_OK}
        
        trials = Trials()
        best = fmin(f, space4kge, algo=partial(tpe.suggest, n_startup_jobs=25), max_evals=5, trials=trials)
        print('best performance:', best)
        
    else:
        reward, rela, struct = reward_rela_struct_list[0][0], reward_rela_struct_list[0][1], reward_rela_struct_list[0][2]
        main(args, struct, rela, reward)
        
    
            
                



