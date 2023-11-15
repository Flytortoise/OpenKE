import openke
import sys
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader



# dataloader for training
train_dataloader = TrainDataLoader(
        in_path = "./benchmarks/FB15K237/",
        nbatches = 100,
        threads = 8,
        sampling_mode = "normal",
        bern_flag = 1,
        filter_flag = 1,
        neg_ent = 25,
        neg_rel = 0)

test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200, 
	p_norm = 1, 
	norm_flag = True)



#for index in range(10):

#    pkg_name = './checkpoint/transe_FB15K237_{}.ckpt'.format(index)
#    print(pkg_name)

    # test the model
#    transe.load_checkpoint(pkg_name)
#    tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
#    tester.run_link_prediction(type_constrain = False)
#    print("***************************")

pkg_name = './my_pkg/my_transE1.ckpt'
print(pkg_name)

    # test the model
transe.load_checkpoint(pkg_name)
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)
print("***************************")
