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

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")

# define the model
transe = TransE(
    ent_tot = train_dataloader.get_ent_tot(),
    rel_tot = train_dataloader.get_rel_tot(),
    dim = 200, 
    p_norm = 1, 
    norm_flag = True)


# define the loss function
model = NegativeSampling(
    model = transe, 
    loss = MarginLoss(margin = 5.0),
    batch_size = train_dataloader.get_batch_size()
)


train_pkg_name = './my_pkg/my_transE{}.ckpt'.format(1)
print(train_pkg_name)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 1.0, use_gpu = True)
trainer.run()
transe.save_checkpoint(train_pkg_name)


# test the model
# test_pkg_name = "./checkpoint/transe_FB15K237_1.ckpt"
test_pkg_name = train_pkg_name
transe.load_checkpoint(test_pkg_name)
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)