import openke
from openke.config import Trainer, Tester
from openke.module.model import TransD
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

# dataloader for training
train_dataloader = TrainDataLoader(
    in_path = "./benchmarks/WN18RR/", 
    batch_size = 2000,
    threads = 8,
    sampling_mode = "cross", 
    bern_flag = 0, 
    filter_flag = 1, 
    neg_ent = 64,
    neg_rel = 0
)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/WN18RR/", "link")

# define the model
transd = TransD(
    ent_tot = train_dataloader.get_ent_tot(),
    rel_tot = train_dataloader.get_rel_tot(),
    dim_e = 1024, 
    dim_r = 1024, 
    p_norm = 1, 
    norm_flag = True)


# define the loss function
model = NegativeSampling(
    model = transd, 
    loss = MarginLoss(margin = 4.0),
    batch_size = train_dataloader.get_batch_size()
)

for index in range(10):
    pkg_name = './checkpoint/transd_WN18RR_{}.ckpt'.format(index)
    print(pkg_name)

    # train the model
    trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 3000, alpha = 0.5, use_gpu = True, opt_method = "adam")
    trainer.run()
    transd.save_checkpoint(pkg_name)

    # test the model
    transd.load_checkpoint(pkg_name)
    tester = Tester(model = transd, data_loader = test_dataloader, use_gpu = True)
    tester.run_link_prediction(type_constrain = False)