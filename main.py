from config import cfg, ConfigManager
import torch
from networks.network_factory import NetworkFactory
from networks.GrImNet import GrImNet
from data_loading.data_loader import CrossValidationDataLoader
from training.trainer import Trainer
from predicting.predictor import Predictor
from evaluation.evaluator import Evaluator

def main():
    torch.multiprocessing.set_start_method('spawn')

    manager = ConfigManager(print_to_file = True)
    manager.start_new_training()

    assert cfg['subset_name'] == 'midface' or cfg['subset_name'] == 'mandible', "The value of 'subset_name' in config can only be 'mandible' or 'midface'."
    cv_dataloader = CrossValidationDataLoader(cfg['data_source'], cfg['cv_fold_num'], cfg['batch_size'], num_workers=cfg['cpu_thread'], subset_name=cfg['subset_name'])

    for cv_fold_id in range(cfg['cv_fold_num']):
        if cv_fold_id < 2:
            print("skip fold:", cv_fold_id)
            continue
        train_loader, test_loader = cv_dataloader.get_dataloader_at_fold(cv_fold_id)
        
        if cfg['subset_name'] == 'midface':
            factory = NetworkFactory(network = GrImNet(geo_feat_dim=8, hidden_dim=256, lmk_num=12), device_ids = cfg['gpu']) # midface
        else:
            factory = NetworkFactory(network = GrImNet(geo_feat_dim=8, hidden_dim=256, lmk_num=24), device_ids = cfg['gpu']) # mandible
        model = factory.get_model()

        trainer = Trainer(model = model, dataloader = train_loader, device_ids = cfg['gpu'])        
        trainer.train(epoch_num = cfg['epoch_num'], cp_filename = manager.get_checkpoint_filename_at_fold(cv_fold_id), loss_filename = manager.get_loss_filename_at_fold(cv_fold_id))
        del trainer

        predictor = Predictor(model = model, testloader = test_loader, trainloader=train_loader, device_ids = cfg['gpu'])
        predictor.predict(result_path = '{0:s}/results_cv_{1:d}'.format(manager.test_result_path, cv_fold_id))
        del predictor

        del model
        del factory
        del train_loader, test_loader

    manager.finish_training_or_testing()

    evaluator = Evaluator(input_path = manager.test_result_path, target_path = cfg['data_source'], roi=cfg['subset_name'])
    evaluator.evaluate()

if __name__ == '__main__':
    main()