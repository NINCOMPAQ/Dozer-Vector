import time
import numpy as np
import torch
import os
from libcity.executor.scheduler import CosineLRScheduler
from ray import tune
from libcity.executor.traffic_state_executor import TrafficStateExecutor
import scipy.sparse as sp
from libcity.utils import reduce_array
from tqdm import tqdm
from fvcore.nn import flop_count


class PDFormerExecutor(TrafficStateExecutor):

    def __init__(self, config, model):
        self.no_load = config.get('no_load', [])
        self.lr_warmup_epoch = config.get("lr_warmup_epoch", 5)
        self.lr_warmup_init = config.get("lr_warmup_init", 1e-6)
        self.lape_dim = config.get('lape_dim', 200)
        self.adj_mx = model.get_data_feature().get('adj_mx')
        self.VF = torch.tensor(model.get_data_feature().get('VF'))
        #print('self.VF is !!!!!!!!!!!', self.VF.shape)
        super().__init__(config, model)
        self.lap_mx = self._cal_lape(self.adj_mx).to(self.device)
        self.random_flip = config.get('random_flip', True)
        self.set_loss = config.get('set_loss', 'masked_mae')
        self.evaluate_count = 0
        self.cal_flops = False

    def check_noload(self, k):
        for no_load_para in self.no_load:
            if no_load_para in k:
                return True
        return False

    def load_model_with_initial_ckpt(self, initial_ckpt):
        assert os.path.exists(initial_ckpt), 'Weights at %s not found' % initial_ckpt
        model_state, optimizer_state = torch.load(initial_ckpt, map_location=torch.device('cpu'))
        model_keys = self.model.state_dict()
        state_dict_load = {}
        unexpect_keys = []
        for k, v in model_state.items():
            if k not in model_keys.keys() or v.shape != model_keys[k].shape or self.check_noload(k):
                unexpect_keys.append(k)
            else:
                state_dict_load[k] = v
        for k, v in model_keys.items():
            if k not in model_state.keys():
                unexpect_keys.append(k)
        self._logger.info("unexpected keys: {}".format(unexpect_keys))
        self.model.load_state_dict(state_dict_load, strict=False)
        self._logger.info("Initialize model from {}".format(initial_ckpt))

    def _calculate_normalized_laplacian(self, adj):
        adj = sp.coo_matrix(adj)
        d = np.array(adj.sum(1))
        isolated_point_num = np.sum(np.where(d, 0, 1))
        self._logger.info(f"Number of isolated points: {isolated_point_num}")
        d_inv_sqrt = np.power(d, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
        return normalized_laplacian, isolated_point_num

    def _calculate_random_walk_laplacian(self, adj):
        adj = sp.coo_matrix(adj)
        d = np.array(adj.sum(1))
        isolated_point_num = np.sum(np.where(d, 0, 1))
        d_inv = np.power(d, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        random_walk_mx = sp.eye(adj.shape[0]) - d_mat_inv.dot(adj).tocoo()
        return random_walk_mx, isolated_point_num

    def _cal_lape(self, adj_mx):
        L, isolated_point_num = self._calculate_normalized_laplacian(adj_mx)
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort()
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

        laplacian_pe = torch.from_numpy(EigVec[:, isolated_point_num + 1: self.lape_dim + isolated_point_num + 1]).float()
        laplacian_pe.require_grad = False
        return laplacian_pe

    def _build_optimizer(self):
        self._logger.info('You select `{}` optimizer.'.format(self.learner.lower()))
        if self.learner.lower() == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                         eps=self.lr_epsilon, betas=self.lr_betas, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                        momentum=self.lr_momentum, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate,
                                            eps=self.lr_epsilon, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate,
                                            alpha=self.lr_alpha, eps=self.lr_epsilon,
                                            momentum=self.lr_momentum, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sparse_adam':
            optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=self.learning_rate,
                                               eps=self.lr_epsilon, betas=self.lr_betas)
        elif self.learner.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                          eps=self.lr_epsilon, betas=self.lr_betas, weight_decay=self.weight_decay)
        else:
            self._logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                         eps=self.lr_epsilon, weight_decay=self.weight_decay)
        return optimizer

    def _build_lr_scheduler(self):
        if self.lr_decay:
            self._logger.info('You select `{}` lr_scheduler.'.format(self.lr_scheduler_type.lower()))
            if self.lr_scheduler_type.lower() == 'multisteplr':
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer, milestones=self.milestones, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'steplr':
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=self.step_size, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'exponentiallr':
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'cosineannealinglr':
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.lr_T_max, eta_min=self.lr_eta_min)
            elif self.lr_scheduler_type.lower() == 'lambdalr':
                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, lr_lambda=self.lr_lambda)
            elif self.lr_scheduler_type.lower() == 'reducelronplateau':
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', patience=self.lr_patience,
                    factor=self.lr_decay_ratio, threshold=self.lr_threshold)
            elif self.lr_scheduler_type.lower() == 'cosinelr':
                lr_scheduler = CosineLRScheduler(
                    self.optimizer, t_initial=self.epochs, lr_min=self.lr_eta_min, decay_rate=self.lr_decay_ratio,
                    warmup_t=self.lr_warmup_epoch, warmup_lr_init=self.lr_warmup_init)
            else:
                self._logger.warning('Received unrecognized lr_scheduler, '
                                     'please check the parameter `lr_scheduler`.')
                lr_scheduler = None
        else:
            lr_scheduler = None
        return lr_scheduler


    def train(self, train_dataloader, eval_dataloader):
        self._logger.info('Start training ...')
        min_val_loss = float('inf')
        wait = 0
        best_epoch = 0
        train_time = []
        eval_time = []
        num_batches = len(train_dataloader)
        self._logger.info("num_batches:{}".format(num_batches))

        batches_seen = num_batches * self._epoch_num
        for epoch_idx in range(self._epoch_num, self.epochs):
            start_time = time.time()
            losses, batches_seen = self._train_epoch(train_dataloader, epoch_idx, batches_seen, self.loss_func)
            t1 = time.time()
            train_time.append(t1 - start_time)
            train_loss = np.mean(losses)
            if self.distributed:
                train_loss = reduce_array(train_loss, self.world_size, self.device)
            self._writer.add_scalar('training loss', train_loss, batches_seen)
            self._logger.info("epoch complete!")

            self._logger.info("evaluating now!")
            t2 = time.time()
            val_loss = self._valid_epoch(eval_dataloader, epoch_idx, batches_seen, self.loss_func)
            end_time = time.time()
            eval_time.append(end_time - t2)

            epoch_time = end_time - start_time
            if self.distributed:
                epoch_time = reduce_array(np.array(epoch_time), self.world_size, self.device)

            if self.lr_scheduler is not None:
                if self.lr_scheduler_type.lower() == 'reducelronplateau':
                    self.lr_scheduler.step(val_loss)
                elif self.lr_scheduler_type.lower() == 'cosinelr':
                    self.lr_scheduler.step(epoch_idx + 1)
                else:
                    self.lr_scheduler.step()

            if (epoch_idx % self.log_every) == 0:
                log_lr = self.optimizer.param_groups[0]['lr']
                message = 'Epoch [{}/{}] ({}) train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.2f}s'. \
                    format(epoch_idx, self.epochs, batches_seen, train_loss, val_loss, log_lr, epoch_time)
                self._logger.info(message)

            if self.hyper_tune:
                with tune.checkpoint_dir(step=epoch_idx) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    self.save_model(path)
                tune.report(loss=val_loss)

            if val_loss < min_val_loss:
                wait = 0
                if self.saved:
                    model_file_name = self.save_model_with_epoch(epoch_idx)
                    self._logger.info('Val loss decrease from {:.4f} to {:.4f}, '
                                      'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss
                best_epoch = epoch_idx
            else:
                wait += 1
                if wait == self.patience and self.use_early_stop:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_idx)
                    break
        if len(train_time) > 0:
            average_train_time = sum(train_time) / len(train_time)
            average_eval_time = sum(eval_time) / len(eval_time)
            if self.distributed:
                average_train_time = reduce_array(average_train_time, self.world_size, self.device)
                average_eval_time = reduce_array(average_eval_time, self.world_size, self.device)
            self._logger.info('Trained totally {} epochs, average train time is {:.3f}s, '
                              'average eval time is {:.3f}s'.
                              format(len(train_time), average_train_time, average_eval_time))
        if self.load_best_epoch:
            self.load_model_with_epoch(best_epoch)
        return min_val_loss

    def _train_epoch(self, train_dataloader, epoch_idx, batches_seen=None, loss_func=None):
        self.model.train()
        if loss_func is None:
            if self.distributed:
                loss_func = self.model.module.calculate_loss_without_predict
            else:
                loss_func = self.model.calculate_loss_without_predict
        losses = []
        
        for batch in train_dataloader:
            batch.to_tensor(self.device)
            batch_lap_pos_enc = self.lap_mx.to(self.device)
            batch_VF = self.VF.to(self.device)
            if self.random_flip:
                sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(self.device)
                sign_flip[sign_flip >= 0.5] = 1.0
                sign_flip[sign_flip < 0.5] = -1.0
                batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)

            if self.evaluate_count==0 and self.cal_flops:
                self.evaluate_count = 1
                print('evaluating FLOPS')
                Cal_FLOPs(model = self.model,inputs = (batch['X'],batch['y'], batch_lap_pos_enc, batch_VF))                 


            y_true = batch['y']
            y_predicted = self.model(batch['X'],batch['y'], batch_lap_pos_enc, batch_VF)
            

            loss = loss_func(y_true, y_predicted, batches_seen=batches_seen, set_loss=self.set_loss)
            self._logger.debug(loss.item())
            losses.append(loss.item())
            batches_seen += 1
            loss = loss / self.grad_accmu_steps
            loss.backward()
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            if batches_seen % self.grad_accmu_steps == 0:
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    if self.lr_scheduler_type.lower() == 'cosinelr':
                        self.lr_scheduler.step_update(num_updates=batches_seen)
                self.optimizer.zero_grad()       
        return losses, batches_seen

    def _valid_epoch(self, eval_dataloader, epoch_idx, batches_seen=None, loss_func=None):
        with torch.no_grad():
            self.model.eval()
            if loss_func is None:
                if self.distributed:
                    loss_func = self.model.module.calculate_loss_without_predict
                else:
                    loss_func = self.model.calculate_loss_without_predict
            losses = []
            for batch in eval_dataloader:
                batch.to_tensor(self.device)
                y_true = batch['y']
                y_predicted = self.model(batch['X'], batch['y'], self.lap_mx, self.VF)
                loss = loss_func(y_true, y_predicted, batches_seen=batches_seen, set_loss=self.set_loss)
                self._logger.debug(loss.item())
                losses.append(loss.item())
            mean_loss = np.mean(losses)
            if self.distributed:
                mean_loss = reduce_array(mean_loss, self.world_size, self.device)
            self._writer.add_scalar('eval loss', mean_loss, batches_seen)
            return mean_loss

    def evaluate(self, test_dataloader):
        self._logger.info('Start evaluating ...')
        with torch.no_grad():
            self.model.eval()
            y_truths = []
            y_preds = []
            for batch in test_dataloader:
                batch.to_tensor(self.device)
                output = self.model.predict(batch['X'], batch['y'], lap_mx=self.lap_mx, VF=self.VF)
                y_true = self._scaler.inverse_transform(batch['y'][..., :self.output_dim])
                y_pred = self._scaler.inverse_transform(output[..., :self.output_dim])
                y_truths.append(y_true.cpu().numpy())
                y_preds.append(y_pred.cpu().numpy())
            y_preds = np.concatenate(y_preds, axis=0)
            y_truths = np.concatenate(y_truths, axis=0)
            outputs = {'prediction': y_preds, 'truth': y_truths}
            filename = \
                time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) + '_' \
                + self.config['model'] + '_' + self.config['dataset'] + '_predictions.npz'
            np.savez_compressed(os.path.join(self.evaluate_res_dir, filename), **outputs)
            self.evaluator.clear()
            self.evaluator.collect({'y_true': torch.tensor(y_truths), 'y_pred': torch.tensor(y_preds)})
            test_result = self.evaluator.save_result(self.evaluate_res_dir)
            return test_result
        

def Cal_FLOPs(model, inputs):

    model = model.cuda()

    # print(model)


    flops, unsupported_operations = flop_count(model, inputs)

    gflops = sum(flops.values())

    n_param = sum([p.nelement() for p in model.parameters()])

    print(f'GMac:{gflops}')

    print(f'Params:{n_param}')
    print(flops)
