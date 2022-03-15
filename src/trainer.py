import os
import torch
import utils


class Trainer:
    def __init__(self, args, loaderTrain, loaderTest, moduleModel, intStartEpoch=0):
        self.args = args
        self.dev = torch.device('cuda' if args.use_cuda else 'cpu')

        self.loaderTrain = loaderTrain
        self.intMaxStep = self.loaderTrain.__len__()
        self.loaderTest = loaderTest
        self.model = moduleModel.to(self.dev)
        self.intCurrentEpoch = intStartEpoch

        data_name_list = ['MSE', 'Warp', 'MC', 'Bpp', 'BppMP', 'BppM', 'BppRP', 'BppR', 'Aux']
        self.moving_avg_meter = MovingAverageMeter(data_name_list=data_name_list)

        parameters = set(p for n, p in self.model.named_parameters() if not n.endswith(".quantiles"))
        aux_parameters = set(p for n, p in self.model.named_parameters() if n.endswith(".quantiles"))
        
        self.optimizer = utils.make_optimizer(args.optimizer, args.lr, args.weight_decay, parameters)
        self.aux_optimizer = utils.make_optimizer(args.optimizer, args.aux_lr, args.weight_decay, aux_parameters)

        self.scheduler = utils.make_scheduler(args, self.optimizer)
        self.aux_scheduler = utils.make_scheduler(args, self.aux_optimizer)

        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        self.ckpt_dir = args.out_dir + '/checkpoint'
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        self.logfile = open(args.out_dir + '/log.txt', 'w')
        self.model.eval()
        self.loaderTest.Test(self.model, self.intCurrentEpoch, self.logfile)

    def load_state_dict(self):
        intStartEpoch = 0
        if self.args.load is not None:
            checkpoint = torch.load(self.args.load)
            self.model.load(checkpoint['state_dict'])
            # self.optimizer.load(checkpoint['optimizer'])
            # self.aux_optimizer.load(checkpoint['aux_optimizer'])
            # self.scheduler.load(checkpoint['lr_scheduler'])
            # self.aux_scheduler.load(checkpoint['aux_lr_scheduler'])
            intStartEpoch = checkpoint['epoch']
        return intStartEpoch

    def train(self):
        # Train
        self.model.train()
        for intBatchIdx, (tenInput, tenRef) in enumerate(self.loaderTrain):
            intGlobalStep = self.intCurrentEpoch * self.intMaxStep + intBatchIdx
            tenInput = tenInput.to(self.dev)
            tenRef = tenRef.to(self.dev)

            self.optimizer.zero_grad()
            self.aux_optimizer.zero_grad()

            tenLossMSE, tenLossWarp, tenLossMotionCompensation, tenBpp, tenBppMotionPrior, tenBppMotion, tenBppResPrior, tenBppRes = self.model(tenInput, tenRef)
            tenLossDistribution = tenBpp
            if intGlobalStep < 500000:
                fltWarpWeight = 0.1
            else:
                fltWarpWeight = 0
            tenLossDistortion = tenLossMSE + fltWarpWeight * (tenLossWarp + tenLossMotionCompensation)
            tenLossRD = self.args.lmbda * tenLossDistortion + tenLossDistribution

            tenLossRD.backward()

            if self.args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_max_norm)
            self.optimizer.step()

            aux_loss = self.model.model.aux_loss()
            aux_loss.backward()
            self.aux_optimizer.step()

            data_list = [
                tenLossMSE.item(),
                tenLossWarp.item(),
                tenLossMotionCompensation.item(),
                tenBpp.item(),
                tenBppMotionPrior.item(),
                tenBppMotion.item(),
                tenBppResPrior.item(),
                tenBppRes.item(),
                aux_loss.item()
            ]
            self.moving_avg_meter.update(data_list)

            if intBatchIdx % 100 == 0:
                data = self.moving_avg_meter.get_value()
                lr = self.scheduler.get_last_lr()[0]
                
                print('{:<13s}{:<12s}{:<6s}{:<16s}{:<5s}{:<12.8f}{:<6s}{:<16.12f}{:<4s}{:<16.12f}{:<5s}{:<12.8f}{:<6s}{:<12.8f}{:<6s}{:<12.8f}{:<6s}{:<12.8f}{:<6s}{:<12.8f}{:<4s}{:<12.4f}{:<4s}{:<8.6f}'.format(
                    'Train Epoch:', '[' + str(self.intCurrentEpoch) + '/' + str(self.args.epochs) + ']',
                    'Step: ', '[' + str(intBatchIdx) + '/' + str(self.intMaxStep) + ']',
                    'MSE: ', data['MSE'],
                    'Warp: ', data['Warp'],
                    'MC: ', data['MC'],
                    'Bpp: ', data['Bpp'],
                    'BppMP: ', data['BppMP'],
                    'BppM: ', data['BppM'],
                    'BppRP: ', data['BppRP'],
                    'BppR: ', data['BppR'],
                    'Aux:  ', data['Aux'],
                    'lr:  ', lr
                ))

        self.intCurrentEpoch += 1
        self.scheduler.step()
        self.aux_scheduler.step()

    def test(self):
        checkpoint = dict()
        checkpoint['epoch'] = self.intCurrentEpoch
        checkpoint['state_dict'] = self.model.get_state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()
        checkpoint['aux_optimizer'] = self.aux_optimizer.state_dict()
        checkpoint['lr_scheduler'] = self.scheduler.state_dict()
        checkpoint['aux_lr_scheduler'] = self.aux_scheduler.state_dict()
        torch.save(checkpoint, self.ckpt_dir + '/model_epoch' + str(self.intCurrentEpoch).zfill(3) + '.pth')
        self.model.eval()
        self.loaderTest.Test(self.model, self.intCurrentEpoch, self.logfile)
        self.logfile.write('\n')

    def terminate(self):
        # return self.intCurrentEpoch >= self.args.epochs
        return False

    def close(self):
        self.logfile.close()


class MovingAverageMeter(object):
    def __init__(self, gamma=0.98, data_name_list=None):
        self.gamma = gamma
        self.buffer = dict()
        self.data_name_list = None
        if data_name_list is not None:
            self.set_data_name_list(data_name_list)
    
    def set_data_name_list(self, data_name_list):
        self.data_name_list = data_name_list

        for data_name in data_name_list:
            self.buffer[data_name] = 0.0

    def update(self, data_list):
        for data_name, value in zip(self.data_name_list, data_list):
            self.buffer[data_name] = self.buffer[data_name] * self.gamma + value * (1 - self.gamma)

    def get_value(self):
        return self.buffer
