# -*- coding: utf-8 -*-
# @Time : 2019/12/10 16:20
# @Author : Ruiqi Wang

import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from models.ResNet12_embedding import resnet12
from models.classification_heads import ClassificationHead, ScheduledClassificationHead
from models.criterion import protoloss
import models.ClassForge as ClassForge

from utils import str2bool, set_gpu, Timer, count_accuracy, check_dir, log


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)

    return encoded_indicies


def get_model(options):
    # Choose the embedding network
    # if options.network == 'ProtoNet':
    #     network = ProtoNetEmbedding().cuda()
    # elif options.network == 'R2D2':
    #     network = R2D2Embedding().cuda()
    if options.network == 'ResNet':
        if options.dataset == 'miniImageNet' or options.dataset == 'tieredImageNet':
            network = resnet12(avg_pool=options.avg, drop_rate=options.drop_rate, dropblock_size=5).cuda()
            # network = torch.nn.DataParallel(network, device_ids=[0, 1, 2, 3])
            # print(os.environ['CUDA_VISIBLE_DEVICES'])
            # print(list(range(len(options.gpu_ids))))
            network = torch.nn.DataParallel(network, device_ids=list(range(len(options.gpu_ids))))
        else:
            network = resnet12(avg_pool=options.avg, drop_rate=options.drop_rate, dropblock_size=2).cuda()
            network = torch.nn.DataParallel(network, device_ids=list(range(len(options.gpu_ids))))
    else:
        print("Cannot recognize the network type")

    # Choose the classification head
    try:
        if options.scale_const:
            cls_head = ScheduledClassificationHead(base_learner=options.head, enable_scale=options.scale,
                                                   scale=options.scale_const, fn=lambda x: options.scale_const,).cuda()
                                                   # norm=options.norm, power=options.power).cuda()
            print('Use const scale {}.'.format(options.scale_const))
        elif options.scale_schedule:
            cls_head = ScheduledClassificationHead(base_learner=options.head, enable_scale=options.scale, scale=5.0,
                                                   fn=lambda e: 10. if e < 14000 else (20. if e < 16000 else 30. if e < 18000 else (50.)),).cuda()
                                                   # norm=options.norm, power=options.power).cuda()
            print('Use scheduled scale.')
        else:
            cls_head = ClassificationHead(base_learner=options.head, enable_scale=options.scale,).cuda()
                                          # norm=options.norm, power=options.power).cuda()
            print('Use learnable scale')

    except Exception:
        print("Cannot recognize the classification head")

    return (network, cls_head)

def get_dataset(options):
    # Choose the embedding network
    if options.dataset == 'miniImageNet':
        from data.miniImagenet import MiniImageNet, FewShotDataloader
        dataset_train = MiniImageNet(phase='train')
        dataset_val = MiniImageNet(phase='val')
        dataset_test = MiniImageNet(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'tieredImageNet':
        from data.tiered_imagenet import tieredImageNet, FewShotDataloader
        dataset_train = tieredImageNet(phase='train')
        dataset_val = tieredImageNet(phase='val')
        dataset_test = tieredImageNet(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'CIFAR_FS':
        from data.CIFAR_FS import CIFAR_FS, FewShotDataloader
        dataset_train = CIFAR_FS(phase='train')
        dataset_val = CIFAR_FS(phase='val')
        dataset_test = CIFAR_FS(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'FC100':
        from data.FC100 import FC100, FewShotDataloader
        dataset_train = FC100(phase='train')
        dataset_val = FC100(phase='val')
        dataset_test = FC100(phase='test')
        data_loader = FewShotDataloader
    else:
        print("Cannot recognize the dataset type")
        assert (False)

    return (dataset_train, dataset_val, dataset_test, data_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epoch', type=int, default=30,
                        help='number of training epochs')
    parser.add_argument('--save-epoch', type=int, default=1,
                        help='frequency of model saving')
    parser.add_argument('--train-shot', type=int, default=15,
                        help='number of support examples per training class')
    parser.add_argument('--val-shot', type=int, default=5,
                        help='number of support examples per validation class')
    parser.add_argument('--train-query', type=int, default=6,
                        help='number of query examples per training class')
    parser.add_argument('--val-episode', type=int, default=2000,
                        help='number of episodes per validation')
    parser.add_argument('--val-query', type=int, default=15,
                        help='number of query examples per validation class')
    parser.add_argument('--train-way', type=int, default=5,
                        help='number of classes in one training episode')
    parser.add_argument('--new-way', type=int, default=15)
    parser.add_argument('--test-way', type=int, default=5,
                        help='number of classes in one test (or validation) episode')
    parser.add_argument('--save-path', type=str,  default='test')
    # parser.add_argument('--gpu', default='0, 1, 2, 3')
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=[0,1,2,3])
    parser.add_argument('--network', type=str, default='ResNet',
                        help='choose which embedding network to use. ProtoNet, R2D2, ResNet')
    parser.add_argument('--head', type=str, default='ProtoNetHead', choices=['ProtoNetHead', 'DotHead', 'L1ProtoNetHead', 'AttProtoHead', 'SVM'],
                        help='choose which classification head to use. ProtoNet, Ridge, R2D2, SVM')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                        help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS, FC100')
    parser.add_argument('--episodes-per-batch', type=int, default=8,
                        help='number of episodes per batch')
    parser.add_argument('--eps', type=float, default=0.0,
                        help='epsilon of label smoothing')

    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--drop-rate', type=float, default=0.0,
                        help='dropout ratio')
    parser.add_argument('--scale', type=str2bool, default=True,
                        help='scale softmax or not')
    parser.add_argument('--scale-schedule', type=str2bool, default=False,
                        help='use scheduled scale or not')
    parser.add_argument('--scale-const', type=float, default=1.0,
                        help='the const value of scale')
    # parser.add_argument('--smooth', type=str2bool, default=False,
    #                     help='smoothed soft label or not')
    parser.add_argument('--avg', type=str2bool, default=False,
                        help='avgpool or not')
    parser.add_argument('--loss-type', type=str, default='protoloss', choices=['protoloss', 'SVMloss', 'celoss', 'focalloss', 'myloss'],
                        help='which loss to use as criterion')
    # parser.add_argument('--gamma', type=float, default= 0.0)
    # parser.add_argument('--norm', type=str2bool, default='false')
    # parser.add_argument('--power', type=int, default=2)
    parser.add_argument('--CF', type=str2bool, default='True')
    parser.add_argument('--CF-type', type=str, default='CF_H')

    opt = parser.parse_args()

    CF = getattr(ClassForge, opt.CF_type)

    (dataset_train, dataset_val, dataset_test, data_loader) = get_dataset(opt)

    # Dataloader of Gidaris & Komodakis (CVPR 2018)
    dloader_train = data_loader(
        dataset=dataset_train,
        nKnovel=opt.train_way,
        nKbase=0,
        nExemplars=opt.train_shot,  # num training examples per novel category
        nTestNovel=opt.train_way * opt.train_query,  # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=opt.episodes_per_batch,
        num_workers=4,
        epoch_size=opt.episodes_per_batch * 1000,  # num of batches per epoch
    )

    dloader_val = data_loader(
        dataset=dataset_val,
        nKnovel=opt.test_way,
        nKbase=0,
        nExemplars=opt.val_shot,  # num training examples per novel category
        nTestNovel=opt.val_query * opt.test_way,  # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.val_episode,  # num of batches per epoch
    )

    dloader_test = data_loader(
        dataset=dataset_test,
        nKnovel=opt.test_way,
        nKbase=0,
        nExemplars=opt.val_shot,  # num training examples per novel category
        nTestNovel=opt.val_query * opt.test_way,  # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.val_episode,  # num of batches per epoch
    )

    set_gpu(str(opt.gpu_ids)[1:-1])
    seed = opt.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    check_dir('./experiments/')
    opt.save_path = os.path.join('experiments', opt.save_path)
    check_dir(opt.save_path)

    log_file_path = os.path.join(opt.save_path, "train_log.txt")
    log(log_file_path, str(vars(opt)))


    (embedding_net, cls_head) = get_model(opt)
    optimizer = torch.optim.SGD([{'params': embedding_net.parameters()},
                                 {'params': cls_head.parameters()}],
                                lr=opt.lr, momentum=0.9,weight_decay=5e-4, nesterov=True)



    # # lambda_epoch = lambda e: 1.0 if e < 20 else (0.06 if e < 40 else 0.012 if e < 50 else (0.0024))
    # # lambda_epoch = lambda e: 1.0 if e < 15 else (0.1 if e < 20 else 0.012 if e < 25 else (0.0024))
    lambda_epoch = lambda e: 1.0 if e < 14 else (0.1 if e < 16 else 0.01 if e < 18 else (0.001))
    # lambda_epoch = lambda e: 1.0 if e < 15 else (0.06 if e < 20 else 0.012 if e < 25 else (0.0024))

    # # lambda_epoch = lambda e: 1.0 if e < 14 else (0.3 if e < 16 else 0.1 if e < 18 else (0.005))
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch)

    # lambda_epoch = lambda e: 1.0 if e < 20 else (0.06 if e < 40 else 0.012 if e < 50 else (0.0024))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)

    max_val_acc = 0.0

    timer = Timer()
    critertion = globals()[opt.loss_type]
    writer = SummaryWriter(os.path.join(opt.save_path, 'run'))

    for epoch in range(1, opt.num_epoch + 1):
        # Train on the training split
        lr_scheduler.step()

        # Fetch the current epoch's learning rate
        epoch_learning_rate = 0.1
        for param_group in optimizer.param_groups:
            epoch_learning_rate = param_group['lr']

        log(log_file_path, 'Train Epoch: {}\tLearning Rate: {:.4f}'.format(
            epoch, epoch_learning_rate))

        _, _ = [x.train() for x in (embedding_net, cls_head)]

        train_accuracies = []
        train_losses = []

        for i, batch in enumerate(tqdm(dloader_train(epoch)), 1):
            data_support, labels_support, data_query, labels_query, global_label, _ = [x.cuda() for x in batch]

            train_n_support = opt.train_way * opt.train_shot
            train_n_query = opt.train_way * opt.train_query

            if opt.CF:
                data_support, labels_support, data_query, labels_query = CF(data_support, labels_support, data_query, labels_query, opt.train_way, opt.new_way)

            emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            # emb_support = emb_support.reshape(opt.episodes_per_batch, train_n_support, -1)
            # emb_query = emb_query.reshape(opt.episodes_per_batch, train_n_query, -1)


            emb_support = emb_support.reshape(opt.episodes_per_batch, -1, emb_support.size(-1))
            emb_query = emb_query.reshape(opt.episodes_per_batch, -1, emb_query.size(-1))

            if opt.CF:
                train_way = opt.train_way + opt.new_way
                logit_query = cls_head(emb_query, emb_support, labels_support, train_way, opt.train_shot)
                loss = critertion(logit_query, labels_query, train_way, ).view(opt.episodes_per_batch, -1, train_way)
                acc, bacc = count_accuracy(logit_query.reshape(-1, train_way), labels_query.reshape(-1))
            else:

                logit_query = cls_head(emb_query, emb_support, labels_support, opt.train_way, opt.train_shot)
                loss = critertion(logit_query, labels_query, opt.train_way,).view(opt.episodes_per_batch, -1, opt.train_way)
                acc, bacc = count_accuracy(logit_query.reshape(-1, opt.train_way), labels_query.reshape(-1))

            loss = loss.mean()

            train_accuracies.append(acc.item())
            train_losses.append(loss.item())

            if (i % 100 == 0):
                train_acc_avg = np.mean(np.array(train_accuracies))
                log(log_file_path,
                    'Train Epoch: {}\tBatch: [{}/{}]\tLoss: {:.4f}\tAccuracy: {:.2f} % ({:.2f} %)'.format(
                        epoch, i, len(dloader_train), loss.item(), train_acc_avg, acc))


                writer.add_scalar('Acc/train', train_acc_avg, (epoch-1)*1000+i-1)
                time.sleep(1e-5)
                writer.add_scalar('Loss/train', loss.item(), (epoch-1)*1000+i-1)
                if opt.scale:
                    time.sleep(1e-5)
                    if isinstance(cls_head.scale, torch.Tensor):
                        writer.add_scalar('Scale', cls_head.scale.item(), (epoch-1)*1000+i)
                    else:
                        writer.add_scalar('Scale', cls_head.scale, (epoch-1)*1000+i)

                    # writer.add_scalar('Norm_val', cls_head.norm_val, (epoch-1)*1000+i)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # Evaluate on the validation split
        _, _ = [x.eval() for x in (embedding_net, cls_head)]
        with torch.no_grad():
            print('\n=====val=====')
            val_accuracies = []
            val_losses = []

            for i, batch in enumerate(tqdm(dloader_val(epoch)), 1):
                data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]

                test_n_support = opt.test_way * opt.val_shot
                test_n_query = opt.test_way * opt.val_query

                emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
                emb_support = emb_support.reshape(1, test_n_support, -1)

                emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
                emb_query = emb_query.reshape(1, test_n_query, -1)

                logit_query = cls_head(emb_query, emb_support, labels_support, opt.test_way, opt.val_shot)

                vloss = critertion(logit_query, labels_query, opt.test_way).mean()
                vacc, _ = count_accuracy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))

                val_accuracies.append(vacc.item())
                val_losses.append(vloss.item())

            val_acc_avg = np.mean(np.array(val_accuracies))
            val_acc_ci95 = 1.96 * np.std(np.array(val_accuracies)) / np.sqrt(opt.val_episode)

            val_loss_avg = np.mean(np.array(val_losses))

            if val_acc_avg > max_val_acc:
                max_val_acc = val_acc_avg
                torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()}, \
                           os.path.join(opt.save_path, 'best_model.pth'))
                log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} % (Best)' \
                    .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))
            else:
                log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} %' \
                    .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))

            torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()} \
                       , os.path.join(opt.save_path, 'last_epoch.pth'))

            if epoch % opt.save_epoch == 0:
                torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()} \
                           , os.path.join(opt.save_path, 'epoch_{}.pth'.format(epoch)))


            print('=====val=====')
            print('=====test=====')

            test_accuracies = []
            test_losses = []

            for i, batch in enumerate(tqdm(dloader_test(epoch)), 1):
                data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]

                test_n_support = opt.test_way * opt.val_shot
                test_n_query = opt.test_way * opt.val_query

                emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
                emb_support = emb_support.reshape(1, test_n_support, -1)

                emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
                emb_query = emb_query.reshape(1, test_n_query, -1)

                logit_query = cls_head(emb_query, emb_support, labels_support, opt.test_way, opt.val_shot)

                tloss = critertion(logit_query, labels_query, opt.test_way).mean()
                tacc, _ = count_accuracy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))

                test_accuracies.append(tacc.item())
                test_losses.append(tloss.item())

            test_acc_avg = np.mean(np.array(test_accuracies))
            test_acc_ci95 = 1.96 * np.std(np.array(test_accuracies)) / np.sqrt(opt.val_episode)

            test_loss_avg = np.mean(np.array(test_losses))
            log(log_file_path, 'Test Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} %' \
                .format(epoch, test_loss_avg, test_acc_avg, test_acc_ci95))

            writer.add_scalars('Acc/',
                               {'train': acc.item(),
                                'val': val_acc_avg,
                                'test':test_acc_avg}, (epoch-1) * 1000 + i)
            time.sleep(1e-5)
            writer.add_scalars('Loss/',
                               {'train':loss.item(),
                                'val': val_loss_avg,
                                'test': test_loss_avg}, (epoch-1) * 1000 + i)

            print('=====test=====')
            log(log_file_path,
                'Elapsed Time: {}/{}'.format(timer.measure(), timer.measure(epoch / float(opt.num_epoch))))

    writer.close()