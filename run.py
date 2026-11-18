from __future__ import print_function, division
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import torch
from itertools import chain
import contrastive_loss
import torch_clustering
import os
import logging
import random
from Network import Network
import torch.nn.functional as F
from clusteringPerformance import clusteringMetrics
from Dataloader import MultiViewDatasetLoader
from torch.utils.data import DataLoader
from loss import RateDistortionUnconstrained, AlhpaDivergenceLoss, target_distribution
from sklearn.cluster import KMeans

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Configure logging
def setup_logging(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s: %(message)s'))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def plot_TSNET(args, z, label, num_cluster, epoch):
    tsne = TSNE(n_components=2, random_state=10, learning_rate=200)
    # label = label.cpu().detach().numpy()
    z_embedded = tsne.fit_transform(z.cpu().detach().numpy())
    plt.figure()
    colors = [
        '#1f77b4', '#2ca02c', '#9467bd',
        '#e377c2', '#bcbd22',
        '#aec7e8', '#98df8a', '#c5b0d5',
        '#f7b6d2', '#dbdb8d',
        '#c7e9c0', '#8c6d31',
        '#e7ba52', '#ce6dbd',
        '#3182bd', '#31a354', '#636363',
        '#dadaeb', '#fccde5',
        '#ff7f0e', '#d62728', '#8c564b', '#7f7f7f', '#17becf', '#ffbb78', '#ff9896', '#c49c94', '#c7c7c7',
        '#9edae5', '#fddaec', '#ad494a', '#6b4c9a', '#a55194', '#de9ed6', '#e6550d', '#756bb1', '#9e9ac8',
        '#bdbdbd'
    ]
    for j in range(num_cluster):
        plt.scatter(z_embedded[label == j, 0], z_embedded[label == j, 1], s=3, color=colors[j % len(colors)])
    plt.axis('off')
    plt.savefig('./visual/{}/{}.pdf'.format(args.dataset, epoch),
                format='pdf', bbox_inches='tight', dpi=400)


@torch.no_grad()
def _collect_all_features(model,data_loader, device,args):
    model.eval()
    all_consensus_feature = []
    for batch_idx, (data, true_label, idx) in enumerate(data_loader):
        for k in range(args.V):
            data[k] = data[k].to(device)
        out = model(data)
        all_consensus_feature.append(out['consensus_feature'].detach().cpu())
    total_feature = torch.cat(all_consensus_feature,dim=0)
    model.train()
    return total_feature


def update_pseudo_labels(model, data_loader, device, args, num_sample, current_labels=None):
    total_features = _collect_all_features(model, data_loader, device,args)
    if not total_features.is_cuda:
        total_features = total_features.to(device)

    kwargs = {
        'metric': 'cosine',
        'distributed': False,
        'random_state': args.seed,
        'n_clusters': args.num_cluster,
        'verbose': False
    }
    km_torch = torch_clustering.PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
    new_psedo_labels = km_torch.fit_predict(total_features)
    if current_labels is not None:
        label_change_rate = (current_labels.cpu()!=new_psedo_labels.cpu()).sum().item() / num_sample
        print(f'Pseudo-label update done. Change Rate:{label_change_rate*100:.2f}%')
    return new_psedo_labels


def main():
    # Ensure environment variables and other setup
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    use_cuda = torch.cuda.is_available()
    logging.info(f"GPU available: {use_cuda}")

    device = torch.device('cuda:0' if use_cuda else 'cpu')
    # best_scores_kmeans = [0, 0, 0, 0, 0, 0, 0, 0]
    best_epoch = 0
    best_epoch_s = 0
    args = parser.parse_args()

    # Set up logging file
    log_path = f"./result_log/{args.dataset}.log"
    setup_logging(log_path)

    args.cuda = torch.cuda.is_available()
    # logging.info(f"Using CUDA: {args.cuda}")
    args.device = torch.device("cuda" if args.cuda else "cpu")

    set_seed(args.seed)
    dataloader, num_sample, dim_list, y = MultiViewDatasetLoader(args)
    train_loader = DataLoader(dataloader, batch_size=args.batch_size, shuffle=True, drop_last=True)
    if args.dataset == 'YTF10_4-1':
        test_loader = DataLoader(dataloader, batch_size=args.batch_size, shuffle=False)
    else:
        test_loader = DataLoader(dataloader, batch_size=num_sample, shuffle=False)
    args.num_cluster = np.size(np.unique(y))
    V = len(dim_list)
    args.V = V
    print('num_cluster:', args.num_cluster)
    best_scores_kmeans = [0, 0, 0, 0, 0, 0, 0, 0]
    best_scores_sema = [0, 0, 0, 0, 0, 0, 0, 0]

    logging.info(
        'lr:{} seed:{}  alpha_RD:{}  alpha_cluster:{} alpha_div {} warmup_epochs {}'.format( args.lr_train,  args.seed,  args.alpha_rd, args.alpha_cluster, args.alpha_div, args.warmup_epochs))
    network = Network(args, dim_list).to(device)

    rate_distortion_loss = RateDistortionUnconstrained(gam1=1.0,eps=0.01).to(device)


    # criterion_cluster = contrastive_loss.ClusterLoss(args.num_cluster, args.cluster_temperature, device).to(device)
    alpha_div_loss = AlhpaDivergenceLoss(alpha=args.alpha_div).to(device)
    optimizer = torch.optim.Adam(
        chain(network.parameters()),
        lr=args.lr_train,
        weight_decay=0.0000001
    )
    # psedo_labels = update_pseudo_labels(network,test_loader,device, args, num_sample)
    cluster_initialized = False
    for epoch in range(args.con_epochs):
        # if (epoch+1)% args.update_interval ==0 and epoch >0:
        #     psedo_labels = update_pseudo_labels(network,test_loader, device,args,num_sample,psedo_labels)

        loss_all = []
        rec_loss_all = []
        L_RD_all = []
        L_div_all = []
        if epoch == args.warmup_epochs:
            total_feature = _collect_all_features(network, test_loader, device, args)
            km = KMeans(n_clusters=args.num_cluster, n_init=20)
            km.fit(total_feature.cpu().numpy())
            cluster_centers = torch.tensor(km.cluster_centers_, dtype=torch.float32).to(device)
            with torch.no_grad():
                network.cluster[0].weight.copy_(cluster_centers)
                if network.cluster[0].bias is not None:
                    network.cluster[0].bias.fill_(0.0)

        global_P = None
        if epoch>=args.warmup_epochs:
            network.eval()
            with torch.no_grad():
                all_Q = []
                for _, (data_eval, _, _) in enumerate(test_loader):
                    for k in range(args.V):
                        data_eval[k] = data_eval[k].to(device)
                    out_eval = network(data_eval)
                    all_Q.append(out_eval['bia_logits'])
                global_Q = torch.cat(all_Q, dim=0)
                global_P = target_distribution(global_Q)

        network.train()
        for batch_idx, (data, true_label, idx) in enumerate(train_loader):
            for k in range(args.V):
                data[k] = data[k].to(device)

            # batch_pseudo_labels = psedo_labels[idx].to(device)
            optimizer.zero_grad()

            out = network(data)
            L_RD = 0.0
            L_rec = 0.0
            L_div = 0.0

            for v in range(V):
                z_c_low = out['biases_low_dim'][v]
                x_rec = out['x_rec'][v]
                L_rec += F.mse_loss(x_rec, data[v])
                # L_RD + L_Conf
                J_u_v= rate_distortion_loss(z_c_low,device)
                L_RD_v = -J_u_v
                L_RD +=L_RD_v
            if epoch >= args.warmup_epochs:
                bias_logit = out['bia_logits']
                p_target = target_distribution(bias_logit).detach()
                L_div_fused = alpha_div_loss(p_target, bias_logit)
                p_target_batch = global_P[idx].to(device)
                L_div_views = 0.0
                for v in range(V):

                    L_div_views += alpha_div_loss(p_target_batch, out['bias_only_logits'][v])
                L_div = (L_div_fused + L_div_views) / (V + 1)


            # z_both = torch.stack(out['debiased_featrue'], dim=0).mean(dim=0)

            # for v in range(V):
            #     z_debiased = out['debiased_featrue'][v]
            #     bias_only_logit = out['bias_only_logits'][v]
            #     bias_logit = out['bia_logits']
            #     # L_contrast += instance_contrastive_loss(z_both,z_debiased)
            #     L_cluster += criterion_cluster(bias_logit,bias_only_logit)
            # L_div = L_div / (V+1)
            L_total =L_rec  + args.alpha_rd * L_RD  + args.alpha_cluster * L_div
            L_total.backward()
            optimizer.step()
            loss_all.append(L_total.item())
            if epoch >= args.warmup_epochs:
                L_div_all.append(L_div.item())
            else:
                L_div_all.append(L_div)

            L_RD_all.append(L_RD.item())
            # L_Con_all.append(L_contrast.item())
            rec_loss_all.append(L_rec.item())




        print(
            f"Epoch {epoch} loss: {np.mean(loss_all)} Rec_loss: {np.mean(rec_loss_all)} L_RD:{np.mean(L_RD_all)} L_contrast_cluster:{np.mean(L_div_all)}")
        network.eval()
        soft_vector = []
        Zs = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, true_label, idx) in enumerate(test_loader):
                for k in range(args.V):
                    data[k] = data[k].to(device)

                out = network(data)

                # z_both = torch.cat(out['debiased_featrue'],dim=1)
                z_both = torch.stack(out['debiased_featrue'], dim=0).mean(dim=0)

                # 收集标签
                y_true.extend(true_label.numpy())

                # 获取soft labels（预测的类别）
                sem_label = np.argmax(np.array(out['bia_logits'].cpu().detach().numpy()), axis=1)

                # 收集特征和soft labels
                Zs.append(z_both.cpu())  # 将tensor移到CPU并添加到列表
                soft_vector.extend(sem_label)

            # 转换格式
            y_true = np.array(y_true).reshape(num_sample)

            # 处理 Zs：将列表中的tensors拼接成一个大tensor
            Zs = torch.cat(Zs, dim=0)  # shape: [num_sample, feature_dim]

            # 处理 soft_vector：转换为numpy数组并reshape
            soft_vector = np.array(soft_vector).reshape(num_sample)  # shape: [num_sample]
            kwargs = {
                'metric': 'cosine',
                'distributed': False,
                'random_state': args.seed,
                'n_clusters': args.num_cluster,
                'verbose': False
            }
            km_torch = torch_clustering.PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)

            psedo_labels = km_torch.fit_predict(Zs)
            if args.dataset == 'Wiki_fea' or args.dataset =='Mfeat':
                plot_TSNET(args,Zs,psedo_labels,args.num_cluster,epoch)
            ACC, NMI, ARI, Purity, Fscore, Precision, Recall, AMI = clusteringMetrics(y_true,
                                                                                      psedo_labels.cpu().numpy())
            # ACC_s, NMI_s, ARI_s, Purity_s, Fscore_s, Precision_s, Recall_s, AMI_s = clusteringMetrics(y_true,
            #                                                                         soft_vector)
            ACC = np.round(ACC, 4).item()
            NMI = np.round(NMI, 4).item()
            ARI = np.round(ARI, 4).item()
            Purity = np.round(Purity, 4).item()
            Fscore = np.round(Fscore, 4).item()
            Precision = np.round(Precision, 4).item()
            Recall = np.round(Recall, 4).item()
            AMI = np.round(AMI, 4).item()
            scores = [ACC, NMI, ARI, Purity, Fscore, Precision, Recall, AMI]
            ss = dict(
                {'KMean Epoch': epoch, 'ACC': ACC, 'NMI': NMI, 'ARI': ARI, 'Purity': Purity, 'F-score': Fscore,
                 'Precision': Precision,
                 'Recall': Recall, 'AMI': AMI})

            print(ss)
            # ACC_s = np.round(ACC_s, 4).item()
            # NMI_s = np.round(NMI_s, 4).item()
            # ARI_s = np.round(ARI_s, 4).item()
            # Purity_s = np.round(Purity_s, 4).item()
            # Fscore_s = np.round(Fscore_s, 4).item()
            # Precision_s = np.round(Precision_s, 4).item()
            # Recall_s = np.round(Recall_s, 4).item()
            # AMI_s = np.round(AMI_s, 4).item()
            # scores_s = [ACC_s, NMI_s, ARI_s, Purity_s, Fscore_s, Precision_s, Recall_s, AMI_s]
            # sss = dict(
            #     {'Semantic Epoch': epoch, 'ACC': ACC_s, 'NMI': NMI_s, 'ARI': ARI_s, 'Purity': Purity_s, 'F-score': Fscore_s,
            #      'Precision': Precision_s,
            #      'Recall': Recall_s, 'AMI': AMI_s})
            #
            # print(sss)
            print('-'*50)

        # plot_TSNET(args,z_both,psedo_labels.cpu().detach().numpy(),args.num_cluster,epoch)
        if scores[0] + scores[1] + scores[2] > best_scores_kmeans[0] + best_scores_kmeans[1] + \
                best_scores_kmeans[2] and epoch>=args.warmup_epochs:
            best_scores_kmeans = scores
            best_epoch = epoch
        # if scores_s[0] + scores_s[1] + scores_s[2] > best_scores_sema[0] + best_scores_sema[1] + \
        #         best_scores_sema[2]  and epoch>=args.warmup_epochs:
        #     best_scores_sema = scores
        #     best_epoch_s = epoch


    logging.info(
        'Kmeans Epoch: {} ACC = {} NMI = {}   ARI = {},Purity = {}, Fscore = {},Precision = {}, Recall = {}, AMI = {}'.format(
            best_epoch,
            best_scores_kmeans[0], best_scores_kmeans[1], best_scores_kmeans[2],
            best_scores_kmeans[3], best_scores_kmeans[4], best_scores_kmeans[5],
            best_scores_kmeans[6], best_scores_kmeans[7]))
    # logging.info(
    #     'Semantic Epoch: {} ACC = {} NMI = {}   ARI = {},Purity = {}, Fscore = {},Precision = {}, Recall = {}, AMI = {}'.format(
    #         best_epoch_s,
    #         best_scores_sema[0], best_scores_sema[1], best_scores_sema[2],
    #         best_scores_sema[3], best_scores_sema[4], best_scores_sema[5],
    #         best_scores_sema[6], best_scores_sema[7]))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_z', default=128, type=int, help='choose from [32, 64]')
    parser.add_argument('--lr_train', default=0.0005, type=float, help='choose from [0.0001~0.001]')
    parser.add_argument('--batch_size', default=256, type=int, help='choose from [512, 1024, 2048]')  # fix
    # Data
    parser.add_argument('--dataset', default='DHA', type=str,
                        help='choose dataset from 0-Scene15, 1-Reuters, 2-BDGP, 3-RGBD')
    parser.add_argument('--V',default=2,type=int)
    # Train
    parser.add_argument('--pre_epochs', type=int, default=0)
    parser.add_argument('--con_epochs', type=int, default=300)
    parser.add_argument('--temper', type=float, default=0.5)
    parser.add_argument('--cluster_temperature', default=1, type=float)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--num_cluster', type=int, default=10)
    # parser.add_argument('--alpha', type=float, default=0.1)

    parser.add_argument("--bias_dim_factor", type=int, default=4)
    parser.add_argument('--update_interval',type=int, default=5)
    parser.add_argument('--alpha_con', type=float, default=0)
    parser.add_argument('--alpha_rd', type=float, default=1)
    parser.add_argument('--alpha_conf',type=float, default=0.5)
    parser.add_argument('--alpha_cluster',type=float,default=0.00001)
    parser.add_argument('--alpha_div',type=float, default=1.2,help='Alpha parameter for Alpha-Divergence. 1.0=KL, >1.0=Mode-seeking')
    parser.add_argument('--warmup_epochs', type=int, default=150)

    args = parser.parse_args()

    main()
