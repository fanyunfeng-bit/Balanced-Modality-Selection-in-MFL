import random
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import OrderedDict, Union, List, Tuple

import numpy as np
import torch
from path import Path
from copy import deepcopy
import torch.nn.functional as F


def fix_random_seed(seed: int) -> None:
    # # torch.cuda.empty_cache()
    # torch.random.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--global_epochs", type=int, default=40)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--local_lr", type=float, default=1e-3)
    parser.add_argument("--lr_min", type=float, default=1e-3)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["CREMAD", "AVE", "CGMNIST", "CrisisMMD"],
        default="CREMAD",
    )

    #
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--log", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--client_num_per_round", type=int, default=2)
    parser.add_argument("--client_num", type=int, default=10)
    parser.add_argument("--save_period", type=int, default=10)
    parser.add_argument("--load", action='store_true')
    parser.add_argument("--fusion_method", type=str, default='concat', choices=['sum', 'concat', 'film', 'gated'])
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--num_frame", type=int, default=2)
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')
    parser.add_argument('--optimizer', default='SGD', type=str)
    parser.add_argument('--fl_method', default='FedAvg', type=str)
    parser.add_argument('--multi_ratio', default=0.5, type=float, help='the proportion of multimodal clients')
    parser.add_argument('--embed_dim', default=512, type=int)
    parser.add_argument("--alpha", type=float, default=5)

    #
    parser.add_argument("--audio_only", action='store_true', help='select when ratio=0')
    parser.add_argument("--visual_only", action='store_true', help='')

    # modality imbalance adjust
    parser.add_argument("--OGM", action='store_true', help='OGM-GE')
    parser.add_argument("--PMR", action='store_true', help='prototypical modal rebalance')

    parser.add_argument('--momentum_coef', default=0.0, type=float, help='momentum_coef')
    parser.add_argument("--modulation_starts", type=int, default=0)
    parser.add_argument("--modulation_ends", type=int, default=70)
    parser.add_argument('--MI_alpha', default=1.0, type=float)

    # fedprox
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--prox", action='store_true', help='')

    # fedmi
    parser.add_argument("--relation_distill", action='store_true')
    parser.add_argument('--temp', default=1.0, type=float)

    # fedcmd
    parser.add_argument('--branch_type', default='linear', type=str, choices=['linear', 'mlp'])
    parser.add_argument("--cross_modal_distill", action='store_true')
    parser.add_argument("--personalized_branch", action='store_true')
    parser.add_argument("--class_wise_w", action='store_true')
    parser.add_argument("--class_wise_t", action='store_true')
    parser.add_argument('--warmup_epoch', default=1, type=int)

    # fedproto
    parser.add_argument("--fedproto", action='store_true', help='')

    # FedMDrop
    parser.add_argument("--drop_thresh", default=2.0, type=float)
    parser.add_argument("--modality_drop", action='store_true')

    # select clients
    parser.add_argument("--grad_epochs", type=int, default=1, help="the epochs for getting gradients in the first "
                                                                   "epoch to construct matrix")
    parser.add_argument("--clientsel_algo", type=str, default='random')
    parser.add_argument("--balansubmod_thresh", default=1.0, type=float)

    # CrisisMMD
    # data processing
    parser.add_argument('--load_size', default=228, type=int)
    parser.add_argument('--crop_size', default=224, type=int)
    parser.add_argument('--max_dataset_size', default=2147483648, type=int)
    parser.add_argument('--dim_img_repr', default=1000, type=int)
    parser.add_argument('--dim_text_repr', default=768, type=int)

    # pow-d
    parser.add_argument('--init_set_client_num', type=int, default=10)

    # AGM
    parser.add_argument('--fusion_type', type=str, default='late_fusion')
    parser.add_argument('--mode', type=str, default='train')

    return parser.parse_args()


# def clone_parameters(
#     src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module]
# ) -> OrderedDict[str, torch.Tensor]:
#     if isinstance(src, OrderedDict):
#         return OrderedDict(
#             {
#                 name: param.clone().detach()
#                 for name, param in src.items()
#             }
#         )
#     if isinstance(src, torch.nn.Module):
#         return OrderedDict(
#             {
#                 name: param.clone().detach()
#                 for name, param in src.state_dict().items()
#             }
#         )


def allocate_client_modality(client_id_indices, multi_ratio, audio_only=False, visual_only=False):
    if not audio_only and not visual_only:
        mm_client = random.sample(client_id_indices, int(multi_ratio * len(client_id_indices)))
        um_client = deepcopy(client_id_indices)
        for cc in mm_client:
            um_client.remove(cc)

        if len(um_client) > 0:
            audio_client = random.sample(um_client, len(um_client) // 2)
            visual_client = deepcopy(um_client)
            for cc in audio_client:
                visual_client.remove(cc)
        else:
            audio_client = []
            visual_client = []
    elif audio_only:
        assert multi_ratio == 0
        mm_client = []
        audio_client = deepcopy(client_id_indices)
        visual_client = []
    elif visual_only:
        assert multi_ratio == 0
        mm_client = []
        audio_client = []
        visual_client = deepcopy(client_id_indices)
    else:
        raise ValueError('error.')
    return mm_client, audio_client, visual_client


def common_loss(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    cost = torch.mean((cov1 - cov2)**2)
    return cost


def trainable_params(
    src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module], requires_name=False
) -> Union[List[torch.Tensor], Tuple[List[str], List[torch.Tensor]]]:
    parameters = []
    keys = []
    if isinstance(src, OrderedDict):
        for name, param in src.items():
            if param.requires_grad:
                parameters.append(param)
                keys.append(name)
    elif isinstance(src, torch.nn.Module):
        for name, param in src.state_dict(keep_vars=True).items():
            if param.requires_grad:
                parameters.append(param)
                keys.append(name)

    if requires_name:
        return keys, parameters
    else:
        return parameters


def calculate_prototype(args, model, dataloader, device, epoch, ratio=0.5, a_proto=None, v_proto=None):
    if args.dataset == 'VGGSound':
        n_classes = 309
    elif args.dataset == 'KineticSound':
        n_classes = 31
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'AVE':
        n_classes = 28
    elif args.dataset == 'CGMNIST':
        n_classes = 10
    elif args.dataset == 'CrisisMMD':
        n_classes = 2
        label_key = 'label'
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    if args.dataset == 'CrisisMMD':
        audio_prototypes = torch.zeros(n_classes, args.dim_img_repr).to(device)
        visual_prototypes = torch.zeros(n_classes, args.dim_text_repr).to(device)
    else:
        audio_prototypes = torch.zeros(n_classes, args.embed_dim).to(device)
        visual_prototypes = torch.zeros(n_classes, args.embed_dim).to(device)
    count_class = [0 for _ in range(n_classes)]

    # calculate prototype
    model.eval()
    with torch.no_grad():
        sample_count = 0
        if args.dataset == 'CrisisMMD':
            for data in dataloader:
                x = (data['image'].to(device),
                     {k: v.to(device) for k, v in data['text_tokens'].items()})
                y = data[label_key].to(device)

                a, v, _ = model(x)  # image, text

                for c, l in enumerate(y):
                    l = l.long()
                    count_class[l] += 1
                    audio_prototypes[l, :] += a[c, :]
                    visual_prototypes[l, :] += v[c, :]
                sample_count += 1
        else:
            for step, (spec, image, label) in enumerate(dataloader):
                if step >= int(len(dataloader) * ratio):
                    break
                spec = spec.to(device)  # B x 257 x 1004
                image = image.to(device)  # B x 3(image count) x 3 x 224 x 224
                label = label.to(device)  # B

                # TODO: make it simpler and easier to extend
                if args.fl_method == 'FedCMD' or args.fl_method == 'FedCMI':
                    a, _, _, v, _, _, _ = model(spec.unsqueeze(1).float(), image.float())
                else:
                    if args.dataset != 'CGMNIST':
                        a, v, out = model(spec.unsqueeze(1).float(), image.float())
                    else:
                        a, v, out = model(spec, image)  # gray colored

                for c, l in enumerate(label):
                    l = l.long()
                    count_class[l] += 1
                    audio_prototypes[l, :] += a[c, :]
                    visual_prototypes[l, :] += v[c, :]

                sample_count += 1

    for c in range(audio_prototypes.shape[0]):
        if count_class[c] == 0:
            if args.dataset == 'CrisisMMD':
                audio_prototypes[c, :] = torch.tensor([1e30 for _ in range(args.dim_img_repr)])
                visual_prototypes[c, :] = torch.tensor([1e30 for _ in range(args.dim_text_repr)])
            else:
                audio_prototypes[c, :] = torch.tensor([1e30 for _ in range(args.embed_dim)])
                visual_prototypes[c, :] = torch.tensor([1e30 for _ in range(args.embed_dim)])
        else:
            audio_prototypes[c, :] /= count_class[c]
            visual_prototypes[c, :] /= count_class[c]

    if epoch <= 0:
        audio_prototypes = audio_prototypes
        visual_prototypes = visual_prototypes
    else:
        audio_prototypes = (1 - args.momentum_coef) * audio_prototypes + args.momentum_coef * a_proto
        visual_prototypes = (1 - args.momentum_coef) * visual_prototypes + args.momentum_coef * v_proto
    return audio_prototypes, visual_prototypes


def EU_dist(x1, x2):
    d_matrix = torch.zeros(x1.shape[0], x2.shape[0]).to(x1.device)
    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            d = torch.sqrt(torch.dot((x1[i] - x2[j]), (x1[i] - x2[j])))
            d_matrix[i, j] = d
    return d_matrix


def dot_product_angle_tensor(v1, v2):
    vector_dot_product = torch.dot(v1, v2)
    arccos = torch.acos(vector_dot_product / (torch.norm(v1, p=2) * torch.norm(v2, p=2)))
    angle = np.degrees(arccos.data.cpu().numpy())
    return arccos, angle


def grad_amplitude_diff(v1, v2):
    len_v1 = torch.norm(v1, p=2)
    len_v2 = torch.norm(v2, p=2)
    return len_v1, len_v2, len_v1 - len_v2


def relation_loss(a, v, audio_proto, visual_proto, label, n_classes, temp, a_detach=True):
    """
    calculate relation distillation loss---cos similarity
    :param a: bsz x embed_dim
    :param v: bsz x embed_dim
    :param audio_proto: n_class x embed_dim
    :param visual_proto: n_class x embed_dim
    :param label: bsz  tensor
    :param class_relation: classes specified
    :param n_classes: the number of classes
    :param a_detach: a is detached or v is.
    :return:
    """

    # instance-wise
    all_classes = [i for i in range(n_classes)]

    a = F.normalize(a, p=2, dim=1)
    v = F.normalize(v, p=2, dim=1)

    if not a_detach:
        audio_f_sim = torch.div(torch.matmul(a, a.T), temp)
        mask = torch.scatter(torch.ones_like(audio_f_sim).to(a.device), 1, torch.arange(audio_f_sim.size(0)).view(-1, 1).to(a.device), 0)
        # audio_max, _ = torch.max(audio_f_sim * mask, dim=1, keepdim=True)
        # audio_f_sim = audio_f_sim - audio_max.detach()
        row_size = audio_f_sim.size(0)
        logits_audio = torch.exp(audio_f_sim[mask.bool()].view(row_size, -1)) / torch.exp(
            audio_f_sim[mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

        v = v.detach()
        visual_f_sim = torch.div(torch.matmul(v, v.T), temp)
        # visual_max, _ = torch.max(visual_f_sim*mask, dim=1, keepdim=True)
        # visual_f_sim = visual_f_sim - visual_max.detach()
        logits_visual = torch.exp(visual_f_sim[mask.bool()].view(row_size, -1)) / torch.exp(
            visual_f_sim[mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)
        loss_distill = (-logits_visual * torch.log(logits_audio)).sum(1).mean()
    else:
        visual_f_sim = torch.div(torch.matmul(v, v.T), temp)
        mask = torch.scatter(torch.ones_like(visual_f_sim).to(a.device), 1, torch.arange(visual_f_sim.size(0)).view(-1, 1).to(a.device), 0)
        # visual_max, _ = torch.max(visual_f_sim * mask, dim=1, keepdim=True)
        # visual_f_sim = visual_f_sim - visual_max.detach()
        row_size = visual_f_sim.size(0)
        logits_visual = torch.exp(visual_f_sim[mask.bool()].view(row_size, -1)) / torch.exp(
            visual_f_sim[mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

        a = a.detach()
        audio_f_sim = torch.div(torch.matmul(a, a.T), temp)
        # audio_max, _ = torch.max(audio_f_sim * mask, dim=1, keepdim=True)
        # audio_f_sim = audio_f_sim - audio_max.detach()
        logits_audio = torch.exp(audio_f_sim[mask.bool()].view(row_size, -1)) / torch.exp(
            audio_f_sim[mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)
        loss_distill = (-logits_audio * torch.log(logits_visual)).sum(1).mean()
    return loss_distill


    # class-wise

    # count = 0
    # all_class = [i for i in range(n_classes)]
    # loss = 0
    # for idx in range(a.size(0)):
    #     if label[idx] in class_relation:
    #         class_other = deepcopy(all_class)
    #         class_other.remove(label[idx])
    #         if a_detach:
    #             cs = 0
    #             for c in class_other:
    #                 cs += torch.abs(F.cosine_similarity(v[idx], visual_proto[c], dim=0)-F.cosine_similarity(audio_proto[label[idx]], audio_proto[c], dim=0))
    #         else:
    #             cs = 0
    #             for c in class_other:
    #                 cs += torch.abs(F.cosine_similarity(a[idx], audio_proto[c], dim=0) - F.cosine_similarity(
    #                     visual_proto[label[idx]], visual_proto[c], dim=0))
    #         count += 1
    #         loss += cs
    # if count == 0:
    #     return 0
    # else:
    #     return loss / count


def evaluate_modality_acc(args, model, dataloader, device, epoch, ratio=0.2, a_proto=None, v_proto=None, r_proto=False):
    audio_proto, visual_proto = calculate_prototype(args, model, dataloader, device, epoch, ratio=ratio, a_proto=None, v_proto=None)

    #
    model.eval()
    num = [0.0 for _ in range(audio_proto.shape[0])]
    acc_a = [0.0 for _ in range(audio_proto.shape[0])]
    acc_v = [0.0 for _ in range(audio_proto.shape[0])]
    with torch.no_grad():
        if args.dataset == 'CrisisMMD':
            for step, data in enumerate(dataloader):
                x = (data['image'].to(device),
                     {k: v.to(device) for k, v in data['text_tokens'].items()})
                label = data['label'].to(device)

                # TODO: make it simpler and easier to extend
                a, v, out = model(x)
                ad_matrix = EU_dist(a, audio_proto)  # bsz x C
                vd_matrix = EU_dist(v, visual_proto)
                for i in range(label.shape[0]):
                    am = np.argmin(ad_matrix[i].cpu().data.numpy())
                    vm = np.argmin(vd_matrix[i].cpu().data.numpy())
                    num[label[i]] += 1.0
                    if np.asarray(label[i].cpu()) == am:
                        acc_a[label[i]] += 1.0
                    if np.asarray(label[i].cpu()) == vm:
                        acc_v[label[i]] += 1.0
        else:
            for step, (spec, image, label) in enumerate(dataloader):
                if step >= int(len(dataloader) * ratio):
                    break
                spec = spec.to(device)  # B x 257 x 1004
                image = image.to(device)  # B x 3(image count) x 3 x 224 x 224
                label = label.to(device)  # B

                # TODO: make it simpler and easier to extend
                if args.fl_method == 'FedCMD' or args.fl_method == 'FedCMI':
                    a, _, _, v, _, _, _ = model(spec.unsqueeze(1).float(), image.float())
                else:
                    if args.dataset != 'CGMNIST':
                        a, v, out = model(spec.unsqueeze(1).float(), image.float())
                    else:
                        a, v, out = model(spec, image)  # gray colored

                ad_matrix = EU_dist(a, audio_proto)  # bsz x C
                vd_matrix = EU_dist(v, visual_proto)
                for i in range(image.shape[0]):
                    am = np.argmin(ad_matrix[i].cpu().data.numpy())
                    vm = np.argmin(vd_matrix[i].cpu().data.numpy())
                    num[label[i]] += 1.0
                    if np.asarray(label[i].cpu()) == am:
                        acc_a[label[i]] += 1.0
                    if np.asarray(label[i].cpu()) == vm:
                        acc_v[label[i]] += 1.0
    # print('step: ', step, len(dataloader))
    if r_proto:
        return sum(acc_a) / sum(num), sum(acc_v) / sum(num), audio_proto, visual_proto
    else:
        return sum(acc_a) / sum(num), sum(acc_v) / sum(num)


def process_grad(grads):
    client_grads = np.asarray(flatten2list(grads))
    return client_grads


def flatten2list(grads):
    gather = np.array([])
    for grad in grads:
        grad = torch.flatten(grad).cpu().data.numpy()
        gather = np.append(gather, grad)
    return gather


def process_param_grad(model, modality='multi', dataset='CREMAD'):
    gather = np.array([])
    if dataset != 'CGMNIST':
        if modality == 'multi':
            for params in model.audio_net.parameters():
                grad = torch.flatten(params.grad).cpu().data.numpy()
                gather = np.append(gather, grad)
            for params in model.visual_net.parameters():
                grad = torch.flatten(params.grad).cpu().data.numpy()
                gather = np.append(gather, grad)
        elif modality == 'audio':
            for params in model.audio_net.parameters():
                grad = torch.flatten(params.grad).cpu().data.numpy()
                gather = np.append(gather, grad)
        elif modality == 'visual':
            for params in model.visual_net.parameters():
                grad = torch.flatten(params.grad).cpu().data.numpy()
                gather = np.append(gather, grad)
    else:
        if modality == 'multi':
            for params in model.gray_net.parameters():
                grad = torch.flatten(params.grad).cpu().data.numpy()
                gather = np.append(gather, grad)
            for params in model.colored_net.parameters():
                grad = torch.flatten(params.grad).cpu().data.numpy()
                gather = np.append(gather, grad)
        elif modality == 'audio':
            for params in model.gray_net.parameters():
                grad = torch.flatten(params.grad).cpu().data.numpy()
                gather = np.append(gather, grad)
        elif modality == 'visual':
            for params in model.colored_net.parameters():
                grad = torch.flatten(params.grad).cpu().data.numpy()
                gather = np.append(gather, grad)
    return gather





