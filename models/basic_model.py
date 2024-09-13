import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import resnet18, resnet34, resnet101
from .fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion
from transformers import BertModel, BertConfig
from torch.nn.modules.dropout import Dropout


class VVClassifier(nn.Module):
    def __init__(self, args):
        super(VVClassifier, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        elif args.dataset == 'UCF':
            n_classes = 101
        elif args.dataset == 'ModelNet40':
            n_classes = 40
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        self.flow_net = resnet18(modality='visual')
        self.visual_net = resnet18(modality='visual')

    def forward(self, flow, visual, B):

        f = self.flow_net(flow)
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1)
        v = torch.flatten(v, 1)

        (_, C, H, W) = f.size()
        f = f.view(B, -1, C, H, W)
        f = f.permute(0, 2, 1, 3, 4)
        f = F.adaptive_avg_pool3d(f, 1)
        f = torch.flatten(f, 1)

        f, v, out = self.fusion_module(f, v)

        return f, v, out

    def forward_audio(self, flow, B=None):

        f = self.flow_net(flow)

        (_, C, H, W) = f.size()
        f = f.view(B, -1, C, H, W)
        f = f.permute(0, 2, 1, 3, 4)
        f = F.adaptive_avg_pool3d(f, 1)
        f = torch.flatten(f, 1)
        out = self.fusion_module.forward_uni_audio(f)
        return f, out

    def forward_visual(self, visual, B=None):

        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1)
        v = torch.flatten(v, 1)
        out = self.fusion_module.forward_uni_visual(v)
        return v, out


class AClassifier(nn.Module):
    def __init__(self, args):
        super(AClassifier, self).__init__()
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        self.net = resnet18(modality='audio')
        self.classifier = nn.Linear(args.embed_dim, n_classes)

    def forward(self, audio):
        a = self.net(audio)
        a = F.adaptive_avg_pool2d(a, 1)
        a = torch.flatten(a, 1)
        out = self.classifier(a)
        return out


class VClassifier(nn.Module):
    def __init__(self, args):
        super(VClassifier, self).__init__()
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        self.net = resnet18(modality='visual')
        self.classifier = nn.Linear(args.embed_dim, n_classes)

    def forward(self, visual, B):
        v = self.net(visual)
        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1)
        v = torch.flatten(v, 1)
        out = self.classifier(v)
        return out


class AVClassifier(nn.Module):
    def __init__(self, args):
        super(AVClassifier, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')

    def forward(self, audio, visual, bsz=None):
        a = self.audio_net(audio)
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        a, v, out = self.fusion_module(a, v)

        return a, v, out

    def forward_audio(self, audio, bsz=None):
        a = self.audio_net(audio)
        a = F.adaptive_avg_pool2d(a, 1)
        a = torch.flatten(a, 1)
        out = self.fusion_module.forward_uni_audio(a)
        return a, out

    def forward_visual(self, visual, bsz=None):
        v = self.visual_net(visual)
        (_, C, H, W) = v.size()
        B = bsz
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1)

        v = torch.flatten(v, 1)
        out = self.fusion_module.forward_uni_visual(v)
        return v, out


class AVLateFusion(nn.Module):
    def __init__(self, args):
        super(AVLateFusion, self).__init__()

        self.args = args

        # fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')

        self.audio_classifier = nn.Linear(args.embed_dim, n_classes)
        self.visual_classifier = nn.Linear(args.embed_dim, n_classes)

    def forward(self, audio, visual, bsz=None):
        a = self.audio_net(audio)
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        a_out = self.audio_classifier(a)
        v_out = self.visual_classifier(a)

        audio_energy = torch.log(torch.sum(torch.exp(a_out), dim=1))
        visual_energy = torch.log(torch.sum(torch.exp(v_out), dim=1))

        a_conf = audio_energy / 10
        v_conf = visual_energy / 10
        a_conf = torch.reshape(a_conf, (-1, 1))
        v_conf = torch.reshape(v_conf, (-1, 1))

        if self.args.df:
            a_v_out = (a_out * a_conf.detach() + v_out * v_conf.detach())
        else:
            a_conf.detach()
            v_conf.detach()
            a_v_out = 0.5 * a_out + 0.5 * v_out

        return a_v_out, a_out, v_out, a_conf, v_conf


class AVClassifier_34(nn.Module):
    def __init__(self, args):
        super(AVClassifier_34, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        self.audio_net = resnet34(modality='audio')
        self.visual_net = resnet34(modality='visual')

    def forward(self, audio, visual):

        a = self.audio_net(audio)
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        # print('concat: ', v.shape)
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        # print('dis: ', v.shape)
        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        a, v, out = self.fusion_module(a, v)

        return a, v, out


class AVClassifier_101(nn.Module):
    def __init__(self, args):
        super(AVClassifier_101, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        self.audio_net = resnet101(modality='audio')
        self.visual_net = resnet101(modality='visual')

    def forward(self, audio, visual):

        a = self.audio_net(audio)
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        # print('concat: ', v.shape)
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        # print('dis: ', v.shape)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)
        # print('avg: ', v.shape)
        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        a, v, out = self.fusion_module(a, v)

        return a, v, out


class CLClassifier(nn.Module):
    def __init__(self, args):
        super(CLClassifier, self).__init__()

        self.fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if self.fusion == 'concat':
            self.fc_out = nn.Linear(args.embed_dim * 2, n_classes)
        elif self.fusion == 'sum':
            self.fc_x = nn.Linear(args.embed_dim, n_classes)
            self.fc_y = nn.Linear(args.embed_dim, n_classes)

    def forward(self, x, y):
        if self.fusion == 'concat':
            output = torch.cat((x, y), dim=1)
            output = self.fc_out(output)
        return output


# Colored-and-gray-MNIST
class convnet(nn.Module):
    def __init__(self, num_classes=10, modal='gray'):
        super(convnet, self).__init__()

        self.modal = modal

        if modal == 'gray':
            in_channel = 1
        elif modal == 'colored':
            in_channel = 3
        else:
            raise ValueError('non exist modal')
        self.bn0 = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(64, 512)

    def forward(self, x):
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.relu(x)  # 28x28
        x = self.maxpool(x)  # 14x14

        x = self.conv2(x)
        x = self.relu(x)  # 14x14
        x = self.conv3(x)
        x = self.relu(x)  # 7x7
        x = self.conv4(x)
        x = self.relu(x)  # 7x7

        feat = x
        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)
        feat = self.fc(feat)

        return feat


class CGClassifier(nn.Module):
    def __init__(self, args):
        super(CGClassifier, self).__init__()

        fusion = args.fusion_method

        n_classes = 10

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        self.gray_net = convnet(modal='gray')
        self.colored_net = convnet(modal='colored')

    def forward(self, gray, colored):
        g = self.gray_net(gray)
        c = self.colored_net(colored)

        g = torch.flatten(g, 1)
        c = torch.flatten(c, 1)

        g, c, out = self.fusion_module(g, c)
        return g, c, out

    def forward_gray(self, gray):
        g = self.gray_net(gray)
        g = torch.flatten(g, 1)

        out = self.fusion_module.forward_uni_audio(g)
        return g, out

    def forward_colored(self, colored):
        c = self.colored_net(colored)
        c = torch.flatten(c, 1)

        out = self.fusion_module.forward_uni_visual(c)
        return c, out


class GrayClassifier(nn.Module):
    def __init__(self, args):
        super(GrayClassifier, self).__init__()
        if args.dataset == 'CGMNIST':
            n_classes = 10

        self.net = convnet(modal='gray')
        self.classifier = nn.Linear(args.embed_dim, n_classes)

    def forward(self, gray):
        g = self.net(gray)
        g = torch.flatten(g, 1)
        g_out = self.classifier(g)
        return g_out


class ColoredClassifier(nn.Module):
    def __init__(self, args):
        super(ColoredClassifier, self).__init__()
        if args.dataset == 'CGMNIST':
            n_classes = 10

        self.net = convnet(modal='colored')
        self.classifier = nn.Linear(args.embed_dim, n_classes)

    def forward(self, color):
        c = self.net(color)
        c = torch.flatten(c, 1)
        c_out = self.classifier(c)
        return c_out


class DecomposedAVClassifier(nn.Module):
    def __init__(self, args):
        super(DecomposedAVClassifier, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')

        if args.branch_type == 'linear':
            self.audio_a_branch = nn.Linear(args.embed_dim, args.embed_dim)
            self.audio_v_branch = nn.Linear(args.embed_dim, args.embed_dim)
            self.visual_v_branch = nn.Linear(args.embed_dim, args.embed_dim)
            self.visual_a_branch = nn.Linear(args.embed_dim, args.embed_dim)
        elif args.branch_type == 'mlp':
            self.audio_a_branch = nn.Sequential(
                nn.Linear(args.embed_dim, args.embed_dim),
                nn.ReLU(),
                nn.Linear(args.embed_dim, args.embed_dim),
                nn.ReLU(),
                nn.Linear(args.embed_dim, args.embed_dim),
                # nn.ReLU(),
                # nn.Linear(args.embed_dim, args.embed_dim),
                # nn.ReLU(),
                # nn.Linear(args.embed_dim, args.embed_dim),
            )
            self.audio_v_branch = nn.Sequential(
                nn.Linear(args.embed_dim, args.embed_dim),
                nn.ReLU(),
                nn.Linear(args.embed_dim, args.embed_dim),
                nn.ReLU(),
                nn.Linear(args.embed_dim, args.embed_dim),
                # nn.ReLU(),
                # nn.Linear(args.embed_dim, args.embed_dim),
                # nn.ReLU(),
                # nn.Linear(args.embed_dim, args.embed_dim),
            )
            self.visual_a_branch = nn.Sequential(
                nn.Linear(args.embed_dim, args.embed_dim),
                nn.ReLU(),
                nn.Linear(args.embed_dim, args.embed_dim),
                nn.ReLU(),
                nn.Linear(args.embed_dim, args.embed_dim),
                # nn.ReLU(),
                # nn.Linear(args.embed_dim, args.embed_dim),
                # nn.ReLU(),
                # nn.Linear(args.embed_dim, args.embed_dim),
            )
            self.visual_v_branch = nn.Sequential(
                nn.Linear(args.embed_dim, args.embed_dim),
                nn.ReLU(),
                nn.Linear(args.embed_dim, args.embed_dim),
                nn.ReLU(),
                nn.Linear(args.embed_dim, args.embed_dim),
                # nn.ReLU(),
                # nn.Linear(args.embed_dim, args.embed_dim),
                # nn.ReLU(),
                # nn.Linear(args.embed_dim, args.embed_dim),
            )
        else:
            raise ValueError('No such branch type.')

        self.av_classifier = nn.Linear(args.embed_dim, n_classes)
        self.va_classifier = nn.Linear(args.embed_dim, n_classes)

    def forward(self, audio, visual, bsz=None):
        a = self.audio_net(audio)
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        aa = self.audio_a_branch(a)
        av = self.audio_v_branch(a)
        vv = self.visual_v_branch(v)
        va = self.visual_a_branch(v)

        av_out = self.av_classifier(torch.cat((aa, av), 0))
        va_out = self.va_classifier(torch.cat((va, vv), 0))

        aa, vv, out = self.fusion_module(aa, vv)

        return aa, av, va, vv, out, av_out, va_out

    def forward_audio(self, audio, bsz=None):
        a = self.audio_net(audio)
        a = F.adaptive_avg_pool2d(a, 1)
        a = torch.flatten(a, 1)

        aa = self.audio_a_branch(a)
        av = self.audio_v_branch(a)

        av_out = self.av_classifier(torch.cat((aa, av), 0))
        _, _, out = self.fusion_module(aa, av)
        return aa, av, out, av_out

    def forward_visual(self, visual, bsz=None):
        v = self.visual_net(visual)
        (_, C, H, W) = v.size()
        B = bsz
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1)
        v = torch.flatten(v, 1)

        vv = self.visual_v_branch(v)
        va = self.visual_a_branch(v)

        va_out = self.va_classifier(torch.cat((va, vv), 0))
        _, _, out = self.fusion_module(va, vv)
        return va, vv, out, va_out


class BaseModel(nn.Module):
    def __init__(self, save_dir):
        super(BaseModel, self).__init__()
        self.save_dir = save_dir

    def save(self, filename):
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(self.save_dir, filename + '.pt'))

    def load(self, filepath):
        state_dict = torch.load(filepath)
        self.load_state_dict(state_dict, strict=False)


class MMModel(BaseModel):
    def __init__(self, imageEncoder, textEncoder, save_dir):
        super(MMModel, self).__init__(save_dir=save_dir)
        self.imageEncoder = imageEncoder
        self.textEncoder = textEncoder

    def forward(self, x):
        raise NotImplemented


class TextOnlyModel(BaseModel):
    def __init__(self, save_dir, dim_text_repr=768, num_class=2):
        super(TextOnlyModel, self).__init__(save_dir)
        config = BertConfig()
        self.dropout = nn.Dropout()

        self.textEncoder = BertModel(
            config).from_pretrained('bert-base-uncased')

        self.linear = nn.Linear(dim_text_repr, num_class)

    def forward(self, x):
        _, text = x

        hidden_states = self.textEncoder(**text)  # N, T, dim_text_repr
        e_i = self.dropout(hidden_states[1])  # N, dim_text_repr

        return self.linear(e_i)


class ImageOnlyModel(BaseModel):
    def __init__(self, save_dir, dim_visual_repr=1000, num_class=2):
        super(ImageOnlyModel, self).__init__(save_dir=save_dir)

        self.imageEncoder = torch.hub.load(
            'pytorch/vision:v0.8.0', 'densenet121', pretrained=False)
        self.flatten_vis = nn.Flatten()
        self.linear = nn.Linear(dim_visual_repr, num_class)
        self.dropout = nn.Dropout()

    def forward(self, x):
        image, _ = x

        f_i = self.dropout(self.flatten_vis(self.imageEncoder(image)))

        return self.linear(f_i)


class DenseNetBertMMModel(MMModel):
    def __init__(self, args, dim_visual_repr=1000, dim_text_repr=768, dim_proj=100, num_class=2, decomposed=False):
        self.args = args
        self.decomposed = decomposed

        self.dim_visual_repr = dim_visual_repr
        self.dim_text_repr = dim_text_repr

        # DenseNet: https://pytorch.org/hub/pytorch_vision_densenet/
        # The authors did not mention which one they used.
        # imageEncoder = torch.hub.load(
        #     'pytorch/vision:v0.8.0', 'densenet121', pretrained=True)
        # imageEncoder = torch.hub.load('pytorch/vision:v0.8.0', 'densenet169', pretrained=True)
        imageEncoder = torch.hub.load(
            'pytorch/vision:v0.8.0', 'densenet121', pretrained=False)
        # imageEncoder= torch.hub.load('pytorch/vision:v0.8.0', 'densenet161', pretrained=True)

        # Bert model: https://huggingface.co/transformers/model_doc/auto.html
        config = BertConfig()
        textEncoder = BertModel(config).from_pretrained('bert-base-uncased')

        super(DenseNetBertMMModel, self).__init__(imageEncoder, textEncoder, save_dir=None)
        self.dropout = Dropout()

        # Flatten image features to 1D array
        self.flatten_vis = torch.nn.Flatten()

        # Linear layers used to project embeddings to fixed dimension (eqn. 3)
        self.proj_visual = nn.Linear(dim_visual_repr, dim_proj)
        self.proj_text = nn.Linear(dim_text_repr, dim_proj)

        self.proj_visual_bn = nn.BatchNorm1d(dim_proj)
        self.proj_text_bn = nn.BatchNorm1d(dim_proj)

        # Linear layers to produce attention masks (eqn. 4)
        self.layer_attn_visual = nn.Linear(dim_visual_repr, dim_proj)
        self.layer_attn_text = nn.Linear(dim_text_repr, dim_proj)

        # An extra fully-connected layer for classification
        # The authors wrote "we add self-attention in the fully-connected networks"
        # Here it is assumed that they mean 'we added a fully-connected layer as self-attention'.
        self.fc_as_self_attn = nn.Linear(2 * dim_proj, 2 * dim_proj)
        self.self_attn_bn = nn.BatchNorm1d(2 * dim_proj)

        # Classification layer
        self.cls_layer = nn.Linear(self.dim_visual_repr + self.dim_text_repr, num_class)

        # self.imageEncoder = nn.Sequential(self.imageEncoder,
        #                                    self.flatten_vis,
        #                                    # self.dropout,
        #                                    # self.proj_visual,
        #                                    # self.proj_visual_bn,
        #                                    )
        # self.textEncoder = nn.Sequential(self.textEncoder,
        #                                   # self.proj_text,
        #                                   # self.proj_text_bn
        #                                   )

        if self.decomposed:
            if args.branch_type == 'linear':
                self.audio_a_branch = nn.Linear(dim_proj, dim_proj)
                self.audio_v_branch = nn.Linear(dim_proj, dim_proj)
                self.visual_v_branch = nn.Linear(dim_proj, dim_proj)
                self.visual_a_branch = nn.Linear(dim_proj, dim_proj)
            elif args.branch_type == 'mlp':
                self.audio_a_branch = nn.Sequential(
                    nn.Linear(dim_proj, dim_proj),
                    nn.ReLU(),
                    nn.Linear(dim_proj, dim_proj),
                    # nn.ReLU(),
                    # nn.Linear(dim_proj, dim_proj),
                    # nn.ReLU(),
                    # nn.Linear(dim_proj, dim_proj),
                    # nn.ReLU(),
                    # nn.Linear(dim_proj, dim_proj),
                )
                self.audio_v_branch = nn.Sequential(
                    nn.Linear(dim_proj, dim_proj),
                    nn.ReLU(),
                    nn.Linear(dim_proj, dim_proj),
                    # nn.ReLU(),
                    # nn.Linear(dim_proj, dim_proj),
                    # nn.ReLU(),
                    # nn.Linear(dim_proj, dim_proj),
                    # nn.ReLU(),
                    # nn.Linear(dim_proj, dim_proj),
                )
                self.visual_a_branch = nn.Sequential(
                    nn.Linear(dim_proj, dim_proj),
                    nn.ReLU(),
                    nn.Linear(dim_proj, dim_proj),
                    # nn.ReLU(),
                    # nn.Linear(dim_proj, dim_proj),
                    # nn.ReLU(),
                    # nn.Linear(dim_proj, dim_proj),
                    # nn.ReLU(),
                    # nn.Linear(dim_proj, dim_proj),
                )
                self.visual_v_branch = nn.Sequential(
                    nn.Linear(dim_proj, dim_proj),
                    nn.ReLU(),
                    nn.Linear(dim_proj, dim_proj),
                    # nn.ReLU(),
                    # nn.Linear(dim_proj, dim_proj),
                    # nn.ReLU(),
                    # nn.Linear(dim_proj, dim_proj),
                    # nn.ReLU(),
                    # nn.Linear(dim_proj, dim_proj),
                )
            else:
                raise ValueError('No such branch type.')

            self.av_classifier = nn.Linear(dim_proj, num_class)
            self.va_classifier = nn.Linear(dim_proj, num_class)

    def forward(self, x):
        image, text = x

        # Getting feature map (eqn. 1)
        # N, dim_visual_repr
        # f_i = self.dropout(self.flatten_vis(self.imageEncoder(image)))

        # Getting sentence representation (eqn. 2)
        hidden_states = self.textEncoder(**text)  # N, T, dim_text_repr
        # The authors used embedding associated with [CLS] to represent the whole sentence
        # e_i = self.dropout(hidden_states[1])  # N, dim_text_repr
        e_i = hidden_states[1]

        # Getting linear projections (eqn. 3)
        f_i = self.imageEncoder(image)
        f_i = self.flatten_vis(f_i)
        # f_i = self.imageEncoder(image)
        # f_i_tilde = F.relu(self.imageEncoder(image))  # N, dim_proj
        # e_i_tilde = F.relu(self.proj_text_bn(self.proj_text(e_i)))  # N, dim_proj

        joint_repr = torch.cat((f_i, e_i), dim=1)  # N, 2*dim_proj
        return f_i, e_i, self.cls_layer(joint_repr)

    def forward_image(self, x):
        image, _ = x
        f_i = self.imageEncoder(image)  # N, dim_proj
        f_i = self.flatten_vis(f_i)

        e_i = torch.zeros(f_i.shape[0], self.dim_text_repr).to(f_i.device)
        joint_repr = torch.cat((f_i, e_i), dim=1)
        return f_i, self.cls_layer(joint_repr)

    def forward_text(self, x):
        _, text = x

        # Getting sentence representation (eqn. 2)
        hidden_states = self.textEncoder(**text)  # N, T, dim_text_repr
        # The authors used embedding associated with [CLS] to represent the whole sentence
        e_i = hidden_states[1]  # N, dim_text_repr
        # e_i_tilde = F.relu(self.proj_text_bn(
        #     self.proj_text(e_i)))  # N, dim_proj

        f_i = torch.zeros(e_i.shape[0], self.dim_visual_repr).to(e_i.device)
        joint_repr = torch.cat((f_i, e_i), dim=1)
        return e_i, self.cls_layer(joint_repr)
