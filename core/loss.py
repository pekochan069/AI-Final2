import torch
import torch.nn as nn
import torchvision
from torchvision.models.vgg import vgg16, VGG16_Weights, vgg19, VGG19_Weights


class VGGPerceptualLoss(torch.nn.Module):
    # https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
    def __init__(self, resize=False):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(
            torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            .features[:4]
            .eval()
        )
        blocks.append(
            torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            .features[4:9]
            .eval()
        )
        blocks.append(
            torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            .features[9:16]
            .eval()
        )
        blocks.append(
            torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            .features[16:23]
            .eval()
        )
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(
                input, mode="bilinear", size=(224, 224), align_corners=False
            )
            target = self.transform(
                target, mode="bilinear", size=(224, 224), align_corners=False
            )
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


class PerceptualLoss(nn.Module):
    def __init__(self, vgg_coef, adversarial_coef):
        super(PerceptualLoss, self).__init__()
        _vgg19 = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.vgg19 = nn.Sequential(*_vgg19.features).eval()
        for p in self.vgg19.parameters():
            p.requires_grad = False
        self.euclidean_distance = nn.MSELoss()
        self.vgg_coef = vgg_coef
        self.adversarial_coef = adversarial_coef

    def forward(self, sr_img, hr_img, output_labels):
        adversarial_loss = torch.mean(1 - output_labels)
        vgg_loss = self.euclidean_distance(self.vgg19(sr_img), self.vgg19(hr_img))
        pixel_loss = self.euclidean_distance(sr_img, hr_img)
        return (
            pixel_loss
            + self.adversarial_coef * adversarial_loss
            + self.vgg_coef * vgg_loss
        )


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(
            self.loss_network(out_images), self.loss_network(target_images)
        )
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        return (
            image_loss
            + 0.001 * adversarial_loss
            + 0.006 * perception_loss
            + 2e-8 * tv_loss
        )


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]
