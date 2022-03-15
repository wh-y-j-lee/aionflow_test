import torch
from opticalflow.spynet import Network as Spynet
from models.dvc.GDN import GDN
import math
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional


def make_model(args):
    return Network()


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, output_padding=stride - 1, padding=kernel_size // 2)



class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        out_channel_N = 64
        out_channel_mv = 128
        self.moduleOpticalFlow = Spynet()
        self.moduleMVEncoder = MVEncoder()
        self.moduleMVDecoder = MVDecoder()
        self.moduleMotionCompensation = MotionCompensation()
        self.moduleResEncoder = ResEncoder()
        self.moduleResPriorEncoder = ResPriorEncoder()
        self.moduleResDecoder = ResDecoder()
        self.moduleResPriorDecoder = ResPriorDecoder()
        self.moduleMotionEntropy = EntropyBottleneck(out_channel_mv)
        self.moduleResEntropy = EntropyBottleneck(out_channel_N)
        self.moduleResPriorGaussian = GaussianConditional(None)
        self.mxrange = 150

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss
    
    def update(self, force=False):
        """Updates the entropy bottleneck(s) CDF values.
        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.
        Args:
            force (bool): overwrite previous values (default: False)
        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.
        """
        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def load_state_dict(self, state_dict):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.moduleMotionEntropy,
            "moduleMotionEntropy",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.moduleResEntropy,
            "moduleResEntropy",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.moduleResPriorGaussian,
            "moduleResPriorGaussian",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.moduleResPriorGaussian.update_scale_table(scale_table, force=force)
        updated |= self.entropy_update(force=force)
        return updated

    def entropy_update(self, force=False):
        """Updates the entropy bottleneck(s) CDF values.
        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.
        Args:
            force (bool): overwrite previous values (default: False)
        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.
        """
        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def forward(self, tenInput, tenRef):
        intBatch, intChannel, intHeight, intWidth = tenInput.size()

        # get optical flow
        tenMotionVector = self.moduleOpticalFlow(tenInput, tenRef)
        
        # motion feature encoding
        tenMotionFeature = self.moduleMVEncoder(tenMotionVector)
        # motion feature quantization
        tenMotionFeatureQuantized, tenMotionFeatureLikelihoods = self.moduleMotionEntropy(tenMotionFeature)
        # motion feature decoding
        tenMotionVectorDecoded = self.moduleMVDecoder(tenMotionFeatureQuantized)

        # motion compensation
        tenMotionCompensation, tenWarpFrame = self.moduleMotionCompensation(tenRef, tenMotionVectorDecoded)

        # get residual
        tenResFrame = tenInput - tenMotionCompensation

        # residual feature encoding
        tenResFeature = self.moduleResEncoder(tenResFrame)
        tenPriorFeature = self.moduleResPriorEncoder(tenResFeature)   # prior encoding

        # prior quantization
        tenPriorFeatureQunatized, tenPriorFeatureLikelihoods = self.moduleResEntropy(tenPriorFeature)
        tenPriorFeatureRecon = self.moduleResPriorDecoder(tenPriorFeatureQunatized)   # prior decoding

        # residual quantization
        tenResFeatureQuantized, tenResFeatureLikelihoods = self.moduleResPriorGaussian(tenResFeature, tenPriorFeatureRecon)

        # residual feature decoding
        tenResRecon = self.moduleResDecoder(tenResFeatureQuantized)

        # synthesis
        tenFrameRecon = tenMotionCompensation + tenResRecon
        tenFrameReconClipped = tenFrameRecon.clamp(0., 1.)

        tenLossMSE = torch.mean((tenFrameRecon - tenInput).pow(2))
        tenLossWarp = torch.mean((tenWarpFrame - tenInput).pow(2))
        tenLossMotionCompensation = torch.mean((tenMotionCompensation - tenInput).pow(2))

        # bpp loss
        num_pixels = intBatch * intHeight * intWidth
        tenBppMotion = torch.log(tenMotionFeatureLikelihoods).sum() / (-math.log(2) * num_pixels)
        tenBppPrior = torch.log(tenPriorFeatureLikelihoods).sum() / (-math.log(2) * num_pixels)
        tenBppFeature = torch.log(tenResFeatureLikelihoods).sum() / (-math.log(2) * num_pixels)

        tenBpp = tenBppMotion + tenBppPrior + tenBppFeature

        return tenLossMSE, tenLossWarp, tenLossMotionCompensation, tenBpp, torch.FloatTensor([0.0]), tenBppMotion, tenBppPrior, tenBppFeature
        
    def compress(self, tenInput, tenRef):
        # get optical flow
        tenMotionVector = self.moduleOpticalFlow(tenInput, tenRef)
        
        # motion feature encoding
        tenMotionFeature = self.moduleMVEncoder(tenMotionVector)

        # quantization
        tenMotionFeatureQuantizedStrings = self.moduleMotionEntropy.compress(tenMotionFeature)
        tenMotionFeatureQuantized = self.moduleMotionEntropy.decompress(tenMotionFeatureQuantizedStrings, tenMotionFeature.size()[-2:])

        # motion feature decoding
        tenMotionVectorDecoded = self.moduleMVDecoder(tenMotionFeatureQuantized)

        # motion compensation
        tenMotionCompensation, _ = self.moduleMotionCompensation(tenRef, tenMotionVectorDecoded)

        # get residual
        tenResFrame = tenInput - tenMotionCompensation

        # residual feature encoding
        tenResFeature = self.moduleResEncoder(tenResFrame)
        tenPriorFeature = self.moduleResPriorEncoder(tenResFeature)   # prior encoding

        # prior quantization
        tenPriorFeatureQunatizedStrings = self.moduleResEntropy.compress(tenPriorFeature)
        tenPriorFeatureQunatized = self.moduleResEntropy.decompress(tenPriorFeatureQunatizedStrings, tenPriorFeature.size()[-2:])
        tenPriorFeatureRecon = self.moduleResPriorDecoder(tenPriorFeatureQunatized)   # prior decoding

        # residual quantization
        indexes = self.moduleResPriorGaussian.build_indexes(tenPriorFeatureRecon)
        tenResFeatureQuantizedStings = self.moduleResPriorGaussian.compress(tenResFeature, indexes)

        result = {
            "strings": [tenMotionFeatureQuantizedStrings,
                        tenPriorFeatureQunatizedStrings,
                        tenResFeatureQuantizedStings],
            "shape": [tenMotionFeature.size()[-2:],
                        tenPriorFeature.size()[-2:]]
        }
        return result

    def decompress(self, tenRef, strings, shape):
        tenMotionFeatureQuantized = self.moduleMotionEntropy.decompress(strings[0], shape[0])

        # motion feature decoding
        tenMotionVectorDecoded = self.moduleMVDecoder(tenMotionFeatureQuantized)

        # motion compensation
        tenMotionCompensation, _ = self.moduleMotionCompensation(tenRef, tenMotionVectorDecoded)

        # Prior decompress
        tenPriorFeatureQunatized = self.moduleResEntropy.decompress(strings[1], shape[1])
        tenPriorFeatureRecon = self.moduleResPriorDecoder(tenPriorFeatureQunatized)   # prior decoding

        # residual dequantization
        indexes = self.moduleResPriorGaussian.build_indexes(tenPriorFeatureRecon)
        tenResFeatureQuantized = self.moduleResPriorGaussian.decompress(strings[2], indexes)

        # residual feature decoding
        tenResRecon = self.moduleResDecoder(tenResFeatureQuantized)

        # synthesis
        tenFrameRecon = tenMotionCompensation + tenResRecon
        tenFrameReconClipped = tenFrameRecon.clamp(0., 1.)

        return tenFrameReconClipped


# class MVEncoder(torch.nn.Module):
#     def __init__(self):
#         super(MVEncoder, self).__init__()
#         out_channel_mv = 128
#         self.encoder = nn.Sequential(
#             conv(2, out_channel_mv, kernel_size=3),
#             GDN(out_channel_mv),
#             conv(out_channel_mv, out_channel_mv, kernel_size=3),
#             GDN(out_channel_mv),
#             conv(out_channel_mv, out_channel_mv, kernel_size=3),
#             GDN(out_channel_mv),
#             conv(out_channel_mv, out_channel_mv, kernel_size=3)
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         return x


# class MVDecoder(torch.nn.Module):
#     def __init__(self):
#         super(MVDecoder, self).__init__()
#         out_channel_mv = 128
#         self.decoder = nn.Sequential(
#             deconv(out_channel_mv, out_channel_mv, kernel_size=3),
#             GDN(out_channel_mv, inverse=True),
#             deconv(out_channel_mv, out_channel_mv, kernel_size=3),
#             GDN(out_channel_mv, inverse=True),
#             deconv(out_channel_mv, out_channel_mv, kernel_size=3),
#             GDN(out_channel_mv, inverse=True),
#             deconv(out_channel_mv, 2)
#         )

#     def forward(self, x):
#         x = self.decoder(x)
#         return x


class MVEncoder(torch.nn.Module):
    def __init__(self):
        super(MVEncoder, self).__init__()
        out_channel_mv = 128
        self.conv1 = torch.nn.Conv2d(2, out_channel_mv, 3, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (2 + out_channel_mv) / (4))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.relu1 = torch.nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = torch.nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.relu2 = torch.nn.LeakyReLU(negative_slope=0.1)
        self.conv3 = torch.nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.relu3 = torch.nn.LeakyReLU(negative_slope=0.1)
        self.conv4 = torch.nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv4.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)
        self.relu4 = torch.nn.LeakyReLU(negative_slope=0.1)
        self.conv5 = torch.nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv5.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv5.bias.data, 0.01)
        self.relu5 = torch.nn.LeakyReLU(negative_slope=0.1)
        self.conv6 = torch.nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv6.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv6.bias.data, 0.01)
        self.relu6 = torch.nn.LeakyReLU(negative_slope=0.1)
        self.conv7 = torch.nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv7.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv7.bias.data, 0.01)
        self.relu7 = torch.nn.LeakyReLU(negative_slope=0.1)
        self.conv8 = torch.nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv8.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv8.bias.data, 0.01)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        x = self.relu7(self.conv7(x))
        return self.conv8(x)


class MVDecoder(torch.nn.Module):
    def __init__(self):
        super(MVDecoder, self).__init__()
        out_channel_mv = 128
        self.deconv1 = torch.nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.relu1 = torch.nn.LeakyReLU(negative_slope=0.1)
        self.deconv2 = torch.nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.relu2 = torch.nn.LeakyReLU(negative_slope=0.1)
        self.deconv3 = torch.nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        self.relu3 = torch.nn.LeakyReLU(negative_slope=0.1)
        self.deconv4 = torch.nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.deconv4.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv4.bias.data, 0.01)
        self.relu4 = torch.nn.LeakyReLU(negative_slope=0.1)
        self.deconv5 = torch.nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv5.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv5.bias.data, 0.01)
        self.relu5 = torch.nn.LeakyReLU(negative_slope=0.1)
        self.deconv6 = torch.nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.deconv6.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv6.bias.data, 0.01)
        self.relu6 = torch.nn.LeakyReLU(negative_slope=0.1)
        self.deconv7 = torch.nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv7.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv7.bias.data, 0.01)
        self.relu7 = torch.nn.LeakyReLU(negative_slope=0.1)
        self.deconv8 = torch.nn.Conv2d(out_channel_mv, 2, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.deconv8.weight.data, (math.sqrt(2 * 1 * (out_channel_mv + 2) / (out_channel_mv + out_channel_mv))))
        torch.nn.init.constant_(self.deconv8.bias.data, 0.01)

    def forward(self, x):
        x = self.relu1(self.deconv1(x))
        x = self.relu2(self.deconv2(x))
        x = self.relu3(self.deconv3(x))
        x = self.relu4(self.deconv4(x))
        x = self.relu5(self.deconv5(x))
        x = self.relu6(self.deconv6(x))
        x = self.relu7(self.deconv7(x))
        return self.deconv8(x)



class ResEncoder(torch.nn.Module):
    def __init__(self):
        super(ResEncoder, self).__init__()
        out_channel_N = 64
        out_channel_M = 96
        self.conv1 = torch.nn.Conv2d(3, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (3 + out_channel_N) / (6))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.gdn1 = GDN(out_channel_N)
        self.conv2 = torch.nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.gdn2 = GDN(out_channel_N)
        self.conv3 = torch.nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.gdn3 = GDN(out_channel_N)
        self.conv4 = torch.nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv4.weight.data, (math.sqrt(2 * (out_channel_M + out_channel_N) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)

    def forward(self, x):
        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        x = self.gdn3(self.conv3(x))
        return self.conv4(x)


class ResPriorEncoder(torch.nn.Module):
    def __init__(self):
        super(ResPriorEncoder, self).__init__()
        out_channel_N = 64
        out_channel_M = 96
        self.conv1 = torch.nn.Conv2d(out_channel_M, out_channel_N, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (out_channel_M + out_channel_N) / (out_channel_M + out_channel_M))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)

    def forward(self, x):
        x = torch.abs(x)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return self.conv3(x)


class ResDecoder(torch.nn.Module):
    def __init__(self):
        super(ResDecoder, self).__init__()
        out_channel_N = 64
        out_channel_M = 96
        self.deconv1 = torch.nn.ConvTranspose2d(out_channel_M, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, (math.sqrt(2 * 1 * (out_channel_M + out_channel_N) / (out_channel_M + out_channel_M))))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.igdn1 = GDN(out_channel_N, inverse=True)
        self.deconv2 = torch.nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.igdn2 = GDN(out_channel_N, inverse=True)
        self.deconv3 = torch.nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        self.igdn3 = GDN(out_channel_N, inverse=True)
        self.deconv4 = torch.nn.ConvTranspose2d(out_channel_N, 3, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv4.weight.data, (math.sqrt(2 * 1 * (out_channel_N + 3) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.deconv4.bias.data, 0.01)

    def forward(self, x):
        x = self.igdn1(self.deconv1(x))
        x = self.igdn2(self.deconv2(x))
        x = self.igdn3(self.deconv3(x))
        x = self.deconv4(x)
        return x


class ResPriorDecoder(torch.nn.Module):
    def __init__(self):
        super(ResPriorDecoder, self).__init__()
        out_channel_N = 64
        out_channel_M = 96
        self.deconv1 = torch.nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.relu1 = torch.nn.ReLU()
        self.deconv2 = torch.nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.relu2 = torch.nn.ReLU()
        self.deconv3 = torch.nn.ConvTranspose2d(out_channel_N, out_channel_M, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, (math.sqrt(2 * 1 * (out_channel_M + out_channel_N) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)

    def forward(self, x):
        x = self.relu1(self.deconv1(x))
        x = self.relu2(self.deconv2(x))
        return torch.exp(self.deconv3(x))


class MotionCompensation(torch.nn.Module):
    def __init__(self):
        super(MotionCompensation, self).__init__()
        self.backwarp_tenGrid = {}

        channelnum = 64

        self.feature_ext = torch.nn.Conv2d(6, channelnum, 3, padding=1)  # feature_ext
        self.f_relu = torch.nn.ReLU()
        torch.nn.init.xavier_uniform_(self.feature_ext.weight.data)
        torch.nn.init.constant_(self.feature_ext.bias.data, 0.0)
        self.conv0 = ResBlock(channelnum, channelnum, 3)  # c0
        self.conv0_p = torch.nn.AvgPool2d(2, 2)  # c0p
        self.conv1 = ResBlock(channelnum, channelnum, 3)  # c1
        self.conv1_p = torch.nn.AvgPool2d(2, 2)  # c1p
        self.conv2 = ResBlock(channelnum, channelnum, 3)  # c2
        self.conv3 = ResBlock(channelnum, channelnum, 3)  # c3
        self.conv4 = ResBlock(channelnum, channelnum, 3)  # c4
        self.conv5 = ResBlock(channelnum, channelnum, 3)  # c5
        self.conv6 = torch.nn.Conv2d(channelnum, 3, 3, padding=1)
        torch.nn.init.xavier_uniform_(self.conv6.weight.data)
        torch.nn.init.constant_(self.conv6.bias.data, 0.0)

    def forward(self, tenInput, tenFlow):
        def warpnet(x):
            feature_ext = self.f_relu(self.feature_ext(x))
            c0 = self.conv0(feature_ext)
            c0_p = self.conv0_p(c0)
            c1 = self.conv1(c0_p)
            c1_p = self.conv1_p(c1)
            c2 = self.conv2(c1_p)
            c3 = self.conv3(c2)
            c3_u = c1 + bilinearupsacling2(c3)  # torch.nn.functional.interpolate(input=c3, scale_factor=2, mode='bilinear', align_corners=True)
            c4 = self.conv4(c3_u)
            c4_u = c0 + bilinearupsacling2(c4)  # torch.nn.functional.interpolate(input=c4, scale_factor=2, mode='bilinear', align_corners=True)
            c5 = self.conv5(c4_u)
            res = self.conv6(c5)
            return res

        warpframe = self.backwarp(tenInput, tenFlow)
        inputfeature = torch.cat((warpframe, tenInput), 1)
        prediction = warpnet(inputfeature) + warpframe

        return prediction, warpframe

    def backwarp(self, tenInput, tenFlow):
        device = tenFlow.device
        if str(tenFlow.shape) not in self.backwarp_tenGrid:
            tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
            tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])

            self.backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([tenHor, tenVer], 1).to(device)

        tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

        tenOutput = torch.nn.functional.grid_sample(input=tenInput, grid=(self.backwarp_tenGrid[str(tenFlow.shape)].to(device) + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)

        return tenOutput


class ResBlock(torch.nn.Module):
    def __init__(self, inputchannel, outputchannel, kernel_size, stride=1):
        super(ResBlock, self).__init__()
        self.relu1 = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(inputchannel, outputchannel, kernel_size, stride, padding=kernel_size//2)
        torch.nn.init.xavier_uniform_(self.conv1.weight.data)
        torch.nn.init.constant_(self.conv1.bias.data, 0.0)
        self.relu2 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(outputchannel, outputchannel, kernel_size, stride, padding=kernel_size//2)
        torch.nn.init.xavier_uniform_(self.conv2.weight.data)
        torch.nn.init.constant_(self.conv2.bias.data, 0.0)

        if inputchannel != outputchannel:
            self.adapt_conv = torch.nn.Conv2d(inputchannel, outputchannel, 1)
            torch.nn.init.xavier_uniform_(self.adapt_conv.weight.data)
            torch.nn.init.constant_(self.adapt_conv.bias.data, 0.0)
        else:
            self.adapt_conv = None

    def forward(self, x):
        x_1 = self.relu1(x)
        firstlayer = self.conv1(x_1)
        firstlayer = self.relu2(firstlayer)
        seclayer = self.conv2(firstlayer)
        if self.adapt_conv is None:
            return x + seclayer
        else:
            return self.adapt_conv(x) + seclayer


def bilinearupsacling2(inputfeature):
    inputheight = inputfeature.size()[2]
    inputwidth = inputfeature.size()[3]
    outfeature = torch.nn.functional.interpolate(inputfeature, (inputheight * 2, inputwidth * 2), mode='bilinear', align_corners=True)
    return outfeature


class Bitparm(torch.nn.Module):
    def __init__(self, channel, final=False):
        super(Bitparm, self).__init__()
        self.final = final
        self.h = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        self.b = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        if not final:
            self.a = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        else:
            self.a = None

    def forward(self, x):
        if self.final:
            return torch.sigmoid(x * torch.nn.functional.softplus(self.h) + self.b)
        else:
            x = x * torch.nn.functional.softplus(self.h) + self.b
            return x + torch.tanh(x) * torch.tanh(self.a)


class BitEstimator(torch.nn.Module):
    def __init__(self, channel):
        super(BitEstimator, self).__init__()
        self.f1 = Bitparm(channel)
        self.f2 = Bitparm(channel)
        self.f3 = Bitparm(channel)
        self.f4 = Bitparm(channel, True)

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return self.f4(x)


def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`
    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find
    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)


def _update_registered_buffer(
    module,
    buffer_name,
    state_dict_key,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)

    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')

        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))

    else:
        raise ValueError(f'Invalid policy "{policy}"')


def update_registered_buffers(
    module,
    module_name,
    buffer_names,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.
    (There's no way in torch to directly load a buffer with a dynamic size)
    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}",
            state_dict,
            policy,
            dtype,
        )


def get_scale_table(min=0.11, max=256, levels=64):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))
