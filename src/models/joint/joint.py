import torch
from opticalflow.spynet import Network as Spynet
import math
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.layers import GDN


def make_model(args):
    return Network()


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, output_padding=stride - 1, padding=kernel_size // 2)



class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        out_channel_N = 192
        out_channel_M = 192
        self.moduleOpticalFlow = Spynet()
        self.moduleMVEncoder = MVEncoder()
        self.moduleMVPriorEncoder = MVPriorEncoder()
        self.moduleMVDecoder = MVDecoder()
        self.moduleMVPriorDecoder = MVPriorDecoder()
        self.moduleMotionCompensation = MotionCompensation()
        self.moduleResEncoder = ResEncoder()
        self.moduleResPriorEncoder = ResPriorEncoder()
        self.moduleResDecoder = ResDecoder()
        self.moduleResPriorDecoder = ResPriorDecoder()
        self.moduleMotionEntropy = EntropyBottleneck(out_channel_N)
        self.moduleResEntropy = EntropyBottleneck(out_channel_N)

        self.moduleMotionEntropyParameter = nn.Sequential(
            nn.Conv2d(out_channel_M * 12 // 3, out_channel_M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_M * 10 // 3, out_channel_M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_M * 8 // 3, out_channel_M * 6 // 3, 1),
        )
        self.moduleResEntropyParameter = nn.Sequential(
            nn.Conv2d(out_channel_M * 12 // 3, out_channel_M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_M * 10 // 3, out_channel_M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_M * 8 // 3, out_channel_M * 6 // 3, 1),
        )
        self.moduleMotionContextPrediction = MaskedConv2d(out_channel_M, 2 * out_channel_M, kernel_size=5, padding=2, stride=1)
        self.moduleResContextPrediction = MaskedConv2d(out_channel_M, 2 * out_channel_M, kernel_size=5, padding=2, stride=1)

        self.moduleMotionPriorGaussian = GaussianConditional(None)
        self.moduleResPriorGaussian = GaussianConditional(None)
        
        self.mxrange = 150
        self.out_channel_M = out_channel_M
        self.out_channel_N = out_channel_N

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
            self.moduleMotionPriorGaussian,
            "moduleMotionPriorGaussian",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
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
        updated = self.moduleMotionPriorGaussian.update_scale_table(scale_table, force=force)
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

        # motion compress part
        # motion feature encoding
        tenMotionFeature = self.moduleMVEncoder(tenMotionVector)
        tenMotionPriorFeature = self.moduleMVPriorEncoder(tenMotionFeature)

        # motion feature quantization
        tenMotionPriorQuantized, tenMotionPriorLikelihoods = self.moduleMotionEntropy(tenMotionPriorFeature)
        tenMotionParams = self.moduleMVPriorDecoder(tenMotionPriorQuantized)

        tenMotionFeatureHat = self.moduleMotionPriorGaussian.quantize(tenMotionFeature, "noise" if self.training else "dequantize")
        tenMotionCTXParams = self.moduleMotionContextPrediction(tenMotionFeatureHat)
        tenMotionGaussianParams = self.moduleMotionEntropyParameter(torch.cat([tenMotionParams, tenMotionCTXParams], dim=1))

        scales_hat, means_hat = tenMotionGaussianParams.chunk(2, 1)
        _, tenMotionFeatureLikelihoods = self.moduleMotionPriorGaussian(tenMotionFeature, scales_hat, means=means_hat)
        tenMotionVectorDecoded = self.moduleMVDecoder(tenMotionFeatureHat)

        # motion compensation
        tenMotionCompensation, tenWarpFrame = self.moduleMotionCompensation(tenRef, tenMotionVectorDecoded)

        # get residual
        tenResFrame = tenInput - tenMotionCompensation

        # residual feature encoding
        tenResFeature = self.moduleResEncoder(tenResFrame)
        tenResPriorFeature = self.moduleResPriorEncoder(tenResFeature)   # prior encoding

        # prior quantization
        tenResPriorQunatized, tenResPriorLikelihoods = self.moduleResEntropy(tenResPriorFeature)
        tenResParams = self.moduleResPriorDecoder(tenResPriorQunatized)   # prior decoding

        # residual quantization
        tenResFeatureHat = self.moduleResPriorGaussian.quantize(tenResFeature, "noise" if self.training else "dequantize")
        tenResCTXParams = self.moduleResContextPrediction(tenResFeatureHat)
        tenResGaussianParams = self.moduleResEntropyParameter(torch.cat([tenResParams, tenResCTXParams], dim=1))

        scales_hat, means_hat = tenResGaussianParams.chunk(2, 1)
        _, tenResFeatureLikelihoods = self.moduleResPriorGaussian(tenResFeature, scales_hat, means=means_hat)
        tenResRecon = self.moduleResDecoder(tenResFeatureHat)

        # synthesis
        tenFrameRecon = tenMotionCompensation + tenResRecon
        tenFrameReconClipped = tenFrameRecon.clamp(0., 1.)

        # mse loss
        tenLossMSE = torch.mean((tenFrameRecon - tenInput).pow(2))
        tenLossWarp = torch.mean((tenWarpFrame - tenInput).pow(2))
        tenLossMotionCompensation = torch.mean((tenMotionCompensation - tenInput).pow(2))

        # bpp loss
        num_pixels = intBatch * intHeight * intWidth
        tenBppMotion = torch.log(tenMotionFeatureLikelihoods).sum() / (-math.log(2) * num_pixels)
        tenBppMotionPrior = torch.log(tenMotionPriorLikelihoods).sum() / (-math.log(2) * num_pixels)
        tenBppRes = torch.log(tenResFeatureLikelihoods).sum() / (-math.log(2) * num_pixels)
        tenBppResPrior = torch.log(tenResPriorLikelihoods).sum() / (-math.log(2) * num_pixels)

        tenBpp = tenBppMotion + tenBppMotionPrior + tenBppRes + tenBppResPrior

        return tenLossMSE, tenLossWarp, tenLossMotionCompensation, tenBpp, tenBppRes, tenBppResPrior, tenBppMotion
        
    def compress(self, tenInput, tenRef):
        # get optical flow
        tenMotionVector = self.moduleOpticalFlow(tenInput, tenRef)
        
        # motion feature encoding
        tenMotionFeature = self.moduleMVEncoder(tenMotionVector)
        tenMotionPriorFeature = self.moduleMVPriorEncoder(tenMotionFeature)

        tenMotionPriorQuantizedStrings = self.moduleMotionEntropy.compress(tenMotionPriorFeature)
        tenMotionPriorQuantized = self.moduleMotionEntropy.decompress(tenMotionPriorQuantizedStrings, tenMotionPriorFeature.size()[-2:])
        tenMotionParams = self.moduleMVPriorDecoder(tenMotionPriorQuantized)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        tenMotionFeatureHeight = tenMotionPriorQuantized.size(2) * s
        tenMotionFeatureWidth = tenMotionPriorQuantized.size(3) * s

        tenMotionFeatureHat = F.pad(tenMotionFeature, (padding, padding, padding, padding))

        tenMotionFeatureStrings = []
        for i in range(tenMotionFeature.size(0)):
            string = self._motion_compress_ar(
                tenMotionFeatureHat[i : i + 1],
                tenMotionParams[i : i + 1],
                tenMotionFeatureHeight,
                tenMotionFeatureWidth,
                kernel_size,
                padding,
            )
            tenMotionFeatureStrings.append(string)

        tenMotionFeatureHat = torch.zeros(
            (tenMotionPriorQuantized.size(0), self.out_channel_M, tenMotionFeatureHeight + 2 * padding, tenMotionFeatureWidth + 2 * padding),
            device=tenMotionPriorQuantized.device,
        )

        for i, y_string in enumerate(tenMotionFeatureStrings):
            self._motion_decompress_ar(
                y_string,
                tenMotionFeatureHat[i : i + 1],
                tenMotionParams[i : i + 1],
                tenMotionFeatureHeight,
                tenMotionFeatureWidth,
                kernel_size,
                padding,
            )

        tenMotionFeatureHat = F.pad(tenMotionFeatureHat, (-padding, -padding, -padding, -padding))
        tenMotionVectorDecoded = self.moduleMVDecoder(tenMotionFeatureHat)

        # motion compensation
        tenMotionCompensation, _ = self.moduleMotionCompensation(tenRef, tenMotionVectorDecoded)

        # get residual
        tenResFrame = tenInput - tenMotionCompensation

        # residual feature encoding
        tenResFeature = self.moduleResEncoder(tenResFrame)
        tenResPriorFeature = self.moduleResPriorEncoder(tenResFeature)   # prior encoding

        # prior quantization
        tenPriorFeatureQunatizedStrings = self.moduleResEntropy.compress(tenResPriorFeature)
        tenPriorFeatureQunatized = self.moduleResEntropy.decompress(tenPriorFeatureQunatizedStrings, tenResPriorFeature.size()[-2:])
        tenResParams = self.moduleResPriorDecoder(tenPriorFeatureQunatized)   # prior decoding

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        tenResFeatureHeight = tenPriorFeatureQunatized.size(2) * s
        tenResFeatureWidth = tenPriorFeatureQunatized.size(3) * s

        tenResFeatureHat = F.pad(tenResFeature, (padding, padding, padding, padding))

        tenResFeatureStrings = []
        for i in range(tenMotionFeature.size(0)):
            string = self._residual_compress_ar(
                tenResFeatureHat[i : i + 1],
                tenResParams[i : i + 1],
                tenResFeatureHeight,
                tenResFeatureWidth,
                kernel_size,
                padding,
            )
            tenResFeatureStrings.append(string)


        result = {
            "strings": [tenMotionPriorQuantizedStrings,
                        tenMotionFeatureStrings,
                        tenPriorFeatureQunatizedStrings,
                        tenResFeatureStrings],
            "shape": [tenMotionPriorFeature.size()[-2:],
                        tenResPriorFeature.size()[-2:]]
        }
        return result

    def decompress(self, tenRef, strings, shape):
        tenMotionPriorQuantized = self.moduleMotionEntropy.decompress(strings[0], shape[0])
        tenMotionParams = self.moduleMVPriorDecoder(tenMotionPriorQuantized)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        tenMotionFeatureHeight = tenMotionPriorQuantized.size(2) * s
        tenMotionFeatureWidth = tenMotionPriorQuantized.size(3) * s

        tenMotionFeatureHat = torch.zeros(
            (tenMotionPriorQuantized.size(0), self.out_channel_M, tenMotionFeatureHeight + 2 * padding, tenMotionFeatureWidth + 2 * padding),
            device=tenMotionPriorQuantized.device,
        )

        for i, y_string in enumerate(strings[1]):
            self._motion_decompress_ar(
                y_string,
                tenMotionFeatureHat[i : i + 1],
                tenMotionParams[i : i + 1],
                tenMotionFeatureHeight,
                tenMotionFeatureWidth,
                kernel_size,
                padding,
            )

        tenMotionFeatureHat = F.pad(tenMotionFeatureHat, (-padding, -padding, -padding, -padding))
        tenMotionVectorDecoded = self.moduleMVDecoder(tenMotionFeatureHat)

        # motion compensation
        tenMotionCompensation, _ = self.moduleMotionCompensation(tenRef, tenMotionVectorDecoded)

        tenResPriorQuantized = self.moduleResEntropy.decompress(strings[2], shape[1])
        tenResParams = self.moduleResPriorDecoder(tenResPriorQuantized)   # prior decoding

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        tenResFeatureHeight = tenResPriorQuantized.size(2) * s
        tenResFeatureWidth = tenResPriorQuantized.size(3) * s

        tenResFeatureHat = torch.zeros(
            (tenResPriorQuantized.size(0), self.out_channel_M, tenResFeatureHeight + 2 * padding, tenResFeatureWidth + 2 * padding),
            device=tenResPriorQuantized.device,
        )

        for i, y_string in enumerate(strings[1]):
            self._residual_decompress_ar(
                y_string,
                tenResFeatureHat[i : i + 1],
                tenResParams[i : i + 1],
                tenResFeatureHeight,
                tenResFeatureWidth,
                kernel_size,
                padding,
            )

        tenResFeatureHat = F.pad(tenResFeatureHat, (-padding, -padding, -padding, -padding))
        tenResRecon = self.moduleResDecoder(tenResFeatureHat)

        # synthesis
        tenFrameRecon = tenMotionCompensation + tenResRecon
        tenFrameReconClipped = tenFrameRecon.clamp(0., 1.)

        return tenFrameReconClipped

    def _motion_compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.moduleMotionPriorGaussian.quantized_cdf.tolist()
        cdf_lengths = self.moduleMotionPriorGaussian.cdf_length.tolist()
        offsets = self.moduleMotionPriorGaussian.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        masked_weight = self.moduleMotionContextPrediction.weight * self.moduleMotionContextPrediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.moduleMotionContextPrediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.moduleMotionEntropyParameter(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.moduleMotionPriorGaussian.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.moduleMotionPriorGaussian.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string

    def _residual_compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.moduleResPriorGaussian.quantized_cdf.tolist()
        cdf_lengths = self.moduleResPriorGaussian.cdf_length.tolist()
        offsets = self.moduleResPriorGaussian.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        masked_weight = self.moduleResContextPrediction.weight * self.moduleResContextPrediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.moduleResContextPrediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.moduleResEntropyParameter(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.moduleResPriorGaussian.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.moduleResPriorGaussian.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string

    def _motion_decompress_ar(self, y_string, y_hat, params, height, width, kernel_size, padding):
        cdf = self.moduleMotionPriorGaussian.quantized_cdf.tolist()
        cdf_lengths = self.moduleMotionPriorGaussian.cdf_length.tolist()
        offsets = self.moduleMotionPriorGaussian.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.moduleMotionContextPrediction.weight,
                    bias=self.moduleMotionContextPrediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.moduleMotionEntropyParameter(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.moduleMotionPriorGaussian.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.moduleMotionPriorGaussian.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv
    
    def _residual_decompress_ar(self, y_string, y_hat, params, height, width, kernel_size, padding):
        cdf = self.moduleResPriorGaussian.quantized_cdf.tolist()
        cdf_lengths = self.moduleResPriorGaussian.cdf_length.tolist()
        offsets = self.moduleResPriorGaussian.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.moduleResContextPrediction.weight,
                    bias=self.moduleResContextPrediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.moduleResEntropyParameter(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.moduleResPriorGaussian.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.moduleResPriorGaussian.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv


class MaskedConv2d(nn.Conv2d):
    r"""Masked 2D convolution implementation, mask future "unseen" pixels.
    Useful for building auto-regressive network components.
    Introduced in `"Conditional Image Generation with PixelCNN Decoders"
    <https://arxiv.org/abs/1606.05328>`_.
    Inherits the same arguments as a `nn.Conv2d`. Use `mask_type='A'` for the
    first layer (which also masks the "current pixel"), `mask_type='B'` for the
    following layers.
    """

    def __init__(self, *args, mask_type: str = "A", **kwargs):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, h, w = self.mask.size()
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B") :] = 0
        self.mask[:, :, h // 2 + 1 :] = 0

    def forward(self, x):
        # TODO(begaintj): weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return super().forward(x)


class MVEncoder(torch.nn.Module):
    def __init__(self):
        super(MVEncoder, self).__init__()
        out_channel_N = 192
        out_channel_M = 192
        self.encoder = nn.Sequential(
            conv(2, out_channel_N),
            GDN(out_channel_N),
            conv(out_channel_N, out_channel_N),
            GDN(out_channel_N),
            conv(out_channel_N, out_channel_N),
            GDN(out_channel_N),
            conv(out_channel_N, out_channel_M)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class MVPriorEncoder(torch.nn.Module):
    def __init__(self):
        super(MVPriorEncoder, self).__init__()
        N = 192
        M = 192
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )

    def forward(self, x):
        x = torch.abs(x)
        x = self.h_a(x)
        return x


class MVDecoder(torch.nn.Module):
    def __init__(self):
        super(MVDecoder, self).__init__()
        out_channel_N = 192
        out_channel_M = 192
        self.decoder = nn.Sequential(
            deconv(out_channel_M, out_channel_N),
            GDN(out_channel_N, inverse=True),
            deconv(out_channel_N, out_channel_N),
            GDN(out_channel_N, inverse=True),
            deconv(out_channel_N, out_channel_N),
            GDN(out_channel_N, inverse=True),
            deconv(out_channel_N, 2)
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


class MVPriorDecoder(torch.nn.Module):
    def __init__(self):
        super(MVPriorDecoder, self).__init__()
        N = 192
        M = 192
        self.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

    def forward(self, x):
        x = self.h_s(x)
        return x


class ResEncoder(torch.nn.Module):
    def __init__(self):
        super(ResEncoder, self).__init__()
        N = 192
        M = 192
        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
        )

    def forward(self, x):
        x = self.g_a(x)
        return x


class ResPriorEncoder(torch.nn.Module):
    def __init__(self):
        super(ResPriorEncoder, self).__init__()
        N = 192
        M = 192
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )

    def forward(self, x):
        x = torch.abs(x)
        x = self.h_a(x)
        return x


class ResDecoder(torch.nn.Module):
    def __init__(self):
        super(ResDecoder, self).__init__()
        N = 192
        M = 192
        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )

    def forward(self, x):
        x = self.g_s(x)
        return x


class ResPriorDecoder(torch.nn.Module):
    def __init__(self):
        super(ResPriorDecoder, self).__init__()
        N = 192
        M = 192
        self.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

    def forward(self, x):
        x = self.h_s(x)
        return x


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
