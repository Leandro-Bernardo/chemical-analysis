from ._utils import check_mutually_exclusive_kwargs
from .typing import CalibratedDistributions, Distribution, Intervals, Value, Values
from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime, timezone
from fft_conv_pytorch import fft_conv
from typing import Any, Callable, Optional, Tuple, Type, Union
import torch


class Network(ABC, torch.nn.Module):
    def __init__(self, expected_range: Tuple[float, float], **_: Any) -> None:
        super(Network, self).__init__()
        now = datetime.now(timezone.utc)
        self._datetime = torch.nn.Parameter(torch.as_tensor((now.year, now.month, now.day, now.hour, now.minute, now.second), dtype=torch.int32), requires_grad=False)
        self._expected_range = torch.nn.Parameter(torch.as_tensor(expected_range, dtype=torch.float32), requires_grad=False)

    @property
    def expected_range(self) -> Tuple[float, float]:
        return tuple(map(float, self._expected_range))

    @property
    @abstractmethod
    def input_range(self) -> Tuple[float, float]:
        raise NotImplementedError  # To be implemented by the subclass.
    
    @property
    @abstractmethod
    def input_roi(self) -> Tuple[Tuple[int, int], ...]:
        raise NotImplementedError  # To be implemented by the subclass.

    @property
    @abstractmethod
    def training_mad(self) -> float:
        raise NotImplementedError  # To be implemented by the subclass.

    @property
    @abstractmethod
    def training_median(self) -> float:
        raise NotImplementedError  # To be implemented by the subclass.
    
    @property
    def version(self) -> str:
        year, month, day, hour, minute, second = self._datetime
        return f'{self.__class__.__name__}-{year:04d}.{month:02d}.{day:02d}-{hour:02d}:{minute:02d}:{second:02d}'

    @classmethod
    def load_from_checkpoint(cls, *args: Any, **kwargs: Any) -> "Network":
        state_dict = torch.load(*args, **kwargs)
        hyper_parameters = state_dict["hyper_parameters"]
        net: Network = hyper_parameters["network_class"](**hyper_parameters)
        net.load_state_dict(OrderedDict([(key.lstrip("net."), value) for (key, value) in state_dict["state_dict"].items() if key.startswith("net.")]))
        return net


class ContinuousNetwork(Network):
    def __init__(self, **kwargs: Any) -> None:
        super(ContinuousNetwork, self).__init__(**kwargs)


class IntervalNetwork(Network):
    def __init__(self, num_divisions: int, **kwargs: Any) -> None:
        super(IntervalNetwork, self).__init__(**kwargs)
        assert num_divisions >= 0
        expected_begin, expected_end = self.expected_range
        interval_begin = torch.cat((torch.as_tensor([-float("Inf"), expected_begin], dtype=torch.float32), torch.linspace(0.0, expected_end, 2**num_divisions + 1, dtype=torch.float32)[1:]), dim=0)
        assert interval_begin[1] < interval_begin[2]
        interval_end = torch.cat((interval_begin[1:], torch.as_tensor([float("Inf")], dtype=torch.float32)), dim=0)
        self.intervals = torch.stack((interval_begin, interval_end), dim=-1)

    @classmethod
    def num_intervals(cls, num_divisions: int) -> int:
        return 2**num_divisions + 2


class EstimationFunction(ABC, torch.nn.Module):
    def __init__(self, network_class: Type[Network], *, checkpoint: Optional[str] = None, net: Optional[Network] = None) -> None:
        super(EstimationFunction, self).__init__()
        check_mutually_exclusive_kwargs(checkpoint=checkpoint, net=net)
        if checkpoint is not None:
            self.net = network_class.load_from_checkpoint(checkpoint)
        elif net is not None:
            self.net = net
        else:
            raise NotImplementedError

    @abstractmethod
    def _check_and_reshape_calibrated_pmf(self, **kwargs: CalibratedDistributions) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError  # To be implemented by the subclass.

    @abstractmethod
    def _check_and_reshape_pmfs(self, **kwargs: Distribution) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError  # To be implemented by the subclass.
    
    def forward(self, *, blank_pmf: Optional[Distribution] = None, calibrated_pmf: Optional[CalibratedDistributions] = None, sample_pmf: Optional[Distribution] = None) -> Union[Value, Values, Intervals]:
        check_mutually_exclusive_kwargs(blank_pmf=blank_pmf, calibrated_pmf=calibrated_pmf)
        check_mutually_exclusive_kwargs(sample_pmf=blank_pmf, calibrated_pmf=calibrated_pmf)
        if sample_pmf is not None and blank_pmf is not None:
            ndim = sample_pmf.ndim
            # Reshape input.
            reshaped_sample_pmf, reshaped_blank_pmf = self._check_and_reshape_pmfs(sample_pmf=sample_pmf, blank_pmf=blank_pmf)
            # Compute C = A - B, where A is the random variable representing the sample and B is the random variable representing the blank sample.
            if ndim == 1:
                reshaped_calibrated_pmf = torch.abs(fft_conv(reshaped_sample_pmf.unsqueeze(-2), reshaped_blank_pmf.unsqueeze(-2), padding=(reshaped_blank_pmf.shape[-1] // 2,)))  # Flip is not required because fft_conv implements the cross-correlation operator.
                reshaped_calibrated_pmf = reshaped_calibrated_pmf[:, :, :reshaped_sample_pmf.shape[-1]].squeeze(1)
            elif ndim == 2:
                reshaped_calibrated_pmf = torch.abs(fft_conv(reshaped_sample_pmf, reshaped_blank_pmf, padding=(reshaped_blank_pmf.shape[-2] // 2, reshaped_blank_pmf.shape[-1] // 2)))  # Flip is not required because fft_conv implements the cross-correlation operator.
                reshaped_calibrated_pmf = reshaped_calibrated_pmf[:, :, :reshaped_sample_pmf.shape[-2], :reshaped_sample_pmf.shape[-1]].squeeze(1)
            else:
                raise NotImplementedError
            # Predict value.
            if isinstance(self.net, ContinuousNetwork):
                value, _ = self.net(reshaped_calibrated_pmf)
                return value.squeeze(0)
            # Predict interval.
            elif isinstance(self.net, IntervalNetwork):
                logits = self.net(reshaped_calibrated_pmf)
                return self.net.intervals.to(logits.device)[torch.argmax(logits, dim=-1), ...]
            else:
                raise NotImplementedError
        elif calibrated_pmf is not None:
            # Reshape input.
            reshaped_calibrated_pmf, = self._check_and_reshape_calibrated_pmf(calibrated_pmf=calibrated_pmf)
            # Predict value.
            if isinstance(self.net, ContinuousNetwork):
                value, _ = self.net(reshaped_calibrated_pmf)
                return value
            # Predict interval.
            elif isinstance(self.net, IntervalNetwork):
                logits = self.net(reshaped_calibrated_pmf)
                return self.net.intervals.to(logits.device)[torch.argmax(logits, dim=-1), ...]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError


class SqueezeExcitation(torch.nn.Module):
    def __init__(self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
        scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
    ) -> None:
        super(SqueezeExcitation, self).__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: torch.Tensor) -> torch.Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        scale = self._scale(input)
        return scale * input


class InvertedResidual(torch.nn.Module):
    def __init__(self, *,
        in_channels: int,
        expand_channels: int,
        out_channels: int,
        squeeze_channels: int,
        kernel_size: int,
        padding: int,
        stride: int,
        resnet: bool = False,
        activation_func: Callable[..., torch.nn.Module] = torch.nn.ReLU,
        excite_squeeze: bool = False,
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.resnet = resnet
        self.expand = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, expand_channels, 1, bias=True),
            activation_func(inplace=True),
        )
        self.indepth = torch.nn.Sequential(
            torch.nn.Conv2d(expand_channels, expand_channels, kernel_size, stride=stride, padding=padding, groups=expand_channels, bias=True),
            torch.nn.ReLU(inplace=True),
        )
        self.project = torch.nn.Conv2d(expand_channels, out_channels, 1, bias=True)
        self.excite_squeeze = SqueezeExcitation(input_channels=expand_channels, squeeze_channels=squeeze_channels, activation=torch.nn.Hardsigmoid) if excite_squeeze else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = self.indepth(self.expand(input))
        if self.excite_squeeze is not None:
            result = self.excite_squeeze(result)
        return self.project(result) + (input if self.resnet else 0)


class MobileNetV3Small(torch.nn.Module):
    def __init__(self, *,
        in_channels: int = 3,
        num_classes: int = 1000
    ) -> None:
        super(MobileNetV3Small, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 16, 3, stride=2, padding=1, bias=True),
            torch.nn.Hardswish(inplace=True),
            torch.nn.Conv2d(16, 16, 3, stride=2, padding=1, groups=16, bias=True),
            torch.nn.ReLU(inplace=True),
            SqueezeExcitation(input_channels=16, squeeze_channels=8, activation=torch.nn.ReLU),
            torch.nn.Conv2d(16, 16, 1, stride=1, bias=True),
            # block 2
            torch.nn.Conv2d(16, 72, 1, stride=1, padding=1, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(72, 72, 3, stride=2, padding=1, groups=72, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(72, 24, 1, stride=1, bias=False),
            torch.nn.ReLU(inplace=True),
            # block 3
            torch.nn.Conv2d(24, 88, 1, stride=1, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(88, 88, 3, stride=1, padding=1, groups=88, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(88, 24, 1, stride=1, bias=False),
            torch.nn.ReLU(inplace=True),
            # remaining blocks with resnet.
            InvertedResidual(in_channels=24, expand_channels=96, out_channels=40, squeeze_channels=16, kernel_size=5, padding=2, stride=2, resnet=False, activation_func=torch.nn.Hardswish),
            InvertedResidual(in_channels=40, expand_channels=240, out_channels=40, squeeze_channels=16, kernel_size=5, padding=2, stride=1, resnet=True, activation_func=torch.nn.Hardswish),
            InvertedResidual(in_channels=40, expand_channels=240, out_channels=40, squeeze_channels=16, kernel_size=5, padding=2, stride=1, resnet=True, activation_func=torch.nn.Hardswish),
            InvertedResidual(in_channels=40, expand_channels=128, out_channels=48, squeeze_channels=16, kernel_size=5, padding=2, stride=1, resnet=False, activation_func=torch.nn.Hardswish),
            InvertedResidual(in_channels=48, expand_channels=144, out_channels=48, squeeze_channels=16, kernel_size=5, padding=2, stride=1, resnet=True, activation_func=torch.nn.Hardswish),
            InvertedResidual(in_channels=48, expand_channels=288, out_channels=48, squeeze_channels=16, kernel_size=5, padding=2, stride=2, resnet=False, activation_func=torch.nn.Hardswish),
            InvertedResidual(in_channels=48, expand_channels=288, out_channels=48, squeeze_channels=16, kernel_size=5, padding=2, stride=1, resnet=True, activation_func=torch.nn.Hardswish),
            InvertedResidual(in_channels=48, expand_channels=288, out_channels=48, squeeze_channels=16, kernel_size=5, padding=2, stride=1, resnet=True, activation_func=torch.nn.Hardswish),
            # final block.
            torch.nn.Conv2d(48, 288, 1, bias=True),
            torch.nn.Hardswish(inplace=True),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(288, 16),
            torch.nn.Hardswish(inplace=True),
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(16, num_classes),
        )

    def forward(self, input) -> torch.Tensor:
        return self.classifier(torch.flatten(self.avgpool(self.features(input)), 1))


class Fire(torch.nn.Module):
    def __init__(self,
        inplanes: int,
        squeeze_planes: int,
        expand1x1_planes: int,
        expand3x3_planes: int,
    ) -> None:
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = torch.nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = torch.nn.ReLU(inplace=True)
        self.expand1x1 = torch.nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = torch.nn.ReLU(inplace=True)
        self.expand3x3 = torch.nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = torch.nn.ReLU(inplace=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.squeeze_activation(self.squeeze(input))
        return torch.cat((self.expand1x1_activation(self.expand1x1(output)), self.expand3x3_activation(self.expand3x3(output))), 1)


class SqueezeNet1_1(torch.nn.Module):
    def __init__( self,
        in_channels: int = 3,
        num_classes: int = 1000,
        scale_chs: float = 1.0,
    ) -> None:
        super(SqueezeNet1_1, self).__init__()
        # Resize must be in sequence (2, 4, 8, 16, 32, ...) or (1/2, 1/4, 1/8, 1/16, 1/32, ...).
        if (scale_chs % 2) != 0 and bin(int(1 / scale_chs)).count("1") != 1:
            raise ValueError("Option 'scale_chs' must be in sequence (2, 4, 8, 16, 32, ...) or (1/2, 1/4, 1/8, 1/16, 1/32, ...).")
        # Define 'feature' extraction backbone.
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, int(64 * scale_chs), kernel_size=(3, 3), stride=(2, 2)),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True),
            Fire(int(64 * scale_chs), int(16 * scale_chs), int(64 * scale_chs), int(64 * scale_chs)),
            Fire(int(128 * scale_chs), int(16 * scale_chs), int(64 * scale_chs), int(64 * scale_chs)),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True),
            Fire(int(128 * scale_chs), int(32 * scale_chs), int(128 * scale_chs), int(128 * scale_chs)),
            Fire(int(256 * scale_chs), int(32 * scale_chs), int(128 * scale_chs), int(128 * scale_chs)),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True),
            Fire(int(256 * scale_chs), int(48 * scale_chs), int(192 * scale_chs), int(192 * scale_chs)),
            Fire(int(384 * scale_chs), int(48 * scale_chs), int(192 * scale_chs), int(192 * scale_chs)),
            Fire(int(384 * scale_chs), int(64 * scale_chs), int(256 * scale_chs), int(256 * scale_chs)),
            Fire(int(512 * scale_chs), int(64 * scale_chs), int(256 * scale_chs), int(256 * scale_chs)),
        )
        # Define 'classifier' backbone.
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5, inplace=False),
            torch.nn.Conv2d(int(512 * scale_chs), num_classes, kernel_size=(1, 1), stride=(1, 1)),
            # torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        # Initialize weights and bias.
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
        # Last 'conv' is initialized differently.
        torch.nn.init.normal_(self.classifier[1].weight, mean=0.0, std=0.01)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(input)).flatten(1)


class Fire_1d(torch.nn.Module):
    def __init__(self,
        inplanes: int,
        squeeze_planes: int,
        expand1x1_planes: int,
        expand3x3_planes: int,
    ) -> None:
        super(Fire_1d, self).__init__()
        self.inplanes = inplanes
        self.squeeze = torch.nn.Conv1d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = torch.nn.ReLU(inplace=True)
        self.expand1x1 = torch.nn.Conv1d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = torch.nn.ReLU(inplace=True)
        self.expand3x3 = torch.nn.Conv1d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = torch.nn.ReLU(inplace=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.squeeze_activation(self.squeeze(input))
        return torch.cat((self.expand1x1_activation(self.expand1x1(output)), self.expand3x3_activation(self.expand3x3(output))), 1)


class SqueezeNet1_1_1d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        scale_chs: float = 1.0,
    ) -> None:
        # resize must be in sequence (2, 4, 8, 16, 32, ...) or (1/2, 1/4, 1/8, 1/16, 1/32, ...).
        if (scale_chs % 2) != 0 and bin(int(1 / scale_chs)).count("1") != 1:
            raise ValueError("Option 'scale_chs' must be in sequence (2, 4, 8, 16, 32, ...) or (1/2, 1/4, 1/8, 1/16, 1/32, ...).")
        super(SqueezeNet1_1_1d, self).__init__()
        # Define 'feature' extraction backbone.
        scale_chs_ = {
            16: int(16 * scale_chs),
            32: int(32 * scale_chs),
            48: int(48 * scale_chs),
            64: int(64 * scale_chs),
            128: int(128 * scale_chs),
            192: int(192 * scale_chs),
            256: int(256 * scale_chs),
            384: int(384 * scale_chs),
            512: int(512 * scale_chs),
        }
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, scale_chs_[64], kernel_size=3, stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True),
            Fire_1d(scale_chs_[64], scale_chs_[16], scale_chs_[64], scale_chs_[64]),
            Fire_1d(scale_chs_[128], scale_chs_[16], scale_chs_[64], scale_chs_[64]),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True),
            Fire_1d(scale_chs_[128], scale_chs_[32], scale_chs_[128], scale_chs_[128]),
            Fire_1d(scale_chs_[256], scale_chs_[32], scale_chs_[128], scale_chs_[128]),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True),
            Fire_1d(scale_chs_[256], scale_chs_[48], scale_chs_[192], scale_chs_[192]),
            Fire_1d(scale_chs_[384], scale_chs_[48], scale_chs_[192], scale_chs_[192]),
            Fire_1d(scale_chs_[384], scale_chs_[64], scale_chs_[256], scale_chs_[256]),
            Fire_1d(scale_chs_[512], scale_chs_[64], scale_chs_[256], scale_chs_[256]),
        )
        # Define 'classifier' backbone.
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5, inplace=False),
            torch.nn.Conv1d(scale_chs_[512], num_classes, kernel_size=1, stride=1),
            # torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool1d(output_size=1),
        )
        # Initialize weights and bias.
        for module in self.modules():
            if isinstance(module, torch.nn.Conv1d):
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
        # Last 'conv' is initialized differently.
        torch.nn.init.normal_(self.classifier[1].weight, mean=0.0, std=0.01)

    def forward(self, input) -> torch.Tensor:
        return self.classifier(self.features(input)).flatten(1)


class Vgg11_1d(torch.nn.Module):
    def __init__(self, *,
        in_channels: int = 3,
        num_classes: int = 1000
    ) -> None:
        super(Vgg11_1d, self).__init__()
        # Define 'feature' extraction backbone.
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
            torch.nn.Conv1d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
            torch.nn.Conv1d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
            torch.nn.Conv1d(256, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
        )
        # Define 'classifier' backbone.
        self.avgpool = torch.nn.AdaptiveAvgPool1d(7)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * 7, 2048),
            torch.nn.ReLU(True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(2048, num_classes),
        )
        # Initialize weights and bias.
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
            elif isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, 0, 0.01)
                torch.nn.init.constant_(module.bias, 0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.flatten(self.avgpool(self.features(input)), 1))


class Vgg19_1d(torch.nn.Module):
    def __init__(self, *,
        in_channels: int = 3,
        num_classes: int = 1000
    ) -> None:
        super(Vgg19_1d, self).__init__()
        # Define 'feature' extraction backbone.
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
            torch.nn.Conv1d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
            torch.nn.Conv1d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
            torch.nn.Conv1d(256, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
        )
        # Define 'classifier' backbone.
        self.avgpool = torch.nn.AdaptiveAvgPool1d(7)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * 7, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, num_classes),
        )
        # Initialize weights and bias.
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
            elif isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, 0, 0.01)
                torch.nn.init.constant_(module.bias, 0)

    def forward(self, input):
        return self.classifier(torch.flatten(self.avgpool(self.features(input)), 1))


class VGG11(torch.nn.Module):
    def __init__(self, *,
        in_channels: int = 3,
        num_classes: int = 1000
    ) -> None:
        super(VGG11, self).__init__()
        # Define 'feature' extraction backbone.
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Define 'classifier' backbone.
        self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * 7 * 7, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, num_classes),
        )
        # Initialize weights and bias.
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
            elif isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, 0, 0.01)
                torch.nn.init.constant_(module.bias, 0)

    def forward(self, input):
        return self.classifier(torch.flatten(self.avgpool(self.features(input)), 1))
