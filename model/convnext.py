from functools import partial
from typing import Callable, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from model.layers.weight_init import trunc_normal_
from model.layers.pool2d_same import AvgPool2dSame
from model.layers.drop import DropPath
from model.layers.mlp import Mlp
from model.layers.mlp import GlobalResponseNormMlp
from model.layers.norm import LayerNorm2d, LayerNorm, RmsNorm2d, RmsNorm
from model.layers.create_conv2d import create_conv2d
from model.layers.create_act import get_act_layer
from model.layers.create_norm import get_norm_layer
from model.layers.helpers import make_divisible, to_ntuple
from model.layers.classifier import ClassifierHead, NormMlpClassifierHead
from model.layers.trace_utils import _assert
from itertools import chain
from torch.utils.checkpoint import checkpoint

def _init_weights(module, name=None, head_init_scale=1.0):
    if isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        nn.init.zeros_(module.bias)
        if name and 'head.' in name:
            module.weight.data.mul_(head_init_scale)
            module.bias.data.mul_(head_init_scale)

def named_apply(
        fn: Callable,
        module: nn.Module, name='',
        depth_first: bool = True,
        include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module

def checkpoint_seq(
        functions,
        x,
        every=1,
        flatten=False,
        skip_last=False,
        preserve_rng_state=True
):
    r"""A helper function for checkpointing sequential models.

    Sequential models execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a sequence into segments
    and checkpoint each segment. All segments except run in :func:`torch.no_grad`
    manner, i.e., not storing the intermediate activations. The inputs of each
    checkpointed segment will be saved for re-running the segment in the backward pass.

    See :func:`~torch.utils.checkpoint.checkpoint` on how checkpointing works.

    .. warning::
        Checkpointing currently only supports :func:`torch.autograd.backward`
        and only if its `inputs` argument is not passed. :func:`torch.autograd.grad`
        is not supported.

    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients.

    Args:
        functions: A :class:`torch.nn.Sequential` or the list of modules or functions to run sequentially.
        x: A Tensor that is input to :attr:`functions`
        every: checkpoint every-n functions (default: 1)
        flatten (bool): flatten nn.Sequential of nn.Sequentials
        skip_last (bool): skip checkpointing the last function in the sequence if True
        preserve_rng_state (bool, optional, default=True):  Omit stashing and restoring
            the RNG state during each checkpoint.

    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`

    Example:
        >>> model = nn.Sequential(...)
        >>> input_var = checkpoint_seq(model, input_var, every=2)
    """
    def run_function(start, end, functions):
        def forward(_x):
            for j in range(start, end + 1):
                _x = functions[j](_x)
            return _x
        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = functions.children()
    if flatten:
        functions = chain.from_iterable(functions)
    if not isinstance(functions, (tuple, list)):
        functions = tuple(functions)

    num_checkpointed = len(functions)
    if skip_last:
        num_checkpointed -= 1
    end = -1
    for start in range(0, num_checkpointed, every):
        end = min(start + every - 1, num_checkpointed - 1)
        x = checkpoint(run_function(start, end, functions), x, preserve_rng_state=preserve_rng_state)
    if skip_last:
        return run_function(end + 1, len(functions) - 1, functions)(x)
    return x

def feature_take_indices(
        num_features: int,
        indices: Optional[Union[int, List[int]]] = None,
        as_set: bool = False,
) -> Tuple[List[int], int]:
    """ Determine the absolute feature indices to 'take' from.

    Note: This function can be called in forwar() so must be torchscript compatible,
    which requires some incomplete typing and workaround hacks.

    Args:
        num_features: total number of features to select from
        indices: indices to select,
          None -> select all
          int -> select last n
          list/tuple of int -> return specified (-ve indices specify from end)
        as_set: return as a set

    Returns:
        List (or set) of absolute (from beginning) indices, Maximum index
    """
    if indices is None:
        indices = num_features  # all features if None

    if isinstance(indices, int):
        # convert int -> last n indices
        _assert(0 < indices <= num_features, f'last-n ({indices}) is out of range (1 to {num_features})')
        take_indices = [num_features - indices + i for i in range(indices)]
    else:
        take_indices: List[int] = []
        for i in indices:
            idx = num_features + i if i < 0 else i
            _assert(0 <= idx < num_features, f'feature index {idx} is out of range (0 to {num_features - 1})')
            take_indices.append(idx)

    if not torch.jit.is_scripting() and as_set:
        return set(take_indices), max(take_indices)

    return take_indices, max(take_indices)


class Downsample(nn.Module):

    def __init__(self, in_chs, out_chs, stride=1, dilation=1):
        super().__init__()
        avg_stride = stride if dilation == 1 else 1
        if stride > 1 or dilation > 1:
            avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
            self.pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)
        else:
            self.pool = nn.Identity()

        if in_chs != out_chs:
            self.conv = create_conv2d(in_chs, out_chs, 1, stride=1)
        else:
            self.conv = nn.Identity()

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class ConvNeXtBlock(nn.Module):
    """ ConvNeXt Block
    There are two equivalent implementations:
      (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
      (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back

    Unlike the official impl, this one allows choice of 1 or 2, 1x1 conv can be faster with appropriate
    choice of LayerNorm impl, however as model size increases the tradeoffs appear to change and nn.Linear
    is a better choice. This was observed with PyTorch 1.10 on 3090 GPU, it could change over time & w/ different HW.
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: Optional[int] = None,
            kernel_size: int = 7,
            stride: int = 1,
            dilation: Union[int, Tuple[int, int]] = (1, 1),
            mlp_ratio: float = 4,
            conv_mlp: bool = False,
            conv_bias: bool = True,
            use_grn: bool = False,
            ls_init_value: Optional[float] = 1e-6,
            act_layer: Union[str, Callable] = 'gelu',
            norm_layer: Optional[Callable] = None,
            drop_path: float = 0.,
    ):
        """

        Args:
            in_chs: Block input channels.
            out_chs: Block output channels (same as in_chs if None).
            kernel_size: Depthwise convolution kernel size.
            stride: Stride of depthwise convolution.
            dilation: Tuple specifying input and output dilation of block.
            mlp_ratio: MLP expansion ratio.
            conv_mlp: Use 1x1 convolutions for MLP and a NCHW compatible norm layer if True.
            conv_bias: Apply bias for all convolution (linear) layers.
            use_grn: Use GlobalResponseNorm in MLP (from ConvNeXt-V2)
            ls_init_value: Layer-scale init values, layer-scale applied if not None.
            act_layer: Activation layer.
            norm_layer: Normalization layer (defaults to LN if not specified).
            drop_path: Stochastic depth probability.
        """
        super().__init__()
        out_chs = out_chs or in_chs
        dilation = to_ntuple(2)(dilation)
        act_layer = get_act_layer(act_layer)
        if not norm_layer:
            norm_layer = LayerNorm2d if conv_mlp else LayerNorm
        mlp_layer = partial(GlobalResponseNormMlp if use_grn else Mlp, use_conv=conv_mlp)
        self.use_conv_mlp = conv_mlp
        self.conv_dw = create_conv2d(
            in_chs,
            out_chs,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation[0],
            depthwise=True,
            bias=conv_bias,
        )
        self.norm = norm_layer(out_chs)
        self.mlp = mlp_layer(out_chs, int(mlp_ratio * out_chs), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(out_chs)) if ls_init_value is not None else None
        if in_chs != out_chs or stride != 1 or dilation[0] != dilation[1]:
            self.shortcut = Downsample(in_chs, out_chs, stride=stride, dilation=dilation[0])
        else:
            self.shortcut = nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        if self.use_conv_mlp:
            x = self.norm(x)
            x = self.mlp(x)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = self.mlp(x)
            x = x.permute(0, 3, 1, 2)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))

        x = self.drop_path(x) + self.shortcut(shortcut)
        return x


class ConvNeXtStage(nn.Module):

    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size=7,
            stride=2,
            depth=2,
            dilation=(1, 1),
            drop_path_rates=None,
            ls_init_value=1.0,
            conv_mlp=False,
            conv_bias=True,
            use_grn=False,
            act_layer='gelu',
            norm_layer=None,
            norm_layer_cl=None
    ):
        super().__init__()
        self.grad_checkpointing = False

        if in_chs != out_chs or stride > 1 or dilation[0] != dilation[1]:
            ds_ks = 2 if stride > 1 or dilation[0] != dilation[1] else 1
            pad = 'same' if dilation[1] > 1 else 0  # same padding needed if dilation used
            self.downsample = nn.Sequential(
                norm_layer(in_chs),
                create_conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=ds_ks,
                    stride=stride,
                    dilation=dilation[0],
                    padding=pad,
                    bias=conv_bias,
                ),
            )
            in_chs = out_chs
        else:
            self.downsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(ConvNeXtBlock(
                in_chs=in_chs,
                out_chs=out_chs,
                kernel_size=kernel_size,
                dilation=dilation[1],
                drop_path=drop_path_rates[i],
                ls_init_value=ls_init_value,
                conv_mlp=conv_mlp,
                conv_bias=conv_bias,
                use_grn=use_grn,
                act_layer=act_layer,
                norm_layer=norm_layer if conv_mlp else norm_layer_cl,
            ))
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x

# Make color cho vui
class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  - https://arxiv.org/pdf/2201.03545.pdf
    """

    def __init__(
            self,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            output_stride: int = 32,
            depths: Tuple[int, ...] = (3, 3, 9, 3),
            dims: Tuple[int, ...] = (96, 192, 384, 768),
            kernel_sizes: Union[int, Tuple[int, ...]] = 7,
            ls_init_value: Optional[float] = 1e-6,
            stem_type: str = 'patch',
            patch_size: int = 4,
            head_init_scale: float = 1.,
            head_norm_first: bool = False,
            head_hidden_size: Optional[int] = None,
            conv_mlp: bool = False,
            conv_bias: bool = True,
            use_grn: bool = False,
            act_layer: Union[str, Callable] = 'gelu',
            norm_layer: Optional[Union[str, Callable]] = None,
            norm_eps: Optional[float] = None,
            drop_rate: float = 0.,
            drop_path_rate: float = 0.,
    ):
        """
        Args:
            in_chans: Number of input image channels.
            num_classes: Number of classes for classification head.
            global_pool: Global pooling type.
            output_stride: Output stride of network, one of (8, 16, 32).
            depths: Number of blocks at each stage.
            dims: Feature dimension at each stage.
            kernel_sizes: Depthwise convolution kernel-sizes for each stage.
            ls_init_value: Init value for Layer Scale, disabled if None.
            stem_type: Type of stem.
            patch_size: Stem patch size for patch stem.
            head_init_scale: Init scaling value for classifier weights and biases.
            head_norm_first: Apply normalization before global pool + head.
            head_hidden_size: Size of MLP hidden layer in head if not None and head_norm_first == False.
            conv_mlp: Use 1x1 conv in MLP, improves speed for small networks w/ chan last.
            conv_bias: Use bias layers w/ all convolutions.
            use_grn: Use Global Response Norm (ConvNeXt-V2) in MLP.
            act_layer: Activation layer type.
            norm_layer: Normalization layer type.
            drop_rate: Head pre-classifier dropout rate.
            drop_path_rate: Stochastic depth drop rate.
        """
        super().__init__()
        assert output_stride in (8, 16, 32)
        kernel_sizes = to_ntuple(4)(kernel_sizes)
        use_rms = isinstance(norm_layer, str) and norm_layer.startswith('rmsnorm')
        if norm_layer is None or use_rms:
            norm_layer = RmsNorm2d if use_rms else LayerNorm2d
            norm_layer_cl = norm_layer if conv_mlp else (RmsNorm if use_rms else LayerNorm)
            if norm_eps is not None:
                norm_layer = partial(norm_layer, eps=norm_eps)
                norm_layer_cl = partial(norm_layer_cl, eps=norm_eps)
        else:
            assert conv_mlp,\
                'If a norm_layer is specified, conv MLP must be used so all norm expect rank-4, channels-first input'
            norm_layer = get_norm_layer(norm_layer)
            norm_layer_cl = norm_layer
            if norm_eps is not None:
                norm_layer_cl = partial(norm_layer_cl, eps=norm_eps)
        act_layer = get_act_layer(act_layer)

        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.feature_info = []

        assert stem_type in ('patch', 'overlap', 'overlap_tiered', 'overlap_act')
        if stem_type == 'patch':
            # NOTE: this stem is a minimal form of ViT PatchEmbed, as used in SwinTransformer w/ patch_size = 4
            self.stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=patch_size, stride=patch_size, bias=conv_bias),
                norm_layer(dims[0]),
            )
            stem_stride = patch_size
        else:
            mid_chs = make_divisible(dims[0] // 2) if 'tiered' in stem_type else dims[0]
            self.stem = nn.Sequential(*filter(None, [
                nn.Conv2d(in_chans, mid_chs, kernel_size=3, stride=2, padding=1, bias=conv_bias),
                act_layer() if 'act' in stem_type else None,
                nn.Conv2d(mid_chs, dims[0], kernel_size=3, stride=2, padding=1, bias=conv_bias),
                norm_layer(dims[0]),
            ]))
            stem_stride = 4

        self.stages = nn.Sequential()
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages = []
        prev_chs = dims[0]
        curr_stride = stem_stride
        dilation = 1
        # 4 feature resolution stages, each consisting of multiple residual blocks
        for i in range(4):
            stride = 2 if curr_stride == 2 or i > 0 else 1
            if curr_stride >= output_stride and stride > 1:
                dilation *= stride
                stride = 1
            curr_stride *= stride
            first_dilation = 1 if dilation in (1, 2) else 2
            out_chs = dims[i]
            stages.append(ConvNeXtStage(
                prev_chs,
                out_chs,
                kernel_size=kernel_sizes[i],
                stride=stride,
                dilation=(first_dilation, dilation),
                depth=depths[i],
                drop_path_rates=dp_rates[i],
                ls_init_value=ls_init_value,
                conv_mlp=conv_mlp,
                conv_bias=conv_bias,
                use_grn=use_grn,
                act_layer=act_layer,
                norm_layer=norm_layer,
                norm_layer_cl=norm_layer_cl,
            ))
            prev_chs = out_chs
            # NOTE feature_info use currently assumes stage 0 == stride 1, rest are stride 2
            self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=f'stages.{i}')]
        self.stages = nn.Sequential(*stages)
        self.num_features = self.head_hidden_size = prev_chs

        # if head_norm_first == true, norm -> global pool -> fc ordering, like most other nets
        # otherwise pool -> norm -> fc, the default ConvNeXt ordering (pretrained FB weights)
        if head_norm_first:
            assert not head_hidden_size
            self.norm_pre = norm_layer(self.num_features)
            self.head = ClassifierHead(
                self.num_features,
                num_classes,
                pool_type=global_pool,
                drop_rate=self.drop_rate,
            )
        else:
            self.norm_pre = nn.Identity()
            self.head = NormMlpClassifierHead(
                self.num_features,
                num_classes,
                hidden_size=head_hidden_size,
                pool_type=global_pool,
                drop_rate=self.drop_rate,
                norm_layer=norm_layer,
                act_layer='gelu',
            )
            self.head_hidden_size = self.head.num_features
        named_apply(partial(_init_weights, head_init_scale=head_init_scale), self)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^stem',
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+)\.downsample', (0,)),  # blocks
                (r'^stages\.(\d+)\.blocks\.(\d+)', None),
                (r'^norm_pre', (99999,))
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head.fc

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        self.head.reset(num_classes, global_pool)

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            norm: Apply norm layer to compatible intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        Returns:

        """
        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.stages) + 1, indices)

        # forward pass
        feat_idx = 0  # stem is index 0
        x = self.stem(x)
        if feat_idx in take_indices:
            intermediates.append(x)

        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            stages = self.stages
        else:
            stages = self.stages[:max_index]
        for stage in stages:
            feat_idx += 1
            x = stage(x)
            if feat_idx in take_indices:
                # NOTE not bothering to apply norm_pre when norm=True as almost no models have it enabled
                intermediates.append(x)

        if intermediates_only:
            return intermediates

        x = self.norm_pre(x)

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.stages) + 1, indices)
        self.stages = self.stages[:max_index]  # truncate blocks w/ stem as idx 0
        if prune_norm:
            self.norm_pre = nn.Identity()
        if prune_head:
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm_pre(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x