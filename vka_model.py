import time
import math
from functools import partial
from typing import Optional, Callable
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class KANLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Calculate the grid step size
        grid_step = 2 / grid_size

        # Create the grid tensor
        grid_range = torch.arange(-spline_order, grid_size + spline_order + 1)
        grid_values = grid_range * grid_step - 1
        self.grid = grid_values.expand(in_features, -1).contiguous()

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.base_activation = nn.SiLU()

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize the base weight tensor with Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

        with torch.no_grad():
            # Generate random noise for initializing the spline weights
            noise_shape = (self.grid_size + 1, self.in_features, self.out_features)
            random_noise = (torch.rand(noise_shape) - 0.5) * 0.1 / self.grid_size

            # Compute the spline weight coefficients from the random noise
            grid_points = self.grid.T[self.spline_order : -self.spline_order]
            spline_coefficients = self.curve2coeff(grid_points, random_noise)

            # Copy the computed coefficients to the spline weight tensor
            self.spline_weight.data.copy_(spline_coefficients)

        # Initialize the spline scaler tensor with Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5))

    def b_splines(self, x: torch.Tensor):

        # Expand the grid tensor to match the input tensor's dimensions
        expanded_grid = (
            self.grid.unsqueeze(0).expand(x.size(0), *self.grid.size()).to(device)
        )  # (batch_size, in_features, grid_size + 2 * spline_order + 1)

        # Add an extra dimension to the input tensor for broadcasting
        input_tensor_expanded = x.unsqueeze(-1).to(
            device
        )  # (batch_size, in_features, 1)

        # Initialize the bases tensor with boolean values
        bases = (
            (input_tensor_expanded >= expanded_grid[:, :, :-1])
            & (input_tensor_expanded < expanded_grid[:, :, 1:])
        ).to(x.dtype)  # (batch_size, in_features, grid_size + spline_order)

        # Compute the B-spline bases recursively
        for order in range(1, self.spline_order + 1):
            left_term = (
                (input_tensor_expanded - expanded_grid[:, :, : -order - 1])
                / (expanded_grid[:, :, order:-1] - expanded_grid[:, :, : -order - 1])
            ) * bases[:, :, :-1]

            right_term = (
                (expanded_grid[:, :, order + 1 :] - input_tensor_expanded)
                / (expanded_grid[:, :, order + 1 :] - expanded_grid[:, :, 1:-order])
            ) * bases[:, :, 1:]

            bases = left_term + right_term

        return bases.contiguous()

    def curve2coeff(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor):

        # Compute the B-spline bases for the input tensor
        b_splines_bases = self.b_splines(
            input_tensor
        )  # (batch_size, input_dim, grid_size + spline_order)

        # Transpose the B-spline bases and output tensor for matrix multiplication
        transposed_bases = b_splines_bases.transpose(
            0, 1
        )  # (input_dim, batch_size, grid_size + spline_order)
        transposed_output = output_tensor.transpose(
            0, 1
        )  # (input_dim, batch_size, output_dim)

        # Convert tensor into the current device type
        transposed_bases = transposed_bases.to(device)
        transposed_output = transposed_output.to(device)

        # Solve the least-squares problem to find the coefficients
        coefficients_solution = torch.linalg.lstsq(
            transposed_bases, transposed_output
        ).solution
        # (input_dim, grid_size + spline_order, output_dim)

        # Permute the coefficients to match the expected shape
        coefficients = coefficients_solution.permute(
            2, 0, 1
        )  # (output_dim, input_dim, grid_size + spline_order)

        return coefficients.contiguous()

    def forward(self, x: torch.Tensor):
        # Save the original shape
        original_shape = x.shape

        # Flatten the last two dimensions of the input
        x = x.contiguous().view(-1, self.in_features)

        base_output = F.linear(
            self.base_activation(x).to(device), self.base_weight.to(device)
        )

        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1).to(device),
            self.spline_weight.view(self.out_features, -1).to(device),
        )

        # Apply the linear transformation
        output = base_output + spline_output

        # Reshape the output to have the same shape as the input
        output = output.view(*original_shape[:-1], -1)

        return output


# Kolmogorov-Arnold Networks
class KAN(nn.Module):

    def __init__(self, dim, intermediate_dim, dropout=0.0, grid_size=5, spline_order=3):
        super().__init__()
        self.kan = nn.Sequential(
            KANLinear(dim, intermediate_dim, grid_size, spline_order),
            KANLinear(intermediate_dim, dim, grid_size, spline_order),
        )

    def forward(self, x):
        return self.kan(x)


class Transformation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.permute(x, (0, 2, 1))

class TCKanLayer(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_patch,
        token_intermediate_dim,
        channel_intermediate_dim,
        dropout=0.0,
        grid_size=5,
        spline_order=3,
    ):
        super().__init__()

        self.token_mixer = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            Transformation(),
            KAN(num_patch, token_intermediate_dim, dropout, grid_size, spline_order),
            Transformation(),
        )

        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            KAN(
                embedding_dim,
                channel_intermediate_dim,
                dropout,
                grid_size,
                spline_order,
            ),
        )

    def forward(self, x):
        val_token_mixer = self.token_mixer(x).to(device)
        val_channel_mixer = self.channel_mixer(x).to(device)
        x = x.to(device)
        x = x + val_token_mixer + val_channel_mixer        

        return x

class PatchEmbed2D(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

def channel_shuffle(x: Tensor, groups: int) -> Tensor:

    batch_size, height, width, num_channels = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batch_size, height, width, groups, channels_per_group)
    x = torch.transpose(x, 3, 4).contiguous()
    x = x.view(batch_size, height, width, -1)

    return x
    
    
class Attention(nn.Module):
    def __init__(self,
                 dim,  
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
        
class LocalSelfAttention(nn.Module):
    def __init__(self,
                 dim,  
                 window_size,  
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(LocalSelfAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.window_size = window_size

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5) 
        x = x.view(B, H, W, C)

        win_H = (H + self.window_size - 1) // self.window_size
        win_W = (W + self.window_size - 1) // self.window_size

        x = x.unfold(1, self.window_size, self.window_size).unfold(2, self.window_size, self.window_size)
        x = x.contiguous().view(B, win_H, win_W, self.window_size, self.window_size, C)

        x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(-1, self.window_size * self.window_size, C)

        qkv = self.qkv(x).reshape(-1, self.window_size * self.window_size, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(-1, self.window_size * self.window_size, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.view(B, win_H, win_W, self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
        x = x.view(B, N, C)

        return x

class LGAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 window_size=7):
        super(LGAttention, self).__init__()
        self.global_attention = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_ratio=attn_drop_ratio,
            proj_drop_ratio=proj_drop_ratio
        )
        self.local_attention = LocalSelfAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_ratio=attn_drop_ratio,
            proj_drop_ratio=proj_drop_ratio
        )

    def forward(self, x):
        global_out = self.global_attention(x)
        local_out = self.local_attention(x)
        out = global_out + local_out
        return out


class Convkan_LGatten(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        num_patch: int = 16,
        token_intermediate_dim: int = 32,
        channel_intermediate_dim: int = 32,
        dropout: float = 0.0,
        grid_size: int = 5,
        spline_order: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim // 2)

        self.self_attention = LGAttention(
            dim=hidden_dim // 2,
            num_heads=8, 
            qkv_bias=False,
            qk_scale=None,
            attn_drop_ratio=attn_drop_rate,
            proj_drop_ratio=attn_drop_rate,
            window_size=7  
        )
        self.drop_path = DropPath(drop_path)

        self.conv33conv33conv11 = nn.Sequential(
            nn.BatchNorm2d(hidden_dim // 2),
            nn.Conv2d(in_channels=hidden_dim // 2, out_channels=hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim // 2, out_channels=hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim // 2, out_channels=hidden_dim // 2, kernel_size=1, stride=1),
            nn.ReLU()
        )

        self.mixer_layer = TCKanLayer(
            embedding_dim=hidden_dim // 2,
            num_patch=num_patch,
            token_intermediate_dim=token_intermediate_dim,
            channel_intermediate_dim=channel_intermediate_dim,
            dropout=dropout,
            grid_size=grid_size,
            spline_order=spline_order,
        )

    def forward(self, input: torch.Tensor):
        input_left, input_right = input.chunk(2, dim=-1)
        B, H, W, C = input_right.shape
        input_right = input_right.view(B, H * W, C)  

        input_right_normalized = self.ln_1(input_right)

        x = self.drop_path(self.self_attention(input_right_normalized))

        input_left = input_left.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        conv_out = self.conv33conv33conv11(input_left)  # [B, C, H, W]
        B, C, H, W = conv_out.shape

        mixer_input = conv_out.flatten(2).transpose(1, 2)  # [B, HW, C]
        mixer_out = self.mixer_layer(mixer_input)  # [B, HW, C]
        mixer_out = mixer_out.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]

        input_left = mixer_out.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]

        x = x.view(B, H, W, C)  # [B, H, W, C]

        output = torch.cat((input_left, x), dim=-1)
        output = channel_shuffle(output, groups=2)

        return output + input

class VKALayer(nn.Module):
    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            Convkan_LGatten(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x

class VKA(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 4, 2], depths_decoder=[2, 9, 2, 2],
                 dims=[96,192,384,768], dims_decoder=[768, 384, 192, 96], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        # WASTED absolute position embedding ======================
        self.ape = False
        # self.ape = False
        # drop_rate = 0.0
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VKALayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)


        # self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_backbone(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        x = self.forward_backbone(x)
        x = x.permute(0,3,1,2)
        x = self.avgpool(x)
        x = torch.flatten(x,start_dim=1)
        x = self.head(x)
        return x
