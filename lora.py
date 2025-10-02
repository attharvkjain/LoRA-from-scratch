# LoRA.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear = None, in_features: int = None, out_features: int = None,
                 r: int = 4, lora_alpha: int = 1, lora_dropout: float = 0.0, bias: bool = True):
        super().__init__()
        # derive in/out from original_linear if provided
        if original_linear is not None:
            in_features = original_linear.in_features
            out_features = original_linear.out_features
            bias_tensor = original_linear.bias.clone() if original_linear.bias is not None else None
        else:
            bias_tensor = None

        assert in_features is not None and out_features is not None

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = (lora_alpha / r) if r > 0 else 1.0

        # Base weight (frozen). Keep it as a Parameter but requires_grad=False for device moves convenience
        if original_linear is not None:
            self.weight = nn.Parameter(original_linear.weight.data.clone(), requires_grad=False)
        else:
            # If no original provided, initialize normally but frozen
            self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias_tensor is not None:
            self.bias = nn.Parameter(bias_tensor.clone(), requires_grad=False)
        else:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False) if bias else None

        # LoRA paramaters
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
            self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
            self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.lora_A = None
            self.lora_B = None
            self.lora_dropout = nn.Identity()

    def forward(self, x: torch.Tensor):
        # base output (frozen weights)
        base_out = F.linear(x, self.weight, bias=self.bias)

        if self.r > 0:
            # LoRA contribution: (dropout(x) @ A.T @ B.T) * scaling
            # shapes:
            # x: (batch, seq, in) or (batch, in)
            # compute x_proj = dropout(x) @ A.T  -> (..., r)
            x_drop = self.lora_dropout(x)
            # allow for 2D or 3D tensors: linear handles them; same here via matmul
            # compute (.., in) @ (in, r) -> (..., r)
            xA = x_drop.matmul(self.lora_A.t())
            # (..., r) @ (r, out) -> (..., out) where B.T is (r, out) so B is (out, r)
            lora_out = xA.matmul(self.lora_B.t()) * self.scaling
            return base_out + lora_out
        else:
            return base_out

def replace_linear_with_lora(model: nn.Module, r: int = 4, lora_alpha: int = 16, lora_dropout: float = 0.0,
                             modules_to_replace: tuple = (nn.Linear,)):
    """
    Walk `model`, replacing instances of `nn.Linear` (or other types passed in modules_to_replace)
    with LoRALinear that copies the original weight/bias.

    Returns count of replaced modules.
    """
    replaced = 0
    for name, module in list(model.named_children()):
        # If the immediate child is to be replaced
        if isinstance(module, modules_to_replace):
            lora_layer = LoRALinear(original_linear=module, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                                    bias=(module.bias is not None))
            setattr(model, name, lora_layer)
            replaced += 1
        else:
            # Recurse
            replaced += replace_linear_with_lora(module, r, lora_alpha, lora_dropout, modules_to_replace)
    return replaced

def lora_parameters(model: nn.Module):
    for n, p in model.named_parameters():
        if ('lora_A' in n) or ('lora_B' in n):
            yield p

def save_lora_state_dict(model: nn.Module, save_path: str):
    """
    Save only LoRA parameters to a file (state_dict).
    """
    lora_state = {k: v.cpu() for k, v in model.state_dict().items() if ('lora_A' in k or 'lora_B' in k)}
    torch.save(lora_state, save_path)

def load_lora_state_dict(model: nn.Module, load_path: str, strict: bool = True):
    sd = torch.load(load_path, map_location='cpu')
    model_sd = model.state_dict()
    model_sd.update(sd)
    model.load_state_dict(model_sd, strict=strict)
