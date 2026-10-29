"""Check all block/input gradients with native Wan and ActionDiT implementations."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from openwam.model.video_backbone.wan.models.dit import WanModel
from openwam.model.video_backbone.wan_backbone import Wan21
from openwam.model.video_backbone.base import BlockLoopState
from openwam.model.action_backbone.separate_action_dit import ActionDiT
from openwam.model.architectures.dual_system.mot_driver import DualSystemMoTDriver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    results = []
    for dtype in (torch.float32, torch.bfloat16):
        torch.manual_seed(718)
        dim, heads, layers = 128, 4, 12
        dit = WanModel(dim=dim, in_dim=16, ffn_dim=dim*4, out_dim=16,
                       text_dim=dim, freq_dim=dim, eps=1e-6, patch_size=(1, 2, 2),
                       num_heads=heads, num_layers=layers, has_image_input=False)
        vb = Wan21(SimpleNamespace(dit=dit), text_dim=dim).to("cuda", dtype).train()
        ab = ActionDiT(action_dim=80, dim=dim, ffn_dim=dim*4, num_heads=heads,
                       num_layers=layers, video_dim=dim, bridge_layers=tuple(range(layers)),
                       variant="joint_self_attn", text_dim=dim).to("cuda", dtype).train()
        driver = DualSystemMoTDriver(vb, ab, mot_checkpoint_mixed_attn=True)
        for seed in (51, 52, 53):
            reference = None
            for pattern in ("1", "1110", "110", "10", "1"):
                os.environ["OPENWAM_CHECKPOINT_PATTERN"] = pattern
                vb.zero_grad(set_to_none=True)
                ab.zero_grad(set_to_none=True)
                torch.manual_seed(seed)
                def rand(*shape):
                    return torch.randn(*shape, device="cuda", dtype=dtype, requires_grad=True)
                video = rand(2, 12, dim)
                actions = rand(2, 8, 80)
                context = rand(2, 512, dim)
                tmod = rand(2, 6, dim)
                # Unit complex rotations exercise the native RoPE code path.
                freq = torch.ones(12, 1, dim//heads//2, device="cuda", dtype=torch.complex64)
                vs = BlockLoopState(hidden_states=video, time_mod=tmod, rope_freqs=freq,
                                    context=context, grid_frames=3, grid_height=2, grid_width=2,
                                    extras={"dit": dit})
                ast = ab.prepare_state(actions, torch.ones(2, device="cuda", dtype=dtype),
                                       context=context,
                                       context_mask=torch.ones(2, 512, device="cuda", dtype=torch.bool))
                vs, ast = driver.run_joint_loop(vs, ast, use_gradient_checkpointing=True)
                loss = vs.hidden_states.float().square().mean() + ast.payload.x_action.float().square().mean()
                loss.backward()
                tensors = {"video": vs.hidden_states, "action": ast.payload.x_action, "loss": loss}
                tensors.update({f"input_grad.{name}": x.grad for name, x in
                                (("video", video), ("actions", actions), ("context", context), ("time", tmod))})
                for prefix, module in (("video", vb), ("action", ab)):
                    tensors.update({f"gradient.{prefix}.{name}": p.grad
                                    for name, p in module.named_parameters() if p.grad is not None})
                actual = {name: x.detach().cpu().clone() for name, x in tensors.items()}
                if reference is None:
                    reference = actual
                assert actual.keys() == reference.keys()
                mismatched = [name for name in actual if not torch.equal(actual[name], reference[name])]
                row = dict(dtype=str(dtype), seed=seed, pattern=pattern,
                           compared_tensors=len(actual), exact_equal=not mismatched,
                           mismatched=mismatched,
                           max_abs=max(float((actual[n].float()-reference[n].float()).abs().max())
                                       for n in actual))
                results.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "mismatched"}), flush=True)
        del driver, ab, vb, dit
        torch.cuda.empty_cache()
    output = dict(status="passed" if all(r["exact_equal"] for r in results) else "failed",
                  implementation="native Wan DiT and native ActionDiT, 12 layers, 512 context tokens",
                  rows=results)
    Path(args.output).write_text(json.dumps(output, indent=2))
    assert output["status"] == "passed"


if __name__ == "__main__":
    main()
