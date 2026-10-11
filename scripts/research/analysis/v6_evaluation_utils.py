"""Checkpoint compatibility and shared, testable evaluation preprocessing."""
import torch

from train_promoter_atlas_v6 import PromoterAtlasNet, infer_arch, ESB_COLS, OMICS_COLS, DIST_COLS


def standardize_targets(df, zstats):
    """Return one normalized frame; callers must split it AFTER this operation."""
    df = df.copy()
    for col in ESB_COLS + OMICS_COLS + DIST_COLS + ["ortholog_E"]:
        if col in df and col in zstats:
            mu, sd = zstats[col]
            if not sd > 0:
                raise ValueError(f"Invalid checkpoint target scale: {col}={sd}")
            df[col] = (df[col] - mu) / sd
    return df


def load_evaluation_model(ck, device):
    """Map known legacy [local, context, species] weights to current layout."""
    arch = infer_arch(ck)
    sd = dict(ck["state_dict"])
    layout = ck.get("fuse_layout")
    if layout is None:
        # The archived v6.1 trainer uses abs; v6.2's 10kb trainer uses sab.
        if arch["ctx10k"]:
            layout = "sab"
        elif "head_mpra.weight" not in sd:
            layout = "abs"
        else:
            raise ValueError("Ambiguous checkpoint fuse layout; supply verified metadata")
    if layout == "abs":
        if arch["ctx10k"]:
            raise ValueError("Legacy abs layout with 10kb is unsupported")
        nsp = sd["species_emb.weight"].shape[1]
        w = sd["fuse.0.weight"]
        sd["fuse.0.weight"] = torch.cat([w[:, -nsp:], w[:, :-nsp]], dim=1)
    elif layout != "sab":
        raise ValueError(f"Unknown fuse layout: {layout}")
    n_dist = sd["head_dist.weight"].shape[0] if "head_dist.weight" in sd else 0
    model = PromoterAtlasNet(n_tf=ck.get("n_tf", 0), n_omics=ck.get("n_omics", 0),
                             n_dist=n_dist, **arch).to(device)
    result = model.load_state_dict(sd, strict=False)
    mpra_keys = {"head_mpra.weight", "head_mpra.bias"}
    allowed = mpra_keys if not (mpra_keys & sd.keys()) and "mpra_activity" not in ck.get("zstats", {}) else set()
    if set(result.missing_keys) - allowed or result.unexpected_keys:
        raise RuntimeError(f"Incompatible checkpoint: {result}")
    if result.missing_keys:
        with torch.no_grad():
            model.head_mpra.weight.zero_()
            model.head_mpra.bias.zero_()
    model.eval()
    return model


def model_embedding(model, x400, x2000, species):
    """Capture the exact pre-head representation from the normal forward path."""
    captured = []
    handle = model.head_esb.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0]))
    try:
        model(x400, None, x2000, species)
    finally:
        handle.remove()
    return captured[0]
