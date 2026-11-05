"""Average Mishkefet checkpoints fine-tuned from one start (a 'model soup': the weights' mean; the first's config and
charset, which must be the same in all) - steadier than any one run (5.11: two seeds of the same training read Tal's
page 11.3% and 14.0% wrong; the two and a longer run averaged, 11.5%, and the best or as good as the best on every
other test):

    python tools/soup_mishkefet.py models/mishkefet-soup.pt models/a.pt models/b.pt [models/c.pt ...]"""
import sys


def soup(checkpoints: list) -> dict:
    """the checkpoints (dicts with 'model' and 'charset') with their weights averaged"""
    import torch
    first = checkpoints[0]
    if any(c["charset"] != first["charset"] or c["model"].keys() != first["model"].keys() for c in checkpoints):
        raise ValueError("checkpoints of different models")
    model = {k: (sum(c["model"][k].float() for c in checkpoints) / len(checkpoints)).to(first["model"][k].dtype)
             for k in first["model"]}
    return dict(first, model=model)


def main():
    import torch
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    out, paths = sys.argv[1], sys.argv[2:]
    ck = soup([torch.load(p, map_location="cpu", weights_only=True) for p in paths])
    ck["soup_of"] = paths
    torch.save(ck, out)
    print(f"{len(paths)} checkpoints averaged -> {out}")


if __name__ == "__main__":
    main()
