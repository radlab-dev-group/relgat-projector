RELGAT_SMALL = "small"
RELGAT_MEDIUM = "medium"
RELGAT_LARGE = "large"

_DEFS = {
    "architectures": {
        RELGAT_SMALL: {"gat_out_dim": 128, "layers": 2, "heads": 8},
        RELGAT_MEDIUM: {"gat_out_dim": 128, "layers": 3, "heads": 10},
        RELGAT_LARGE: {"gat_out_dim": 256, "layers": 4, "heads": 12},
    }
}
