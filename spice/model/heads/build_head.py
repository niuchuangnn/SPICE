from .sem_head import SemHead
from .sem_head_multi import SemHeadMulti


def build_head(head_cfg_ori):
    head_cfg = head_cfg_ori.copy()
    head_type = head_cfg.pop("type")
    if head_type == "sem":
        return SemHead(**head_cfg)
    elif head_type == "sem_multi":
        return SemHeadMulti(**head_cfg)
    else:
        raise TypeError
