# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""鲲鹏（aarch64）专用优化总开关。

以下路径优化只在「架构是 aarch64」且「显式设置 OPENVIKING_KUNPENG_OPT=1」时生效：

  - 写入侧：prompt schema 指纹缓存（OPENVIKING_PROMPT_SCHEMA_CACHE）
  - P0    ：read 探测合并（先读后 stat）
  - P1    ：/read 响应预编码（OPENVIKING_READ_FAST_JSON）
  - P2    ：读内容缓存（OPENVIKING_CONTENT_CACHE）
  - 鉴权  ：账号 ACL 开关缓存（OPENVIKING_ACL_ENABLED_CACHE）

默认关闭；x86 上即使误设该变量也不会生效，行为与 0.5.0 原生一致。
"""

import os
import platform

_ENV = "OPENVIKING_KUNPENG_OPT"
_ON_VALUES = {"1", "true", "on", "yes", "enable", "enabled"}
_ARM_ARCHES = {"aarch64", "arm64"}


def requested() -> bool:
    """是否显式要求开启（只看环境变量）。"""
    return os.getenv(_ENV, "").strip().lower() in _ON_VALUES


def is_kunpeng() -> bool:
    """当前是否运行在 ARM64（鲲鹏 950 等）上。"""
    return platform.machine().strip().lower() in _ARM_ARCHES


def enabled() -> bool:
    """总开关：必须同时满足「ARM64 架构」与「显式开启」。"""
    return requested() and is_kunpeng()
