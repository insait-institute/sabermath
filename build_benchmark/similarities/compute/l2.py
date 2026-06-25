import os
import glob
import ctypes
import platform


def _parse_linux_cpu_list(s: str) -> list[int]:
    cpus = []
    for part in s.split(","):
        part = part.strip()
        try:
            if "-" in part:
                a, b = part.split("-")
                cpus.extend(range(int(a), int(b)))
            else:
                cpus.append(int(part))
        except:
            continue
    return sorted(cpus)


def _l2_cache_groups_linux() -> list[list[int]]:
    groups = set()

    for cpu_path in glob.glob("/sys/devices/system/cpu/cpu[0-9]*"):
        try:
            cpu = int(os.path.basename(cpu_path)[3:])
        except:
            continue

        cache_path = os.path.join(cpu_path, "cache")

        if not os.path.isdir(cache_path):
            continue

        for index in os.listdir(cache_path):
            entry = os.path.join(cache_path, index)
            if not os.path.isdir(entry):
                continue

            try:
                with open(os.path.join(entry, "level")) as f:
                    level = int(f.read().strip())
                if level != 2:
                    continue

                with open(os.path.join(entry, "type")) as f:
                    ctype = f.read().strip()

                if ctype not in ("Data", "Unified"):
                    continue

                with open(os.path.join(entry, "shared_cpu_list")) as f:
                    shared = _parse_linux_cpu_list(f.read().strip())

                groups.add(tuple(shared))

            except FileNotFoundError:
                pass

    return [list(g) for g in groups]


def _l2_cache_groups_windows() -> list[list[int]]:
    from ctypes import wintypes

    RelationProcessorCore = 0

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    GetLogicalProcessorInformationEx = getattr(
        kernel32, "GetLogicalProcessorInformationEx", None
    )

    if GetLogicalProcessorInformationEx is None:
        return _l2_groups_windows_single_group(kernel32)

    class GROUP_AFFINITY(ctypes.Structure):
        _fields_ = [
            ("Mask", ctypes.c_ulonglong),  # KAFFINITY (64-bit)
            ("Group", wintypes.WORD),
            ("Reserved", wintypes.WORD * 3),
        ]

    class PROCESSOR_RELATIONSHIP(ctypes.Structure):
        _fields_ = [
            ("Flags", ctypes.c_ubyte),
            ("EfficiencyClass", ctypes.c_ubyte),
            ("Reserved", ctypes.c_ubyte * 20),
            ("GroupCount", wintypes.WORD),
            ("GroupMask", GROUP_AFFINITY * 1),  # ANYSIZE_ARRAY placeholder
        ]

    class SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX(ctypes.Structure):
        _fields_ = [
            ("Relationship", wintypes.DWORD),
            ("Size", wintypes.DWORD),
            # Followed in memory by a union, but we only care about Processor.
            # We'll manually cast the tail as PROCESSOR_RELATIONSHIP.
        ]

    GetLogicalProcessorInformationEx.argtypes = [
        wintypes.DWORD,  # Relationship
        ctypes.c_void_p,  # Buffer
        ctypes.POINTER(wintypes.DWORD),  # ReturnedLength
    ]
    GetLogicalProcessorInformationEx.restype = wintypes.BOOL

    RelationAll = 0xFFFF
    length = wintypes.DWORD(0)

    res = GetLogicalProcessorInformationEx(RelationAll, None, ctypes.byref(length))
    ERROR_INSUFFICIENT_BUFFER = 122

    if not res and ctypes.get_last_error() != ERROR_INSUFFICIENT_BUFFER:
        err = ctypes.get_last_error()
        raise OSError(err, "GetLogicalProcessorInformationEx: cannot get buffer size")

    buffer = (ctypes.c_byte * length.value)()

    if not GetLogicalProcessorInformationEx(
        RelationAll, ctypes.byref(buffer), ctypes.byref(length)
    ):
        err = ctypes.get_last_error()
        raise OSError(err, "GetLogicalProcessorInformationEx: call failed")

    groups: list[list[int]] = []

    base = ctypes.addressof(buffer)
    end = base + length.value

    while base < end:
        info_ex = ctypes.cast(
            base, ctypes.POINTER(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)
        ).contents

        if info_ex.Relationship == RelationProcessorCore:
            payload_addr = base + ctypes.sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)
            proc_rel = ctypes.cast(
                payload_addr, ctypes.POINTER(PROCESSOR_RELATIONSHIP)
            ).contents

            group_masks_type = GROUP_AFFINITY * proc_rel.GroupCount
            group_masks = group_masks_type.from_address(
                ctypes.addressof(proc_rel.GroupMask)
            )

            logicals_for_core: list[int] = []

            for gm in group_masks:
                mask = gm.Mask
                group = gm.Group

                for cpu in range(64):
                    if mask & (1 << cpu):
                        global_index = group * 64 + cpu
                        logicals_for_core.append(global_index)

            if logicals_for_core:
                groups.append(sorted(logicals_for_core))

        base += info_ex.Size

    groups.sort(key=lambda g: g[0])
    return groups


def _l2_groups_windows_single_group(kernel32=None) -> list[list[int]]:
    from ctypes import wintypes

    RelationProcessorCore = 0

    class CACHE_DESCRIPTOR(ctypes.Structure):
        _fields_ = [
            ("Level", ctypes.c_ubyte),
            ("Associativity", ctypes.c_ubyte),
            ("LineSize", wintypes.WORD),
            ("Size", wintypes.DWORD),
            ("Type", ctypes.c_ubyte),
        ]

    class PROCESSORCORE(ctypes.Structure):
        _fields_ = [
            ("Flags", ctypes.c_ubyte),
        ]

    class NUMANODE(ctypes.Structure):
        _fields_ = [
            ("NodeNumber", wintypes.DWORD),
        ]

    class SYSTEM_LOGICAL_PROCESSOR_INFORMATION_UNION(ctypes.Union):
        _fields_ = [
            ("ProcessorCore", PROCESSORCORE),
            ("NumaNode", NUMANODE),
            ("Cache", CACHE_DESCRIPTOR),
            ("Reserved", ctypes.c_ulonglong * 2),
        ]

    class SYSTEM_LOGICAL_PROCESSOR_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("ProcessorMask", ctypes.c_ulonglong),  # KAFFINITY (64-bit)
            ("Relationship", wintypes.DWORD),
            ("u", SYSTEM_LOGICAL_PROCESSOR_INFORMATION_UNION),
        ]

    if kernel32 is None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    GetLogicalProcessorInformation = kernel32.GetLogicalProcessorInformation
    GetLogicalProcessorInformation.argtypes = [
        ctypes.POINTER(SYSTEM_LOGICAL_PROCESSOR_INFORMATION),
        ctypes.POINTER(wintypes.DWORD),
    ]
    GetLogicalProcessorInformation.restype = wintypes.BOOL

    length = wintypes.DWORD(0)
    res = GetLogicalProcessorInformation(None, ctypes.byref(length))
    ERROR_INSUFFICIENT_BUFFER = 122

    if not res and ctypes.get_last_error() != ERROR_INSUFFICIENT_BUFFER:
        err = ctypes.get_last_error()
        raise OSError(err, "GetLogicalProcessorInformation: cannot get buffer size")

    count = length.value // ctypes.sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)
    buffer = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION * count)()

    if not GetLogicalProcessorInformation(buffer, ctypes.byref(length)):
        err = ctypes.get_last_error()
        raise OSError(err, "GetLogicalProcessorInformation: call failed")

    groups: list[list[int]] = []

    for info in buffer:
        if info.Relationship != RelationProcessorCore:
            continue

        mask = info.ProcessorMask
        logicals: list[int] = []

        for cpu in range(64):
            if mask & (1 << cpu):
                logicals.append(cpu)

        if logicals:
            groups.append(logicals)

    return groups


def _l2_cache_groups_macos() -> list[list[int]]:
    raise NotImplementedError("MacOS is not supported")


def get_l2_cache_groups() -> list[list[int]]:
    system = platform.system()
    if system == "Linux":
        return _l2_cache_groups_linux()
    elif system == "Windows":
        return _l2_cache_groups_windows()
    elif system == "Darwin":
        return _l2_cache_groups_macos()
    else:
        raise NotImplementedError(f"Unsupported OS for L2 grouping: {system}")
