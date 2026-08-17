"""Communication policies from Table I of NEXUS-MI."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Policy:
    id: str
    name: str
    label: str
    family: str
    selection_policy: str
    buffer_policy: str
    stale_policy: str
    download_policy: str


POLICIES = {
    "P1": Policy(
        "P1", "fixed_all_no_buffer", "Non-adaptive all-gateway scheduling, no buffering",
        "fixed_selection", "all", "none", "drop", "always",
    ),
    "P2": Policy(
        "P2", "fixed_all_fifo_accept_all",
        "Non-adaptive all-gateway scheduling, FIFO buffering, accept-if-base-retained delayed admission",
        "fixed_selection", "all", "fifo", "accept_all", "always",
    ),
    "P3": Policy(
        "P3", "fixed_all_fifo_stale_drop",
        "Non-adaptive all-gateway scheduling, FIFO buffering, stale-drop admission",
        "fixed_selection", "all", "fifo", "drop", "always",
    ),
    "P4": Policy(
        "P4", "fixed_all_latest_stale_drop",
        "Non-adaptive all-gateway scheduling, latest-pending buffering, stale-drop admission",
        "fixed_selection", "all", "latest", "drop", "always",
    ),
    "P5": Policy(
        "P5", "gateway_fifo_stale_drop",
        "Communication-aware gateway scheduling, FIFO buffering, stale-drop admission",
        "gateway_scheduled", "comm_aware", "fifo", "drop", "stale_only",
    ),
    "P6": Policy(
        "P6", "gateway_latest_stale_drop",
        "Communication-aware gateway scheduling, latest-update buffering, stale-drop admission",
        "gateway_scheduled", "comm_aware", "latest", "drop", "stale_only",
    ),
}
