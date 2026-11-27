from __future__ import annotations

from collections import defaultdict

from retrieval.npa_mas.schemas import Channel, ChannelBudget, QueryBudget, budget_status


class BudgetError(ValueError):
    pass


class BudgetLedger:
    def __init__(self, channel_targets: dict[Channel, int]):
        self._targets = dict(channel_targets)
        self._queries: dict[Channel, list[QueryBudget]] = defaultdict(list)

    def record(
        self,
        *,
        channel: Channel,
        query: str,
        query_key: str,
        target: int,
        requested: int,
        actual: int,
        error: str | None = None,
    ) -> QueryBudget:
        if channel not in self._targets:
            raise BudgetError(f"channel {channel} has no declared target")
        if min(target, requested, actual) < 0:
            raise BudgetError("budget values cannot be negative")
        if not actual <= requested <= target:
            raise BudgetError("actual <= requested <= target is required")
        if self.requested(channel) + requested > self._targets[channel]:
            raise BudgetError("query reservations exceed the channel target")
        item = QueryBudget(
            channel=channel,
            query=query,
            query_key=query_key,
            target_raw_hits=target,
            requested_raw_hits=requested,
            actual_raw_hits=actual,
            unspent_raw_hits=requested - actual,
            status=budget_status(requested, actual, error),
            error=error,
        )
        self._queries[channel].append(item)
        return item

    def requested(self, channel: Channel) -> int:
        return sum(item.requested_raw_hits for item in self._queries[channel])

    def actual(self, channel: Channel) -> int:
        return sum(item.actual_raw_hits for item in self._queries[channel])

    def channel_budget(self, channel: Channel) -> ChannelBudget:
        queries = list(self._queries[channel])
        target = self._targets[channel]
        requested = self.requested(channel)
        actual = self.actual(channel)
        if any(item.status == "error" for item in queries):
            status = "error"
        else:
            status = budget_status(requested, actual)
        return ChannelBudget(
            channel=channel,
            target_raw_hits=target,
            requested_raw_hits=requested,
            actual_raw_hits=actual,
            unspent_raw_hits=requested - actual,
            unrequested_raw_hits=target - requested,
            status=status,
            queries=queries,
        )

    def snapshot(self) -> dict[str, ChannelBudget]:
        return {channel: self.channel_budget(channel) for channel in self._targets}
