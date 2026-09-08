"""Prospective admission estimates; never interrupt an admitted experiment."""
from datetime import datetime, timezone
import math

ESTIMATED_HOURS = {'hindsight_calibration':1., 'clara':.05, 'compensation':3.,
                   'reference':2., 'monitor':2., 'tabular_drift':2.}


def remaining_hours(start, spent, now=None):
    moment=datetime.fromisoformat(start.replace('Z','+00:00'))
    now=now or datetime.now(timezone.utc)
    if moment.tzinfo is None or moment>now:raise ValueError('invalid allocation start')
    if not math.isfinite(spent) or not 0<=spent<50:raise ValueError('invalid previous hours')
    return 50-spent-(now-moment).total_seconds()/3600


def admission(stage,start,spent,estimate=None,now=None):
    estimate=ESTIMATED_HOURS[stage] if estimate is None else estimate
    if not math.isfinite(estimate) or estimate<=0:raise ValueError('invalid runtime estimate')
    remaining=remaining_hours(start,spent,now)
    required=1.5*estimate+.5
    return {'admit':remaining>=required,'estimated_hours':estimate,'remaining_hours':remaining,
            'required_hours_with_margin':required,'estimate_multiplier':1.5,'reserve_hours':.5,
            'mid_run_timeout':False,'budget_target_hours':50}
