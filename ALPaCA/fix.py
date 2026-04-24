import numpy as np
from darts import TimeSeries
from darts.models import Theta
from darts.utils.utils import SeasonalityMode
import logging
logging.disable(logging.CRITICAL)
np.random.seed(0)
series = np.random.randn(100)
ts = TimeSeries.from_values(series)
model = Theta(theta=2, season_mode=SeasonalityMode.ADDITIVE)
model.fit(ts[:50])
preds = model.historical_forecasts(ts, start=50, forecast_horizon=1, stride=1, retrain=False)
print(len(preds))
