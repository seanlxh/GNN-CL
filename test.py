# import statsmodels.stats.api as sms
# import stats
import numpy as np
# data = [0,1,2]
# ci = stats.t.interval(alpha=0.95, df=len(data) - 1, loc=np.mean(data), scale=stats.sem(data))
intersection_list = [2,2,3]
random_interp_place = np.random.rand(len(intersection_list))
ratio = float(1.0 / sum(random_interp_place))
interp_places = random_interp_place * ratio

print(random_interp_place)
print(interp_places)