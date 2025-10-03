import matplotlib.pyplot as plt
import numpy as np

data = {
    "Baseline": np.float64(0.20680845383103147),
    "cCompCor": np.float64(0.22362809018705548),
    "Everything": np.float64(0.3534571538732255),
    "ICAAROMA": np.float64(0.2368991752180009),
    "ICAAROMACCompCor": np.float64(0.30166478403759334),
    "ICAAROMAGSR": np.float64(0.24510807858070302),
    "ICAAROMAScrubbing": np.float64(0.337699454154822),
    "ICAAROMAScrubbingGSR": np.float64(0.36351322436105427),
    "motionParameters": np.float64(0.23451091595981569),
    "motionParametersGSR": np.float64(0.23577587797063168),
    "motionParametersScrubbing": np.float64(0.2670413049905428),
    "motionParametersScrubbingGSR": np.float64(0.2843016532875028),
    "Wang2023aCompCor": np.float64(0.3781950543279907),
    "Wang2023aCompCorGSR": np.float64(0.36433109128025587),
    "Wang2023Scrubbing": np.float64(0.29028118979869566),
    "Wang2023ScrubbingGSR": np.float64(0.2965675979946612),
    "Wang2023Simple": np.float64(0.2381074781870411),
    "Wang2023SimpleGSR": np.float64(0.23106772093894778),
}

plt.bar(range(len(data)), list(data.values()), align="center")
plt.xticks(range(len(data)), list(data.keys()))
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
