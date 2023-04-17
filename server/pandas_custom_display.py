import pandas as pd

pd.set_option("display.max_columns", 200)
pd.set_option("display.max_colwidth", 20000)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: "%.1f" % x)
