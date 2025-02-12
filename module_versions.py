import sys
import pandas as pd
import plotly
import platform

print(f"Python version: {sys.version}")
print(f"Pandas version: {pd.__version__}")
print(f"Plotly version: {plotly.__version__}")
print(f"Platform: {platform.platform()}")
print(f"Timezone: {pd.Timestamp.now().tz}")