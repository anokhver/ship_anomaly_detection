import sklearn as sk
import pandas as pd
import numpy as np
import json
import django

print("sklearn version: ", sk.__version__)
print("pandas version: ", pd.__version__)
print("numpy version: ", np.__version__)
print("json version: ", json.__version__)
print("django version: ", django.get_version())
