import pandas as pd
from eval.plugins.dnsmos_plugin import DNSMOSPlugin
import numpy as np
import soundfile as sf
import os

# Create dummy audio
sf.write("dummy.wav", np.random.randn(16000), 16000)

plugin = DNSMOSPlugin()
df = pd.DataFrame([{"audio_path": "dummy.wav"}])
out = plugin.run(df)
print(out)
os.remove("dummy.wav")
