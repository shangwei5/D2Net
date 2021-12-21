#! bash

# Detector
python test_detector.py  

# Reconstruct without event data
python inference_d2net.py 

# Reconstruct with event data
#python inference_d2net_event.py
