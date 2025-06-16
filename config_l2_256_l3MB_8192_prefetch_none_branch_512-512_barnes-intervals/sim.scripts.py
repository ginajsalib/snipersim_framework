import sys
sys.argv = [ "/root/sniper/scripts/periodicins-stats.py", "500000" ]
execfile("/root/sniper/scripts/periodicins-stats.py")
sys.argv = [ "/root/sniper/scripts/periodic-stats.py", "1000:2000" ]
execfile("/root/sniper/scripts/periodic-stats.py")
sys.argv = [ "/root/sniper/scripts/markers.py", "markers" ]
execfile("/root/sniper/scripts/markers.py")
