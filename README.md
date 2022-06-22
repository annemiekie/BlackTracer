# Black-Hole Visualization

A. Verbraeck and E. Eisemann, "Interactive Black-Hole Visualization", 
in IEEE Transactions on Visualization and Computer Graphics, 
vol. 27, no. 2, pp. 796-805, Feb. 2021, 
doi: 10.1109/TVCG.2020.3030452.

The demo.zip file contains a WIP demo version. You need cuda > 5.0 to run it.
Note there are some issues with this version, which I'll try to address as soon as possible!
- I need to adjust diffraction as that takes too much time in the current setup.
- The colors are BGR instead of RGB in the runtime version.
- Moving the camera should not be possible in 360 view, but it is and it crashes the program.
- Can not input own image yet.

Controls:
- l : turn redshift and lensing effects on and off
- m : turn the environment map on and off
- s : turn the stars on and off
- d : turn diffraction on and off (turn this off if the simulation does not run smoothly
