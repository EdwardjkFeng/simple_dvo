# simple_dvo
To refresh my mind and also get my hands on the master thesis project I'm working on, I started this overnight project to implemnet a simple dense visual odometry based on the idea of krrish94 as well as some public repos.

## Overall Planning
- [x] Read references
- [x] Prepare all necessary libraries (OpenCV, Eigen, Ceres, G2O, Sophus ....)
- [x] Code & Debug (on a pair of images)
- [x] Get the code run on TUM RGB-D dataset!
- [ ] Benchmark (time and accuracy)
- [ ] Finish up documentation and README


## Plans in details
- [x] Read in a pair of images (RGB as well as Depth)
- [x] From RGB-D to pointcloud
- [ ] Plot pointcloud
- [x] Build image pyramid
- [x] SE(3) operators
- [x] Direct image alignment
- [x] Simple gradient descent solver
- [x] Test on two conjutive frames
    To showcase a simple test, run the following command:
    `python3 vo.py -datapath ../data/ -startFrameRGB rgb -startFrameDepth depth -endFrameRGB rgb2 -endFrameDepth depth2`
- [ ] Test on a seqeuence