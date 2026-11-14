"""navstack v2 -- LiDAR + RTK + IMU mapping and navigation stack.

Module map
----------
``kitti``    data conventions (frame0, calibration, IMU/RTK streams), fixed once
``lio``      point-to-plane ICP registration front end
``imu``      pre-integration bookkeeping and frame0 attitude
``graph``    GTSAM factor graph: LIO between-factors, RTK priors, IMU, loops
``schema``   on-disk map contract, chunk keys, LOD levels
``tiles``    voxel decimation, LOD, byte-bounded chunk cache
``store``    incremental map store: atomic writes, journal, cursor, region load
``levels``   floor segmentation, per-level bake, global minimap
``relocal``  Scan Context retrieval (L1) + multi-hypothesis verification (L2)
``nav``      coarse global layer, regional fine layer, hierarchical A*
``vision``   dynamic masks, semantic cost, stereo depth, visual descriptors
``bundle``   numpy bundle loading and the body<->frame0 conversion

The six themes of the navigation-stack plan map onto these modules one to one;
see ``docs/RESULTS.md`` for the measured result behind each.
"""

__version__ = "2.0.0"
