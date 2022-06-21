#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
### Simple demo with at least 2 cameras for triangulation
### Input : Live videos of face / hand / body
###       : Calibrated camera intrinsics and extrinsics
### Output: 2D/3D (triangulated) display of hand, body keypoint/joint
### Usage : python 07_triangulate.py -m body --use_panoptic_dataset
###############################################################################

# INSTALLATION
# Needs at least python3.7 (do 3.9?) --> sudo apt install python3.9-dev python3.9-venv
# then modify symlinks in /usr/bin if necessary. Modify both python3 and python3-config


import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray

import cv2
import sys
import time
import numpy as np
import open3d as o3d
import time

from cv_bridge import CvBridge
bridge = CvBridge()

from utils_display import DisplayHand, DisplayBody, DisplayHolistic
from utils_mediapipe import MediaPipeHand, MediaPipeBody, MediaPipeHolistic
from utils_3d_reconstruct import Triangulation

def main():
  # User select mode
  rospy.loginfo('[mediapipe] main() startup')

  use_panoptic_dataset = False
  mode = 'body'

  # Define list of camera index
  # cam_idx = [4,10] # Note: Hardcoded for my setup
  # Read from .mp4 file
  if use_panoptic_dataset:
      # Test with 2 views
      cam_idx = [#'../data/171204_pose1_sample/hdVideos/hd_00_00.mp4',
                 '../data/171204_pose1_sample/hdVideos/hd_00_11.mp4',
                 '../data/171204_pose1_sample/hdVideos/hd_00_28.mp4']
  else:
      cam_idx = [0,2]
      # # Test with n views
      # num_views = 5 # Note: Maximum 31 hd cameras but processing time will be extremely slow
      # cam_idx = []
      # for i in range(num_views):
      #     cam_idx.append(
      #         '../data/171204_pose1_sample/hdVideos/hd_00_'+str(i).zfill(2)+'.mp4')

  # Start video capture
  cap = [cv2.VideoCapture(cam_idx[i]) for i in range(len(cam_idx))]
  rospy.loginfo('[mediapipe] Cameras Opening')
  # cap[0].set(3,160)
  # cap[0].set(4,120)
  # cap[1].set(3,160)
  # cap[1].set(4,120)


  # Define list of other variable
  img   = [None for i in range(len(cam_idx))] # Store image
  pipe  = [None for i in range(len(cam_idx))] # MediaPipe class
  disp  = [None for i in range(len(cam_idx))] # Display class
  param = [None for i in range(len(cam_idx))] # Store pose parameter
  prev_time = [time.time() for i in range(len(cam_idx))]

  print("Num cams:", len(cam_idx))

  # Open3D visualization
  vis = o3d.visualization.Visualizer()
  vis.create_window(width=640, height=480)
  vis.get_render_option().point_size = 5.0

  # Load triangulation class
  tri = Triangulation(cam_idx=cam_idx, vis=vis,
      use_panoptic_dataset=use_panoptic_dataset, filename='CYWwebcams', JSONpath='/home/cwong/src_UdeS/catkin_ws/src/ros_mediapipe/code/')

  # Load mediapipe and display class
  if mode=='hand':
      for i in range(len(cam_idx)):
          pipe[i] = MediaPipeHand(static_image_mode=False, max_num_hands=1)
          disp[i] = DisplayHand(draw3d=True, max_num_hands=1, vis=vis)
  elif mode=='body':
      for i in range(len(cam_idx)):
          pipe[i] = MediaPipeBody(static_image_mode=False, model_complexity=1)
          disp[i] = DisplayBody(draw3d=True, vis=vis)
  elif mode=='holistic':
      for i in range(len(cam_idx)):
          pipe[i] = MediaPipeHolistic(static_image_mode=False, model_complexity=1)
          disp[i] = DisplayHolistic(draw3d=True, vis=vis)
  else:
      print('Undefined mode only the following modes are available: \n hand / body / holistic')
      sys.exit()

  #ROS Part
  rospy.init_node('mediapipeTriangulate', anonymous=False)


  img0_pub = rospy.Publisher('mediapipe/img0', Image, queue_size=1)
  img1_pub = rospy.Publisher('mediapipe/img1', Image, queue_size=1)
  skeleton_pub = rospy.Publisher('mediapipe/skeleton', Marker, queue_size=1)

  t_start = time.time()

  rospy.loginfo('[mediapipe] Node Initialization Finished')

  while not rospy.is_shutdown():
      # Loop through video capture
      # v4l2-ctl --list-devices
      for i, c in enumerate(cap):
          if not c.isOpened():
              print('Cam',i,'not opened.')
              break
          # else:
          #     print("Cam",i, "opened")
          ret, img[i] = c.read()
          # print(i, len(img[i]), len(img[i][0]))
          # img[i] = cv2.flip(img[i], 1)
          if not ret:
              print("Cam",i,"read failed")
              break

          # Preprocess image if necessary
          # img[i] = cv2.flip(img[i], 1) # Flip image for 3rd person view

          # To improve performance, optionally mark image as not writeable to pass by reference
          img[i].flags.writeable = False

          # Feedforward to extract keypoint
          param[i] = pipe[i].forward(img[i])

          img[i].flags.writeable = True

          # Compute FPS
          curr_time = time.time()
          fps = 1/(curr_time-prev_time[i])
          if mode=='body':
              param[i]['fps'] = fps
          elif mode=='hand':
              param[i][0]['fps'] = fps
          elif mode=='holistic':
              for p in param[i]:
                  p['fps'] = fps
          prev_time[i] = curr_time
          # print('========================================')
          # print(param)
          # print(len(param), len(param[0]))

      # cv2.imshow('img'+str(i), img[0])
      # cv2.imshow('img', img[0])

      # Perform triangulation
      # if use_panoptic_dataset:
      if len(cam_idx)==2:
          param = tri.triangulate_2views(param, mode)
      else:
          param = tri.triangulate_nviews(param, mode)

      for i in range(len(cam_idx)):
          # Display 2D keypoint
          img[i] = disp[i].draw2d(img[i].copy(), param[i])
          img[i] = cv2.resize(img[i], None, fx=0.5, fy=0.5)
          cv2.imshow('img'+str(i), img[i])
          # Display 3D
          disp[i].draw3d(param[i])
          print('==========================', i,'==========================')
          print(param[i])

      vis.update_geometry(None)
      vis.poll_events()
      vis.update_renderer()

      # img0 = Image()
      # img0 =
      im = np.frombuffer(image_data.data, dtype=np.uint8).reshape(image_data.height, image_data.width, -1)

      img0_pub.publish(bridge.cv2_to_imgmsg(img[0], encoding="8UC3"))

      # img1 = Image()
      # img1 = bridge.cv2_to_imgmsg(img[1], encoding="bgr8")
      # img1_pub.publish(bridge.cv2_to_imgmsg(img[1], encoding="mono8"))

      key = cv2.waitKey(1)
      if key==27:
          break


  # vis.run() # Keep 3D display for visualization

  for p, c in zip(pipe, cap):
      p.pipe.close()
      c.release()

main()
