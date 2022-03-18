import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()
wrapper = rs.pipeline_wrapper(pipeline)
profile = config.resolve(wrapper)
device = profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

print(device_product_line)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale
print('clipping_distance =', clipping_distance)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Streaming
frames = pipeline.wait_for_frames()
aligned_frames = align.process(frames)

aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
color_frame = aligned_frames.get_color_frame()
pipeline.stop()

color_image = np.asanyarray(color_frame.get_data())
# print(color.shape)

depth = np.asanyarray(aligned_depth_frame.get_data())
depth_meters = depth * depth_scale
dist, _, _, _ = cv2.mean(depth_meters)
print(dist)
print(depth_meters[220:260, 300:340])
# print(depth.shape)
# print(depth[:10, :25])

