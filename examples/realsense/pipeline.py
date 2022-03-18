import pyrealsense2 as rs


pipeline = rs.pipeline()
profile = pipeline.start()
try:
    for i in range(100):
        frames = pipeline.wait_for_frames()
        for f in frames:
            print(f.profile)
finally:
    pipeline.stop()
