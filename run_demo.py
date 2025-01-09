# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse

import zmq
import cv2

def flush_queue(socket_in):
    while True:
        try:
            msg = socket_in.recv(flags=zmq.NOBLOCK)
            #print("Discarding a leftover message.")
        except zmq.Again:
            break


if __name__=='__main__':
  cameraMatrix = np.array([
        [428.559,    0, 317.942],
        [   0, 427.2, 241.016],
        [   0,    0,   1.0]
    ], dtype=np.float64)

  context = zmq.Context(2)
  socket_in = context.socket(zmq.PULL)
  socket_in.bind("tcp://*:5555")
  socket_out = context.socket(zmq.PUSH)
  socket_out.bind("tcp://*:5554")



  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  mesh = trimesh.load(args.mesh_file)

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")
  i = 0
  try:
    while True:
        flush_queue(socket_in)
        msg_color = socket_in.recv(flags=0)
        msg_depth = None
        msg_timestamp = None
        try:
            msg_depth = socket_in.recv(flags=zmq.NOBLOCK)
            msg_timestamp = socket_in.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            pass
        compressed_cframe = np.frombuffer(msg_color, dtype=np.uint8)
        cframe = cv2.imdecode(compressed_cframe, cv2.IMREAD_COLOR)
        
        H, W = cframe.shape[:2]
        
        compressed_dframe = np.frombuffer(msg_depth, dtype=np.uint8)
        dframe = cv2.imdecode(compressed_dframe, cv2.IMREAD_UNCHANGED)
        depth = dframe / 1e3  
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
        depth[(depth < 0.001) | (depth >= np.inf)] = 0
        if i == 0:
            cframe_hsv = cv2.cvtColor(cframe, cv2.COLOR_BGR2HSV)
            object_ub = np.array([35,255,255])
            object_lb = np.array([14, 80,0])
            mask = cv2.inRange(cframe_hsv, object_lb, object_ub)
            #cv2.imshow("pre bool mask", mask)

            mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST).astype(bool)
            #cv2.imshow("mask", mask.astype(np.uint8)*255)

            pose = est.register(K=cameraMatrix, rgb=cframe, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
        else: 
            pose = est.track_one(rgb=cframe, depth=depth, K=cameraMatrix, iteration=args.track_refine_iter)
        i += 1
        
        if debug>=1:
            center_pose = pose@np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(K=cameraMatrix, img=cframe, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(cframe, ob_in_cam=center_pose, scale=0.1, K=cameraMatrix, thickness=3, transparency=0, is_input_rgb=True)
            cv2.imshow('1', vis[...,::-1])
                
        rotation_vector, _ = cv2.Rodrigues(pose[:3, :3])
        pose_6d = np.concatenate((rotation_vector.flatten(), pose[:3, 3])).astype(np.float64)

        #print(pose[:3, 3])
        socket_out.send(pose_6d.tobytes(), zmq.SNDMORE)
        socket_out.send(msg_timestamp)
    
        cv2.waitKey(1)
  except KeyboardInterrupt:
        pass
  finally:
        cv2.destroyAllWindows()
        socket_in.close()
        socket_out.close()
        context.term()
