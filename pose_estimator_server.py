#!/usr/bin/env python3
from estimater import *
from datareader import *
import argparse

import msgpack
import zmq
import cv2
import struct

import os
import socket

FIVEG_ENABLED = (os.environ["FIVEG_ENABLED"] == 'true')
ipv6_address = os.environ["CLIENT_IP6"] if FIVEG_ENABLED else None 

print(zmq.has('draft'))
if FIVEG_ENABLED:
    print("Using 5g")
    sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
    local_port = 5555
    remote_port = 5555
    sock.bind(('', local_port))
    sock.sendto("hello".encode(), (ipv6_address, remote_port))
    del sock


def flush_queue(socket_in):
    while True:
        try:
            msg = socket_in.recv(flags=zmq.NOBLOCK)
            #print("Discarding a leftover message.")
        except zmq.Again:
            break


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = param[y, x]
        print(f"H: {pixel[0]}; S: {pixel[1]}; V: {pixel[2]}")


if __name__=='__main__':
  cameraMatrix = np.array([
        [428.559,    0, 317.942],
        [   0, 427.2, 241.016],
        [   0,    0,   1.0]
    ], dtype=np.float64)

  context = zmq.Context(2)

  socket_in = context.socket(zmq.PULL)
  socket_out = context.socket(zmq.PUSH)
  socket_control = context.socket(zmq.REP)
  socket_in.bind("tcp://0.0.0.0:5555")
  socket_out.bind("tcp://0.0.0.0:5554")
  socket_control.bind("tcp://0.0.0.0:5553")
  #socket_in.bind("tcp://129.97.71.51:5555")
  #socket_out.bind("tcp://129.97.71.51:5554")
  #socket_control.bind("tcp://129.97.71.51:5553")
  
  poller = zmq.Poller()
  poller.register(socket_in, zmq.POLLIN)
  poller.register(socket_control, zmq.POLLIN)

  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  args = parser.parse_args()

  import logging
  set_logging_format(logging.WARNING)
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
  mode = 'registration'
  frames_storage = []
  run_cnt = 0

  global_frame_cnt = -1
  black_image = np.zeros((480, 640, 3), dtype=np.uint8)
  if debug >= 1:
      cv2.imshow(f'frame', black_image)

  try:
    while True:
        flush_queue(socket_in)
        events = dict(poller.poll(timeout=-1))
        if socket_control in events:
            msg_restart_tag = socket_control.recv(flags=0)
            mode = 'registration'
             
            #output_folder = f'frames/output_frames_run{run_cnt}'
            #os.makedirs(output_folder, exist_ok=True)
            #print("saving", len(frames_storage))
            
            #for i, frame in enumerate(frames_storage):
            #    filename = os.path.join(output_folder, f'frame_{i:04d}_{frame["number"]}_{frame["iter"]}.png')
            #    cv2.imwrite(filename, frame["frame"])
            frames_storage = []
            run_cnt += 1
            
            socket_control.send(b"reset")
            print("restarted")
            continue

        msg_ee_transform = None 
        msg_timestamp = None
        msg_frame_number = None
        msg_iteration_idx = None

        msg_color = socket_in.recv(flags=0)
        msg_depth = None
        msg_timestamp = None
        msg_restart_tag = None
        try:
            msg_depth = socket_in.recv(flags=zmq.NOBLOCK)
            msg_frame_number = socket_in.recv(flags=zmq.NOBLOCK)
            msg_timestamp = socket_in.recv(flags=zmq.NOBLOCK)
            msg_ee_transform = socket_in.recv(flags=zmq.NOBLOCK)
            msg_iteration_idx = socket_in.recv(flags=zmq.NOBLOCK)
        except zmq.Again as e:
            print(e)
            continue
        
        t = time.time()
  
      
        frame_number = struct.unpack('i', msg_frame_number)[0]
        iteration_idx = struct.unpack('i', msg_iteration_idx)[0]
        timestamp_in = struct.unpack('Q',msg_timestamp)[0]
        #print(f"ts: {timestamp_in}, R: {iteration_idx}, F{frame_number}")

        #if frame_number < global_frame_cnt:
        #    continue
        # global_frame_cnt = frame_number

        compressed_cframe = np.frombuffer(msg_color, dtype=np.uint8)
        compressed_dframe = np.frombuffer(msg_depth, dtype=np.uint8)
          

        cframe = cv2.imdecode(compressed_cframe, cv2.IMREAD_COLOR)
        #frames_storage.append({"frame": cframe.copy(), "number": frame_number, "iter": iteration_idx})

        H, W = cframe.shape[:2] 
        dframe = cv2.imdecode(compressed_dframe, cv2.IMREAD_UNCHANGED)
        depth = dframe / 1e3  
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
        depth[(depth < 0.001) | (depth >= np.inf)] = 0
        global was_registration 
        if mode == 'registration':
            cframe_hsv = cv2.cvtColor(cframe, cv2.COLOR_BGR2HSV)
            #cv2.imshow("hsv", cframe_hsv)
            #cv2.setMouseCallback("hsv", mouse_callback, cframe_hsv)
            
            object_ub = np.array([40,255,255])
            object_lb = np.array([0, 50,150])
            mask = cv2.inRange(cframe_hsv, object_lb, object_ub)
            mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST)

            rect_mask = np.zeros_like(mask, dtype=np.uint8)
            verticies = np.array([[272, 170], [223, 475], [487, 476], [405, 169]], dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(rect_mask, [verticies], True)

            start_rect_mask = np.zeros_like(mask, dtype=np.uint8)
            verticies_start = np.array([[260, 200], [223, 475], [487, 476], [419, 200]], dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(start_rect_mask, [verticies_start], True)
            
            start_mask = np.where(start_rect_mask, mask, 0)
            contours, _ = cv2.findContours(start_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #print("------------>detecting")
            detected = False
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                #cv2.rectangle(cframe, (x, y), (x + w, y + h), (0, 0, 255), 2)
                if w >= 20 and h >= 40:
                    detected = True
                    break
            
            if not detected:
                continue
            mask = np.where(rect_mask, mask, 0)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            fin_mask = np.zeros_like(mask, dtype=bool)
            best_bbox = {
                    "sq": 0,
                    "location": None
                    }
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                #cv2.rectangle(cframe, (x, y), (x + w, y + h), (0, 255, 0), 2)    
                if w >= 20 and h >= 40:
                    if (w * h > best_bbox["sq"]):
                        best_bbox["sq"] = w* h
                        best_bbox["location"] = (x, y, w, h)
                 
            if not best_bbox["location"]:
                continue
            x, y, w, h = best_bbox["location"]
            fin_mask[y:y+h, x:x+w] = True

            fake_pose_6d = np.zeros(6, dtype=np.float64)
            zero = 0
            zero_bytes = zero.to_bytes(4, byteorder='little', signed=True)
             
       
            socket_out.send(fake_pose_6d.tobytes(), zmq.SNDMORE)
            socket_out.send(msg_frame_number, zmq.SNDMORE)
            socket_out.send(msg_timestamp, zmq.SNDMORE)
            socket_out.send(msg_ee_transform, zmq.SNDMORE)
            socket_out.send(msg_iteration_idx, zmq.SNDMORE)
            socket_out.send(zero_bytes)
            
            if debug >= 1:
                cv2.imshow("cframe", fin_mask.astype(np.uint8)*255)
                cv2.imshow("cframe1", mask)
                cv2.imshow("rl", cframe)

                cv2.waitKey(0)
                pass
            mask = mask.astype(bool)
            
            #fp_input = cv2.cvtColor(cframe, cv2.COLOR_BGR2RGB)    
            pose = est.register(K=cameraMatrix, rgb=cframe, depth=depth, ob_mask=fin_mask, iteration=args.est_refine_iter)
            print("Starting registration")
            print(f'registration time: {time.time() - t}') 
            was_registration = True
            mode = 'tracking'
            #TODO: REMOVE 
            """
            frames_storage.append({"frame": cframe.copy(), "number": 1})
            depth_norm = cv2.normalize(dframe.copy(), None, 0, 255, cv2.NORM_MINMAX)
            depth_norm = np.uint8(depth_norm.copy())
            frames_storage.append({"frame": depth_norm.copy(), "number": 2})
            frames_storage.append({"frame": mask.astype(np.uint8)*255, "number": 3})
            frames_storage.append({"frame": fin_mask.astype(np.uint8)*255, "number": 4})
            center_pose = pose@np.linalg.inv(to_origin)

            vis = draw_posed_3d_box(K=cameraMatrix, img=cframe, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(cframe, ob_in_cam=center_pose, scale=0.1, K=cameraMatrix, thickness=3, transparency=0, is_input_rgb=False)
            frames_storage.append({"frame": vis.copy(), "number": 5})
            
            output_folder = f'frames/output_frames_run{run_cnt}'
            os.makedirs(output_folder, exist_ok=True)
            

            for i, frame in enumerate(frames_storage):
                print(frame["number"])
                filename = os.path.join(output_folder, f'frame_{i:04d}_{frame["number"]}.png')
                cv2.imwrite(filename, frame["frame"])
            """

        else:
            was_registration = False
            
            #fp_input = cv2.cvtColor(cframe, cv2.COLOR_BGR2RGB)
            pose = est.track_one(rgb=cframe, depth=depth, K=cameraMatrix, iteration=args.track_refine_iter)
            #print(f'tracking time: {time.time() - t}')

        center_pose = pose@np.linalg.inv(to_origin)

        
        #frames_storage.append({"frame": cframe.copy(), "number": frame_number, "iter": iteration_idx})

        
        #frames_storage.append({"frame": cframe.copy(), "number": frame_number, "iter": iteration_idx})
        #vis = draw_posed_3d_box(K=cameraMatrix, img=cframe, ob_in_cam=center_pose, bbox=bbox) 
        #vis = draw_xyz_axis(cframe, ob_in_cam=center_pose, scale=0.1, K=cameraMatrix, thickness=3, transparency=0, is_input_rgb=False)
        #frames_storage.append({"frame": vis, "number": frame_number, "iter": iteration_idx})
        
        if debug >= 1:           
            #vis = draw_posed_3d_box(K=cameraMatrix, img=cframe, ob_in_cam=center_pose, bbox=bbox)
            #vis = draw_xyz_axis(cframe, ob_in_cam=center_pose, scale=0.1, K=cameraMatrix, thickness=3, transparency=0, is_input_rgb=False)
            timestamp = struct.unpack('Q', msg_timestamp)[0]
            if was_registration:
                cv2.imshow(f'reg', vis) 
            cv2.imshow(f'tracking', vis) 

        rotation_vector, _ = cv2.Rodrigues(center_pose[:3, :3])
        pose_6d = np.concatenate((rotation_vector.flatten(), center_pose[:3, 3])).astype(np.float64)
        
        processing_time = int((time.time() - t) * 1000)
        
        socket_out.send(pose_6d.tobytes(), zmq.SNDMORE)
        socket_out.send(msg_frame_number, zmq.SNDMORE)
        socket_out.send(msg_timestamp, zmq.SNDMORE)
        socket_out.send(msg_ee_transform, zmq.SNDMORE)
        socket_out.send(msg_iteration_idx, zmq.SNDMORE)
        socket_out.send(processing_time.to_bytes(4, byteorder='little', signed=True))
        if debug >= 1:
            cv2.waitKey(1)

  except KeyboardInterrupt:
        pass
  finally:
        cv2.destroyAllWindows()
        socket_in.close()
        socket_out.close()
        context.term()
