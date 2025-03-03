from estimater import *
from datareader import *
import argparse

import zmq
import cv2
import struct

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
  socket_in.bind("tcp://129.97.71.51:5555")
  socket_out = context.socket(zmq.PUSH)
  socket_out.bind("tcp://129.97.71.51:5554")



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
  i = 0

  black_image = np.zeros((480, 640, 3), dtype=np.uint8)
  cv2.imshow(f'frame', black_image)

  try:
    while True:
        flush_queue(socket_in)
        msg_color = socket_in.recv(flags=0)
        
        msg_depth = None
        msg_timestamp = None
        try:
            msg_depth = socket_in.recv(flags=zmq.NOBLOCK)
            msg_frame_number = socket_in.recv(flags=zmq.NOBLOCK)
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
            #cv2.imshow("hsv", cframe_hsv)
            #cv2.setMouseCallback("hsv", mouse_callback, cframe_hsv)

            object_ub = np.array([40,255,255])
            object_lb = np.array([0, 50,120])
            mask = cv2.inRange(cframe_hsv, object_lb, object_ub)
            mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST)

            rect_mask = np.zeros_like(mask, dtype=np.uint8)
            verticies = np.array([[268, 150], [209, 475], [453, 476], [395, 150]], dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(rect_mask, [verticies], True)

            start_rect_mask = np.zeros_like(mask, dtype=np.uint8)
            verticies_start = np.array([[268, 212], [209, 475], [453, 476], [421, 216]], dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(start_rect_mask, [verticies_start], True)
            
            start_mask = np.where(start_rect_mask, mask, 0)
            contours, _ = cv2.findContours(start_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detected = False
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                #cv2.rectangle(cframe, (x, y), (x + w, y + h), (0, 255, 0), 2)
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
            socket_out.send(fake_pose_6d.tobytes(), zmq.SNDMORE)
            socket_out.send(msg_frame_number, zmq.SNDMORE)
            socket_out.send(msg_timestamp)
                    
            mask = mask.astype(bool)
            pose = est.register(K=cameraMatrix, rgb=cframe, depth=depth, ob_mask=fin_mask, iteration=args.est_refine_iter)
            
        else:
            pose = est.track_one(rgb=cframe, depth=depth, K=cameraMatrix, iteration=args.track_refine_iter)
        
        center_pose = pose@np.linalg.inv(to_origin)

        if debug>=1:
            vis = draw_posed_3d_box(K=cameraMatrix, img=cframe, ob_in_cam=center_pose, bbox=bbox)            
            vis = draw_xyz_axis(cframe, ob_in_cam=center_pose, scale=0.1, K=cameraMatrix, thickness=3, transparency=0, is_input_rgb=True)
            frame_number = struct.unpack('i', msg_frame_number)[0]
            timestamp = struct.unpack('Q', msg_timestamp)[0]
            cv2.imshow(f'frame', vis[...,::-1]) 
            
        if i > 0:
            rotation_vector, _ = cv2.Rodrigues(center_pose[:3, :3])
            pose_6d = np.concatenate((rotation_vector.flatten(), center_pose[:3, 3])).astype(np.float64)
            socket_out.send(pose_6d.tobytes(), zmq.SNDMORE)
            socket_out.send(msg_frame_number, zmq.SNDMORE)
            socket_out.send(msg_timestamp)

        cv2.waitKey(1)
        i += 1

  except KeyboardInterrupt:
        pass
  finally:
        cv2.destroyAllWindows()
        socket_in.close()
        socket_out.close()
        context.term()
