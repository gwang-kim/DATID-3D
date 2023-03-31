import numpy as np
import os
import torch
import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_root', type=str, default="", help='process folder')
    parser.add_argument('--out_root', type=str, default="output", help='output folder')
    args = parser.parse_args()
    in_root = args.in_root

    def compute_rotation(angles):
        """
        Return:
            rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat

        Parameters:
            angles           -- torch.tensor, size (B, 3), radian
        """

        batch_size = angles.shape[0]
        ones = torch.ones([batch_size, 1])
        zeros = torch.zeros([batch_size, 1])
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],
        
        rot_x = torch.cat([
            ones, zeros, zeros,
            zeros, torch.cos(x), -torch.sin(x), 
            zeros, torch.sin(x), torch.cos(x)
        ], dim=1).reshape([batch_size, 3, 3])
        
        rot_y = torch.cat([
            torch.cos(y), zeros, torch.sin(y),
            zeros, ones, zeros,
            -torch.sin(y), zeros, torch.cos(y)
        ], dim=1).reshape([batch_size, 3, 3])

        rot_z = torch.cat([
            torch.cos(z), -torch.sin(z), zeros,
            torch.sin(z), torch.cos(z), zeros,
            zeros, zeros, ones
        ], dim=1).reshape([batch_size, 3, 3])

        rot = rot_z @ rot_y @ rot_x
        return rot.permute(0, 2, 1)[0]

    npys = sorted([x for x in os.listdir(in_root) if x.endswith(".npy")])

    mode = 1 #1 = IDR, 2 = LSX
    outAll={}

    for src_filename in npys:
        src = os.path.join(in_root, src_filename)
        
        print(src)
        dict_load=np.load(src, allow_pickle=True)

        angle = dict_load.item()['angle']
        trans = dict_load.item()['trans'][0]
        R = compute_rotation(torch.from_numpy(angle)).numpy()
    
        trans[2] += -10
        c = -np.dot(R, trans)
        pose = np.eye(4)
        pose[:3, :3] = R

        c *= 0.27 # factor to match tripleganger
        c[1] += 0.006 # offset to align to tripleganger
        c[2] += 0.161 # offset to align to tripleganger
        c = c/np.linalg.norm(c)*2.7  ##yiqian教我放到半球上去
        pose[0,3] = c[0]
        pose[1,3] = c[1]
        pose[2,3] = c[2] 

        focal = 2985.29 # = 1015*1024/224*(300/466.285)#
        pp = 512#112
        w = 1024#224
        h = 1024#224

        if mode==1:
            count = 0
            K = np.eye(3)
            K[0][0] = focal
            K[1][1] = focal
            K[0][2] = w/2.0
            K[1][2] = h/2.0
            K = K.tolist()

            Rot = np.eye(3)
            Rot[0, 0] = 1
            Rot[1, 1] = -1
            Rot[2, 2] = -1        
            pose[:3, :3] = np.dot(pose[:3, :3], Rot)

            pose = pose.tolist()
            out = {}
            out["intrinsics"] = K
            out["pose"] = pose
            out["angle"] = (angle * [1, -1, 1]).flatten().tolist()
            outAll[src_filename.replace(".npy", ".png")] = out

        elif mode==2:

            dst = os.path.join(in_root, src_filename.replace(".npy", "_lscam.txt"))
            outCam = open(dst, "w")
            outCam.write("#focal length\n")
            outCam.write(str(focal) + " " + str(focal) + "\n")

            outCam.write("#principal point\n")
            outCam.write(str(pp) + " " + str(pp) + "\n")

            outCam.write("#resolution\n")
            outCam.write(str(w) + " " + str(h) + "\n")

            outCam.write("#distortion coeffs\n")
            outCam.write("0 0 0 0\n")


            outCam.write("MATRIX :\n")
            for r in range(4):
                outCam.write(str(pose[r, 0]) + " " + str(pose[r, 1]) + " " + str(pose[r, 2]) + " " + str(pose[r, 3]) + "\n")

            outCam.close()

    if mode == 1:
        dst = os.path.join(args.out_root, "cameras.json")
        with open(dst, "w") as outfile:
            json.dump(outAll, outfile, indent=4)
