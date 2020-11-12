# calculate curavature
import os
import h5py
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch




def get_KNN_points(center_point, xyz, k):
    new = np.tile(center_point, (2048, 1))
    delta = xyz - new
    dist = np.sum(delta * delta, 1)
    dist = torch.from_numpy(dist)
    sorteDis, pos = dist.sort()
    knn_points_ids = pos[1:k + 1]
    knn_points = xyz[knn_points_ids, :]


    return knn_points


def get_triangle_angles(center_point, p1, p2):
    p0p1 = math.sqrt((center_point[0] - p1[0]) ** 2 + (center_point[1] - p1[1]) ** 2 + (center_point[2] - p1[2]) ** 2)
    p1p2 = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)
    p0p2 = math.sqrt((center_point[0] - p2[0]) ** 2 + (center_point[1] - p2[1]) ** 2 + (center_point[2] - p2[2]) ** 2)
    a = (p0p1 ** 2 + p1p2 ** 2 - p0p2 ** 2) / 2 / p0p1 / p1p2
    b = (p0p2 ** 2 + p1p2 ** 2 - p0p1 ** 2) / 2 / p0p2 / p1p2
    # ensure not get out the bound of [-1,1]
    if a<-1:
        a = -0.9999999
    if a>1:
        a = 0.9999999
    if b<-1:
        b = -0.9999999
    if b>1:
        b = 0.9999999
        
    angleP1 = math.acos(a) * 180.0 / math.pi
    angleP2 = math.acos(b) * 180.0 / math.pi
    angleP0 = 180 - angleP1 - angleP2
    return angleP0, angleP1, angleP2, p1p2, p0p2, p0p1


def get_triangle_area( angleP0, angleP1, angleP2, p1p2, p0p2, p0p1 ):
    R = p1p2 / (2 * math.sin((angleP0 / 180) * math.pi))
    if angleP0 < 90:
        Area = (math.sqrt(R * R - p0p1 * p0p1 / 4) * p0p1) / 4 + (math.sqrt(R * R - p0p2 * p0p2 / 4) * p0p1) / 4

    else:
        Area = ((p1p2 * p0p1 * p0p2) / (4 * R)) / 2

    return Area



def get_curcature(center_point, center_normal, knn_points):
    KH = 0
    KG = 0
    # k = size(knn_points, 2)
    Area = np.zeros((k, 1))
    angles_P0 = np.zeros((k, 1))
    angles_P1 = np.zeros((k, 1))
    angles_P2 = np.zeros((k, 1))

    # get angles and areas

    for m in range(k):

        p1 = knn_points[m, :]

        if (m == int(k - 1)):

            p2 = knn_points[1, :]
        else:
            p2 = knn_points[m + 1, :]

        # get angels of triangle
        angleP0, angleP1, angleP2, p1p2, p0p2, p0p1 = get_triangle_angles(center_point, p1, p2)
        angles_P0[m, :] = angleP0 * math.pi / 180
        angles_P1[m, :] = angleP1 * math.pi / 180
        angles_P2[m, :] = angleP2 * math.pi / 180
        # get Area of triangle
        if angleP0 == 0:
            angleP0 = 0.000001
            print(angleP0)
        Area[m, :] = get_triangle_area(angleP0, angleP1, angleP2, p1p2, p0p2, p0p1)

    # cal curcature
    for m in range(k):

        if m == 1:
            aplha = angles_P1[k-1,:]
            beta = angles_P2[m,:]
        else:

            aplha = angles_P1[m - 1,:]
            beta = angles_P2[m,:]
        P1P0 = knn_points[m,:] - center_point
        try:
            KH = KH + (1/math.tan(aplha) + 1/math.tan(beta) ) * (sum(P1P0 * center_normal))
        except ZeroDivisionError:
            KH = KH
        KG = KG + angles_P0[m,:]


    Am = sum(Area)
    KH = KH / (4 * Am)
    KG = (2 * math.pi - KG) / Am

    K1 = KH + (KH ** 2 - KG) ** (1 / 2)
    K2 = KH - (KH ** 2 - KG) ** (1 / 2)

    return KH,KG,K1,K2


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data']
    n = f['normal']
    return (data, n)

xyz1, normal1 = load_h5('ply_data_test1.h5')
#  you can download the dataset from 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'


for i in range(len(xyz1)):
    print(i)
    k = 10
    xyz, normal = xyz1[i], normal1[i]
    KH = np.zeros((2048, 1))
    KG = np.zeros((2048, 1))
    K1 = np.zeros((2048, 1))
    K2 = np.zeros((2048, 1))

    for m in range(2048):
        center_point = np.squeeze(xyz[m, :])
        center_normal = np.squeeze(normal[m, :])
        knn_points = get_KNN_points(center_point, xyz, k)
        KH[m, :], KG[m, :], K1[m, :], K2[m, :] = get_curcature(center_point, center_normal, knn_points)


    # where K1,k2 are the Principal curvature, KH is Mean curvature， KG is Gaussian curvature



