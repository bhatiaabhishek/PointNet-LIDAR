import os
import sys
import numpy as np

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def main():

    bb_pred = np.load('BB_pred.txt.npy')
    bb_org = np.load('BB_org.txt.npy')

    print (bb_pred.shape)
    print (bb_org.shape)


    dist_arr = []
    iou_list = []
    for f in range(bb_pred.shape[0]):
   
        frame = bb_pred[f]
        frame_gt = bb_org[f]
        dist_cls = []
        for cls in range(bb_pred.shape[1]):
            mycls = frame[cls]
            mycls_gt = frame_gt[cls]
            for p_b in range(8):
                bb1 = {}
                bb2 = {}
                iou = -1
                if (mycls[p_b][0] > 0.8):
                    xyz = mycls[p_b][1:4]
                    xyz = xyz.reshape((1,3))
                    mycls_gt_filter = np.all([mycls_gt[:,0]==1], axis=0)
                    mycls_gt_f = mycls_gt[mycls_gt_filter]
                    xyz_gt = mycls_gt_f[:,1:4]
                    dist = np.linalg.norm(xyz_gt-xyz, axis=1)
                    min_idx = np.argmin(dist)
                    #print(dist[min_idx])
                    # calcullate lxw IOU, bird's eye
                    # top-left = xc-w/2,yc-l/2
                    # bottom right = xc+w/2, yc+l/2
                    xc_p = mycls[p_b][1]
                    yc_p = mycls[p_b][2]
                    l_p = mycls[p_b][4]
                    w_p = mycls[p_b][5]

                    x_org = mycls_gt_f[min_idx,1]
                    y_org = mycls_gt_f[min_idx,2]
                    l_org = mycls_gt_f[min_idx,4]
                    w_org = mycls_gt_f[min_idx,5]

                    bb1['x1'] = xc_p-(w_p/2)
                    bb1['x2'] = xc_p+(w_p/2) 
                    bb1['y1'] = yc_p-(l_p/2)
                    bb1['y2'] = yc_p+(l_p/2)
                    
                    bb2['x1'] = x_org-(w_org/2)
                    bb2['x2'] = x_org+(w_org/2) 
                    bb2['y1'] = y_org-(l_org/2)
                    bb2['y2'] = y_org+(l_org/2)
         
                    iou = get_iou(bb1,bb2)
                    iou_list.append(iou)


    ids = np.all([np.array(iou_list) > 0], axis = 0)
    print("mean_iou= ", np.mean(np.array(iou_list)[ids]))
    print (sorted(iou_list)[-10:])
    ids_best = np.all([np.array(iou_list) > 0.5], axis = 0)
    print ("IOUs = ", np.array(iou_list)[ids_best]) 
if __name__ == "__main__":
    main()
