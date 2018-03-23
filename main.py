import os.path
import sys
import tensorflow as tf
import warnings
import tf_util
import pykitti
import numpy as np
import parseTrackletXML as xmlParser
import cv2
from sklearn.metrics import f1_score, accuracy_score




## Check for a GPU
#if not tf.test.gpu_device_name():
#    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
#else:
#    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def input_transform_net(point_cloud, is_training, bn_decay=None, K=3):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix of size 3xK """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    input_image = tf.expand_dims(point_cloud, -1)
    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_XYZ') as sc:
        assert(K==3)
        weights = tf.get_variable('weights', [256, 3*K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [3*K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant([1,0,0,0,1,0,0,0,1], dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, 3, K])
    return transform




def MyModel(point_cloud, is_training,bn_decay=None):
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    
    # Point functions (MLP implemented as conv2d)
    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)
    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    net_out_ = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net_out_, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
    
    # Symmetric function: max pooling
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool')
    
    net_tiled = tf.tile(net,[1,num_point,1,1])
    net_concat = tf.concat([net_out_,net_tiled],3)
    net = tf_util.conv2d(net_concat, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv6', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv7', bn_decay=bn_decay)
    # MLP on global point cloud vector
    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training,
                                  scope='fc3', bn_decay=bn_decay)
    net_dp = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net_dp, 512,  activation_fn=None,  scope='fc4')
    net_count = tf_util.fully_connected(net_dp, 8, scope='fc4_1')
    
    
    
    return net, end_points, net_count

def load_tracklets_for_frames(n_frames, xml_path):
    """
    Loads dataset labels also referred to as tracklets, saving them individually for each frame.

    Parameters
    ----------
    n_frames    : Number of frames in the dataset.
    xml_path    : Path to the tracklets XML.

    Returns
    -------
    Tuple of dictionaries with integer keys corresponding to absolute frame numbers and arrays as values. First array
    contains coordinates of bounding box vertices for each object in the frame, and the second array contains objects
    types as strings.
    """
    tracklets = xmlParser.parseXML(xml_path)

    frame_tracklets = {}
    frame_tracklets_types = {}
    frame_tracklets_yaw = {}
    for i in range(n_frames):
        frame_tracklets[i] = []
        frame_tracklets_types[i] = []
        frame_tracklets_yaw[i] = []

    # loop over tracklets
    for i, tracklet in enumerate(tracklets):
        # this part is inspired by kitti object development kit matlab code: computeBox3D
        h, w, l = tracklet.size
        # in velodyne coordinates around zero point and without orientation yet
        trackletBox = np.array([
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0.0, 0.0, 0.0, 0.0, h, h, h, h]
        ])
        # loop over all data in tracklet
        for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in tracklet:
            # determine if object is in the image; otherwise continue
            if truncation not in (xmlParser.TRUNC_IN_IMAGE, xmlParser.TRUNC_TRUNCATED):
                continue
            # re-create 3D bounding box in velodyne coordinate system
            yaw = rotation[2]  # other rotations are supposedly 0
            assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
            rotMat = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0]
            ])
            cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T
            frame_tracklets[absoluteFrameNumber] = frame_tracklets[absoluteFrameNumber] + [cornerPosInVelo]
            frame_tracklets_types[absoluteFrameNumber] = frame_tracklets_types[absoluteFrameNumber] + [
                tracklet.objectType]
            frame_tracklets_yaw[absoluteFrameNumber] = frame_tracklets_yaw[absoluteFrameNumber] + [yaw]

    return (frame_tracklets, frame_tracklets_types, frame_tracklets_yaw)


def load_dataset(basedir, date,drive, num_points=2048, calibrated=False):

    dataset = pykitti.raw(basedir, date, drive)
    tracklet_rects, tracklet_types, tracklet_yaw = load_tracklets_for_frames(len(list(dataset.velo)), '{}/{}/{}_drive_{}_sync/tracklet_labels.xml'.format(basedir,date, date, drive))
    calib_data = None
    if calibrated:
        dataset._load_calib()
        calib_data = dataset.calib
    dataset_velo = list(dataset.velo)
    print(" loading ", len(list(dataset.velo)), " frames")
    # Sample x% of the point clouds
    velo_data = np.zeros((len(dataset_velo),num_points,3))
    labels = np.zeros((len(dataset_velo),8))
    labels_bbox = np.zeros((len(dataset_velo),8,8,8)) # 4 bounding boxes per frame per class, 8 values - prob,x,y,z,l,w,h,yaw
    class_ids = {
        'Car': 0,
        'Tram': 1,
        'Cyclist': 2,
        'Van': 3,
        'Truck': 4,
        'Pedestrian': 5,
        'Person (sitting)': 6,
        'Misc': 7
    }
    for f in range(len(dataset_velo)):
        frame = dataset_velo[f]
        # Cutting down the region of interest
        # Select only X from 10 to 40
        velo_front_index = np.all([frame[:,0]>10, frame[:,0]<40],axis=0)
        velo_filter = frame[velo_front_index]
        sample = num_points/float(velo_filter.shape[0]) # fraction of the points to sample
        points_step = int(1. / sample)
        velo_range = range(0,velo_filter.shape[0],points_step)
        velo_sampled = velo_filter[velo_range]
        #print("velo_sampled shape =", velo_sampled.shape)
        velo_data[f,:,:] = velo_sampled[:num_points,:3] # remove reflectance

        # prepare labels
        # select objects only in the above view
        # trects is of the format 
        # x1 x2 x3....
        # y1 y2 y3....
        # z1 z2 z3
       
        # Ground truth label of [xc,yc,zc,l,w,h]
        for t_rects, t_type, t_yaw in zip(tracklet_rects[f], tracklet_types[f], tracklet_yaw[f]):
            if(max(t_rects[0,:]) < 10 or min(t_rects[0,:])>40):
                continue
            labels[f,class_ids[t_type]] += 1
            xc = (t_rects[0,1] + t_rects[0,2])/2
            yc = (t_rects[1,2] + t_rects[1,3])/2
            zc = (t_rects[2,3] + t_rects[2,4])/2
            hg = abs(t_rects[2,4] - t_rects[2,3])
            wg = abs(t_rects[0,1] - t_rects[0,2])
            lg = abs(t_rects[1,2] - t_rects[1,3])
            #print ( min(labels[f,class_ids[t_type]]-1,3))
            
            labels_bbox[f,class_ids[t_type],int(min(labels[f,class_ids[t_type]]-1,7))] = np.array([1, xc, yc, zc, lg, wg, hg, abs(t_yaw)])
        #sum_labels = sum(labels[f])
        #if (sum_labels != 0):
        #    labels[f] = labels[f]/np.sum(labels[f])
        #print ("labels per frame =", labels[f])
    print ("max number of boxes = ", max(labels.flatten()))
    return velo_data, labels, labels_bbox, calib_data


def optimize(nn_last_layer, correct_label, learning_rate, num_classes, end_points,  net_count, count_label_pl,reg_weight=0.001, use_l2_loss=True):
    
    print (nn_last_layer.shape)
    print (correct_label.shape)
    logits_count = tf.reshape(net_count, (-1,num_classes),name='logits_count')
    labels_count = tf.reshape(count_label_pl, (-1, num_classes))
    logits = tf.reshape(nn_last_layer, (-1, num_classes),name='logits')
    labels = tf.reshape(correct_label, (-1, num_classes),name='labels')
    print(logits.shape)
    print(labels.shape)
    if use_l2_loss:
        diff = logits_count - labels_count
        loss_l2 = tf.nn.l2_loss(diff)
        loss = tf.reduce_mean( tf.losses.huber_loss(labels,logits)) + 0.01*loss_l2
    else:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, name = 'train_op')


    

    return logits, train_op, loss, logits_count


def train_nn_one_epoch(sess, current_data, current_label, batch_size, num_batches, num_points, train_op, cross_entropy_loss, point_cloud_pl, correct_label, is_training_pl, train_writer, merged, logits, count_label_pl, current_label_count):
    
    total_loss = 0
    for batch in range(num_batches):
        start_idx = batch*batch_size
        end_idx = (batch+1)*batch_size
        _, loss, summary, pred = sess.run([train_op,cross_entropy_loss, merged, logits],feed_dict={point_cloud_pl:current_data[start_idx:end_idx,:,:], correct_label:current_label[start_idx:end_idx],count_label_pl:current_label_count[start_idx:end_idx], is_training_pl:True})
        train_writer.add_summary(summary)
        #pred_error_per_batch = np.sum((current_label[start_idx:end_idx] - pred)**2)/(batch_size*current_label.shape[1])
        #print("pred error = ", pred_error_per_batch)
        #print("pred one frame = ", pred[0])
        #print("label one frame = ", current_label[start_idx])
        total_loss = total_loss + loss
    return total_loss/(num_batches*batch_size)




def eval_nn_one_epoch(sess, current_data, current_label, batch_size, num_batches, num_points, train_op, cross_entropy_loss, point_cloud_pl, correct_label, is_training_pl, test_writer, merged, logits, count_label_pl, current_label_count):
    total_loss = 0
    for batch in range(num_batches):
        start_idx = batch*batch_size
        end_idx = (batch+1)*batch_size
        _, loss, summary, pred = sess.run([train_op,cross_entropy_loss, merged, logits],feed_dict={point_cloud_pl:current_data[start_idx:end_idx,:,:], correct_label:current_label[start_idx:end_idx], count_label_pl:current_label_count[start_idx:end_idx], is_training_pl:False})
        #print("logits shape = ", logits.shape)
        #pred_error_per_batch = np.sum((current_label[start_idx:end_idx] - pred)**2)/(batch_size*current_label.shape[1])
        #print("pred error = ", pred_error_per_batch)
        total_loss = total_loss + loss
    return total_loss/(num_batches*batch_size)


def run(train=True):
    num_classes = 8  
    num_point = 2048*2
    data_dir = './data'
    runs_dir = './runs'
    epochs = 50
    batch_size = 32 

    if (train == True):
        point_cloud_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3),name = 'point_cloud_pl')
        count_label_pl = tf.placeholder(tf.float32, shape=(batch_size,num_classes), name='count_label_pl')
        correct_label = tf.placeholder(tf.float32, shape=(batch_size,num_classes,8,8),name='correct_label')
        is_training_pl = tf.placeholder(tf.bool, shape=(),name='is_training_pl')

        learning_rate = tf.constant(0.001)

        with tf.Session() as sess:
            date = '2011_09_26'
            drive = '0009'
            velo_data_0, label_count_0, labels_0, _ = load_dataset(data_dir,date, drive, num_point)
            drive = '0051'
            velo_data_1, label_count_1, labels_1, _ = load_dataset(data_dir,date, drive, num_point)
            drive = '0091'
            velo_data_2, label_count_2, labels_2, _ = load_dataset(data_dir,date, drive, num_point)
            
            velo_data = np.concatenate((velo_data_0,velo_data_1,velo_data_2),axis=0)
            label_count = np.concatenate((label_count_0,label_count_1, label_count_2),axis=0)
            labels = np.concatenate((labels_0,labels_1, labels_2),axis=0)
            print ("velo data shape = ", velo_data.shape)
            print ("labels shape = ", labels.shape)
            # Shuffle velo data
            frame_idxs = np.arange(0,velo_data.shape[0])
            np.random.shuffle(frame_idxs)
            shuffled_data = velo_data[frame_idxs]
            shuffled_labels = labels[frame_idxs]
            shuffled_label_count = label_count[frame_idxs]

            # grab 20% test data
            num_test_data = int(0.2*shuffled_data.shape[0])
            test_data = shuffled_data[:num_test_data]
            train_data = shuffled_data[num_test_data:]
            test_label_data = shuffled_labels[:num_test_data]
            test_label_count_data = shuffled_label_count[:num_test_data]
            train_label_data = shuffled_labels[num_test_data:]
            train_label_count_data = shuffled_label_count[num_test_data:]


            output_layer, end_points, net_count = MyModel(point_cloud_pl, is_training_pl) 
            logits, train_op, cross_entropy_loss, logits_count = optimize(output_layer, correct_label, learning_rate, num_classes, end_points,net_count, count_label_pl)
            tf.summary.scalar('loss', cross_entropy_loss)

            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(os.path.join('log', 'train'),
                                      sess.graph)
            test_writer = tf.summary.FileWriter(os.path.join('log', 'test'))
  

            # TODO: Train NN using the train_nn function
            sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                frame_idxs = np.arange(0,train_data.shape[0])
                np.random.shuffle(frame_idxs)
                current_data = train_data[frame_idxs]
                current_labels = train_label_data[frame_idxs]
                current_label_count = train_label_count_data[frame_idxs]
                num_batches = current_data.shape[0]//batch_size

                loss = train_nn_one_epoch(sess, current_data,  current_labels, batch_size, num_batches, num_point, train_op, cross_entropy_loss, point_cloud_pl, correct_label, is_training_pl, train_writer,merged, logits, count_label_pl, current_label_count)
                print("Epoch {}/{}...".format(epoch+1, epochs), "Training Loss: {:.4f}...".format(loss))
  
                ## Use test data to validate
                #current_data = test_data
                #current_labels = test_label_data
                #current_label_count = test_label_count_data
                #num_batches = current_data.shape[0]//batch_size
                #test_loss = eval_nn_one_epoch(sess, current_data, current_labels, batch_size, num_batches, num_point, train_op, cross_entropy_loss, point_cloud_pl, correct_label, is_training_pl, test_writer,merged, logits, count_label_pl, current_label_count) 
                #print ("Mean test loss on test data = ", test_loss)
            

            # Save model
            saver = tf.train.Saver()
            saver.save(sess, 'pointnet_new')
    else:        
            # Test
            date = '2011_09_26'
            drive = '0009'
            velo_data,label_count, labels, calib_data = load_dataset(data_dir,date, drive,num_point, True)

            frame_idxs = np.arange(0,velo_data.shape[0])
            #np.random.shuffle(frame_idxs)
            current_data = velo_data[frame_idxs]
            current_labels = labels[frame_idxs]
            current_label_count = label_count[frame_idxs]
            print("current_data shape = ", current_data.shape)
            num_batches = velo_data.shape[0]//batch_size

            # Restore saved model
            sess = tf.Session()
            saver = tf.train.import_meta_graph('pointnet_new.meta')
            saver.restore(sess, tf.train.latest_checkpoint('./'))
            graph = tf.get_default_graph()
            
            point_cloud_pl = graph.get_tensor_by_name("point_cloud_pl:0")
            count_label_pl = graph.get_tensor_by_name("count_label_pl:0")
            correct_label = graph.get_tensor_by_name("correct_label:0")
            is_training_pl = graph.get_tensor_by_name("is_training_pl:0")
            logits = graph.get_tensor_by_name("logits:0")
            logits_count = graph.get_tensor_by_name("logits_count:0")
             
            #test_loss = eval_nn_one_epoch(sess, current_data, current_labels, batch_size, num_batches, num_point, train_op, cross_entropy_loss, point_cloud_pl, correct_label, is_training_pl, test_writer,merged, logits, count_label_pl, current_label_count) 
            #print ("Mean test loss on new drive data = ", test_loss)
            pred, count_pred = sess.run([logits, logits_count],feed_dict={point_cloud_pl:current_data[74:106,:,:], correct_label:current_labels[74:106], count_label_pl:current_label_count[74:106], is_training_pl:False})
            print (pred.shape)
            #print (loss)
            prediction  = pred.reshape(32,num_classes,8,8)
            count_prediction = count_pred.reshape(32, num_classes)

            P_rect_30 = calib_data.P_rect_30
            R_rect_00 = calib_data.R_rect_00
            T_cam3_velo = calib_data.T_cam3_velo
            for j in range(num_classes):
                print("class = ", j)
                print("count = ", count_prediction[-1][j])
                for b in range(8):
                    print("confidence = ", prediction[-1][j][b][0])
                    print("x y l w h yaw =", prediction[-1][j][b][1:])
                    print("GT =", current_labels[105][j][b][0:])

            class_ids = [
                'Car',
                'Tram',
                'Cyclist',
                'Van',
                'Truck',
                'Pedestrian',
                'Sitter',
                'Misc'
            ]
            y_true = []
            y_pred = []
            cls_pred = []
            cls_org = []
            count_error = []
            BB_pred = []
            BB_org = []
            count_error_f = open("error_file.txt", 'w')
            for f in range(0,current_data.shape[0],32):
                if (current_data[f:f+32,:,:].shape[0] !=32):
                    break
                pred, count_pred = sess.run([logits, logits_count],feed_dict={point_cloud_pl:current_data[f:f+32,:,:], correct_label:current_labels[f:f+32], count_label_pl:current_label_count[f:f+32], is_training_pl:False})
                count_pred.reshape((32,num_classes))
                count_org = current_label_count[f:f+32].reshape(32,num_classes)
                prediction  = pred.reshape(32,num_classes,8,8)
                label_org = current_labels[f:f+32]
                #print("Processing "+ str(f)+ " to " + str(f+31))
                for i in range(32):
                    img = cv2.imread(data_dir+"/"+date+"/"+date+"_drive_"+drive+"_sync/image_03/data/"+str(f+i).zfill(10)+".png")
                    cls_list = []
                    cls_org_list = []
                    BB_pred_f = []
                    BB_org_f = []
                    for j in range(num_classes):
                        count = int(round(count_pred[i][j]))
                        count_ref = count_org[i][j]
                        count_error.append(np.abs(count - count_ref))
                        y_true.append(count_ref)
                        y_pred.append(count)
                        if (count > 0):
                            cls_list.append(1)
                        else:
                            cls_list.append(0)
                        if (count_ref >0):
                            cls_org_list.append(1)
                        else:
                            cls_org_list.append(0)
                        cv2.putText(img, class_ids[j] + " = " + str(count)+ "  Org: "+str(count_ref),(25,125+(j*50)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)
                        BB_pred_cls = []
                        BB_org_cls = []
                        for b in range(8):
                            c_org = label_org[i][j][b][0]
                            x_org = label_org[i][j][b][1]
                            y_org = label_org[i][j][b][2]
                            z_org = label_org[i][j][b][3]
                            l_org = label_org[i][j][b][4]
                            w_org = label_org[i][j][b][5]
                            h_org = label_org[i][j][b][6]

                            c = prediction[i][j][b][0]
                            x_3d = prediction[i][j][b][1]
                            y_3d = prediction[i][j][b][2]
                            z_3d = prediction[i][j][b][3]
                            l = prediction[i][j][b][4]
                            w = prediction[i][j][b][5]
                            h = prediction[i][j][b][6]

                            BB_pred_cls.append([c, x_3d, y_3d, z_3d, l, w, h])
                            BB_org_cls.append([c_org, x_org, y_org, z_org, l_org, w_org, h_org])
                            if (label_org[i][j][b][0] == 1):
                                x_3d = prediction[i][j][b][1]
                                y_3d = prediction[i][j][b][2]
                                z_3d = prediction[i][j][b][3]
                                l = prediction[i][j][b][4]
                                w = prediction[i][j][b][5]
                                h = prediction[i][j][b][6]
          


                                X_homo = np.array([x_3d, y_3d, z_3d, 1])
                                x_cam = P_rect_30.dot(R_rect_00).dot(T_cam3_velo).dot(X_homo.T)
                                x_cam_homo = x_cam/x_cam[-1]
                                x_cam = x_cam_homo[0:2]
                                #if (f+i == 148):
                                #    print ( "class, box = "+class_ids[j]+ " , "+str(b))
                                #    pred_str = "Pred Box Coord = " +  "("+str(x_3d)+","+str(y_3d)+","+str(z_3d)+")"
                                #    #cv2.putText(img, pred_str,(250,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)
                                #    pred_str_dim = "Pred (l,w,h) = " + "("+str(l)+","+str(w)+","+str(h)+")"
                                #    #cv2.putText(img, pred_str_dim,(250,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)
                                #    org_str_coord = "Label Coord = " + "("+str(x_org)+","+str(y_org)+","+str(z_org)+")"
                                #    #cv2.putText(img, org_str_coord,(250,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)
                                #    org_str_dim = "Label Dim = " + "("+str(l_org)+","+str(w_org)+","+str(h_org)+")"
                                #    #cv2.putText(img, org_str_dim,(250,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)
                                #    print(pred_str)
                                #    print(pred_str_dim)
                                #    print(org_str_coord)
                                #    print(org_str_dim)
 
                                #print(x_cam)
                                #cv2.circle(img,(int(x_cam[0]),int(x_cam[1])),10, (0,0,255),-1)
                        BB_pred_f.append(BB_pred_cls)
                        BB_org_f.append(BB_org_cls)
                    cv2.imwrite("output/img_"+str(f+i)+".png",img)
                    cls_pred.append(cls_list)
                    cls_org.append(cls_org_list)
                    BB_pred.append(BB_pred_f)
                    BB_org.append(BB_org_f)
                    #count_error_f.write(str(count_error))
                    #count_error_f.write("\n")
                    #count_error = []
 

            print("Count error % = " ,np.sum(count_error)/np.sum(y_true))
            cls_pred_arr = np.array(cls_pred) 
            cls_org_arr = np.array(cls_org)
            cls_error = np.abs(cls_pred_arr-cls_org_arr)
            cls_error_per_class = np.sum(cls_error, axis=0)
            print("Num Cls Error per class = ", cls_error_per_class, " in ", cls_error.shape[0], " frames")
            f1_score_p_class = []
            acc_cls_p_class = []
            

            for i in range(num_classes):
                f1 = f1_score(cls_org_arr[:,i].reshape(-1),cls_pred_arr[:,i].reshape(-1))
                f1_score_p_class.append(f1)
                acc_cls = accuracy_score(cls_org_arr[:,i].reshape(-1),cls_pred_arr[:,i].reshape(-1))
                acc_cls_p_class.append(acc_cls)

            acc_count = accuracy_score(y_true,y_pred)
            print("f1 score per class = ", f1_score_p_class)
            print("Classification accuracy per class = ", acc_cls_p_class)
            print("Counting accuracy across all classes = ", acc_count)
        
            np.save("BB_pred.txt", BB_pred)
            np.save("BB_org.txt", BB_org)
            np.save("f1_score.txt", f1_score_p_class)
            np.save("cls_acc_p_class.txt", acc_cls_p_class)

            count_error_f.close()

if __name__ == '__main__':
    if (len(sys.argv) > 1):
        if (sys.argv[1] == 'train'):
            run(train=True)
    else:
        run(train=False)
