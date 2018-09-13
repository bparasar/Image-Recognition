#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 21:58:37 2018

@author: prayash
"""

import os, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.python.framework import graph_util


def freeze_graph(model_folder,output_graph="prayash_cricket_model.pb"):
    
    model_folder = '/Users/prayash/Users/prayash/Downloads/cricket_CNN_model'
    try:
            checkpoint = tf.train.get_checkpoint_state(model_folder)
            input_checkpoint = checkpoint.model_checkpoint_path
            print("[INFO] input_checkpoint:", input_checkpoint)
    except:
            input_checkpoint = model_folder
            print("[INFO] Model folder", model_folder)

    clear_devices = True

    saver = tf.train.import_meta_graph('/Users/prayash/Users/prayash/Downloads/cricket_CNN_model/model.ckpt-1.meta', clear_devices=clear_devices)



    with tf.Session() as sess:    
        saver = tf.train.import_meta_graph('/Users/prayash/Users/prayash/Downloads/cricket_CNN_model/model.ckpt-1.meta')
        saver.restore(sess,tf.train.latest_checkpoint('/Users/prayash/Users/prayash/Downloads/cricket_CNN_model/'))

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    output_node_names = "softmax_tensor"

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,                        # The session is used to retrieve the weights
            input_graph_def,             # The graph_def is used to retrieve the nodes 
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        )     
        
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph." % len(output_graph_def.node))
        print("[INFO] output_graph:",output_graph)
        print("[INFO] all done")    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tensorflow graph freezer\nConverts trained models to .pb file",
                                     prefix_chars='-')
    parser.add_argument("--mfolder", type=str, help="model folder to export")
    parser.add_argument("--ograph", type=str, help="output graph name", default="prayash_cricket_model.pb")
    
    args = parser.parse_args()
    print(args,"\n")

    freeze_graph(args.mfolder,args.ograph)
        