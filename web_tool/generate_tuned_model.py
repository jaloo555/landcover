#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# pylint: disable=E1137,E1136,E0110

import sys, os, time
import joblib
import argparse
import numpy as np
import glob
import json
import rasterio
import fiona
import fiona.transform

import keras
import keras.backend as K
import keras.callbacks
import keras.utils
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.layers import Input, Dense, Activation, MaxPooling2D, Conv2D, BatchNormalization
from keras.layers import Concatenate, Cropping2D, Lambda
from keras.losses import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator


# Sample execution:
# python generate_tuned_model.py --in_dir ../data/ypejnini/ --out_geo_path ./tiles/temp.geojson --in_model_path ./data/naip_autoencoder.h5 --in_tile_path ./tiles/m_3807537_ne_18_1_20170611.mrf --out_model_path ./data/naip_autoencoder_tuned.h5 --num_classes 5 --gpu 1

# Supports 1 file only
def genGeoJson(in_dir,out_geo_path):
    output = {
        "type": "FeatureCollection",
        "name": out_geo_path.split(".")[0],
        "crs": { "type": "name", "properties": {"name":"urn:ogc:def:crs:EPSG::3857"}},
        "features": []
    }
    fns = [
        fn 
        for fn in glob.glob(in_dir + "*.p") 
        if "request_list" in fn
        ]
    fns = sorted(fns, key=lambda x: int(x.split("_")[1]))
    request_list = joblib.load(fns[-1])
    for request in request_list:
        if request["type"] == "correction":
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": None
                },
                "properties": { "user_label": request["value"]}
            }
            xmin = request["extent"]["xmin"]
            xmax = request["extent"]["xmax"]
            ymin = request["extent"]["ymin"]
            ymax = request["extent"]["ymax"]
            polygon = [[xmin,ymax],[xmax,ymax],[xmax,ymin],[xmin,ymin],[xmin,ymax]]

            point = [
                (xmin+xmax)/2,
                (ymin+ymax)/2
            ]

            feature["geometry"]["coordinates"] = point

            output["features"].append(feature)
    
    with open(out_geo_path, "w") as f:
        f.write(json.dumps(output))

def get_loss(mask_value):
    mask_value = K.variable(mask_value)
    def masked_categorical_crossentropy(y_true, y_pred):
        
        mask = K.all(K.equal(y_true, mask_value), axis=-1)
        mask = 1 - K.cast(mask, K.floatx())

        loss = K.categorical_crossentropy(y_true, y_pred) * mask

        return K.sum(loss) / K.sum(mask)
    return masked_categorical_crossentropy

def get_model(model_path, num_classes):
    K.clear_session()
    tmodel = keras.models.load_model(model_path)
    toutput = tmodel.layers[-2].output
    toutput = Conv2D(num_classes+1, (1,1), padding="same", use_bias=True, activation="softmax", name="output_conv")(toutput)
    model = keras.models.Model(inputs=tmodel.inputs, outputs=[toutput])

    optimizer = Adam(lr=0.001)
    loss_mask = np.zeros(num_classes+1)
    loss_mask[0] = 1
    model.compile(loss=get_loss(loss_mask), optimizer=optimizer)
    
    return model

def train_model_from_points(in_geo_path, in_model_path, in_tile_path, out_model_path, num_classes):
    print("Loading initial model...")
    model = get_model(in_model_path, num_classes)
    model.summary()

    print("Loading tiles...")
    f = rasterio.open(in_tile_path)
    data = np.rollaxis(f.read(), 0, 3)
    profile = f.profile
    transform = f.profile["transform"]
    src_crs = f.crs.to_string()
    f.close

    print("Loading new GeoJson file...")
    f = fiona.open(in_geo_path)
    coords = []
    labels = []
    for line in f:
        label = line["properties"]["user_label"]
        geom = fiona.transform.transform_geom(f.crs["init"], src_crs, line["geometry"])
        lon, lat = geom["coordinates"]
        y, x = ~transform * (lon, lat)
        y = int(y)
        x = int(x)
        coords.append((x,y))
        labels.append(label)
    f.close()

    coords = np.array(coords)
    labels = np.array(labels)

    # x-dim, y-dim, # of bands
    x_train = np.zeros((coords.shape[0], 150, 150, 4), dtype=np.float32)

    # x-dim, y-dim, # of classes + dummy index
    y_train = np.zeros((coords.shape[0], 150, 150, num_classes+1), dtype=np.uint8)

    y_train[:,:,:] = [1] + [0] * (y_train.shape[-1]-1)

    for i in range(coords.shape[0]):
        y,x = coords[i]
        label = labels[i]

        x_train[i] = data[y-75:y+74+1, x-75:x+74+1, :].copy()

        y_train[i,75,75,0] = 0
        y_train[i,75,75,label+1] = 1
        
    x_train = x_train / 255.0

    print("Tuning model")

    model.fit(
        x_train, y_train,
        batch_size=10, epochs=1, verbose=1, validation_split=0
    )

    model.save(out_model_path)


def main():
    parser = argparse.ArgumentParser(description="Generate a model tuned using webtool")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debugging", default=False)
    parser.add_argument("--in_dir", action="store", dest="in_dir", type= str, help="Path to user pickle file folder (takes newest file)", required=True)
    parser.add_argument("--out_geo_path", action="store", dest="out_geo_path", type=str, help="Output geojson path (i.e. ../data/output.geojson)", required=True)
    parser.add_argument("--in_model_path", action="store", dest="in_model_path", type=str, help="Path to model that needs retraining", required=True)
    parser.add_argument("--in_tile_path", action="store", dest="in_tile_path", type=str, help="Path to input tif file", required=True)
    parser.add_argument("--out_model_path", action="store", dest="out_model_path", type=str, help="Output path for tuned model", required=True)
    parser.add_argument("--num_classes", action="store", dest="num_classes", type=str, help="Number of classes", required=True)
    parser.add_argument("--gpu", action="store", dest="gpuid", type=int, help="GPU to use", required=True)
    
    args = parser.parse_args(sys.argv[1:])
    args.batch_size=10
    args.num_epochs=30

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

    start_time = float(time.time())
    
    genGeoJson(args.in_dir, args.out_geo_path)

    print("GeoJson file created at {}".format(args.out_geo_path))

    print("Retraining model from GeoJson")

    train_model_from_points(args.out_geo_path, args.in_model_path, args.in_tile_path, args.out_model_path, int(args.num_classes))

    print("Finished in %0.4f seconds" % (time.time() - start_time))
    
    pass

if __name__ == "__main__":
    main()