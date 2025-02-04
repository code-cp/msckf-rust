#!/usr/bin/env python3
#
# Download and converts EuRoC dataset to JSONL format.
# Requires installing `pyyaml` for YAML support.
#
# Based on <https://github.com/AaltoML/vio_benchmark/blob/main/convert/euroc_to_benchmark.py> (Apache-2.0).

import csv
import json
import os
import pathlib
import re
import shutil
import subprocess

import numpy as np
import yaml

LINK_PREFIX = "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset"
TMP_DIR = "data/raw/euroc"
OUTPUT_DIR ="data/benchmark/euroc"
DATASETS = [
    # {"name": "mh-01-easy", "link": "machine_hall/MH_01_easy/MH_01_easy.zip"},
    # {"name": "mh-02-easy", "link": "machine_hall/MH_02_easy/MH_02_easy.zip"},
    # {"name": "mh-03-medium", "link": "machine_hall/MH_03_medium/MH_03_medium.zip"},
    # {"name": "mh-04-difficult", "link": "machine_hall/MH_04_difficult/MH_04_difficult.zip"},
    # {"name": "mh-05-difficult", "link": "machine_hall/MH_05_difficult/MH_05_difficult.zip"},
    {"name": "v1-01-easy", "link": "vicon_room1/V1_01_easy/V1_01_easy.zip"},
    # {"name": "v1-02-medium", "link": "vicon_room1/V1_02_medium/V1_02_medium.zip"},
    # {"name": "v1-03-difficult", "link": "vicon_room1/V1_03_difficult/V1_03_difficult.zip"},
    # {"name": "v2-01-easy", "link": "vicon_room2/V2_01_easy/V2_01_easy.zip"},
    # {"name": "v2-02-medium", "link": "vicon_room2/V2_02_medium/V2_02_medium.zip"},
    # {"name": "v2-03-difficult", "link": "vicon_room2/V2_03_difficult/V2_03_difficult.zip"}
]

TO_SECONDS = 1000 * 1000 * 1000 # Timestamps are in nanoseconds

def getCalibration(tmpDir):
    calibration = { "cameras": [] }
    for cam in ["cam0", "cam1"]:
        with open("{}/mav0/{}/sensor.yaml".format(tmpDir, cam)) as f:
            p = yaml.load(f, Loader=yaml.FullLoader)
            cameraToImu = np.array(p["T_BS"]["data"])
            cameraToImu.shape = (4, 4)
            camera = {}
            intrinsics = p["intrinsics"]
            # OpenCV radtan model k_1, k_2, p_1, p_2.
            # We use three radial components in our "pinhole" model, so the third is set to 0.
            d = p["distortion_coefficients"]
            calibration["cameras"].append({
                "focalLengthX": intrinsics[0],
                "focalLengthY": intrinsics[1],
                "principalPointX": intrinsics[2],
                "principalPointY": intrinsics[3],
                "imuToCamera": np.linalg.inv(cameraToImu).tolist(),
                "distortionCoefficients": [d[0], d[1], 0],
                "model": "pinhole",
                "imageWidth": p["resolution"][0],
                "imageHeight": p["resolution"][1],
            })
    return calibration

def convertVideo(args, files, output):
    # Use `-crf 0` for lossless compression.
    fps="20"
    subprocess.run(["ffmpeg",
        "-y",
        "-r", fps,
        "-f", "image2",
        "-pattern_type", "glob", "-i", files,
        "-c:v", "libx264",
        "-preset", "ultrafast" if args.fast else "veryslow",
        "-crf", "0",
        "-vf", "format=yuv420p",
        "-an",
        output])

def getCameraParameters(filename):
    with open(filename) as f:
        intrinsics = yaml.load(f, Loader=yaml.FullLoader)["intrinsics"]
        # Syntax: intrinsics: [458.654, 457.296, 367.215, 248.375] #fu, fv, cu, cv
        return {
            "focalLengthX": intrinsics[0],
            "focalLengthY": intrinsics[1],
            "principalPointX": intrinsics[2],
            "principalPointY": intrinsics[3]
        }
    raise Exception("Failed to read camera params")

def makeDataset(outputIn, tmpDir, name, t0, kind):
    assert(kind == "cam0" or kind == "cam1" or kind == "stereo")
    stereo = kind == "stereo"
    output = list(outputIn) # clone

    outputdir = OUTPUT_DIR + "/" + name
    if not stereo:
        outputdir = outputdir + "-" + kind
    pathlib.Path(outputdir).mkdir(parents=True, exist_ok=True)

    if stereo:
        parameters0 = getCameraParameters("{}/mav0/cam0/sensor.yaml".format(tmpDir))
        parameters1 = getCameraParameters("{}/mav0/cam1/sensor.yaml".format(tmpDir))
    else:
        parameters0 = getCameraParameters("{}/mav0/{}/sensor.yaml".format(tmpDir, kind))

    timestamps = []
    if stereo:
        # Use images that are present for both cameras.
        # Rename bad files so that they do not match glob `*.png` given for ffmpeg.
        timestamps0 = []
        timestamps1 = []
        dir0 = "{}/mav0/cam0/data".format(tmpDir)
        dir1 = "{}/mav0/cam1/data".format(tmpDir)
        for filename in os.listdir(dir0):
            timestamps0.append(filename)
        for filename in os.listdir(dir1):
            timestamps1.append(filename)
        for t in timestamps0:
            if t not in timestamps1:
                f = "{}/{}".format(dir0, t)
                os.rename(f, f + "_hdn")
                print(t, "not found in cam1, ignoring")
            else:
                timestamps.append(int(os.path.splitext(t)[0]) / TO_SECONDS)
        for t in timestamps1:
            if t not in timestamps0:
                f = "{}/{}".format(dir1, t)
                os.rename(f, f + "_hdn")
                print(t, "not found in cam0, ignoring")
    else:
        # Use all images.
        for filename in os.listdir("{}/mav0/{}/data".format(tmpDir, kind)):
            timestamps.append(int(os.path.splitext(filename)[0]) / TO_SECONDS)

    timestamps = sorted(timestamps)
    number = 0
    for timestamp in timestamps:
        t = timestamp - t0
        x = {
            "number": number,
            "time": t
        }
        if stereo:
            x["frames"] = [
                {"cameraInd": 0, "cameraParameters": parameters0, "time": t},
                {"cameraInd": 1, "cameraParameters": parameters1, "time": t}
            ]
        else:
            x["frames"] = [
                {"cameraInd": 0, "cameraParameters": parameters0, "number": number, "time": t}
            ]
        output.append(x)
        number += 1

    # Video
    if stereo:
        convertVideo(args, "{}/mav0/cam0/data/*.png".format(tmpDir), outputdir + "/data.mp4")
        convertVideo(args, "{}/mav0/cam1/data/*.png".format(tmpDir), outputdir + "/data2.mp4")
    else:
        convertVideo(args, "{}/mav0/{}/data/*.png".format(tmpDir, kind), outputdir + "/data.mp4")

    # Reverse image name changes.
    if stereo:
        dir0 = "{}/mav0/cam0/data".format(tmpDir)
        dir1 = "{}/mav0/cam1/data".format(tmpDir)
        for directory in [dir0, dir1]:
            for filename in os.listdir(directory):
                f = "{}/{}".format(directory, filename)
                m = re.search("(.+)_hdn", f)
                if m:
                    os.rename(f, m.group(1))

    output = sorted(output, key=lambda row: row["time"]) # Sort by time
    with open(outputdir + "/data.jsonl", "w") as f:
        for obj in output:
            f.write(json.dumps(obj, separators=(',', ':')))
            f.write("\n")

    # All the datasets have the same calibration. Convert for each dataset
    # and save in the output folder root.
    calibration = getCalibration(tmpDir)
    with open("{}/calibration.json".format(OUTPUT_DIR), "w") as f:
        f.write(json.dumps(calibration, indent=4))

def convert(dataset):
    name = dataset['name']
    link = dataset['link']
    output = []
    print("Converting " + name)

    # Setup.
    tmpDir = TMP_DIR + "/" + name
    pathlib.Path(tmpDir).mkdir(parents=True, exist_ok=True)

    # Download.
    subprocess.run(["wget", LINK_PREFIX + "/" + link, "-O", tmpDir + ".zip"])
    subprocess.run(["unzip", tmpDir + ".zip", "-d", tmpDir])
    for f in os.listdir(tmpDir):
        if f.endswith(".zip"):
            os.remove(os.path.join(tmpDir, f))

    # The starting time is very large, shift timestamps to around zero to reduce floating point
    # accuracy issues.
    t0 = None

    # Read groudtruth.
    with open(tmpDir + '/mav0/state_groundtruth_estimate0/data.csv') as csvfile:
        # 0  timestamp,
        # 1  p_RS_R_x [m],
        # 2  p_RS_R_y [m],
        # 3  p_RS_R_z [m],
        # 4  q_RS_w [],
        # 5  q_RS_x [],
        # 6  q_RS_y [],
        # 7  q_RS_z [],
        # 8  v_RS_R_x [m s^-1],
        # 9  v_RS_R_y [m s^-1],
        # 10 v_RS_R_z [m s^-1],
        # 11 b_w_RS_S_x [rad s^-1],
        # 12 b_w_RS_S_y [rad s^-1],
        # 13 b_w_RS_S_z [rad s^-1],
        # 14 b_a_RS_S_x [m s^-2],
        # 15 b_a_RS_S_y [m s^-2],
        # 16 b_a_RS_S_z [m s^-2]
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader) # Skip header
        for row in csvreader:
            t = int(row[0]) / TO_SECONDS
            if not t0:
                t0 = t
            timestamp = t - t0
            output.append({
                "groundTruth": {
                    "position": {
                        "x": float(row[1]), "y": float(row[2]), "z": float(row[3])
                    },
                    "orientation": {
                        "w": float(row[4]), "x": float(row[5]), "y": float(row[6]), "z": float(row[7])
                    }
                },
                "time": timestamp
            })

    # Read IMU
    with open(tmpDir + '/mav0/imu0/data.csv') as csvfile:
        # 0 timestamp [ns],
        # 1 w_RS_S_x [rad s^-1]
        # 2 w_RS_S_y [rad s^-1]
        # 3 w_RS_S_z [rad s^-1]
        # 4 a_RS_S_x [m s^-2]
        # 5 a_RS_S_y [m s^-2]
        # 6 a_RS_S_z [m s^-2]
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader) # Skip header
        for row in csvreader:
            timestamp = int(row[0]) / TO_SECONDS - t0
            output.append({
                "sensor": {
                    "type": "gyroscope",
                    "values": [float(row[1]), float(row[2]), float(row[3])]
                },
                "time": timestamp
            })
            output.append({
                "sensor": {
                    "type": "accelerometer",
                    "values": [float(row[4]), float(row[5]), float(row[6])]
                },
                "time": timestamp
            })

    makeDataset(output, tmpDir, name, t0, "stereo")
    # This dataset has lots of missing frames for cam0, so make monocular dataset from cam1.
    if name == "euroc-v2-03-difficult":
        makeDataset(output, tmpDir, name, t0, "cam1")

def main(args):
    for dataset in DATASETS:
        if args.case and args.case != dataset["name"]: continue
        convert(dataset)
        shutil.rmtree(TMP_DIR)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="Dataset case name. If omitted, all datasets will be downloaded.")
    parser.add_argument("--fast", action="store_true", help="Fast video conversion, could theoretically hurt VIO performance, but unlikely.")
    args = parser.parse_args()
    main(args)