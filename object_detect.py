#!/usr/bin/python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import sys
import argparse
import jetson.inference
import jetson.utils

# from jetson.inference import detectNet
# from jetson.utils import imageSource, imageOutput, Log

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in an image using an object detection DNN.", 
                                 # formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=jetson.inference.detectNet.Usage() + jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.Log.Usage())

parser.add_argument("input", type=str, help="Path to the input image file")
parser.add_argument("--output", type=str, default="", help="Path to the output image file or display URI")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pretrained model to load(see below for options)")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# create image source and optional output
input = jetson.utils.videoSource(args.input, argv=sys.argv)
output = jetson.utils.videoOutput(args.output, argv=sys.argv) if args.output else None
	
# load the object detection network
net = jetson.inference.detectNet(args.network, sys.argv, args.threshold)

# note: to hard-code the paths to load a model, the following API can be used:
# change the detectNet model= parameter to test the different engines tensorRT builds
#net = jetson.inference.detectNet(model="/home/tiku/code/mm1-ssd-to-jetson-nano/SSD-Mobilenet-v2/ssd_mobilenet_v2_coco.uff.1.1.8201.GPU.FP16.engine",
#		labels="SSD-Mobilenet-v2/ssd_coco_labels.txt",
#                input_blob="Input", output_cvg="NMS", output_bbox="NMS_1", 
#                threshold=args.threshold)

# capture the input image
img = input.Capture()
if img is None:
    raise SystemExit("Failed to load input image: {}".format(args.input))

# one real detection call to get results/overlay for display or saving
detections = net.Detect(img, overlay=args.overlay)
print("detected {:d} objects in image".format(len(detections)))

# then repeat inference to get a stable FPS reading
net.Detect(img, overlay="none")


for detection in detections:
    print(detection)


# render or save the image if requested
if output:
	output.Render(img)
	output.SetStatus("{:s} | Network {:.0f} FPS".format("ssd-inception-v2", net.GetNetworkFPS()))
	# print out performance info
	net.PrintProfilerTimes()

log_path = "benchmark_log.csv"
new_file = not os.path.exists(log_path)

with open(log_path, "a", newline="") as f:
    writer = csv.writer(f)
    if new_file:
        writer.writerow(["timestamp", "num_detections",
                          "network_fps", "network_time_ms"])
    writer.writerow([
        time.strftime("%Y-%m-%d %H:%M:%S"),
        len(detections),
        round(net.GetNetworkFPS(), 2),
        round(net.GetNetworkTime(), 2),
    ])

