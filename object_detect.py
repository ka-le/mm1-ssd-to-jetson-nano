#!/usr/bin/env python3
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

from jetson_inference import detectNet
from jetson_utils import imageSource, imageOutput, Log

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in an image using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=detectNet.Usage() + imageSource.Usage() + imageOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, help="Path to the input image file")
parser.add_argument("--output", type=str, default="", help="Path to the output image file or display URI")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# create image source and optional output
input = imageSource(args.input, argv=sys.argv)
output = imageOutput(args.output, argv=sys.argv) if args.output else None
	
# load the object detection network
#  net = detectNet(args.network, sys.argv, args.threshold)

# note: to hard-code the paths to load a model, the following API can be used:

net = detectNet(model="models/mb1-ssd.onnx", labels="models/labels.txt",
                input_blob="input_0", output_cvg="scores", output_bbox="boxes", 
                threshold=args.threshold)

# capture the input image
img = input.Capture()
if img is None:
    raise SystemExit("Failed to load input image: {}".format(args.input))

# detect objects in the image (with overlay)
detections = net.Detect(img, overlay=args.overlay)

# print the detections
print("detected {:d} objects in image".format(len(detections)))
for detection in detections:
    print(detection)

# render or save the image if requested
if output:
    output.Render(img)

# print out performance info
net.PrintProfilerTimes()