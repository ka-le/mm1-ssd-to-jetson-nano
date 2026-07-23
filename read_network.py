import  gfile

with gfile.FastGFile("/models/mb1-ssd_fp32.engine", 'wb') as f:
	f.writeR(baseline.SerializeToString())
	print("TRT model is stored")

