import h5py
filename = 'weights_cnn_sentece.hdf5'
f = h5py.File(filename, 'r')

print(list(f.keys()))
d = f['dense']['dense_1']['kernel:0']
# <HDF5 dataset "kernel:0": shape (128, 1), type "<f4">
d.shape == (128, 1)
d[0] == array([-0.14390108], dtype=float32)
# etc.