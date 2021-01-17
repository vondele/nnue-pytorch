import argparse
import features
import math
import model as M
import numpy
import nnue_bin_dataset
import struct
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from functools import reduce
import operator

def ascii_hist(name, x, bins=6):
  N,X = numpy.histogram(x, bins=bins)
  total = 1.0*len(x)
  width = 50
  nmax = N.max()

  print(name)
  for (xi, n) in zip(X,N):
    bar = '#'*int(n*1.0*width/nmax)
    xi = '{0: <8.4g}'.format(xi).ljust(10)
    print('{0}| {1}'.format(xi,bar))

# hardcoded for now
FC_HASH = 0x63337156
VERSION = 0x7AF32F16

class NNUEWriter():
  """
  All values are stored in little endian.
  """
  def __init__(self, model):
    self.buf = bytearray()

    self.write_header(model)
    self.int32(model.feature_set.hash ^ (M.L1*2)) # Feature transformer hash
    self.write_feature_transformer(model)
    self.int32(FC_HASH) # FC layers hash
    self.write_fc_layer(model.l1)
    self.write_fc_layer(model.l2)
    self.write_fc_layer(model.output, is_output=True)

  def write_header(self, model):
    self.int32(VERSION) # version
    self.int32(FC_HASH ^ model.feature_set.hash ^ (M.L1*2)) # halfkp network hash
    description = b"Features=HalfKP(Friend)[41024->256x2],"
    description += b"Network=AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-32]"
    description += b"(ClippedReLU[32](AffineTransform[32<-512](InputSlice[512(0:512)])))))"
    self.int32(len(description)) # Network definition
    self.buf.extend(description)

  def coalesce_ft_weights(self, model, layer):
    weight = layer.weight.data
    indices = model.feature_set.get_virtual_to_real_features_gather_indices()
    weight_coalesced = weight.new_zeros((weight.shape[0], model.feature_set.num_real_features))
    for i_real, is_virtual in enumerate(indices):
      weight_coalesced[:, i_real] = sum(weight[:, i_virtual] for i_virtual in is_virtual)

    return weight_coalesced

  def write_feature_transformer(self, model):
    # int16 bias = round(x * 127)
    # int16 weight = round(x * 127)
    layer = model.input
    bias = layer.bias.data
    bias = bias.mul(127).round().to(torch.int16)
    ascii_hist('ft bias:', bias.numpy())
    self.buf.extend(bias.flatten().numpy().tobytes())

    weight = self.coalesce_ft_weights(model, layer)
    weight = weight.mul(127).round().to(torch.int16)
    ascii_hist('ft weight:', weight.numpy())
    # weights stored as [41024][256], so we need to transpose the pytorch [256][41024]
    self.buf.extend(weight.transpose(0, 1).flatten().numpy().tobytes())

  def write_fc_layer(self, layer, is_output=False):
    # FC layers are stored as int8 weights, and int32 biases
    kWeightScaleBits = 6
    kActivationScale = 127.0
    if not is_output:
      kBiasScale = (1 << kWeightScaleBits) * kActivationScale # = 8128
    else:
      kBiasScale = 9600.0 # kPonanzaConstant * FV_SCALE = 600 * 16 = 9600
    kWeightScale = kBiasScale / kActivationScale # = 64.0 for normal layers
    kMaxWeight = 127.0 / kWeightScale # roughly 2.0

    # int32 bias = round(x * kBiasScale)
    # int8 weight = round(x * kWeightScale)
    bias = layer.bias.data
    bias = bias.mul(kBiasScale).round().to(torch.int32)
    ascii_hist('fc bias:', bias.numpy())
    self.buf.extend(bias.flatten().numpy().tobytes())
    weight = layer.weight.data
    clipped = torch.count_nonzero(weight.clamp(-kMaxWeight, kMaxWeight) - weight)
    total_elements = torch.numel(weight)
    clipped_max = torch.max(torch.abs(weight.clamp(-kMaxWeight, kMaxWeight) - weight))
    print("layer has {}/{} clipped weights. Exceeding by {} the maximum {}.".format(clipped, total_elements, clipped_max, kMaxWeight))
    weight = weight.clamp(-kMaxWeight, kMaxWeight).mul(kWeightScale).round().to(torch.int8)
    ascii_hist('fc weight:', weight.numpy())
    # Stored as [outputs][inputs], so we can flatten
    self.buf.extend(weight.flatten().numpy().tobytes())

  def int32(self, v):
    self.buf.extend(struct.pack("<i", v))

class NNUEReader():
  def __init__(self, f, feature_set):
    self.f = f
    self.feature_set = feature_set
    self.model = M.NNUE(feature_set)

    self.read_header(feature_set)
    self.read_int32(feature_set.hash ^ (M.L1*2)) # Feature transformer hash
    self.read_feature_transformer(self.model.input)
    self.read_int32(FC_HASH) # FC layers hash
    self.read_fc_layer(self.model.l1)
    self.read_fc_layer(self.model.l2)
    self.read_fc_layer(self.model.output, is_output=True)

  def read_header(self, feature_set):
    self.read_int32(VERSION) # version
    self.read_int32(FC_HASH ^ feature_set.hash ^ (M.L1*2)) # halfkp network hash
    desc_len = self.read_int32() # Network definition
    description = self.f.read(desc_len)

  def tensor(self, dtype, shape):
    d = numpy.fromfile(self.f, dtype, reduce(operator.mul, shape, 1))
    d = torch.from_numpy(d.astype(numpy.float32))
    d = d.reshape(shape)
    return d

  def read_feature_transformer(self, layer):
    layer.bias.data = self.tensor(numpy.int16, layer.bias.shape).divide(127.0)
    # weights stored as [41024][256], so we need to transpose the pytorch [256][41024]
    weights = self.tensor(numpy.int16, layer.weight.shape[::-1])
    layer.weight.data = weights.divide(127.0).transpose(0, 1)

  def read_fc_layer(self, layer, is_output=False):
    # FC layers are stored as int8 weights, and int32 biases
    kWeightScaleBits = 6
    kActivationScale = 127.0
    if not is_output:
      kBiasScale = (1 << kWeightScaleBits) * kActivationScale # = 8128
    else:
      kBiasScale = 9600.0 # kPonanzaConstant * FV_SCALE = 600 * 16 = 9600
    kWeightScale = kBiasScale / kActivationScale # = 64.0 for normal layers

    layer.bias.data = self.tensor(numpy.int32, layer.bias.shape).divide(kBiasScale)
    layer.weight.data = self.tensor(numpy.int8, layer.weight.shape).divide(kWeightScale)

  def read_int32(self, expected=None):
    v = struct.unpack("<i", self.f.read(4))[0]
    if expected is not None and v != expected:
      raise Exception("Expected: %x, got %x" % (expected, v))
    return v

def main():
  parser = argparse.ArgumentParser(description="Converts files between ckpt and nnue format.")
  parser.add_argument("source", help="Source file (can be .ckpt, .pt or .nnue)")
  parser.add_argument("target", help="Target file (can be .pt or .nnue)")
  features.add_argparse_args(parser)
  args = parser.parse_args()

  feature_set = features.get_feature_set_from_name(args.features)

  print('Converting %s to %s' % (args.source, args.target))

  if args.source.endswith(".pt") or args.source.endswith(".ckpt"):
    if not args.target.endswith(".nnue"):
      raise Exception("Target file must end with .nnue")
    if args.source.endswith(".pt"):
      nnue = torch.load(args.source)
    else:
      nnue = M.NNUE.load_from_checkpoint(args.source, feature_set=feature_set)
    nnue.eval()
    writer = NNUEWriter(nnue)
    with open(args.target, 'wb') as f:
      f.write(writer.buf)
  elif args.source.endswith(".nnue"):
    if not args.target.endswith(".pt"):
      raise Exception("Target file must end with .pt")
    with open(args.source, 'rb') as f:
      reader = NNUEReader(f, feature_set)
    torch.save(reader.model, args.target)
  else:
    raise Exception('Invalid filetypes: ' + str(args))

if __name__ == '__main__':
  main()
