# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Yolox heads."""
from loguru import logger
import tensorflow as tf
from tensorflow.keras.models import Sequential
from official.vision.beta.projects.yolo.modeling.layers import nn_blocks
from tensorflow.keras.layers import Concatenate
from official.vision.beta.projects.yolo.ops import box_ops

class YOLOXHead(tf.keras.layers.Layer):
  """YOLOX Prediction Head."""

  def __init__(
      self,
      num_classes,
      width=1.0,
      strides=[8, 16, 32],
      in_channels=[256, 512, 1024],
      act='silu',
      depthwise=False,
      **kwargs
  ):


    """
    Args:
        num_classes: `int`, number of classes per category.
        act (str): activation type of conv. Defalut value: "silu".
        depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
    """

    super().__init__(**kwargs)

    self._key_list = [
        str(key) for key in range(3, 6)
    ]

    self._n_anchors = 1
    self._num_classes = num_classes
    self._decode_in_inference = True

    self._cls_convs = dict()
    self._reg_convs = dict()
    
    self._cls_preds = dict()
    self._reg_preds = dict()
    self._obj_preds = dict()
    self._stems = dict()

    self.prior_prob = 1e-2
    self.bias=-tf.math.log((1-self.prior_prob)/self.prior_prob)

    Conv = nn_blocks.DWConv if depthwise else nn_blocks.BaseConv 
    for k in self._key_list:
      self._stems[k] = nn_blocks.BaseConv(
          filters=int(256 * width),
          kernel_size=1,
          strides=1,
          padding='same', 
          # use_bn = True,
          activation=act,
      )

      self._cls_convs[k] = Sequential(
          [
          Conv(
            filters=int(256 * width),
            kernel_size=3,
            strides=1,
            # use_bn = True,
            activation=act,
          ),
          Conv(
            filters=int(256 * width),
            kernel_size=3,
            strides=1,
            # use_bn = True,
            activation=act,
          ),
          ]
        )

      self._reg_convs[k] = Sequential(
          [
            Conv(
              filters=int(256 * width),
              kernel_size=3,
              strides=1,
              # use_bn = True,
              activation=act,
            ),
            Conv(
              filters=int(256 * width),
              kernel_size=3,
              strides=1,
              # use_bn = True,
              activation=act,
            ),
          ]
        )
      
      self._cls_preds[k] = tf.keras.layers.Conv2D(
          filters=self._n_anchors * self._num_classes,
          kernel_size=1,
          strides=1,
          padding='same',
          bias_initializer=tf.keras.initializers.constant(self.bias)
        )
      
      self._reg_preds[k] = tf.keras.layers.Conv2D(
          filters=4,
          kernel_size=1,
          strides=1,
          padding='same',
        )
      
      self._obj_preds[k] = tf.keras.layers.Conv2D(
          filters=self._n_anchors * 1,
          kernel_size=1,
          strides=1,
          padding='same',
          bias_initializer=tf.keras.initializers.constant(self.bias)
        )


  def call(self, inputs, *args, **kwargs):
    outputs=dict()
    for k in self._key_list:    
        x = self._stems[k](inputs[k])
        cls_x = x
        reg_x = x

        cls_feat = self._cls_convs[k](cls_x)
        cls_output = self._cls_preds[k](cls_feat)

        reg_feat = self._reg_convs[k](reg_x)
        reg_output = self._reg_preds[k](reg_feat)
        obj_output = self._obj_preds[k](reg_feat)
        output=Concatenate(-1)([reg_output,obj_output,cls_output])
        outputs[k] = output
      #TODO flatten

    return outputs
    
