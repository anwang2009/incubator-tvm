# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Test function extraction"""
import tvm
from tvm import relay
from tvm.relay.extract_submodules import extract_hashed_submodules
from tvm.relay.testing.resnet import get_workload


def convoluted_network():
    """This gets the net for a case described in fuse_ops.cc:

            conv2d
            /  |  \
           /   |   \
         op    op   op
          \    |    /
           \   |   /
          elemwise add
               |
    """
    dshape = (1, 1, 5, 1)
    x = relay.var("x", shape=dshape)
    y = relay.nn.conv2d(x, relay.var("w1"),
                        kernel_size=(3, 3),
                        padding=(1, 1),
                        channels=1)

    x1 = relay.nn.conv2d(y, relay.var("w2"),
                         kernel_size=(3, 3),
                         padding=(1, 1),
                         channels=1)
    x2 = relay.nn.conv2d(y, relay.var("w3"),
                         kernel_size=(3, 3),
                         padding=(1, 1),
                         channels=1)
    x3 = relay.nn.conv2d(y, relay.var("w4"),
                         kernel_size=(3, 3),
                         padding=(1, 1),
                         channels=1)

    z = relay.add(x1, x2)
    z = relay.add(x3, z)

    return tvm.IRModule.from_expr(z)


def get_conv2d():
    x = relay.var("x", shape=(1, 56, 56, 64))
    weight1 = relay.var('weight1', shape=(3, 3, 64, 32))
    y = relay.nn.conv2d(x, weight1,
                        channels=32,
                        kernel_size=(3, 3),
                        padding=(1, 1),
                        data_layout='NHWC',
                        kernel_layout='HWIO')
    return tvm.IRModule.from_expr(y)


def test_extract_identity():
    mod = get_conv2d()
    mdict = relay.analysis.extract_functions(mod['main'])
    assert len(mdict) == 1
    relay.analysis.assert_graph_equal(list(mdict.values())[0], mod)


def test_extract_convoluted_network():
    mod = convoluted_network()
    mdict = relay.analysis.extract_functions(mod['main'])
    assert len(mdict) == 2
    xs = list(mdict.values())
    x = xs[0]
    y = xs[1]

    def is_conv(mod: tvm.IRModule):
        conv2d = relay.op.op.get("nn.conv2d")
        call_node = mod["main"].body
        return call_node.op == conv2d

    def is_conv_add(mod: tvm.IRModule):
        add = relay.op.op.get("add")
        call_node = mod["main"].body
        maybe_conv_module = tvm.IRModule.from_expr(call_node.args[0])
        return call_node.op == add and is_conv(maybe_conv_module)

    # Order in dicts isn't consistent, so checking both orders is necessary
    return (is_conv(x) and is_conv_add(y)) or (is_conv_add(x) and is_conv(y))


def test_extract_resnet():
    mod, params = get_workload()
    res = relay.analysis.extract_functions(mod['main'])
    # extract_hashed_submodules(mod)


def test_test():
    print("get conv2d")
    mod, params = convoluted_network()
    print("test test")
    res = relay.analysis.test_test(mod['main'])
    print("res", res)


if __name__ == '__main__':
    # test_extract_identity()
    # test_extract_convoluted_network()
    # test_extract_resnet()
    test_test()
