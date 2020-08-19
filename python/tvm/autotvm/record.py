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
# pylint: disable=superfluous-parens, redefined-outer-name, redefined-outer-name,pointless-string-statement
# pylint: disable=consider-using-enumerate,invalid-name
"""Tuning record and serialization format"""

import argparse
import inspect
import logging
import multiprocessing
import json
import time
import os
import itertools
from collections import OrderedDict
import numpy as np
from google.protobuf import json_format
from google.protobuf import message
from tvm import runtime
from tvm.autotvm.util import get_const_tuple
from tvm.generated import AutoTVMLog_pb2
from tvm.ir import container
from tvm.te import tensor, placeholder
from tvm.tir import expr

from .. import build, lower, target as _target
from .. import __version__
from . import task
from .task import ConfigEntity, ApplyHistoryBest
from .measure import MeasureInput, MeasureResult
from .task.space import SplitEntity, ReorderEntity, AnnotateEntity, OtherOptionEntity

AUTOTVM_LOG_VERSION = "0.3"
_old_version_warning = True
logger = logging.getLogger('autotvm')

try:  # convert unicode to str for python2
    _unicode = unicode
except NameError:
    _unicode = ()

try:
    _long = long
except NameError:
    _long = int


def measure_str_key(inp, include_config=True):
    """ get unique str key for MeasureInput

    Parameters
    ----------
    inp: MeasureInput
        input for the measure
    include_config: bool, optional
        whether includes config in the str key

    Returns
    -------
    key: str
        The str representation of key
    """
    config_str = str(inp.config) if include_config else ""

    # Consolidate args and kwargs to kwargs only for clarity.
    arg_names = inspect.getfullargspec(inp.task.func.fcustomized).args
    assert len(arg_names) == len(inp.task.args) + len(inp.task.kwargs)

    kwargs = {}
    # First translate args to kwargs.
    for i, arg in enumerate(inp.task.args):
        kwargs[arg_names[i]] = arg

    # The remaining arg_names which were not used in translation should
    # exist in inp.task.kwargs.
    for arg_name in arg_names[len(inp.task.args):]:
        assert arg_name in inp.task.kwargs
        kwargs[arg_name] = inp.task.kwargs[arg_name]

    return "".join([str(inp.target), inp.task.name, str(kwargs), config_str])


def encode_task_arg_to_pb(arg):
    """Encodes task argument as TaskArgument protobuf object.

    Parameters
    ----------
    arg : Any
        Argument for an AutoTVM Task.

    Returns
    -------
    ret: TaskArgument
        The TaskArgument protobuf object encoded from `arg`
    """
    # Preprocess arg
    if isinstance(arg, (expr.StringImm, expr.IntImm, expr.FloatImm)):
        arg = arg.value
    if isinstance(arg, runtime.container.String):
        arg = str(arg)

    arg_pb = AutoTVMLog_pb2.TaskArgument()
    if isinstance(arg, (tuple, list, container.Array)):
        for a in arg:
            arg_pb.arg_list.arguments.append(encode_task_arg_to_pb(a))
    if isinstance(arg, tensor.Tensor):
        arg_pb.tensor.shape = get_const_tuple(arg.shape)
        arg_pb.tensor.dtype = arg.dtype
    if isinstance(arg, str):
        arg_pb.string = arg
    if isinstance(arg, (int, np.int)):
        arg_pb.int32 = arg
    if isinstance(arg, (float, np.float)):
        arg_pb.double = arg
    if isinstance(arg, expr.Var):
        arg_pb.expr_var.name = arg.name
        arg_pb.expr_var.dtype = arg.dtype

    return arg_pb


def decode_task_arg_from_pb(arg_pb):
    """Decodes task argument from TaskArgument protobuf object.

    Parameters
    ----------
    arg_pb : AutoTVMLog_pb.TaskArgument
        Protobuf representation of an argument for an AutoTVM Task.

    Returns
    -------
    ret: Any
        Argument for an AutoTVM Task.
    """
    if arg_pb.WhichOneof("arg") == "arg_list":
        return tuple([decode_task_arg_from_pb(a) for a in arg_pb.arg_list])
    if arg_pb.WhichOneof("arg") == "tensor":
        return placeholder(arg_pb.tensor.shape, arg_pb.tensor.dtype)
    if arg_pb.WhichOneof("arg") == "string":
        return arg_pb.string
    if arg_pb.WhichOneof("arg") == "int32":
        return arg_pb.int32
    if arg_pb.WhichOneof("arg") == "double":
        return arg_pb.double
    if arg_pb.WhichOneof("arg") == "expr_var":
        return expr.Var(arg_pb.name, arg_pb.dtype)
    return arg_pb


def encode(inp, result, protocol='json'):
    """Encode (MeasureInput, MeasureResult) pair to a string.

    Parameters
    ----------
    inp: autotvm.tuner.MeasureInput
    result: autotvm.tuner.MeasureResult
        Pair of input/result.
    protocol: str, optional
        Serialization protocol that is one of (protobuf, json), by default json.

    Returns
    -------
    row: str
        A row in the logger file.
    """

    # NOTE: For protobuf-generated objects we cannot directly assign
    # to a non-primitive field.
    log_pb = AutoTVMLog_pb2.AutoTVMLog()
    log_pb.target.target_string = str(inp.target)

    # Encode task info.
    log_pb.task.task_name = inp.task.name
    # From TaskTemplate: When the customized func is registered,
    # compute and schedule function will be ignored
    arg_names = None
    if inp.task.func.fcustomized is not None:
        arg_names = inspect.getfullargspec(inp.task.func.fcustomized).args
    elif inp.task.func.fcompute is not None:
        arg_names = inspect.getfullargspec(inp.task.func.fcompute).args
    else:
        raise RuntimeError("Attempted to produce AutoTVM log but neither default nor " +
                           "custom function is provided to the task.")

    assert len(inp.task.args) == len(arg_names)

    for i in range(len(inp.task.args)):
        task_kwarg_pb = AutoTVMLog_pb2.TaskKeywordArgument()
        task_kwarg_pb.keyword = arg_names[i]
        task_kwarg_pb.arg.CopyFrom(encode_task_arg_to_pb(inp.task.args[i]))
        log_pb.task.kwargs.append(task_kwarg_pb)

    # Encode config info.
    log_pb.config.CopyFrom(inp.config.encode(protocol="protobuf"))

    # Encode tuning result.
    log_pb.result.costs.extend(result.costs if result.error_no == 0 else (1e9,))
    log_pb.result.error_no = result.error_no
    log_pb.result.all_cost = result.all_cost
    log_pb.result.timestamp = result.timestamp

    # Encode version info.
    log_pb.version = AUTOTVM_LOG_VERSION
    log_pb.tvm_version = __version__

    # Serialize protobuf to the requested protocol.
    if protocol == 'protobuf':
        return log_pb.SerializeToString()
    if protocol == 'json':
        return json.dumps(json_format.MessageToDict(log_pb))
    raise RuntimeError("Invalid serialization protocol: " + protocol)


def decode(row, protocol='json'):
    """Decode encoded record string to python object.

    Parameters
    ----------
    row : Optional[str]
        A row in the logger file. None if decoding fails.
    protocol : str, optional
        Serialization protocol that is one of (protobuf, json), by default json.

    Returns
    -------
    ret : tuple(autotvm.tuner.MeasureInput, autotvm.tuner.MeasureResult), or None
        The tuple of input and result, or None if input uses old version log format.
    """
    # pylint: disable=unused-variable
    global _old_version_warning

    if protocol not in ['protobuf', 'json']:
        raise RuntimeError("Invalid deserialization protocol: " + protocol)

    # Decode log from the given protocol to protobuf.
    log_pb = None
    if protocol == 'protobuf':
        try:
            log_pb = AutoTVMLog_pb2.AutoTVMLog.FromString(row)
        except message.DecodeError:
            logger.warning("Failed to decode log to protobuf")
    elif protocol == 'json':
        try:
            # import pdb; pdb.set_trace()
            log_pb = json_format.Parse(row, AutoTVMLog_pb2.AutoTVMLog())
        except json_format.ParseError:
            logger.warning("Failed to parse log to json and convert to protobuf")
            raise ValueError("wtf")

    if log_pb is None:
        if _old_version_warning:
            logger.warning("AutoTVM log version has been updated to 0.3")
            _old_version_warning = False
        return None

    # Decode task info from protobuf.
    kwargs = {}
    for kwarg in log_pb.task.kwargs:
        kwargs[kwarg.keyword] = decode_task_arg_from_pb(kwarg.arg)

    tgt = str(log_pb.target.target_string)
    if "-target" in tgt:
        logger.warning("\"-target\" is deprecated, use \"-mtriple\" instead.")
        tgt = tgt.replace("-target", "-mtriple")
    tgt = _target.create(str(tgt))
    tsk = task.Task(log_pb.task.task_name, args=(), kwargs=kwargs)

    # Decode config info from protobuf.
    config = ConfigEntity.decode(log_pb.config, protocol='protobuf')

    # Decode tuning result from protobuf.
    result = MeasureResult(
        tuple(log_pb.result.costs),
        log_pb.result.error_no,
        log_pb.result.all_cost,
        log_pb.result.timestamp
    )
    config.cost = np.mean(result.costs)
    inp = MeasureInput(tgt, tsk, config)

    return inp, result


def load_from_file(filename):
    """Generator: load records from file.
    This is a generator that yields the records.

    Parameters
    ----------
    filename: str

    Yields
    ------
    input: autotvm.tuner.MeasureInput
    result: autotvm.tuner.MeasureResult
    """
    for row in open(filename):
        if row and not row.startswith('#'):
            ret = decode(row)
            if ret is None:
                continue
            yield ret


def split_workload(in_file, clean=True):
    """Split a log file into separate files, each of which contains only a single workload
    This function can also delete duplicated records in log file

    Parameters
    ----------
    in_file: str
        input filename
    clean: bool
        whether delete duplicated items
    """
    tic = time.time()
    lines = list(open(in_file).readlines())

    logger.info("start converting...")
    pool = multiprocessing.Pool()
    lines = [rec for rec in pool.map(decode, lines) if rec is not None]
    logger.info("map done %.2f", time.time() - tic)

    wkl_dict = OrderedDict()
    for inp, res in lines:
        wkl = measure_str_key(inp, False)
        if wkl not in wkl_dict:
            wkl_dict[wkl] = []
        wkl_dict[wkl].append([inp, res])

    if clean:
        for i, (k, v) in enumerate(wkl_dict.items()):
            # clean duplicated items
            added = set()
            cleaned = []
            for inp, res in v:
                str_key = measure_str_key(inp)
                if str_key in added:
                    continue
                added.add(str_key)
                cleaned.append([inp, res])

            # write to file
            logger.info("Key: %s\tValid: %d\tDup: %d\t", k, len(cleaned), len(v) - len(cleaned))
            with open(args.i + ".%03d.wkl" % i, 'w') as fout:
                for inp, res in cleaned:
                    fout.write(encode(inp, res) + '\n')
    else:
        for i, (k, v) in enumerate(wkl_dict.items()):
            logger.info("Key: %s\tNum: %d", k, len(v))
            with open(args.i + ".%03d.wkl" % i, 'w') as fout:
                for inp, res in v:
                    fout.write(encode(inp, res) + '\n')

def pick_best(in_file, out_file):
    """
    Pick best entries from a file and store it to another file.
    This distill the useful log entries from a large log file.
    If out_file already exists, the best entries from both
    in_file and out_file will be saved.

    Parameters
    ----------
    in_file: str
        The filename of input
    out_file: str or file
        The filename of output
    """
    context = load_from_file(in_file)
    if os.path.isfile(out_file):
        out_context = load_from_file(out_file)
        context = itertools.chain(context, out_context)
    context, context_clone = itertools.tee(context)
    best_context = ApplyHistoryBest(context)
    best_set = set()

    for v in best_context.best_by_model.values():
        best_set.add(measure_str_key(v[0]))

    for v in best_context.best_by_targetkey.values():
        best_set.add(measure_str_key(v[0]))

    logger.info("Extract %d best records from the %s", len(best_set), in_file)
    fout = open(out_file, 'w') if isinstance(out_file, str) else out_file

    for inp, res in context_clone:
        if measure_str_key(inp) in best_set:
            fout.write(encode(inp, res) + "\n")
            best_set.remove(measure_str_key(inp))

"""
Usage:
This record executable module has three modes.

* Print log file in readable format
e.g. python -m tvm.autotvm.record --mode read --i collect_conv.log --begin 0 --end 5 --ir --code

* Extract history best from a large log file
e.g. python -m tvm.autotvm.record --mode pick --i collect.log

* Split a log file into separate files, each of which contains only a single wkl
e.g. python -m tvm.autotvm.record --mode split --i collect.log
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['read', 'pick', 'split'], default='read')
    parser.add_argument("--i", type=str, help="input file")
    parser.add_argument("--o", type=str, default=None, help='output file')
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=5)
    parser.add_argument("--ir", action='store_true')
    parser.add_argument("--code", action='store_true')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.mode == 'pick':
        args.o = args.o or args.i + ".best.log"
        pick_best(args.i, args.o)
    elif args.mode == 'read':
        for i, (inp, result) in enumerate(load_from_file(args.i)):
            if args.begin <= i < args.end:
                with inp.target:
                    s, arg_bufs = inp.task.instantiate(inp.config)

                print("")
                print(inp.target, inp.task, inp.config)
                print(result)

                if args.ir:
                    with inp.target:
                        print(lower(s, arg_bufs, simple_mode=True))

                if args.code:
                    with inp.target:
                        func = build(s, arg_bufs)
                        print(func.imported_modules[0].get_source())
    elif args.mode == 'split':
        split_workload(args.i)
