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
from typing import Dict, List
import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprVisitor


class ExtractSubgraphs(ExprVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.submodule_dict = {}

    def collect_function(self, f: relay.Function):
        """Construct and memoize IRModule from function

        Parameters
        ----------
        f : relay.Function
            function to convert to tvm.IRModule
        """
        f_hash = relay.analysis.structural_hash(f)
        mod = tvm.IRModule.from_expr(f.body)
        self.submodule_dict[f_hash] = mod

    def visit_function(self, f):
        if f.attrs and int(f.attrs.Primitive) == 1:
            self.collect_function(f)
        super().visit_function(f)

    def tunable_task_dict(self) -> Dict[int, tvm.IRModule]:
        return self.submodule_dict


def extract_submodules(mod: tvm.IRModule) -> List[tvm.IRModule]:
    """Get per-fused-layer IRModules from the given IRModule.

    Parameters
    ----------
    mod : tvm.IRModule
        Module to decompose into per-fused-layer IRModules

    Returns
    -------
    List[tvm.IRModule]
        List of subgraph modules
    """
    task_dict = extract_hashed_submodules(mod)
    return list(task_dict.values())


def extract_hashed_submodules(mod: tvm.IRModule) -> Dict[int, tvm.IRModule]:
    """Get per-fused-layer IRModules from the given IRModule.

    Parameters
    ----------
    mod : tvm.IRModule
        Module to decompose into per-fused-layer tasks

    Returns
    -------
    Dict[int, tvm.IRModule]
        Dict of (hash, module)
    """
    mod = relay.transform.SimplifyInference()(mod)
    mod = relay.transform.FuseOps(fuse_opt_level=3)(mod)
    et = ExtractSubgraphs()
    et.visit(mod['main'])
    return et.tunable_task_dict()
