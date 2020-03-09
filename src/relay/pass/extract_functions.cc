/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *
 * \file extract_submodules.cc
 * \brief Extract per-function submodules post-fusion.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

namespace tvm {
namespace relay {

class FunctionExtractor : private ExprVisitor {
 public:
  Map<Integer, Function> Extract(const Expr& expr) {
    std::vector<transform::Pass> passes = {transform::SimplifyInference(), transform::FuseOps(3)};
    auto mod = IRModule::FromExpr(expr);
    auto seq = transform::Sequential(passes);
    mod = seq(mod);

    VisitExpr(mod->Lookup("main"));
    return this->hash_to_function;
  }

 private:
  Map<Integer, Function> hash_to_function;

  void VisitExpr_(const FunctionNode* n) final {
    if (n->IsPrimitive()) {
      Function func = FunctionNode::make(n->params, n->body, n->ret_type, n->type_params, n->attrs);
      size_t hash_ = StructuralHash()(func);
      this->hash_to_function.Set(hash_, func);
    }

    ExprVisitor::VisitExpr_(n);
  }
};

Map<Integer, Function> TestExtractor(const Expr& e) { return FunctionExtractor().Extract(e); }

TVM_REGISTER_GLOBAL("relay._analysis.extract_functions").set_body_typed(TestExtractor);

}  // namespace relay
}  // namespace tvm
