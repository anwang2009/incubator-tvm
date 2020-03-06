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

// class FunctionExtractor : private ExprVisitor {
//  public:
//   std::map<Function, int> Extract(const Expr& expr) {
//     VisitExpr(expr);
//     return std::map<Function, int>();
//   }

//  private:
//   // std::unordered_map<>

//   void VisitExpr_(const FunctionNode* n) final {
//     bool result = true;
//     std::cout << "FUNCTIONNODE " << n << std::endl;
//   }
// };

// std::map<Function, int> TestExtractor(const Expr& e) { return FunctionExtractor().Extract(e); }

// TVM_REGISTER_GLOBAL("relay._analysis.extract_functions").set_body_typed(TestExtractor);


// bool TestTest(const Expr& e) {
//   return true;
// }

// TVM_REGISTER_GLOBAL("relay._analysis.test").set_body_typed(TestTest);

// namespace transform {

// Pass ExtractFunctions() {
//   runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
//     [=](Function f, IRModule m, PassContext pc) {
//     return Downcast<Function>(ExtractFunctions(f, m));
//   };
//   return CreateFunctionPass(pass_func, 1, "ExtractFunctions",
//                             {tir::StringImmNode::make("InferType")});
// }

// } // namespace transform

}  // namespace relay
}  // namespace tvm
