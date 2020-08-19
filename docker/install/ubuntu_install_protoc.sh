#!/bin/bash
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

set -e
set -u
set -o pipefail

# Download protoc release binary and install.
PROTOC_ZIP=protoc-3.12.3-linux-x86_64.zip
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.12.3/$PROTOC_ZIP
unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
chmod +x /usr/local/bin/protoc
unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
rm -f $PROTOC_ZIP

# Generate python code from protobuf schemas
TVM_HOME=/usr/tvm
protoc -I=${TVM_HOME}/proto --python_out=${TVM_HOME}/python/tvm/generated ${TVM_HOME}/proto/*
