/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
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

package com.simiacryptus.mindseye.test.integration;

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.ref.wrappers.RefStream;
import com.simiacryptus.util.test.LabeledObject;

import javax.annotation.Nonnull;
import java.io.IOException;

/**
 * The interface Image problem data.
 */
public interface ImageProblemData {
  /**
   * Training data ref stream.
   *
   * @return the ref stream
   * @throws IOException the io exception
   */
  @Nonnull
  RefStream<LabeledObject<Tensor>> trainingData() throws IOException;

  /**
   * Validation data ref stream.
   *
   * @return the ref stream
   * @throws IOException the io exception
   */
  RefStream<LabeledObject<Tensor>> validationData() throws IOException;
}
