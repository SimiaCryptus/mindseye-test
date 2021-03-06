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
import com.simiacryptus.mindseye.test.data.MNIST;
import com.simiacryptus.ref.wrappers.RefStream;
import com.simiacryptus.ref.wrappers.RefString;
import com.simiacryptus.util.test.LabeledObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;

/**
 * The type Mnist problem data.
 */
public class MnistProblemData implements ImageProblemData {
  /**
   * The Log.
   */
  static final Logger log = LoggerFactory.getLogger(MnistProblemData.class);

  @Nonnull
  @Override
  public RefStream<LabeledObject<Tensor>> trainingData() {
    log.info(RefString.format("Loaded %d items", MNIST.trainingDataStream().count()));
    return MNIST.trainingDataStream();
  }

  @Nonnull
  @Override
  public RefStream<LabeledObject<Tensor>> validationData() {
    return MNIST.validationDataStream();
  }

}
