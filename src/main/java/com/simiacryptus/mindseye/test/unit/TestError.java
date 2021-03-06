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

package com.simiacryptus.mindseye.test.unit;

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.ref.wrappers.RefString;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

/**
 * The type Test error.
 */
public class TestError extends RuntimeException {
  /**
   * The Test.
   */
  @Nullable
  public final String test;
  /**
   * The Layer.
   */
  @Nonnull
  public final String layer;

  /**
   * Instantiates a new Test error.
   *
   * @param cause the cause
   * @param test  the test
   * @param layer the layer
   */
  public TestError(Throwable cause, @Nonnull ComponentTest<?> test, @Nonnull Layer layer) {
    super(RefString.format("Error in %s apply %s", test.addRef(), layer.addRef()), cause);
    this.test = test.toString();
    test.freeRef();
    this.layer = layer.toString();
    layer.freeRef();
  }

}
