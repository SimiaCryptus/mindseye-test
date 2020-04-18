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

package com.simiacryptus.mindseye.test;

import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.ref.lang.ReferenceCounting;

import javax.annotation.Nonnull;
import java.util.UUID;

/**
 * The interface Simple result.
 */
public interface SimpleResult extends ReferenceCounting {
  /**
   * Get input derivative tensor list [ ].
   *
   * @return the tensor list [ ]
   */
  @javax.annotation.Nullable
  TensorList[] getInputDerivative();

  /**
   * Gets layer derivative.
   *
   * @return the layer derivative
   */
  @javax.annotation.Nullable
  DeltaSet<UUID> getLayerDerivative();

  /**
   * Gets output.
   *
   * @return the output
   */
  @javax.annotation.Nullable
  TensorList getOutput();

  /**
   * Free.
   */
  void _free();

  @Nonnull
  SimpleResult addRef();
}
