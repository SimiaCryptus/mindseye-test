/*
 * Copyright (c) 2020 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.test;

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.test.unit.*;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.util.Util;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.concurrent.TimeUnit;

/**
 * The type Layer test base.
 */
public abstract class LayerTestBase extends LayerTests {

  /**
   * Perf test.
   */
  @Test
  @Timeout(value = 15, unit = TimeUnit.MINUTES)
  @DisplayName("Performance")
  public void perfTest() {
    long seed = (long) (Math.random() * Long.MAX_VALUE);
    run(getLog(), getPerformanceTester(), getLargeDims(), seed);
  }

  /**
   * Batching test.
   */
  @Test
  @Timeout(value = 15, unit = TimeUnit.MINUTES)
  @DisplayName("Data Batching Invariance")
  public void batchingTest() {
    long seed = (long) (Math.random() * Long.MAX_VALUE);
    run(getLog(), getBatchingTester(), getLargeDims(), seed);
  }

  /**
   * Reference io test.
   */
  @Test
  @Timeout(value = 15, unit = TimeUnit.MINUTES)
  @DisplayName("Input/Output")
  public void referenceIOTest() {
    long seed = (long) (Math.random() * Long.MAX_VALUE);
    run(getLog(), new ReferenceIO(getReferenceIO()), getLargeDims(), seed);
  }

  @Test
  @Timeout(value = 15, unit = TimeUnit.MINUTES)
  @DisplayName("Network Graph")
  public void graphTest() {
    final Layer layer = getLayer();
    Assumptions.assumeTrue(null != layer, "No Layer");
    Assumptions.assumeTrue(layer instanceof DAGNetwork, "No Layer");
    layer.freeRef();
    run(getLog(),
        new ComponentTestBase<String>() {
          @Override
          public void _free() {
            super._free();
          }

          @Nullable
          @Override
          public String test(NotebookOutput log, Layer component, Tensor... inputPrototype) {
            RefUtil.freeRef(inputPrototype);
            new MermaidGrapher(log, true).mermaid((DAGNetwork) component);
            return "OK";
          }
        },
        getLargeDims(),
        (long) (Math.random() * Long.MAX_VALUE)
    );
  }

  /**
   * Equivalency test.
   */
  @Test
  @Timeout(value = 15, unit = TimeUnit.MINUTES)
  @DisplayName("Equivalency Validation")
  public void equivalencyTest() {
    long seed = (long) (Math.random() * Long.MAX_VALUE);
    EquivalencyTester equivalencyTester = getEquivalencyTester();
    Assumptions.assumeTrue(null != equivalencyTester, "No Reference Layer");
    run(getLog(), equivalencyTester, getLargeDims(), seed);
  }

  /**
   * Json test.
   */
  @Test
  @Timeout(value = 15, unit = TimeUnit.MINUTES)
  @DisplayName("JSON Serialization")
  public void jsonTest() {
    long seed = (long) (Math.random() * Long.MAX_VALUE);
    run(getLog(), new SerializationTest(), getSmallDims(), seed);
  }

  /**
   * Derivative test.
   */
  @Test
  @Timeout(value = 15, unit = TimeUnit.MINUTES)
  @DisplayName("Derivative Validation")
  public void derivativeTest() {
    long seed = (long) (Math.random() * Long.MAX_VALUE);
    run(getLog(), getDerivativeTester(), getSmallDims(), seed);
  }

  /**
   * Training test.
   */
  @Test
  @Timeout(value = 15, unit = TimeUnit.MINUTES)
  @DisplayName("Comparative Training")
  public void trainingTest() {
    long seed = (long) (Math.random() * Long.MAX_VALUE);
    run(getLog(), getTrainingTester(), getLargeDims(), seed);
  }


  @Nonnull
  @Override
  protected Layer lossLayer() {
    try {
      return (Layer) Class.forName("com.simiacryptus.mindseye.layers.java.EntropyLossLayer").getConstructor().newInstance();
    } catch (Exception e) {
      throw Util.throwException(e);
    }
  }

}
