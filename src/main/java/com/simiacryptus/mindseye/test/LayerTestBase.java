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
import com.simiacryptus.mindseye.test.unit.EquivalencyTester;
import com.simiacryptus.mindseye.test.unit.LayerTests;
import com.simiacryptus.mindseye.test.unit.ReferenceIO;
import com.simiacryptus.mindseye.test.unit.SerializationTest;
import com.simiacryptus.util.Util;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import javax.annotation.Nonnull;
import java.util.concurrent.TimeUnit;

public abstract class LayerTestBase extends LayerTests {

  @Test
  @Timeout(value = 15, unit = TimeUnit.MINUTES)
  public void perfTest() {
    long seed = (long) (Math.random() * Long.MAX_VALUE);
    run(getLog(), getPerformanceTester(), getLargeDims(), seed);
  }

  @Test
  @Timeout(value = 15, unit = TimeUnit.MINUTES)
  public void batchingTest() {
    long seed = (long) (Math.random() * Long.MAX_VALUE);
    run(getLog(), getBatchingTester(), getLargeDims(), seed);
  }

  @Test
  @Timeout(value = 15, unit = TimeUnit.MINUTES)
  public void referenceIOTest() {
    long seed = (long) (Math.random() * Long.MAX_VALUE);
    run(getLog(), new ReferenceIO(getReferenceIO()), getLargeDims(), seed);
  }

  @Test
  @Timeout(value = 15, unit = TimeUnit.MINUTES)
  public void equivalencyTest() {
    long seed = (long) (Math.random() * Long.MAX_VALUE);
    EquivalencyTester equivalencyTester = getEquivalencyTester();
    if (null != equivalencyTester) {
      run(getLog(), equivalencyTester, getLargeDims(), seed);
    }
  }

  @Test
  @Timeout(value = 15, unit = TimeUnit.MINUTES)
  public void jsonTest() {
    long seed = (long) (Math.random() * Long.MAX_VALUE);
    run(getLog(), new SerializationTest(), getSmallDims(), seed);
  }

  @Test
  @Timeout(value = 15, unit = TimeUnit.MINUTES)
  public void derivativeTest() {
    long seed = (long) (Math.random() * Long.MAX_VALUE);
    run(getLog(), getDerivativeTester(), getSmallDims(), seed);
  }

  @Test
  @Timeout(value = 15, unit = TimeUnit.MINUTES)
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
