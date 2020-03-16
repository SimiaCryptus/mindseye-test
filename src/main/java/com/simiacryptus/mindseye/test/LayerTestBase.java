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
import com.simiacryptus.ref.wrappers.RefSystem;
import com.simiacryptus.util.Util;
import org.junit.After;
import org.junit.Before;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInfo;
import org.junit.jupiter.api.Timeout;

import javax.annotation.Nonnull;
import java.lang.management.ManagementFactory;
import java.util.Random;
import java.util.concurrent.TimeUnit;

public abstract class LayerTestBase extends LayerTests {

  @Test
  @Timeout(value = 15, unit = TimeUnit.MINUTES)
  public void perfTest(TestInfo testInfo) {
    report(testInfo, log -> {
      long seed = (long) (Math.random() * Long.MAX_VALUE);
      run(log, getPerformanceTester(), getSmallDims(new Random(seed)), seed);
    });
  }

  @Test
  @Timeout(value = 15, unit = TimeUnit.MINUTES)
  public void batchingTest(TestInfo testInfo) {
    report(testInfo, log -> {
      long seed = (long) (Math.random() * Long.MAX_VALUE);
      run(log, getBatchingTester(), getSmallDims(new Random(seed)), seed);
    });
  }

  @Test
  @Timeout(value = 15, unit = TimeUnit.MINUTES)
  public void referenceIOTest(TestInfo testInfo) {
    report(testInfo, log -> {
      long seed = (long) (Math.random() * Long.MAX_VALUE);
      run(log, new ReferenceIO(getReferenceIO()), getSmallDims(new Random(seed)), seed);
    });
  }

  @Test
  @Timeout(value = 15, unit = TimeUnit.MINUTES)
  public void equivalencyTest(TestInfo testInfo) {
    report(testInfo, log -> {
      long seed = (long) (Math.random() * Long.MAX_VALUE);
      EquivalencyTester equivalencyTester = getEquivalencyTester();
      if (null != equivalencyTester) {
        run(log, equivalencyTester, getSmallDims(new Random(seed)), seed);
      }
    });
  }

  @Test
  @Timeout(value = 15, unit = TimeUnit.MINUTES)
  public void jsonTest(TestInfo testInfo) {
    report(testInfo, log -> {
      long seed = (long) (Math.random() * Long.MAX_VALUE);
      run(log, new SerializationTest(), getSmallDims(new Random(seed)), seed);
    });
  }

  @Test
  @Timeout(value = 15, unit = TimeUnit.MINUTES)
  public void derivativeTest(TestInfo testInfo) {
    report(testInfo, log -> {
      long seed = (long) (Math.random() * Long.MAX_VALUE);
      run(log, getDerivativeTester(), getSmallDims(new Random(seed)), seed);
    });
  }

  @Test
  @Timeout(value = 15, unit = TimeUnit.MINUTES)
  public void trainingTest(TestInfo testInfo) {
    report(testInfo, log -> {
      long seed = (long) (Math.random() * Long.MAX_VALUE);
      run(log, getTrainingTester(), getSmallDims(new Random(seed)), seed);
    });
  }

  @Before
  public void setup() {
    reportingFolder = "reports/_reports";
    //GpuController.remove();
  }

  @After
  public void cleanup() {
    RefSystem.gc();
    long used = ManagementFactory.getMemoryMXBean().getHeapMemoryUsage().getUsed();
    logger.info("Total memory after GC: " + used);
//    try {
//      Thread.sleep(5*1000);
//    } catch (InterruptedException e) {
//      e.printStackTrace();
//    }
    //GpuController.remove();
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
