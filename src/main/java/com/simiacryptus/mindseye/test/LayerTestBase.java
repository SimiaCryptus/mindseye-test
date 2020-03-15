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
import com.simiacryptus.mindseye.test.unit.StandardLayerTests;
import com.simiacryptus.ref.wrappers.RefSystem;
import org.junit.After;
import org.junit.Before;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import javax.annotation.Nonnull;
import java.lang.management.ManagementFactory;
import java.util.concurrent.TimeUnit;

public abstract class LayerTestBase extends StandardLayerTests {

  //  @Test(timeout = 15 * 60 * 1000)
  //  public void testMonteCarlo() throws Throwable {
  //    apply(this::monteCarlo);
  //  }

  @Test
  @Timeout(value = 15, unit = TimeUnit.MINUTES)
  public void test() {
    run(this::run);
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
      throw new RuntimeException(e);
    }
  }

}
