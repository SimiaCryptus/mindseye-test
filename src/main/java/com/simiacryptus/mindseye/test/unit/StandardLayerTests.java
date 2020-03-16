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
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.TableOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrayList;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefString;
import com.simiacryptus.util.test.SysOutInterceptor;

import javax.annotation.Nonnull;
import java.util.Random;

public abstract class StandardLayerTests extends LayerTests {

  static {
    SysOutInterceptor.INSTANCE.init();
  }

  public StandardLayerTests() {
    super();
    logger.info("Seed: " + seed);
  }

  @Nonnull
  protected RefList<ComponentTest<?>> getBigTests() {
    return RefArrays.asList(getPerformanceTester(), getBatchingTester(), new ReferenceIO(getReferenceIO()),
        getEquivalencyTester());
  }

  @Nonnull
  protected RefList<ComponentTest<?>> getFinalTests() {
    return RefArrays.asList(getTrainingTester());
  }

  @Nonnull
  protected RefList<ComponentTest<?>> getLittleTests() {
    return RefArrays.asList(new SerializationTest(), getDerivativeTester());
  }

  public final void allTests(@Nonnull final NotebookOutput log) {
    printJavadoc(log);
    long seed = (long) (Math.random() * Long.MAX_VALUE);
    @Nonnull int[][] smallDims = getSmallDims(new Random(seed));
    final Layer smallLayer = getLayer(smallDims, new Random(seed));
    @Nonnull int[][] largeDims = getLargeDims(new Random(seed));
    final Layer largeLayer = getLayer(largeDims, new Random(seed));
    assert smallLayer.getClass() == largeLayer.getClass();

    TableOutput results = new TableOutput();
    try {
      log.h1("Test Modules");
      RefArrayList<TestError> exceptions = new RefArrayList<>();
      renderGraph(log, smallLayer.addRef());
      log.p(RefString.format("Using Seed %d", seed));

      run(log,
          getLittleTests(),
          new LayerTestParameters(
              smallLayer.copy(),
              smallDims),
          exceptions.addRef(), results);

      run(log,
          getBigTests(),
          new LayerTestParameters(
              largeLayer.copy(),
              largeDims),
          exceptions.addRef(), results);

      log.run(RefUtil.wrapInterface(() -> {
        throwException(exceptions.addRef());
      }, exceptions));

      run(log,
          getFinalTests(),
          new LayerTestParameters(
              getLayer(largeDims, new Random(seed)),
              largeDims),
          new RefArrayList<>(), results);

    } finally {
      if (null != largeLayer)
        largeLayer.freeRef();
      if (null != smallLayer)
        smallLayer.freeRef();
    }

    log.h1("Test Matrix");
    log.out(results.toMarkdownTable());
  }

  private void run(@Nonnull final NotebookOutput log,
                   @Nonnull final RefList<ComponentTest<?>> tests,
                   @Nonnull final LayerTestParameters layerTestParameters,
                   @Nonnull final RefList<TestError> out_exceptions,
                   @Nonnull TableOutput out_results) {
    tests.stream().filter(x -> RefUtil.isNotNull(x)).forEach(RefUtil.wrapInterface(test -> {
      log.subreport(String.format("%s (Test: %s)", log.getDisplayName(), getName(test.getClass())), RefUtil.wrapInterface(
          sublog -> {
            run(sublog, test.addRef(), layerTestParameters.addRef(), out_exceptions.addRef(), out_results);
            return null;
          }, test
      ));
    }, out_exceptions, layerTestParameters, tests));
  }

}
