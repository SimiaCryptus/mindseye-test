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

package com.simiacryptus.mindseye.test.unit;

import com.simiacryptus.devutil.Javadoc;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.TableOutput;
import com.simiacryptus.ref.lang.LifecycleException;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.IOUtil;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.File;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.DoubleSupplier;

public abstract class LayerTests extends NotebookReportBase {
  public static final long seed = 51389; //com.simiacryptus.ref.wrappers.RefSystem.nanoTime();
  protected static final Map<String, ? extends NavigableMap<String, String>> javadocs = LayerTests.loadJavadoc();
  private final Random random = getRandom();
  protected int testingBatchSize = 5;
  protected boolean validateDifferentials = true;
  protected double tolerance;

  public LayerTests() {
    tolerance = 1e-3;
  }

  @Nullable
  public ComponentTest<ToleranceStatistics> getBatchingTester() {
    BatchingTester batchingTester = new BatchingTester(1e-2, validateDifferentials) {
      {
      }

      @Override
      public double getRandom() {
        return random();
      }

      public @SuppressWarnings("unused")
      void _free() {
        super._free();
      }
    };
    batchingTester.setBatchSize(testingBatchSize);
    return batchingTester;
  }

  @Nullable
  public ComponentTest<ToleranceStatistics> getDerivativeTester() {
    if (!validateDifferentials)
      return null;
    return new SingleDerivativeTester(tolerance, 1e-4);
  }

  @Nullable
  public ComponentTest<ToleranceStatistics> getEquivalencyTester() {
    @Nullable final Layer referenceLayer = getReferenceLayer();
    if (null == referenceLayer) {
      return null;
    }
    EquivalencyTester temp_07_0002 = new EquivalencyTester(1e-2,
        referenceLayer.addRef());
    referenceLayer.freeRef();
    return temp_07_0002;
  }

  @Nullable
  protected ComponentTest<ToleranceStatistics> getJsonTester() {
    return new SerializationTest();
  }

  @Nullable
  public ComponentTest<ToleranceStatistics> getPerformanceTester() {
    PerformanceTester performanceTester = new PerformanceTester();
    performanceTester.setBatches(this.testingBatchSize);
    return performanceTester;
  }

  @Nonnull
  public Random getRandom() {
    return new Random(seed);
  }

  @Nullable
  protected RefHashMap<Tensor[], Tensor> getReferenceIO() {
    return new RefHashMap<>();
  }

  @Nullable
  protected ComponentTest<ToleranceStatistics> getReferenceIOTester() {
    return new ReferenceIO(getReferenceIO());
  }

  @Nullable
  public Layer getReferenceLayer() {
    return convertToReferenceLayer(getLayer(getSmallDims(new Random()), new Random()));
  }

  @Nullable
  public Class<? extends Layer> getReferenceLayerClass() {
    return null;
  }

  @Nonnull
  @Override
  public ReportType getReportType() {
    return ReportType.Components;
  }

  @Override
  protected Class<?> getTargetClass() {
    Layer layer = getLayer(getSmallDims(new Random()), new Random());
    try {
      assert layer != null;
      return layer.getClass();
    } catch (Throwable e) {
      logger.warn("ERROR", e);
      return getClass();
    } finally {
      layer.freeRef();
    }
  }

  public Class<?> getTestClass() {
    Layer layer = getLayer(getSmallDims(new Random()), new Random());
    assert layer != null;
    Class<?> layerClass = layer.getClass();
    layer.freeRef();
    return layerClass;
  }

  @Nullable
  public ComponentTest<TrainingTester.ComponentResult> getTrainingTester() {
    return new TrainingTester() {

      public @SuppressWarnings("unused")
      void _free() {
        super._free();
      }

      @Nonnull
      @Override
      protected Layer lossLayer() {
        return LayerTests.this.lossLayer();
      }
    };
  }

  public static int[] getDimensions(TensorList tensorList) {
    try {
      return tensorList.getDimensions();
    } finally {
      tensorList.freeRef();
    }
  }

  @NotNull
  public static TensorList getData(Result result) {
    try {
      return result.getData();
    } finally {
      result.freeRef();
    }
  }

  @NotNull
  public static Layer copy(Layer layer) {
    assert layer != null;
    try {
      return layer.copy();
    } finally {
      layer.freeRef();
    }
  }

  @Nonnull
  private static Map<String, ? extends NavigableMap<String, String>> loadJavadoc() {
    try {
      HashMap<String, TreeMap<String, String>> javadocData = Javadoc.loadModelSummary();
      IOUtil.writeJson(new TreeMap<>(javadocData), new File("./javadoc.json"));
      return javadocData;
    } catch (Throwable e) {
      logger.warn("Error loading javadocs", e);
      return new HashMap<>();
    }
  }

  @Nonnull
  public abstract int[][] getSmallDims(Random random);

  @Nullable
  public abstract Layer getLayer(int[][] inputSize, Random random);

  @Nonnull
  public int[][] getLargeDims(Random random) {
    return getSmallDims(new Random());
  }

  public double random() {
    return random(random);
  }

  public double random(@Nonnull Random random) {
    return Math.round(1000.0 * (random.nextDouble() - 0.5)) / 250.0;
  }

  @Nonnull
  public Tensor[] randomize(@Nonnull final int[][] inputDims) {
    return RefArrays.stream(inputDims).map(dim -> {
      Tensor tensor = new Tensor(dim);
      tensor.set((DoubleSupplier) this::random);
      return tensor;
    }).toArray(Tensor[]::new);
  }

  public static void throwException(@Nonnull RefList<TestError> exceptions) {
    exceptions.forEach(exception -> {
      logger.info(RefString.format("LayerBase: %s", exception.layer));
      logger.info("Error", exception.toString());
    });
    try {
      exceptions.forEach(exception -> {
        try {
          ReferenceCountingBase.supressLog = true;
          RefSystem.gc();
          throw new RuntimeException(exception);
        } finally {
          ReferenceCountingBase.supressLog = false;
        }
      });
    } finally {
      exceptions.freeRef();
    }
  }

  @Nullable
  protected final Layer convertToReferenceLayer(@Nullable Layer layer) {
    AtomicInteger counter = new AtomicInteger(0);
    Layer cvt = cvt(layer == null ? null : layer.addRef(), counter);
    if (null != layer)
      layer.freeRef();
    if (counter.get() == 0) {
      if (null != cvt)
        cvt.freeRef();
      return null;
    } else {
      return cvt;
    }
  }

  @Nonnull
  protected abstract Layer lossLayer();

  protected void run(@Nonnull final NotebookOutput log,
                     @Nonnull final RefList<ComponentTest<?>> tests,
                     @Nonnull final Invocation invocation,
                     @Nonnull final RefList<TestError> out_exceptions,
                     @Nonnull TableOutput out_results) {
    tests.stream().filter(x -> {
      boolean notNull = null != x;
      if (null != x) x.freeRef();
      return notNull;
    }).forEach(RefUtil.wrapInterface(test -> {
      run(log, invocation, test, out_exceptions.addRef(), out_results);
    }, out_exceptions, invocation));
    tests.freeRef();
  }

  @Nullable
  private final Layer cvt(Layer layer, @Nonnull AtomicInteger counter) {
    if (layer instanceof DAGNetwork) {
      ((DAGNetwork) layer).visitNodes(node -> {
        node.setLayer(cvt(node.getLayer(), counter));
        node.freeRef();
      });
      return layer;
    } else if (getTestClass().isAssignableFrom(layer.getClass())) {
      @Nullable
      Class<? extends Layer> referenceLayerClass = getReferenceLayerClass();
      if (null == referenceLayerClass) {
        layer.freeRef();
        return null;
      } else {
        @Nonnull
        Layer cast = layer.as(referenceLayerClass);
        layer.freeRef();
        counter.incrementAndGet();
        return cast;
      }
    } else {
      return layer;
    }
  }

  private void run(@Nonnull NotebookOutput log, @Nonnull Invocation invocation, ComponentTest<?> test, @Nonnull RefList<TestError> out_exceptions, @Nonnull TableOutput out_results) {
    @Nonnull Layer layer = LayerTests.copy(invocation.getLayer());
    Tensor[] inputs = randomize(invocation.getDims());
    Map<CharSequence, Object> testResultProps = new LinkedHashMap<>();
    try {
      String testname = test.getClass().getCanonicalName();
      testResultProps.put("class", testname);
      Object result = log.subreport(RefUtil.wrapInterface(
          sublog -> test.test(sublog, layer.addRef(), RefUtil.addRef(inputs)),
          inputs, layer.addRef(), test.addRef()),
          String.format("%s (Test: %s)", log.getDisplayName(), testname));
      testResultProps.put("details", null == result ? null : result.toString());
      RefUtil.freeRef(result);
      testResultProps.put("result", "OK");
    } catch (LifecycleException e) {
      throw e;
    } catch (Throwable e) {
      testResultProps.put("result", e.toString());
      out_exceptions.add(new TestError(e, test.addRef(), layer.addRef()));
    } finally {
      out_results.putRow(testResultProps);
      out_exceptions.freeRef();
      test.freeRef();
      layer.freeRef();
      RefSystem.gc();
    }
  }

}
