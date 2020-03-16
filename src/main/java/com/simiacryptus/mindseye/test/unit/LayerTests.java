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

import com.google.gson.GsonBuilder;
import com.simiacryptus.devutil.Javadoc;
import com.simiacryptus.lang.UncheckedSupplier;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.Explodable;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.util.test.NotebookReportBase;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.TableOutput;
import com.simiacryptus.ref.lang.LifecycleException;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.IOUtil;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.test.SysOutInterceptor;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import guru.nidi.graphviz.model.Graph;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.DoubleSupplier;

public abstract class LayerTests extends NotebookReportBase {
  public static final long seed = 51389; //com.simiacryptus.ref.wrappers.RefSystem.nanoTime();
  protected static final Map<String, ? extends NavigableMap<String, String>> javadocs = LayerTests.loadJavadoc();

  static {
    SysOutInterceptor.INSTANCE.init();
  }

  private final Random random = getRandom();
  protected int testingBatchSize = 5;
  protected double tolerance = 1e-3;

  @Nonnull
  protected BatchingTester getBatchingTester() {
    return getBatchingTester(1e-2, true, this.testingBatchSize);
  }

  @Nullable
  protected SingleDerivativeTester getDerivativeTester() {
    return new SingleDerivativeTester(tolerance, 1e-4);
  }

  @Nullable
  protected EquivalencyTester getEquivalencyTester() {
    @Nullable final Layer referenceLayer = getReferenceLayer();
    if (null == referenceLayer) {
      return null;
    }
    return new EquivalencyTester(1e-2, referenceLayer);
  }

  @Nonnull
  protected int[][] getLargeDims() {
    return getSmallDims();
  }

  @Nullable
  protected abstract Layer getLayer();

  @Nullable
  protected PerformanceTester getPerformanceTester() {
    PerformanceTester performanceTester = new PerformanceTester();
    performanceTester.setBatches(this.testingBatchSize);
    return performanceTester;
  }

  @Nonnull
  protected Random getRandom() {
    return new Random(seed);
  }

  @Nullable
  protected RefHashMap<Tensor[], Tensor> getReferenceIO() {
    return new RefHashMap<>();
  }

  @Nullable
  protected Layer getReferenceLayer() {
    return convertToReferenceLayer(getLayer());
  }

  @Nullable
  protected Class<? extends Layer> getReferenceLayerClass() {
    return null;
  }

  @Nonnull
  @Override
  public ReportType getReportType() {
    return ReportType.Components;
  }

  @Nonnull
  protected abstract int[][] getSmallDims();

  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    Layer layer = getLayer();
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

  @Nonnull
  protected Class<?> getTestClass() {
    Layer layer = getLayer();
    assert layer != null;
    Class<?> layerClass = layer.getClass();
    layer.freeRef();
    return layerClass;
  }

  @Nullable
  protected TrainingTester getTrainingTester() {
    TrainingTester trainingTester = new TrainingTester() {

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
    trainingTester.setBatches(testingBatchSize);
    return trainingTester;
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
    layer.assertAlive();
    try {
      return layer.copy();
    } finally {
      layer.freeRef();
    }
  }

  public static final void renderGraph(@Nonnull NotebookOutput log, Layer layer) {
    if (layer instanceof DAGNetwork) {
      try {
        log.h1("Network Diagram");
        log.p("This is a network apply the following layout:");
        log.eval(RefUtil.wrapInterface((UncheckedSupplier<BufferedImage>) () -> {
          return Graphviz.fromGraph((Graph) TestUtil.toGraph(((DAGNetwork) layer).addRef())).height(400).width(600)
              .render(Format.PNG).toImage();
        }, layer.addRef()));
      } catch (Throwable e) {
        logger.info("Error plotting graph", e);
      }
    } else if (layer instanceof Explodable) {
      try {
        Layer explode = ((Explodable) layer).explode();
        if (explode instanceof DAGNetwork) {
          log.h1("Exploded Network Diagram");
          log.p("This is a network apply the following layout:");
          @Nonnull
          DAGNetwork network = (DAGNetwork) explode.addRef();
          log.eval(RefUtil.wrapInterface((UncheckedSupplier<String>) () -> {
            @Nonnull
            Graphviz graphviz = Graphviz.fromGraph((Graph) TestUtil.toGraph(network.addRef()))
                .height(400).width(600);
            @Nonnull
            File file = new File(log.getResourceDir(), log.getFileName() + "_network.svg");
            graphviz.render(Format.SVG_STANDALONE).toFile(file);
            log.link(file, "Saved to File");
            return graphviz.render(Format.SVG).toString();
          }, network));
        }
        explode.freeRef();
      } catch (Throwable e) {
        logger.info("Error plotting graph", e);
      }
    }
    layer.freeRef();
  }

  public static final void logDetails(@Nonnull NotebookOutput log, LayerTestParameters layerTestParameters, Layer subLayer) {
    assert subLayer != null;
    log.p(RefArrays.deepToString(layerTestParameters.getDims()));
    layerTestParameters.freeRef();
    log.eval(() -> {
      return new GsonBuilder().setPrettyPrinting().create().toJson(
          subLayer.getJson(new HashMap<>(), SerialPrecision.Double)
      );
    });
    subLayer.freeRef();
  }

  @NotNull
  public static String getName(Class<? extends ComponentTest> testClass) {
    String name = testClass.getCanonicalName();
    if (null == name)
      name = testClass.getName();
    if (null == name)
      name = testClass.toString();
    return name;
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
          throw Util.throwException(exception);
        } finally {
          ReferenceCountingBase.supressLog = false;
        }
      });
    } finally {
      exceptions.freeRef();
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

  @NotNull
  protected final BatchingTester getBatchingTester(double tolerance, boolean validateDifferentials, int testingBatchSize) {
    BatchingTester batchingTester = new BatchingTester(tolerance, validateDifferentials) {

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

  protected double random() {
    return random(random);
  }

  protected double random(@Nonnull Random random) {
    return Math.round(1000.0 * (random.nextDouble() - 0.5)) / 250.0;
  }

  @Nonnull
  protected Tensor[] randomTensors(@Nonnull final int[][] inputDims) {
    return RefArrays.stream(inputDims).map(dim -> {
      Tensor tensor = new Tensor(dim);
      tensor.set((DoubleSupplier) this::random);
      return tensor;
    }).toArray(Tensor[]::new);
  }

  protected final void printJavadoc(@Nonnull NotebookOutput log) {
    NavigableMap<String, String> javadoc = javadocs.get(getTargetClass().getCanonicalName());
    if (null != javadoc) {
      log.p("Class Javadoc: " + javadoc.get(":class"));
      javadoc.remove(":class");
      javadoc.forEach((key, doc) -> {
        log.p(RefString.format("Field __%s__: %s", key, doc));
      });
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

  protected void run(@Nonnull NotebookOutput log, ComponentTest<?> test, @Nonnull LayerTestParameters layerTestParameters, @Nonnull RefList<TestError> out_exceptions, @Nonnull TableOutput out_results) {
    @Nonnull RefList<TestError> exceptions = new RefArrayList<>();
    @Nonnull Layer layer = LayerTests.copy(layerTestParameters.getLayer());

    try {
      Map<CharSequence, Object> testResultProps = new LinkedHashMap<>();
      try {
        String testname = test.getClass().getCanonicalName();
        testResultProps.put("class", testname);
        Tensor[] inputs = randomTensors(layerTestParameters.getDims());
        Object result = test.test(log, layer.addRef(), inputs);
        testResultProps.put("details", null == result ? null : result.toString());
        RefUtil.freeRef(result);
        testResultProps.put("result", "OK");
      } catch (LifecycleException e) {
        throw e;
      } catch (Throwable e) {
        testResultProps.put("result", e.toString());
        exceptions.add(new TestError(e, test.addRef(), layer.addRef()));
      }
      out_results.putRow(testResultProps);

      if (!exceptions.isEmpty() && layer instanceof DAGNetwork) {
        log.h1("SubTests: " + layer.getClass().getSimpleName());
        RefCollection<LayerTestParameters> subLayerTestParameters = LayerTestParameters.getNodeTests(layer.addRef(), layerTestParameters.getDims());
        subLayerTestParameters.forEach(sub_layerTestParameters -> {
          logDetails(log, sub_layerTestParameters.addRef(), sub_layerTestParameters.getLayer());
          RefArrayList<TestError> subExceptions = new RefArrayList<>();
          run(log, test.addRef(), sub_layerTestParameters, subExceptions.addRef(), out_results);
          subExceptions.forEach((TestError ex) -> log.eval(() -> {
            return Util.toString(ex);
          }));
          exceptions.addAll(subExceptions);
        });
        subLayerTestParameters.freeRef();
      }
      out_exceptions.addAll(exceptions);
    } finally {
      layer.freeRef();
      test.freeRef();
      out_exceptions.freeRef();
      layerTestParameters.freeRef();
      RefSystem.gc();
    }


  }

  protected void run(@Nonnull NotebookOutput log, ComponentTest<?> test, @Nonnull int[][] dims, long seed) {
    logger.info("Seed: " + seed);
    printJavadoc(log);
    final Layer layer = getLayer();
    TableOutput results = new TableOutput();
    try {
      log.h1("Test Modules");
      RefArrayList<TestError> exceptions = new RefArrayList<>();
      renderGraph(log, layer.addRef());
      log.p(RefString.format("Using Seed %d", seed));

      run(log,
          test, new LayerTestParameters(
              layer.copy(),
              dims
          ),
          exceptions.addRef(),
          results);
      log.run(RefUtil.wrapInterface(() -> {
        throwException(exceptions.addRef());
      }, exceptions));
    } finally {
      layer.freeRef();
    }

    log.h1("Results");
    log.out(results.toMarkdownTable());
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

}
