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

import com.google.gson.JsonObject;
import com.simiacryptus.devutil.Javadoc;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.Explodable;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.TableOutput;
import com.simiacryptus.ref.lang.LifecycleException;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.util.IOUtil;
import com.simiacryptus.util.test.SysOutInterceptor;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import guru.nidi.graphviz.model.Graph;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.File;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

public abstract @com.simiacryptus.ref.lang.RefAware
class StandardLayerTests extends NotebookReportBase {
  public static final long seed = 51389; //System.nanoTime();
  private static final Map<String, ? extends NavigableMap<String, String>> javadocs = loadJavadoc();

  static {
    SysOutInterceptor.INSTANCE.init();
  }

  private final Random random = getRandom();
  protected int testingBatchSize = 5;
  protected boolean validateBatchExecution = true;
  protected boolean validateDifferentials = true;
  protected boolean testTraining = true;
  protected boolean testEquivalency = true;
  protected double tolerance;

  public StandardLayerTests() {
    logger.info("Seed: " + seed);
    tolerance = 1e-3;
  }

  @Nullable
  public ComponentTest<ToleranceStatistics> getBatchingTester() {
    if (!validateBatchExecution)
      return null;
    return new BatchingTester(1e-2, validateDifferentials) {
      @Override
      public double getRandom() {
        return random();
      }

      public @SuppressWarnings("unused")
      void _free() {
      }
    }.setBatchSize(testingBatchSize);
  }

  @Nonnull
  public com.simiacryptus.ref.wrappers.RefList<ComponentTest<?>> getBigTests() {
    return com.simiacryptus.ref.wrappers.RefArrays.asList(getPerformanceTester(), getBatchingTester(),
        getReferenceIOTester(), getEquivalencyTester());
  }

  @Nullable
  public ComponentTest<ToleranceStatistics> getDerivativeTester() {
    if (!validateDifferentials)
      return null;
    return new SingleDerivativeTester(tolerance, 1e-4);
  }

  @Nullable
  public ComponentTest<ToleranceStatistics> getEquivalencyTester() {
    if (!testEquivalency)
      return null;
    @Nullable final Layer referenceLayer = getReferenceLayer();
    if (null == referenceLayer)
      return null;
    return new EquivalencyTester(1e-2, referenceLayer);
  }

  @Nonnull
  public com.simiacryptus.ref.wrappers.RefList<ComponentTest<?>> getFinalTests() {
    return com.simiacryptus.ref.wrappers.RefArrays.asList(getTrainingTester());
  }

  @Nullable
  protected ComponentTest<ToleranceStatistics> getJsonTester() {
    return new SerializationTest();
  }

  @Nonnull
  public com.simiacryptus.ref.wrappers.RefList<ComponentTest<?>> getLittleTests() {
    return com.simiacryptus.ref.wrappers.RefArrays.asList(getJsonTester(), getDerivativeTester());
  }

  @Nullable
  public ComponentTest<ToleranceStatistics> getPerformanceTester() {
    return new PerformanceTester().setBatches(this.testingBatchSize);
  }

  @Nonnull
  public Random getRandom() {
    return new Random(seed);
  }

  protected com.simiacryptus.ref.wrappers.RefHashMap<Tensor[], Tensor> getReferenceIO() {
    return new com.simiacryptus.ref.wrappers.RefHashMap<>();
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
    try {
      Layer layer = getLayer(getSmallDims(new Random()), new Random());
      return layer.getClass();
    } catch (Throwable e) {
      logger.warn("ERROR", e);
      return getClass();
    }
  }

  public Class<?> getTestClass() {
    Layer layer = getLayer(getSmallDims(new Random()), new Random());
    return layer.getClass();
  }

  @Nullable
  public ComponentTest<TrainingTester.ComponentResult> getTrainingTester() {
    return isTestTraining() ? new TrainingTester() {
      public @SuppressWarnings("unused")
      void _free() {
      }

      @Override
      protected Layer lossLayer() {
        return StandardLayerTests.this.lossLayer();
      }
    } : null;
  }

  public boolean isTestTraining() {
    return testTraining;
  }

  @Nonnull
  public void setTestTraining(boolean testTraining) {
    this.testTraining = testTraining;
  }

  public static @SuppressWarnings("unused")
  StandardLayerTests[] addRefs(StandardLayerTests[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(StandardLayerTests::addRef)
        .toArray((x) -> new StandardLayerTests[x]);
  }

  public static @SuppressWarnings("unused")
  StandardLayerTests[][] addRefs(StandardLayerTests[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(StandardLayerTests::addRefs)
        .toArray((x) -> new StandardLayerTests[x][]);
  }

  @Nonnull
  private static Map<String, ? extends NavigableMap<String, String>> loadJavadoc() {
    try {
      HashMap<String, TreeMap<String, String>> javadocData = Javadoc.loadModelSummary();
      IOUtil.writeJson(new com.simiacryptus.ref.wrappers.RefTreeMap<>(javadocData), new File("./javadoc.json"));
      return javadocData;
    } catch (Throwable e) {
      logger.warn("Error loading javadocs", e);
      return new com.simiacryptus.ref.wrappers.RefHashMap<>();
    }
  }

  public abstract int[][] getSmallDims(Random random);

  public abstract Layer getLayer(int[][] inputSize, Random random);

  public int[][] getLargeDims(Random random) {
    return getSmallDims(new Random());
  }

  public double random() {
    return random(random);
  }

  public double random(@Nonnull Random random) {
    return Math.round(1000.0 * (random.nextDouble() - 0.5)) / 250.0;
  }

  public Tensor[] randomize(@Nonnull final int[][] inputDims) {
    return com.simiacryptus.ref.wrappers.RefArrays.stream(inputDims).map(dim -> new Tensor(dim).set(() -> random()))
        .toArray(i -> new Tensor[i]);
  }

  public void run(@Nonnull final NotebookOutput log) {
    NavigableMap<String, String> javadoc = javadocs.get(getTargetClass().getCanonicalName());
    if (null != javadoc) {
      log.p("Class Javadoc: " + javadoc.get(":class"));
      javadoc.remove(":class");
      javadoc.forEach((key, doc) -> {
        log.p(String.format("Field __%s__: %s", key, doc));
      });
    }

    long seed = (long) (Math.random() * Long.MAX_VALUE);
    int[][] smallDims = getSmallDims(new Random(seed));
    final Layer smallLayer = getLayer(smallDims, new Random(seed));
    int[][] largeDims = getLargeDims(new Random(seed));
    final Layer largeLayer = getLayer(largeDims, new Random(seed));

    log.h1("Test Modules");
    TableOutput results = new TableOutput();
    {
      if (smallLayer instanceof DAGNetwork) {
        try {
          log.h1("Network Diagram");
          log.p("This is a network apply the following layout:");
          log.eval(() -> {
            return Graphviz.fromGraph((Graph) TestUtil.toGraph((DAGNetwork) smallLayer)).height(400).width(600)
                .render(Format.PNG).toImage();
          });
        } catch (Throwable e) {
          logger.info("Error plotting graph", e);
        }
      } else if (smallLayer instanceof Explodable) {
        try {
          Layer explode = ((Explodable) smallLayer).explode();
          if (explode instanceof DAGNetwork) {
            log.h1("Exploded Network Diagram");
            log.p("This is a network apply the following layout:");
            @Nonnull
            DAGNetwork network = (DAGNetwork) explode;
            log.eval(() -> {
              @Nonnull
              Graphviz graphviz = Graphviz.fromGraph((Graph) TestUtil.toGraph(network)).height(400).width(600);
              @Nonnull
              File file = new File(log.getResourceDir(), log.getName() + "_network.svg");
              graphviz.render(Format.SVG_STANDALONE).toFile(file);
              log.link(file, "Saved to File");
              return graphviz.render(Format.SVG).toString();
            });
          }
        } catch (Throwable e) {
          logger.info("Error plotting graph", e);
        }
      }
      @Nonnull
      com.simiacryptus.ref.wrappers.RefArrayList<TestError> exceptions = standardTests(log, seed, results);
      if (!exceptions.isEmpty()) {
        if (smallLayer instanceof DAGNetwork) {
          for (@Nonnull
              Invocation invocation : getInvocations(smallLayer, smallDims)) {
            log.h1("Small SubTests: " + invocation.getLayer().getClass().getSimpleName());
            log.p(com.simiacryptus.ref.wrappers.RefArrays.deepToString(invocation.getDims()));
            tests(log, getLittleTests(), invocation, exceptions, results);
          }
        }
        if (largeLayer instanceof DAGNetwork) {
          testEquivalency = false;
          for (@Nonnull
              Invocation invocation : getInvocations(largeLayer, largeDims)) {
            log.h1("Large SubTests: " + invocation.getLayer().getClass().getSimpleName());
            log.p(com.simiacryptus.ref.wrappers.RefArrays.deepToString(invocation.getDims()));
            tests(log, getBigTests(), invocation, exceptions, results);
          }
        }
      }
      log.run(() -> {
        throwException(exceptions);
      });
    }
    getFinalTests().stream().filter(x -> null != x).forEach(test -> {
      final Layer perfLayer;
      perfLayer = getLayer(largeDims, new Random(seed));
      perfLayer.assertAlive();
      @Nonnull
      Layer copy;
      copy = perfLayer.copy();
      Tensor[] randomize = randomize(largeDims);
      com.simiacryptus.ref.wrappers.RefHashMap<CharSequence, Object> testResultProps = new com.simiacryptus.ref.wrappers.RefHashMap<>();
      try {
        Class<? extends ComponentTest> testClass = test.getClass();
        String name = testClass.getCanonicalName();
        if (null == name)
          name = testClass.getName();
        if (null == name)
          name = testClass.toString();
        testResultProps.put("class", name);
        Object result = log.subreport(sublog -> test.test(sublog, copy, randomize), log.getName() + "_" + name);
        testResultProps.put("details", null == result ? null : result.toString());
        testResultProps.put("result", "OK");
      } catch (Throwable e) {
        testResultProps.put("result", e.toString());
        throw new RuntimeException(e);
      } finally {
        results.putRow(testResultProps);
      }
    });
    log.h1("Test Matrix");
    log.out(results.toMarkdownTable());

  }

  @Nonnull
  public com.simiacryptus.ref.wrappers.RefCollection<Invocation> getInvocations(@Nonnull Layer smallLayer,
                                                                                @Nonnull int[][] smallDims) {
    @Nonnull
    DAGNetwork smallCopy = (DAGNetwork) smallLayer.copy();
    @Nonnull
    com.simiacryptus.ref.wrappers.RefHashSet<Invocation> invocations = new com.simiacryptus.ref.wrappers.RefHashSet<>();
    smallCopy.visitNodes(node -> {
      @Nullable
      Layer inner = node.getLayer();
      @Nullable
      Layer wrapper = new LayerBase() {
        @Nullable
        @Override
        public Result eval(@Nonnull Result... array) {
          if (null == inner)
            return null;
          @Nullable
          Result result = inner.eval(array);
          invocations.add(new Invocation(inner, com.simiacryptus.ref.wrappers.RefArrays.stream(array)
              .map(x -> x.getData().getDimensions()).toArray(i -> new int[i][])));
          return result;
        }

        @Override
        public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
                                  DataSerializer dataSerializer) {
          return inner.getJson(resources, dataSerializer).getAsJsonObject();
        }

        @Nullable
        @Override
        public com.simiacryptus.ref.wrappers.RefList<double[]> state() {
          return inner.state();
        }

        public void _free() {
        }
      };
      node.setLayer(wrapper);
    });
    Tensor[] input = com.simiacryptus.ref.wrappers.RefArrays.stream(smallDims).map(i -> new Tensor(i))
        .toArray(i -> new Tensor[i]);
    Result eval = smallCopy.eval(input);
    eval.getData();
    return invocations;
  }

  public void throwException(@Nonnull com.simiacryptus.ref.wrappers.RefArrayList<TestError> exceptions) {
    for (@Nonnull
        TestError exception : exceptions) {
      logger.info(String.format("LayerBase: %s", exception.layer));
      logger.info("Error", exception);
    }
    for (Throwable exception : exceptions) {
      try {
        ReferenceCountingBase.supressLog = true;
        System.gc();
        throw new RuntimeException(exception);
      } finally {
        ReferenceCountingBase.supressLog = false;
      }
    }
  }

  @Nonnull
  public com.simiacryptus.ref.wrappers.RefArrayList<TestError> standardTests(@Nonnull NotebookOutput log, long seed,
                                                                             TableOutput results) {
    log.p(String.format("Using Seed %d", seed));
    @Nonnull
    com.simiacryptus.ref.wrappers.RefArrayList<TestError> exceptions = new com.simiacryptus.ref.wrappers.RefArrayList<>();
    final Layer layer = getLayer(getSmallDims(new Random(seed)), new Random(seed));
    Invocation invocation = new Invocation(layer, getSmallDims(new Random(seed)));
    tests(log, getLittleTests(), invocation, exceptions, results);
    final Layer perfLayer = getLayer(getLargeDims(new Random(seed)), new Random(seed));
    bigTests(log, seed, perfLayer, exceptions, results);
    return exceptions;
  }

  public void bigTests(NotebookOutput log, long seed, @Nonnull Layer perfLayer,
                       @Nonnull com.simiacryptus.ref.wrappers.RefArrayList<TestError> exceptions, TableOutput results) {
    getBigTests().stream().filter(x -> null != x).forEach(test -> {
      @Nonnull
      Layer layer = perfLayer.copy();
      try {
        Tensor[] input = randomize(getLargeDims(new Random(seed)));
        com.simiacryptus.ref.wrappers.RefLinkedHashMap<CharSequence, Object> testResultProps = new com.simiacryptus.ref.wrappers.RefLinkedHashMap<>();
        try {
          String testclass = test.getClass().getCanonicalName();
          if (null == testclass || testclass.isEmpty())
            testclass = test.toString();
          testResultProps.put("class", testclass);
          Object result = log.subreport(sublog -> test.test(sublog, layer, input), log.getName() + "_" + testclass);
          testResultProps.put("details", null == result ? null : result.toString());
          testResultProps.put("result", "OK");
        } catch (Throwable e) {
          testResultProps.put("result", e.toString());
          throw new RuntimeException(e);
        } finally {
          results.putRow(testResultProps);
        }
      } catch (LifecycleException e) {
        throw e;
      } catch (Throwable e) {
        if (e.getClass().getSimpleName().equals("CudaError"))
          throw e;
        exceptions.add(new TestError(e, test, layer));
      } finally {
        System.gc();
      }
    });
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  StandardLayerTests addRef() {
    return (StandardLayerTests) super.addRef();
  }

  protected final Layer convertToReferenceLayer(Layer layer) {
    AtomicInteger counter = new AtomicInteger(0);
    Layer cvt = cvt(layer, counter);
    if (counter.get() == 0) {
      return null;
    } else
      return cvt;
  }

  protected abstract Layer lossLayer();

  private final Layer cvt(Layer layer, AtomicInteger counter) {
    if (layer instanceof DAGNetwork) {
      ((DAGNetwork) layer).visitNodes(node -> {
        Layer cvt = cvt(node.getLayer(), counter);
        node.setLayer(cvt);
      });
      return layer;
    } else if (getTestClass().isAssignableFrom(layer.getClass())) {
      @Nullable
      Class<? extends Layer> referenceLayerClass = getReferenceLayerClass();
      if (null == referenceLayerClass) {
        return null;
      } else {
        @Nonnull
        Layer cast = layer.as(referenceLayerClass);
        counter.incrementAndGet();
        return cast;
      }
    } else {
      return layer;
    }
  }

  private void tests(final NotebookOutput log, final com.simiacryptus.ref.wrappers.RefList<ComponentTest<?>> tests,
                     @Nonnull final Invocation invocation,
                     @Nonnull final com.simiacryptus.ref.wrappers.RefArrayList<TestError> exceptions, TableOutput results) {
    tests.stream().filter(x -> null != x).forEach((ComponentTest<?> test) -> {
      @Nonnull
      Layer layer = invocation.getLayer().copy();
      Tensor[] inputs = randomize(invocation.getDims());
      com.simiacryptus.ref.wrappers.RefLinkedHashMap<CharSequence, Object> testResultProps = new com.simiacryptus.ref.wrappers.RefLinkedHashMap<>();
      try {
        String testname = test.getClass().getCanonicalName();
        testResultProps.put("class", testname);
        Object result = log.subreport(sublog -> test.test(sublog, layer, inputs), log.getName() + "_" + testname);
        testResultProps.put("details", null == result ? null : result.toString());
        testResultProps.put("result", "OK");
      } catch (LifecycleException e) {
        throw e;
      } catch (Throwable e) {
        testResultProps.put("result", e.toString());
        exceptions.add(new TestError(e, test, layer));
      } finally {
        results.putRow(testResultProps);
        System.gc();
      }
    });
  }

  private static @com.simiacryptus.ref.lang.RefAware
  class Invocation extends ReferenceCountingBase {
    private final Layer layer;
    private final int[][] smallDims;

    private Invocation(Layer layer, int[][] smallDims) {
      this.layer = layer;
      this.smallDims = smallDims;
    }

    public int[][] getDims() {
      return smallDims;
    }

    public Layer getLayer() {
      return layer;
    }

    public static @SuppressWarnings("unused")
    Invocation[] addRefs(Invocation[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Invocation::addRef)
          .toArray((x) -> new Invocation[x]);
    }

    @Override
    public boolean equals(Object o) {
      if (this == o)
        return true;
      if (!(o instanceof Invocation))
        return false;

      @Nonnull
      Invocation that = (Invocation) o;

      if (layer != null ? !layer.getClass().equals(that.layer.getClass()) : that.layer != null)
        return false;
      return com.simiacryptus.ref.wrappers.RefArrays.deepEquals(smallDims, that.smallDims);
    }

    @Override
    public int hashCode() {
      int result = layer != null ? layer.getClass().hashCode() : 0;
      result = 31 * result + com.simiacryptus.ref.wrappers.RefArrays.deepHashCode(smallDims);
      return result;
    }

    public void _free() {
      super._free();
    }

    public @Override
    @SuppressWarnings("unused")
    Invocation addRef() {
      return (Invocation) super.addRef();
    }
  }
}
