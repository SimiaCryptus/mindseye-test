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
import com.simiacryptus.lang.UncheckedSupplier;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.Explodable;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.TableOutput;
import com.simiacryptus.ref.lang.*;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.IOUtil;
import com.simiacryptus.util.test.SysOutInterceptor;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import guru.nidi.graphviz.model.Graph;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.function.Function;

public abstract @RefAware
class StandardLayerTests extends NotebookReportBase {
  public static final long seed = 51389; //com.simiacryptus.ref.wrappers.RefSystem.nanoTime();
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
    BatchingTester temp_07_0011 = new BatchingTester(1e-2, validateDifferentials) {
      {
      }

      @Override
      public double getRandom() {
        return random();
      }

      public @SuppressWarnings("unused")
      void _free() {
      }
    };
    BatchingTester temp_07_0010 = temp_07_0011.setBatchSize(testingBatchSize);
    if (null != temp_07_0011)
      temp_07_0011.freeRef();
    return temp_07_0010;
  }

  @Nonnull
  public RefList<ComponentTest<?>> getBigTests() {
    return RefArrays.asList(getPerformanceTester(), getBatchingTester(), getReferenceIOTester(),
        getEquivalencyTester());
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
    if (null == referenceLayer) {
      if (null != referenceLayer)
        referenceLayer.freeRef();
      return null;
    }
    EquivalencyTester temp_07_0002 = new EquivalencyTester(1e-2,
        referenceLayer == null ? null : referenceLayer.addRef());
    if (null != referenceLayer)
      referenceLayer.freeRef();
    return temp_07_0002;
  }

  @Nonnull
  public RefList<ComponentTest<?>> getFinalTests() {
    return RefArrays.asList(getTrainingTester());
  }

  @Nullable
  protected ComponentTest<ToleranceStatistics> getJsonTester() {
    return new SerializationTest();
  }

  @Nonnull
  public RefList<ComponentTest<?>> getLittleTests() {
    return RefArrays.asList(getJsonTester(), getDerivativeTester());
  }

  @Nullable
  public ComponentTest<ToleranceStatistics> getPerformanceTester() {
    PerformanceTester temp_07_0013 = new PerformanceTester();
    PerformanceTester temp_07_0012 = temp_07_0013.setBatches(this.testingBatchSize);
    if (null != temp_07_0013)
      temp_07_0013.freeRef();
    return temp_07_0012;
  }

  @Nonnull
  public Random getRandom() {
    return new Random(seed);
  }

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
    try {
      Layer layer = getLayer(getSmallDims(new Random()), new Random());
      Class<?> temp_07_0003 = layer.getClass();
      if (null != layer)
        layer.freeRef();
      return temp_07_0003;
    } catch (Throwable e) {
      logger.warn("ERROR", e);
      return getClass();
    }
  }

  public Class<?> getTestClass() {
    Layer layer = getLayer(getSmallDims(new Random()), new Random());
    Class<?> temp_07_0004 = layer.getClass();
    if (null != layer)
      layer.freeRef();
    return temp_07_0004;
  }

  @Nullable
  public ComponentTest<TrainingTester.ComponentResult> getTrainingTester() {
    return isTestTraining() ? new TrainingTester() {
      {
      }

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
    return Arrays.stream(array).filter((x) -> x != null).map(StandardLayerTests::addRef)
        .toArray((x) -> new StandardLayerTests[x]);
  }

  public static @SuppressWarnings("unused")
  StandardLayerTests[][] addRefs(StandardLayerTests[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(StandardLayerTests::addRefs)
        .toArray((x) -> new StandardLayerTests[x][]);
  }

  @Nonnull
  private static Map<String, ? extends NavigableMap<String, String>> loadJavadoc() {
    try {
      HashMap<String, TreeMap<String, String>> javadocData = Javadoc.loadModelSummary();
      IOUtil.writeJson(new RefTreeMap<>(javadocData), new File("./javadoc.json"));
      return javadocData;
    } catch (Throwable e) {
      logger.warn("Error loading javadocs", e);
      return new HashMap<>();
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
    return RefArrays.stream(inputDims).map(dim -> {
      return new Tensor(dim).set(() -> random());
    }).toArray(i -> new Tensor[i]);
  }

  public void run(@Nonnull final NotebookOutput log) {
    NavigableMap<String, String> javadoc = javadocs.get(getTargetClass().getCanonicalName());
    if (null != javadoc) {
      log.p("Class Javadoc: " + javadoc.get(":class"));
      javadoc.remove(":class");
      javadoc.forEach((key, doc) -> {
        log.p(RefString.format("Field __%s__: %s", key, doc));
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
          log.eval(RefUtil
              .wrapInterface((UncheckedSupplier<BufferedImage>) () -> {
                return Graphviz.fromGraph((Graph) TestUtil.toGraph(((DAGNetwork) smallLayer).addRef())).height(400)
                    .width(600).render(Format.PNG).toImage();
              }, smallLayer == null ? null : smallLayer.addRef()));
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
            log.eval(RefUtil
                .wrapInterface((UncheckedSupplier<String>) () -> {
                  @Nonnull
                  Graphviz graphviz = Graphviz
                      .fromGraph((Graph) TestUtil.toGraph(network == null ? null : network.addRef())).height(400)
                      .width(600);
                  @Nonnull
                  File file = new File(log.getResourceDir(), log.getName() + "_network.svg");
                  graphviz.render(Format.SVG_STANDALONE).toFile(file);
                  log.link(file, "Saved to File");
                  return graphviz.render(Format.SVG).toString();
                }, network == null ? null : network));
          }
          if (null != explode)
            explode.freeRef();
        } catch (Throwable e) {
          logger.info("Error plotting graph", e);
        }
      }
      @Nonnull
      RefList<TestError> exceptions = standardTests(log, seed, results);
      if (!exceptions.isEmpty()) {
        if (smallLayer instanceof DAGNetwork) {
          for (@Nonnull
              Invocation invocation : getInvocations(smallLayer == null ? null : smallLayer.addRef(), smallDims)) {
            Layer temp_07_0014 = invocation.getLayer();
            log.h1("Small SubTests: " + temp_07_0014.getClass().getSimpleName());
            if (null != temp_07_0014)
              temp_07_0014.freeRef();
            log.p(RefArrays.deepToString(invocation.getDims()));
            tests(log, getLittleTests(), invocation == null ? null : invocation.addRef(),
                exceptions == null ? null : exceptions.addRef(), results);
          }
        }
        if (largeLayer instanceof DAGNetwork) {
          testEquivalency = false;
          for (@Nonnull
              Invocation invocation : getInvocations(largeLayer == null ? null : largeLayer.addRef(), largeDims)) {
            Layer temp_07_0015 = invocation.getLayer();
            log.h1("Large SubTests: " + temp_07_0015.getClass().getSimpleName());
            if (null != temp_07_0015)
              temp_07_0015.freeRef();
            log.p(RefArrays.deepToString(invocation.getDims()));
            tests(log, getBigTests(), invocation == null ? null : invocation.addRef(),
                exceptions == null ? null : exceptions.addRef(), results);
          }
        }
      }
      log.run(RefUtil.wrapInterface(() -> {
        throwException(exceptions == null ? null : exceptions.addRef());
      }, exceptions == null ? null : exceptions));
    }
    if (null != largeLayer)
      largeLayer.freeRef();
    if (null != smallLayer)
      smallLayer.freeRef();
    RefList<ComponentTest<?>> temp_07_0016 = getFinalTests();
    temp_07_0016.stream().filter(x -> {
      boolean temp_07_0005 = null != x;
      if (null != x)
        x.freeRef();
      return temp_07_0005;
    }).forEach(test -> {
      final Layer perfLayer;
      perfLayer = getLayer(largeDims, new Random(seed));
      perfLayer.assertAlive();
      @Nonnull
      Layer copy;
      copy = perfLayer.copy();
      if (null != perfLayer)
        perfLayer.freeRef();
      Tensor[] randomize = randomize(largeDims);
      Map<CharSequence, Object> testResultProps = new HashMap<>();
      try {
        Class<? extends ComponentTest> testClass = test.getClass();
        String name = testClass.getCanonicalName();
        if (null == name)
          name = testClass.getName();
        if (null == name)
          name = testClass.toString();
        testResultProps.put("class", name);
        Object result = log
            .subreport(
                RefUtil.wrapInterface(
                    (Function<NotebookOutput, ?>) sublog -> test.test(
                        sublog, copy == null ? null : copy.addRef(),
                        Tensor.addRefs(randomize)),
                    copy == null ? null : copy.addRef(), Tensor.addRefs(randomize),
                    test == null ? null : test.addRef()),
                log.getName() + "_" + name);
        testResultProps.put("details", null == result ? null : result.toString());
        testResultProps.put("result", "OK");
      } catch (Throwable e) {
        testResultProps.put("result", e.toString());
        throw new RuntimeException(e);
      } finally {
        results.putRow(testResultProps);
      }
      if (null != randomize)
        ReferenceCounting.freeRefs(randomize);
      copy.freeRef();
      if (null != test)
        test.freeRef();
    });
    if (null != temp_07_0016)
      temp_07_0016.freeRef();
    log.h1("Test Matrix");
    log.out(results.toMarkdownTable());

  }

  @Nonnull
  public RefCollection<Invocation> getInvocations(@Nonnull Layer smallLayer, @Nonnull int[][] smallDims) {
    @Nonnull
    DAGNetwork smallCopy = (DAGNetwork) smallLayer.copy();
    smallLayer.freeRef();
    @Nonnull
    RefHashSet<Invocation> invocations = new RefHashSet<>();
    smallCopy.visitNodes(RefUtil
        .wrapInterface(node -> {
          @Nullable
          Layer inner = node.getLayer();
          @Nullable
          Layer wrapper = new LayerBase() {
            {
            }

            @Nullable
            @Override
            public Result eval(@Nonnull Result... array) {
              if (null == inner) {
                ReferenceCounting.freeRefs(array);
                return null;
              }
              @Nullable
              Result result = inner.eval(Result.addRefs(array));
              invocations.add(new Invocation(inner == null ? null : inner.addRef(),
                  RefArrays.stream(Result.addRefs(array)).map(x -> {
                    TensorList temp_07_0017 = x.getData();
                    int[] temp_07_0006 = temp_07_0017.getDimensions();
                    if (null != temp_07_0017)
                      temp_07_0017.freeRef();
                    if (null != x)
                      x.freeRef();
                    return temp_07_0006;
                  }).toArray(i -> new int[i][])));
              ReferenceCounting.freeRefs(array);
              return result;
            }

            @Override
            public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
              return inner.getJson(resources, dataSerializer).getAsJsonObject();
            }

            @Nullable
            @Override
            public RefList<double[]> state() {
              return inner.state();
            }

            public void _free() {
            }
          };
          if (null != inner)
            inner.freeRef();
          node.setLayer(wrapper == null ? null : wrapper.addRef());
          if (null != wrapper)
            wrapper.freeRef();
          if (null != node)
            node.freeRef();
        }, invocations == null ? null : invocations.addRef()));
    Tensor[] input = RefArrays.stream(smallDims).map(i -> new Tensor(i)).toArray(i -> new Tensor[i]);
    Result eval = smallCopy.eval(Tensor.addRefs(input));
    if (null != input)
      ReferenceCounting.freeRefs(input);
    smallCopy.freeRef();
    RefUtil.freeRef(eval.getData());
    if (null != eval)
      eval.freeRef();
    return invocations;
  }

  public void throwException(@Nonnull RefList<TestError> exceptions) {
    for (@Nonnull
        TestError exception : exceptions) {
      logger.info(RefString.format("LayerBase: %s", exception.layer));
      logger.info("Error", exception.toString());
    }
    for (Throwable exception : exceptions) {
      try {
        ReferenceCountingBase.supressLog = true;
        com.simiacryptus.ref.wrappers.RefSystem.gc();
        exceptions.freeRef();
        throw new RuntimeException(exception);
      } finally {
        ReferenceCountingBase.supressLog = false;
      }
    }
    exceptions.freeRef();
  }

  @Nonnull
  public RefArrayList<TestError> standardTests(@Nonnull NotebookOutput log, long seed, TableOutput results) {
    log.p(RefString.format("Using Seed %d", seed));
    @Nonnull
    RefArrayList<TestError> exceptions = new RefArrayList<>();
    final Layer layer = getLayer(getSmallDims(new Random(seed)), new Random(seed));
    Invocation invocation = new Invocation(layer == null ? null : layer.addRef(), getSmallDims(new Random(seed)));
    if (null != layer)
      layer.freeRef();
    tests(log, getLittleTests(), invocation == null ? null : invocation.addRef(),
        exceptions == null ? null : exceptions.addRef(), results);
    if (null != invocation)
      invocation.freeRef();
    final Layer perfLayer = getLayer(getLargeDims(new Random(seed)), new Random(seed));
    bigTests(log, seed, perfLayer == null ? null : perfLayer.addRef(), exceptions == null ? null : exceptions.addRef(),
        results);
    if (null != perfLayer)
      perfLayer.freeRef();
    return exceptions;
  }

  public void bigTests(NotebookOutput log, long seed, @Nonnull Layer perfLayer,
                       @Nonnull RefArrayList<TestError> exceptions, TableOutput results) {
    RefList<ComponentTest<?>> temp_07_0018 = getBigTests();
    temp_07_0018.stream().filter(x -> {
      boolean temp_07_0007 = null != x;
      if (null != x)
        x.freeRef();
      return temp_07_0007;
    }).forEach(RefUtil.wrapInterface(
        (Consumer<? super ComponentTest<?>>) test -> {
          @Nonnull
          Layer layer = perfLayer.copy();
          try {
            Tensor[] input = randomize(getLargeDims(new Random(seed)));
            Map<CharSequence, Object> testResultProps = new LinkedHashMap<>();
            try {
              String testclass = test.getClass().getCanonicalName();
              if (null == testclass || testclass.isEmpty())
                testclass = test.toString();
              testResultProps.put("class", testclass);
              Object result = log.subreport(RefUtil.wrapInterface(
                  (Function<NotebookOutput, ?>) sublog -> test.test(sublog,
                      layer == null ? null : layer.addRef(), Tensor.addRefs(input)),
                  Tensor.addRefs(input), test == null ? null : test.addRef(),
                  layer == null ? null : layer.addRef()), log.getName() + "_" + testclass);
              testResultProps.put("details", null == result ? null : result.toString());
              testResultProps.put("result", "OK");
            } catch (Throwable e) {
              testResultProps.put("result", e.toString());
              throw new RuntimeException(e);
            } finally {
              results.putRow(testResultProps);
            }
            if (null != input)
              ReferenceCounting.freeRefs(input);
          } catch (LifecycleException e) {
            throw e;
          } catch (Throwable e) {
            if (e.getClass().getSimpleName().equals("CudaError"))
              throw e;
            exceptions
                .add(new TestError(e, test == null ? null : test.addRef(), layer == null ? null : layer.addRef()));
          } finally {
            com.simiacryptus.ref.wrappers.RefSystem.gc();
          }
          layer.freeRef();
          if (null != test)
            test.freeRef();
        }, exceptions == null ? null : exceptions, perfLayer == null ? null : perfLayer));
    if (null != temp_07_0018)
      temp_07_0018.freeRef();
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

  protected abstract Layer lossLayer();

  private final Layer cvt(Layer layer, AtomicInteger counter) {
    if (layer instanceof DAGNetwork) {
      ((DAGNetwork) layer).visitNodes(node -> {
        Layer cvt = cvt(node.getLayer(), counter);
        node.setLayer(cvt == null ? null : cvt.addRef());
        if (null != cvt)
          cvt.freeRef();
        if (null != node)
          node.freeRef();
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

  private void tests(final NotebookOutput log, final RefList<ComponentTest<?>> tests,
                     @Nonnull final Invocation invocation, @Nonnull final RefList<TestError> exceptions, TableOutput results) {
    tests.stream().filter(x -> {
      boolean temp_07_0008 = null != x;
      if (null != x)
        x.freeRef();
      return temp_07_0008;
    }).forEach(RefUtil.wrapInterface(
        (Consumer<ComponentTest<?>>) (ComponentTest<?> test) -> {
          Layer temp_07_0019 = invocation.getLayer();
          @Nonnull
          Layer layer = temp_07_0019.copy();
          if (null != temp_07_0019)
            temp_07_0019.freeRef();
          Tensor[] inputs = randomize(invocation.getDims());
          Map<CharSequence, Object> testResultProps = new LinkedHashMap<>();
          try {
            String testname = test.getClass().getCanonicalName();
            testResultProps.put("class", testname);
            Object result = log.subreport(RefUtil.wrapInterface(
                (Function<NotebookOutput, ?>) sublog -> test.test(sublog,
                    layer == null ? null : layer.addRef(), Tensor.addRefs(inputs)),
                Tensor.addRefs(inputs), layer == null ? null : layer.addRef(),
                test == null ? null : test.addRef()), log.getName() + "_" + testname);
            testResultProps.put("details", null == result ? null : result.toString());
            testResultProps.put("result", "OK");
          } catch (LifecycleException e) {
            throw e;
          } catch (Throwable e) {
            testResultProps.put("result", e.toString());
            exceptions
                .add(new TestError(e, test == null ? null : test.addRef(), layer == null ? null : layer.addRef()));
          } finally {
            results.putRow(testResultProps);
            com.simiacryptus.ref.wrappers.RefSystem.gc();
          }
          if (null != test)
            test.freeRef();
          if (null != inputs)
            ReferenceCounting.freeRefs(inputs);
          layer.freeRef();
        }, exceptions == null ? null : exceptions, invocation == null ? null : invocation));
    if (null != tests)
      tests.freeRef();
  }

  private static @RefAware
  class Invocation extends ReferenceCountingBase {
    private final Layer layer;
    private final int[][] smallDims;

    private Invocation(Layer layer, int[][] smallDims) {
      {
        Layer temp_07_0001 = layer == null ? null : layer.addRef();
        this.layer = temp_07_0001 == null ? null : temp_07_0001.addRef();
        if (null != temp_07_0001)
          temp_07_0001.freeRef();
      }
      if (null != layer)
        layer.freeRef();
      this.smallDims = smallDims;
    }

    public int[][] getDims() {
      return smallDims;
    }

    public Layer getLayer() {
      return layer == null ? null : layer.addRef();
    }

    public static @SuppressWarnings("unused")
    Invocation[] addRefs(Invocation[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Invocation::addRef)
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

      if (layer != null ? !layer.getClass().equals(that.layer.getClass()) : that.layer != null) {
        that.freeRef();
        return false;
      }
      boolean temp_07_0009 = RefArrays.deepEquals(smallDims, that.smallDims);
      that.freeRef();
      return temp_07_0009;
    }

    @Override
    public int hashCode() {
      int result = layer != null ? layer.getClass().hashCode() : 0;
      result = 31 * result + RefArrays.deepHashCode(smallDims);
      return result;
    }

    public void _free() {
      if (null != layer)
        layer.freeRef();
      super._free();
    }

    public @Override
    @SuppressWarnings("unused")
    Invocation addRef() {
      return (Invocation) super.addRef();
    }
  }
}
