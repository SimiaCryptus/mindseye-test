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

public abstract class StandardLayerTests extends NotebookReportBase {
  public static final long seed = 51389; //System.nanoTime();
  private static final HashMap<String, TreeMap<String, String>> javadocs = loadJavadoc();

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
    }.setBatchSize(testingBatchSize);
  }

  @Nonnull
  public List<ComponentTest<?>> getBigTests() {
    return Arrays.asList(getPerformanceTester(), getBatchingTester(), getReferenceIOTester(), getEquivalencyTester());
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
  public List<ComponentTest<?>> getFinalTests() {
    return Arrays.asList(getTrainingTester());
  }

  @Nullable
  protected ComponentTest<ToleranceStatistics> getJsonTester() {
    return new SerializationTest();
  }

  @Nonnull
  public List<ComponentTest<?>> getLittleTests() {
    return Arrays.asList(getJsonTester(), getDerivativeTester());
  }

  @Nullable
  public ComponentTest<ToleranceStatistics> getPerformanceTester() {
    return new PerformanceTester().setBatches(this.testingBatchSize);
  }

  @Nonnull
  public Random getRandom() {
    return new Random(seed);
  }

  protected HashMap<Tensor[], Tensor> getReferenceIO() {
    return new HashMap<>();
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

  @Nonnull
  private static HashMap<String, TreeMap<String, String>> loadJavadoc() {
    try {
      HashMap<String, TreeMap<String, String>> javadocData = Javadoc.loadModelSummary();
      IOUtil.writeJson(new TreeMap<>(javadocData), new File("./javadoc.json"));
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
    return Arrays.stream(inputDims).map(dim -> new Tensor(dim).set(() -> random())).toArray(i -> new Tensor[i]);
  }

  public void run(@Nonnull final NotebookOutput log) {
    TreeMap<String, String> javadoc = javadocs.get(getTargetClass().getCanonicalName());
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
      ArrayList<TestError> exceptions = standardTests(log, seed, results);
      if (!exceptions.isEmpty()) {
        if (smallLayer instanceof DAGNetwork) {
          for (@Nonnull
              Invocation invocation : getInvocations(smallLayer, smallDims)) {
            log.h1("Small SubTests: " + invocation.getLayer().getClass().getSimpleName());
            log.p(Arrays.deepToString(invocation.getDims()));
            tests(log, getLittleTests(), invocation, exceptions, results);
          }
        }
        if (largeLayer instanceof DAGNetwork) {
          testEquivalency = false;
          for (@Nonnull
              Invocation invocation : getInvocations(largeLayer, largeDims)) {
            log.h1("Large SubTests: " + invocation.getLayer().getClass().getSimpleName());
            log.p(Arrays.deepToString(invocation.getDims()));
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
      HashMap<CharSequence, Object> testResultProps = new HashMap<>();
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
  public Collection<Invocation> getInvocations(@Nonnull Layer smallLayer, @Nonnull int[][] smallDims) {
    @Nonnull
    DAGNetwork smallCopy = (DAGNetwork) smallLayer.copy();
    @Nonnull
    HashSet<Invocation> invocations = new HashSet<>();
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
          invocations.add(new Invocation(inner,
              Arrays.stream(array).map(x -> x.getData().getDimensions()).toArray(i -> new int[i][])));
          return result;
        }

        @Override
        public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
          return inner.getJson(resources, dataSerializer).getAsJsonObject();
        }

        @Nullable
        @Override
        public List<double[]> state() {
          return inner.state();
        }

        @Override
        protected void _free() {
        }
      };
      node.setLayer(wrapper);
    });
    Tensor[] input = Arrays.stream(smallDims).map(i -> new Tensor(i)).toArray(i -> new Tensor[i]);
    Result eval = smallCopy.eval(input);
    eval.getData();
    return invocations;
  }

  public void throwException(@Nonnull ArrayList<TestError> exceptions) {
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
  public ArrayList<TestError> standardTests(@Nonnull NotebookOutput log, long seed, TableOutput results) {
    log.p(String.format("Using Seed %d", seed));
    @Nonnull
    ArrayList<TestError> exceptions = new ArrayList<>();
    final Layer layer = getLayer(getSmallDims(new Random(seed)), new Random(seed));
    Invocation invocation = new Invocation(layer, getSmallDims(new Random(seed)));
    tests(log, getLittleTests(), invocation, exceptions, results);
    final Layer perfLayer = getLayer(getLargeDims(new Random(seed)), new Random(seed));
    bigTests(log, seed, perfLayer, exceptions, results);
    return exceptions;
  }

  public void bigTests(NotebookOutput log, long seed, @Nonnull Layer perfLayer,
                       @Nonnull ArrayList<TestError> exceptions, TableOutput results) {
    getBigTests().stream().filter(x -> null != x).forEach(test -> {
      @Nonnull
      Layer layer = perfLayer.copy();
      try {
        Tensor[] input = randomize(getLargeDims(new Random(seed)));
        LinkedHashMap<CharSequence, Object> testResultProps = new LinkedHashMap<>();
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

  private void tests(final NotebookOutput log, final List<ComponentTest<?>> tests, @Nonnull final Invocation invocation,
                     @Nonnull final ArrayList<TestError> exceptions, TableOutput results) {
    tests.stream().filter(x -> null != x).forEach((ComponentTest<?> test) -> {
      @Nonnull
      Layer layer = invocation.getLayer().copy();
      Tensor[] inputs = randomize(invocation.getDims());
      LinkedHashMap<CharSequence, Object> testResultProps = new LinkedHashMap<>();
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

  private static class Invocation extends ReferenceCountingBase {
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
      return Arrays.deepEquals(smallDims, that.smallDims);
    }

    @Override
    public int hashCode() {
      int result = layer != null ? layer.getClass().hashCode() : 0;
      result = 31 * result + Arrays.deepHashCode(smallDims);
      return result;
    }

    @Override
    protected void _free() {
      super._free();
    }
  }
}
