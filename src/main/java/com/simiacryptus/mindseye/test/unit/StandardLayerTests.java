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

import com.google.gson.GsonBuilder;
import com.simiacryptus.lang.UncheckedSupplier;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.Explodable;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.TableOutput;
import com.simiacryptus.ref.lang.LifecycleException;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.test.SysOutInterceptor;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import guru.nidi.graphviz.model.Graph;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.*;
import java.util.function.Consumer;
import java.util.function.Function;

public abstract class StandardLayerTests extends LayerTests {
  protected boolean validateBatchExecution = true;
  protected boolean testTraining = true;
  protected boolean testEquivalency = true;

  public boolean isTestTraining() {
    return testTraining;
  }

  @Nonnull
  public RefList<ComponentTest<?>> getFinalTests() {
    return RefArrays.asList(getTrainingTester());
  }

  public void setTestTraining(boolean testTraining) {
    this.testTraining = testTraining;
  }

  @Override
  public @Nullable ComponentTest<TrainingTester.ComponentResult> getTrainingTester() {
    if (!isTestTraining()) {
      return null;
    }
    return super.getTrainingTester();
  }

  @Override
  public @Nullable ComponentTest<ToleranceStatistics> getEquivalencyTester() {
    if (!testEquivalency)
      return null;
    return super.getEquivalencyTester();
  }

  @Override
  public @Nullable ComponentTest<ToleranceStatistics> getBatchingTester() {
    if (!validateBatchExecution)
      return null;
    return super.getBatchingTester();
  }

  static {
    SysOutInterceptor.INSTANCE.init();
  }

  public StandardLayerTests() {
    super();
    logger.info("Seed: " + seed);
  }

  @Nonnull
  public RefList<ComponentTest<?>> getBigTests() {
    return RefArrays.asList(getPerformanceTester(), getBatchingTester(), getReferenceIOTester(),
        getEquivalencyTester());
  }

  @Nonnull
  public RefList<ComponentTest<?>> getLittleTests() {
    return RefArrays.asList(getJsonTester(), getDerivativeTester());
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
    assert smallLayer.getClass() == largeLayer.getClass();
    TableOutput results = new TableOutput();

    try {
      log.h1("Test Modules");
      if (smallLayer instanceof DAGNetwork) {
        try {
          log.h1("Network Diagram");
          log.p("This is a network apply the following layout:");
          log.eval(RefUtil.wrapInterface((UncheckedSupplier<BufferedImage>) () -> {
            return Graphviz.fromGraph((Graph) TestUtil.toGraph(((DAGNetwork) smallLayer).addRef())).height(400).width(600)
                .render(Format.PNG).toImage();
          }, smallLayer.addRef()));
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
      @Nonnull
      RefList<TestError> exceptions = standardTests(log, seed, results);
      if (!exceptions.isEmpty()) {
        if (smallLayer instanceof DAGNetwork) {
          RefCollection<Invocation> smallInvocations = Invocation.getInvocations(smallLayer.addRef(), smallDims);
          smallInvocations.forEach(invocation -> {
            Layer subLayer = invocation.getLayer();
            assert subLayer != null;
            log.h1("Small SubTests: " + subLayer.getClass().getSimpleName());
            log.p(RefArrays.deepToString(invocation.getDims()));
            log.eval(() -> {
              return new GsonBuilder().setPrettyPrinting().create().toJson(
                  subLayer.getJson(new HashMap<>(), SerialPrecision.Double)
              );
            });
            subLayer.freeRef();
            RefArrayList<TestError> subExceptions = new RefArrayList<>();
            run(log, getLittleTests(), invocation, subExceptions.addRef(), results);
            subExceptions.forEach((TestError ex) -> log.eval(() -> {
              return Util.toString(ex);
            }));
            exceptions.addAll(subExceptions);
          });
          smallInvocations.freeRef();
        }
        if (largeLayer instanceof DAGNetwork) {
          testEquivalency = false;
          RefCollection<Invocation> largeInvocations = Invocation.getInvocations(largeLayer.addRef(), largeDims);
          largeInvocations.forEach(invocation -> {
            Layer subLayer = invocation.getLayer();
            assert subLayer != null;
            log.h1("Large SubTests: " + subLayer.getClass().getSimpleName());
            log.p(RefArrays.deepToString(invocation.getDims()));
            log.eval(() -> {
              return new GsonBuilder().setPrettyPrinting().create().toJson(
                  subLayer.getJson(new HashMap<>(), SerialPrecision.Double)
              );
            });
            subLayer.freeRef();
            RefArrayList<TestError> subExceptions = new RefArrayList<>();
            run(log, getBigTests(), invocation, subExceptions.addRef(), results);
            subExceptions.forEach((TestError ex) -> log.eval(() -> {
              return Util.toString(ex);
            }));
            exceptions.addAll(subExceptions);
          });
          largeInvocations.freeRef();
        }
      }
      log.run(RefUtil.wrapInterface(() -> {
        throwException(exceptions.addRef());
      }, exceptions));
    } finally {
      if (null != largeLayer)
        largeLayer.freeRef();
      if (null != smallLayer)
        smallLayer.freeRef();
    }
    RefList<ComponentTest<?>> finalTests = getFinalTests();
    finalTests.stream().filter(x -> {
      boolean temp_07_0005 = null != x;
      if (null != x)
        x.freeRef();
      return temp_07_0005;
    }).forEach(test -> {
      final Layer perfLayer = getLayer(largeDims, new Random(seed));
      assert perfLayer != null;
      perfLayer.assertAlive();
      @Nonnull
      Layer copy = perfLayer.copy();
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
        Object result = log.subreport(RefUtil.wrapInterface(
            (Function<NotebookOutput, ?>) sublog -> test.test(sublog, copy.addRef(),
                RefUtil.addRef(randomize)),
            copy, randomize, test),
            String.format("%s (Test %s)", log.getDisplayName(), name));
        testResultProps.put("details", null == result ? null : result.toString());
        testResultProps.put("result", "OK");
      } catch (Throwable e) {
        testResultProps.put("result", e.toString());
        throw new RuntimeException(e);
      } finally {
        results.putRow(testResultProps);
      }
    });
    finalTests.freeRef();
    log.h1("Test Matrix");
    log.out(results.toMarkdownTable());
  }

  @Nonnull
  public RefArrayList<TestError> standardTests(@Nonnull NotebookOutput log, long seed, @Nonnull TableOutput results) {
    log.p(RefString.format("Using Seed %d", seed));
    @Nonnull
    RefArrayList<TestError> exceptions = new RefArrayList<>();
    final Layer layer = getLayer(getSmallDims(new Random(seed)), new Random(seed));
    Invocation invocation = new Invocation(layer, getSmallDims(new Random(seed)));
    run(log, getLittleTests(), invocation, exceptions.addRef(), results);
    final Layer perfLayer = getLayer(getLargeDims(new Random(seed)), new Random(seed));
    bigTests(log, seed, perfLayer, exceptions.addRef(), results);
    return exceptions;
  }

  public void bigTests(@Nonnull NotebookOutput log, long seed, @Nonnull Layer perfLayer,
                       @Nonnull RefArrayList<TestError> exceptions, @Nonnull TableOutput results) {
    RefList<ComponentTest<?>> bigTests = getBigTests();
    bigTests.stream().filter(x -> {
      boolean temp_07_0007 = null != x;
      if (null != x)
        x.freeRef();
      return temp_07_0007;
    }).forEach(RefUtil.wrapInterface((Consumer<? super ComponentTest<?>>) test -> {
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
              (Function<NotebookOutput, ?>) sublog -> test.test(sublog, layer.addRef(),
                  RefUtil.addRef(input)),
              RefUtil.addRef(input), test.addRef(), layer.addRef()),
              String.format("%s (Test: %s)", log.getDisplayName(), testclass));
          testResultProps.put("details", null == result ? null : result.toString());
          RefUtil.freeRef(result);
          testResultProps.put("result", "OK");
        } catch (Throwable e) {
          testResultProps.put("result", e.toString());
          throw new RuntimeException(e);
        } finally {
          results.putRow(testResultProps);
        }
        RefUtil.freeRef(input);
      } catch (LifecycleException e) {
        throw e;
      } catch (Throwable e) {
        if (e.getClass().getSimpleName().equals("CudaError"))
          throw e;
        exceptions.add(new TestError(e, test == null ? null : test.addRef(), layer.addRef()));
      } finally {
        RefSystem.gc();
        layer.freeRef();
      }
      if (null != test)
        test.freeRef();
    }, exceptions, perfLayer));
    bigTests.freeRef();
  }

}
