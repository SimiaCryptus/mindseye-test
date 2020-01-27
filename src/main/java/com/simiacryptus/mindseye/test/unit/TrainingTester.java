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

import com.simiacryptus.lang.UncheckedSupplier;
import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch;
import com.simiacryptus.mindseye.opt.line.QuadraticSearch;
import com.simiacryptus.mindseye.opt.orient.GradientDescent;
import com.simiacryptus.mindseye.opt.orient.LBFGS;
import com.simiacryptus.mindseye.test.ProblemRun;
import com.simiacryptus.mindseye.test.StepRecord;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import smile.plot.PlotCanvas;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.swing.*;
import java.awt.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.function.IntFunction;

public abstract class TrainingTester extends ComponentTestBase<TrainingTester.ComponentResult> {
  static final Logger logger = LoggerFactory.getLogger(TrainingTester.class);

  private int batches = 3;
  private RandomizationMode randomizationMode = RandomizationMode.Permute;
  private boolean verbose = true;
  private boolean throwExceptions = false;

  public TrainingTester() {
  }

  public int getBatches() {
    return batches;
  }

  public void setBatches(int batches) {
    this.batches = batches;
  }

  public RandomizationMode getRandomizationMode() {
    return randomizationMode;
  }

  public void setRandomizationMode(RandomizationMode randomizationMode) {
    this.randomizationMode = randomizationMode;
  }

  public boolean isThrowExceptions() {
    return throwExceptions;
  }

  public void setThrowExceptions(boolean throwExceptions) {
    this.throwExceptions = throwExceptions;
  }

  public boolean isVerbose() {
    return verbose;
  }

  public void setVerbose(boolean verbose) {
    this.verbose = verbose;
  }

  @Nonnull
  public static TrainingMonitor getMonitor(@Nonnull final RefList<StepRecord> history) {
    try {
      return new TrainingMonitor() {
        @Override
        public void log(final String msg) {
          logger.info(msg);
        }

        @Override
        public void onStepComplete(@Nonnull final Step currentPoint) {
          assert currentPoint.point != null;
          history.add(new StepRecord(currentPoint.point.getMean(), currentPoint.time, currentPoint.iteration));
          currentPoint.freeRef();
        }
      };
    } finally {
      history.freeRef();
    }
  }

  @Nonnull
  public static Tensor[][] append(@Nonnull Tensor[][] left, @Nonnull Tensor[] right) {
    if (left.length != right.length) {
      IllegalArgumentException temp_18_0021 = new IllegalArgumentException(left.length + "!=" + right.length);
      RefUtil.freeRefs(left);
      RefUtil.freeRefs(right);
      throw temp_18_0021;
    }
    Tensor[][] temp_18_0020 = RefIntStream.range(0, left.length)
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor[]>) i -> RefStream
            .concat(RefArrays.stream(RefUtil.addRefs(left[i])), RefStream.of(right[i].addRef()))
            .toArray(j -> new Tensor[j]), RefUtil.addRefs(right), RefUtil.addRefs(left)))
        .toArray(j -> new Tensor[j][]);
    RefUtil.freeRefs(right);
    RefUtil.freeRefs(left);
    return temp_18_0020;
  }

  @Nonnull
  public static Tensor[][] copy(@Nonnull Tensor[][] input_gd) {
    Tensor[][] temp_18_0022 = RefArrays.stream(RefUtil.addRefs(input_gd)).map(t -> {
      Tensor[] temp_18_0001 = RefArrays.stream(RefUtil.addRefs(t)).map(v -> {
        Tensor temp_18_0002 = v.copy();
        v.freeRef();
        return temp_18_0002;
      }).toArray(i -> new Tensor[i]);
      if (null != t)
        RefUtil.freeRefs(t);
      return temp_18_0001;
    }).toArray(i -> new Tensor[i][]);
    RefUtil.freeRefs(input_gd);
    return temp_18_0022;
  }

  @Nonnull
  public static Tensor[][] pop(@Nonnull Tensor[][] data) {
    Tensor[][] temp_18_0023 = RefArrays.stream(RefUtil.addRefs(data)).map(t -> {
      Tensor[] temp_18_0003 = RefArrays.stream(RefUtil.addRefs(t)).limit(t.length - 1).toArray(i -> new Tensor[i]);
      RefUtil.freeRefs(t);
      return temp_18_0003;
    }).toArray(i -> new Tensor[i][]);
    RefUtil.freeRefs(data);
    return temp_18_0023;
  }

  @Nonnull
  public ResultType getResultType(@Nonnull final RefList<StepRecord> lbfgsmin) {
    TrainingTester.ResultType temp_18_0024 = Math.abs(min(lbfgsmin)) < 1e-9
        ? ResultType.Converged
        : ResultType.NonConverged;
    return temp_18_0024;
  }

  @Nonnull
  public JPanel grid(@Nullable final TestResult inputLearning, @Nullable final TestResult modelLearning,
                     @Nullable final TestResult completeLearning) {
    int rows = 0;
    if (inputLearning != null) {
      rows++;
    }
    if (modelLearning != null) {
      rows++;
    }
    if (completeLearning != null) {
      rows++;
    }
    @Nonnull final GridLayout layout = new GridLayout(rows, 2, 0, 0);
    @Nonnull final JPanel jPanel = new JPanel(layout);
    jPanel.setSize(1200, 400 * rows);
    if (inputLearning != null) {
      jPanel.add(inputLearning.iterPlot == null ? new JPanel() : inputLearning.iterPlot);
      jPanel.add(inputLearning.timePlot == null ? new JPanel() : inputLearning.timePlot);
    }
    if (modelLearning != null) {
      jPanel.add(modelLearning.iterPlot == null ? new JPanel() : modelLearning.iterPlot);
      jPanel.add(modelLearning.timePlot == null ? new JPanel() : modelLearning.timePlot);
    }
    if (completeLearning != null) {
      jPanel.add(completeLearning.iterPlot == null ? new JPanel() : completeLearning.iterPlot);
      jPanel.add(completeLearning.timePlot == null ? new JPanel() : completeLearning.timePlot);
    }
    return jPanel;
  }

  public boolean isZero(@Nonnull final RefDoubleStream stream) {
    return isZero(stream, 1e-14);
  }

  public boolean isZero(@Nonnull final RefDoubleStream stream, double zeroTol) {
    final double[] array = stream.toArray();
    if (array.length == 0)
      return false;
    return RefArrays.stream(array).map(x -> Math.abs(x)).sum() < zeroTol;
  }

  @Override
  public ComponentResult test(@Nonnull final NotebookOutput log, @Nonnull final Layer component,
                              @Nonnull final Tensor... inputPrototype) {
    printHeader(log);
    RefList<double[]> temp_18_0033 = component.state();
    assert temp_18_0033 != null;
    final boolean testModel = !temp_18_0033.isEmpty();
    temp_18_0033.freeRef();
    RefList<double[]> temp_18_0034 = component.state();
    assert temp_18_0034 != null;
    if (testModel && isZero(temp_18_0034.stream().flatMapToDouble(x1 -> RefArrays.stream(x1)))) {
      component.freeRef();
      RefUtil.freeRefs(inputPrototype);
      temp_18_0034.freeRef();
      throw new AssertionError("Weights are all zero?");
    }
    temp_18_0034.freeRef();
    if (isZero(RefArrays.stream(RefUtil.addRefs(inputPrototype)).flatMapToDouble(x -> {
      RefDoubleStream temp_18_0004 = RefArrays.stream(x.getData());
      x.freeRef();
      return temp_18_0004;
    }))) {
      component.freeRef();
      RefUtil.freeRefs(inputPrototype);
      throw new AssertionError("Inputs are all zero?");
    }
    @Nonnull final Random random = new Random();
    final boolean testInput = RefArrays.stream(RefUtil.addRefs(inputPrototype)).anyMatch(x -> {
      boolean temp_18_0005 = x.length() > 0;
      x.freeRef();
      return temp_18_0005;
    });
    @Nullable
    TestResult inputLearning;
    if (testInput) {
      log.h2("Input Learning");
      inputLearning = testInputLearning(log, component.addRef(), random,
          RefUtil.addRefs(inputPrototype));
    } else {
      inputLearning = null;
    }
    @Nullable
    TestResult modelLearning;
    if (testModel) {
      log.h2("Model Learning");
      modelLearning = testModelLearning(log, component.addRef(), random,
          RefUtil.addRefs(inputPrototype));
    } else {
      modelLearning = null;
    }
    @Nullable
    TestResult completeLearning;
    if (testInput && testModel) {
      log.h2("Composite Learning");
      completeLearning = testCompleteLearning(log, component.addRef(), random,
          RefUtil.addRefs(inputPrototype));
    } else {
      completeLearning = null;
    }
    RefUtil.freeRefs(inputPrototype);
    component.freeRef();
    log.h2("Results");
    log.eval(() -> {
      return grid(inputLearning, modelLearning, completeLearning);
    });
    ComponentResult result = log.eval(() -> {
      return new ComponentResult(null == inputLearning ? null : inputLearning.value,
          null == modelLearning ? null : modelLearning.value, null == completeLearning ? null : completeLearning.value);
    });
    log.setFrontMatterProperty("training_analysis", result.toString());
    if (throwExceptions) {
      assert result.complete.map.values().stream().allMatch(x -> x.type == ResultType.Converged);
      assert result.input.map.values().stream().allMatch(x -> x.type == ResultType.Converged);
      assert result.model.map.values().stream().allMatch(x -> x.type == ResultType.Converged);
    }
    return result;
  }

  @Nonnull
  public TestResult testCompleteLearning(@Nonnull final NotebookOutput log, @Nonnull final Layer component,
                                         final Random random, @Nonnull final Tensor[] inputPrototype) {
    Layer temp_18_0035 = shuffle(random, component.copy());
    temp_18_0035.freeze();
    @Nonnull final Layer network_target = temp_18_0035.addRef();
    temp_18_0035.freeRef();
    final Tensor[][] input_target = shuffleCopy(random, RefUtil.addRefs(inputPrototype));
    log.p(
        "In this apply, attempt to train a network to emulate a randomized network given an example input/output. The target state is:");
    log.eval(RefUtil.wrapInterface((UncheckedSupplier<String>) () -> {
      RefList<double[]> temp_18_0037 = network_target.state();
      assert temp_18_0037 != null;
      String temp_18_0036 = temp_18_0037.stream().map(RefArrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
      temp_18_0037.freeRef();
      return temp_18_0036;
    }, network_target.addRef()));
    log.p("We simultaneously regress this target input:");
    log.eval(RefUtil.wrapInterface((UncheckedSupplier<String>) () -> {
      return RefArrays.stream(RefUtil.addRefs(input_target)).flatMap(x -> {
        RefStream<Tensor> temp_18_0006 = RefArrays.stream(RefUtil.addRefs(x));
        if (null != x)
          RefUtil.freeRefs(x);
        return temp_18_0006;
      }).map(x -> {
        String temp_18_0007 = x.prettyPrint();
        x.freeRef();
        return temp_18_0007;
      }).reduce((a, b) -> a + "\n" + b).orElse("");
    }, RefUtil.addRefs(input_target)));
    log.p("Which produces the following output:");
    Result[] inputs = ConstantResult.batchResultArray(RefUtil.addRefs(input_target));
    RefUtil.freeRefs(input_target);
    Result temp_18_0038 = network_target.eval(RefUtil.addRefs(inputs));
    assert temp_18_0038 != null;
    TensorList result = temp_18_0038.getData();
    temp_18_0038.freeRef();
    RefUtil.freeRefs(inputs);
    network_target.freeRef();
    final Tensor[] output_target = result.stream().toArray(i -> new Tensor[i]);
    result.freeRef();
    log.eval(RefUtil.wrapInterface((UncheckedSupplier<String>) () -> {
      return RefStream.of(RefUtil.addRefs(output_target)).map(x -> {
        String temp_18_0008 = x.prettyPrint();
        x.freeRef();
        return temp_18_0008;
      }).reduce((a, b) -> a + "\n" + b).orElse("");
    }, RefUtil.addRefs(output_target)));
    //if (output_target.length != inputPrototype.length) return null;
    Tensor[][] trainingInput = append(shuffleCopy(random, RefUtil.addRefs(inputPrototype)),
        RefUtil.addRefs(output_target));
    RefUtil.freeRefs(output_target);
    TrainingTester.TestResult temp_18_0009 = trainAll("Integrated Convergence", log, RefUtil.addRefs(trainingInput),
        shuffle(random, component.copy()), buildMask(inputPrototype.length));
    RefUtil.freeRefs(inputPrototype);
    component.freeRef();
    RefUtil.freeRefs(trainingInput);
    return temp_18_0009;
  }

  @Nullable
  public TestResult testInputLearning(@Nonnull final NotebookOutput log, @Nonnull final Layer component,
                                      final Random random, @Nonnull final Tensor[] inputPrototype) {
    Layer network = shuffle(random, component.copy());
    network.freeze();
    component.freeRef();
    final Tensor[][] input_target = shuffleCopy(random, RefUtil.addRefs(inputPrototype));
    log.p("In this apply, we use a network to learn this target input, given it's pre-evaluated output:");
    log.eval(RefUtil.wrapInterface((UncheckedSupplier<String>) () -> {
      return RefArrays.stream(RefUtil.addRefs(input_target)).flatMap(x -> {
        return RefArrays.stream(x);
      }).map(x -> {
        try {
          return x.prettyPrint();
        } finally {
          x.freeRef();
        }
      }).reduce((a, b) -> a + "\n" + b).orElse("");
    }, RefUtil.addRefs(input_target)));
    Result eval = network.eval(ConstantResult.batchResultArray(input_target));
    TensorList result = eval.getData();
    eval.freeRef();
    int resultLength = result.length();
    if (resultLength != getBatches()) {
      logger.info(RefString.format("Meta layers not supported. %d != %d", resultLength, getBatches()));
      network.freeRef();
      RefUtil.freeRefs(inputPrototype);
      result.freeRef();
      return null;
    }
    final Tensor[] output_target = result.stream().toArray(i -> new Tensor[i]);
    result.freeRef();
    //if (output_target.length != inputPrototype.length) return null;
    int inputPrototypeLength = inputPrototype.length;
    return trainAll("Input Convergence",
        log,
        append(shuffleCopy(random, inputPrototype), output_target),
        network,
        buildMask(inputPrototypeLength));
  }

  @Nullable
  public TestResult testModelLearning(@Nonnull final NotebookOutput log, @Nonnull final Layer component,
                                      final Random random, @Nullable final Tensor[] inputPrototype) {
    Layer temp_18_0040 = shuffle(random, component.copy());
    temp_18_0040.freeze();
    @Nonnull final Layer network_target = temp_18_0040.addRef();
    temp_18_0040.freeRef();
    final Tensor[][] input_target = shuffleCopy(random, RefUtil.addRefs(inputPrototype));
    if (null != inputPrototype)
      RefUtil.freeRefs(inputPrototype);
    log.p(
        "In this apply, attempt to train a network to emulate a randomized network given an example input/output. The target state is:");
    log.eval(RefUtil.wrapInterface((UncheckedSupplier<String>) () -> {
      RefList<double[]> temp_18_0042 = network_target.state();
      assert temp_18_0042 != null;
      String temp_18_0041 = temp_18_0042.stream().map(RefArrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
      temp_18_0042.freeRef();
      return temp_18_0041;
    }, network_target.addRef()));
    Result[] array = ConstantResult.batchResultArray(RefUtil.addRefs(input_target));
    Result eval = network_target.eval(RefUtil.addRefs(array));
    RefUtil.freeRefs(array);
    network_target.freeRef();
    assert eval != null;
    TensorList result = eval.getData();
    eval.freeRef();
    final Tensor[] output_target = result.stream().toArray(i -> new Tensor[i]);
    result.freeRef();
    if (output_target.length != input_target.length) {
      logger.info("Batch layers not supported");
      RefUtil.freeRefs(input_target);
      RefUtil.freeRefs(output_target);
      component.freeRef();
      return null;
    }
    Tensor[][] trainingInput = append(RefUtil.addRefs(input_target), RefUtil.addRefs(output_target));
    RefUtil.freeRefs(output_target);
    RefUtil.freeRefs(input_target);
    TrainingTester.TestResult temp_18_0013 = trainAll("Model Convergence", log, RefUtil.addRefs(trainingInput),
        shuffle(random, component.copy()));
    component.freeRef();
    RefUtil.freeRefs(trainingInput);
    return temp_18_0013;
  }

  public double min(@Nonnull RefList<StepRecord> history) {
    double temp_18_0025 = history.stream().mapToDouble(x -> x.fitness).min().orElse(Double.NaN);
    history.freeRef();
    return temp_18_0025;
  }

  @Nonnull
  public boolean[] buildMask(int length) {
    @Nonnull final boolean[] mask = new boolean[length + 1];
    for (int i = 0; i < length; i++) {
      mask[i] = true;
    }
    return mask;
  }

  @Nonnull
  public TestResult trainAll(CharSequence title, @Nonnull NotebookOutput log, @Nonnull Tensor[][] trainingInput,
                             @Nonnull Layer layer, boolean... mask) {
    log.h3("Gradient Descent");
    final RefList<StepRecord> gd = train(log, this::trainGD, layer.copy(), copy(RefUtil.addRefs(trainingInput)), mask);
    log.h3("Conjugate Gradient Descent");
    final RefList<StepRecord> cjgd = train(log, this::trainCjGD, layer.copy(), copy(RefUtil.addRefs(trainingInput)),
        mask);
    log.h3("Limited-Memory BFGS");
    final RefList<StepRecord> lbfgs = train(log, this::trainLBFGS, layer.copy(), copy(RefUtil.addRefs(trainingInput)),
        mask);
    @Nonnull final ProblemRun[] runs = {
        new ProblemRun("GD", gd.addRef(), Color.GRAY, ProblemRun.PlotType.Line),
        new ProblemRun("CjGD", cjgd.addRef(), Color.CYAN, ProblemRun.PlotType.Line),
        new ProblemRun("LBFGS", lbfgs.addRef(), Color.GREEN, ProblemRun.PlotType.Line)};
    @Nonnull
    ProblemResult result = new ProblemResult();
    result.put("GD",
        new TrainingResult(getResultType(gd.addRef()), min(gd.addRef())));
    gd.freeRef();
    result.put("CjGD", new TrainingResult(getResultType(cjgd.addRef()),
        min(cjgd.addRef())));
    cjgd.freeRef();
    result.put("LBFGS", new TrainingResult(getResultType(lbfgs.addRef()),
        min(lbfgs.addRef())));
    lbfgs.freeRef();
    if (verbose) {
      final PlotCanvas iterPlot = log.eval(() -> {
        return TestUtil.compare(title + " vs Iteration", runs);
      });
      final PlotCanvas timePlot = log.eval(() -> {
        return TestUtil.compareTime(title + " vs Time", runs);
      });
      RefUtil.freeRefs(trainingInput);
      layer.freeRef();
      return new TestResult(iterPlot, timePlot, result);
    } else {
      @Nullable final PlotCanvas iterPlot = TestUtil.compare(title + " vs Iteration", runs);
      @Nullable final PlotCanvas timePlot = TestUtil.compareTime(title + " vs Time", runs);
      RefUtil.freeRefs(trainingInput);
      layer.freeRef();
      return new TestResult(iterPlot, timePlot, result);
    }
  }

  @Nonnull
  public RefList<StepRecord> trainCjGD(@Nonnull final NotebookOutput log, @Nullable final Trainable trainable) {
    log.p(
        "First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.");
    @Nonnull final RefList<StepRecord> history = new RefArrayList<>();
    @Nonnull final TrainingMonitor monitor = TrainingTester.getMonitor(history.addRef());
    try {
      log.eval(RefUtil.wrapInterface((UncheckedSupplier<Double>) () -> {
        IterativeTrainer temp_18_0028 = new IterativeTrainer(trainable == null ? null : trainable.addRef());
        temp_18_0028.setLineSearchFactory(label -> new QuadraticSearch());
        IterativeTrainer temp_18_0043 = temp_18_0028.addRef();
        temp_18_0043.setOrientation(new GradientDescent());
        IterativeTrainer temp_18_0044 = temp_18_0043.addRef();
        temp_18_0044.setMonitor(monitor);
        IterativeTrainer temp_18_0045 = temp_18_0044.addRef();
        temp_18_0045.setTimeout(30, TimeUnit.SECONDS);
        IterativeTrainer temp_18_0046 = temp_18_0045.addRef();
        temp_18_0046.setMaxIterations(250);
        IterativeTrainer temp_18_0047 = temp_18_0046.addRef();
        temp_18_0047.setTerminateThreshold(0);
        IterativeTrainer temp_18_0048 = temp_18_0047.addRef();
        double temp_18_0027 = temp_18_0048.run();
        temp_18_0048.freeRef();
        temp_18_0047.freeRef();
        temp_18_0046.freeRef();
        temp_18_0045.freeRef();
        temp_18_0044.freeRef();
        temp_18_0043.freeRef();
        temp_18_0028.freeRef();
        return temp_18_0027;
      }, trainable == null ? null : trainable.addRef()));
    } catch (Throwable e) {
      if (isThrowExceptions())
        throw new RuntimeException(e);
    }
    if (null != trainable)
      trainable.freeRef();
    return history;
  }

  @Nonnull
  public RefList<StepRecord> trainGD(@Nonnull final NotebookOutput log, @Nullable final Trainable trainable) {
    log.p("First, we train using basic gradient descent method apply weak line search conditions.");
    @Nonnull final RefList<StepRecord> history = new RefArrayList<>();
    @Nonnull final TrainingMonitor monitor = TrainingTester.getMonitor(history.addRef());
    try {
      log.eval(RefUtil.wrapInterface((UncheckedSupplier<Double>) () -> {
        IterativeTrainer temp_18_0030 = new IterativeTrainer(trainable == null ? null : trainable.addRef());
        temp_18_0030.setLineSearchFactory(label -> new ArmijoWolfeSearch());
        IterativeTrainer temp_18_0049 = temp_18_0030.addRef();
        temp_18_0049.setOrientation(new GradientDescent());
        IterativeTrainer temp_18_0050 = temp_18_0049.addRef();
        temp_18_0050.setMonitor(monitor);
        IterativeTrainer temp_18_0051 = temp_18_0050.addRef();
        temp_18_0051.setTimeout(30, TimeUnit.SECONDS);
        IterativeTrainer temp_18_0052 = temp_18_0051.addRef();
        temp_18_0052.setMaxIterations(250);
        IterativeTrainer temp_18_0053 = temp_18_0052.addRef();
        temp_18_0053.setTerminateThreshold(0);
        IterativeTrainer temp_18_0054 = temp_18_0053.addRef();
        double temp_18_0029 = temp_18_0054.run();
        temp_18_0054.freeRef();
        temp_18_0053.freeRef();
        temp_18_0052.freeRef();
        temp_18_0051.freeRef();
        temp_18_0050.freeRef();
        temp_18_0049.freeRef();
        temp_18_0030.freeRef();
        return temp_18_0029;
      }, trainable == null ? null : trainable.addRef()));
    } catch (Throwable e) {
      if (isThrowExceptions())
        throw new RuntimeException(e);
    }
    if (null != trainable)
      trainable.freeRef();
    return history;
  }

  @Nonnull
  public RefList<StepRecord> trainLBFGS(@Nonnull final NotebookOutput log, @Nullable final Trainable trainable) {
    log.p(
        "Next, we apply the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.");
    @Nonnull final RefList<StepRecord> history = new RefArrayList<>();
    @Nonnull final TrainingMonitor monitor = TrainingTester.getMonitor(history.addRef());
    try {
      log.eval(RefUtil.wrapInterface((UncheckedSupplier<Double>) () -> {
        IterativeTrainer temp_18_0032 = new IterativeTrainer(trainable == null ? null : trainable.addRef());
        temp_18_0032.setLineSearchFactory(label -> new ArmijoWolfeSearch());
        IterativeTrainer temp_18_0055 = temp_18_0032.addRef();
        temp_18_0055.setOrientation(new LBFGS());
        IterativeTrainer temp_18_0056 = temp_18_0055.addRef();
        temp_18_0056.setMonitor(monitor);
        IterativeTrainer temp_18_0057 = temp_18_0056.addRef();
        temp_18_0057.setTimeout(30, TimeUnit.SECONDS);
        IterativeTrainer temp_18_0058 = temp_18_0057.addRef();
        temp_18_0058.setIterationsPerSample(100);
        IterativeTrainer temp_18_0059 = temp_18_0058.addRef();
        temp_18_0059.setMaxIterations(250);
        IterativeTrainer temp_18_0060 = temp_18_0059.addRef();
        temp_18_0060.setTerminateThreshold(0);
        IterativeTrainer temp_18_0061 = temp_18_0060.addRef();
        double temp_18_0031 = temp_18_0061.run();
        temp_18_0061.freeRef();
        temp_18_0060.freeRef();
        temp_18_0059.freeRef();
        temp_18_0058.freeRef();
        temp_18_0057.freeRef();
        temp_18_0056.freeRef();
        temp_18_0055.freeRef();
        temp_18_0032.freeRef();
        return temp_18_0031;
      }, trainable == null ? null : trainable.addRef()));
    } catch (Throwable e) {
      if (isThrowExceptions())
        throw new RuntimeException(e);
    }
    if (null != trainable)
      trainable.freeRef();
    return history;
  }

  @Nonnull
  @Override
  public String toString() {
    return "TrainingTester{" + "batches=" + batches + ", randomizationMode=" + randomizationMode + ", verbose="
        + verbose + ", throwExceptions=" + throwExceptions + '}';
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  TrainingTester addRef() {
    return (TrainingTester) super.addRef();
  }

  protected void printHeader(@Nonnull NotebookOutput log) {
    log.h1("Training Characteristics");
  }

  protected abstract Layer lossLayer();

  @Nonnull
  private Layer shuffle(final Random random, @Nonnull final Layer testComponent) {
    RefList<double[]> temp_18_0062 = testComponent.state();
    assert temp_18_0062 != null;
    temp_18_0062.forEach(buffer -> {
      randomizationMode.shuffle(random, buffer);
    });
    temp_18_0062.freeRef();
    return testComponent;
  }

  @Nonnull
  private Tensor[][] shuffleCopy(final Random random, @Nonnull final Tensor... copy) {
    Tensor[][] temp_18_0026 = RefIntStream.range(0, getBatches())
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor[]>) i -> {
          return RefArrays.stream(RefUtil.addRefs(copy)).map(tensor -> {
            @Nonnull final Tensor cpy = tensor.copy();
            tensor.freeRef();
            randomizationMode.shuffle(random, cpy.getData());
            return cpy;
          }).toArray(j -> new Tensor[j]);
        }, RefUtil.addRefs(copy))).toArray(i -> new Tensor[i][]);
    RefUtil.freeRefs(copy);
    return temp_18_0026;
  }

  private RefList<StepRecord> train(@Nonnull NotebookOutput log,
                                    @Nonnull RefBiFunction<NotebookOutput, Trainable, RefList<StepRecord>> opt,
                                    @Nonnull Layer layer,
                                    @Nonnull Tensor[][] data, @Nonnull boolean... mask) {
    int inputs = data[0].length;
    @Nonnull final PipelineNetwork network = new PipelineNetwork(inputs);
    Layer lossLayer = lossLayer();
    assert null != lossLayer : getClass().toString();
    RefUtil.freeRef(network.add(lossLayer.addRef(),
        network.add(layer.addRef(),
            RefIntStream.range(0, inputs - 1)
                .mapToObj(RefUtil.wrapInterface((IntFunction<? extends DAGNode>) i -> network.getInput(i),
                    network.addRef()))
                .toArray(i -> new DAGNode[i])),
        network.getInput(inputs - 1)));
    lossLayer.freeRef();
    @Nonnull
    ArrayTrainable trainable = new ArrayTrainable(RefUtil.addRefs(data), network.addRef());
    if (0 < mask.length)
      trainable.setMask(mask);
    RefList<StepRecord> history = opt.apply(log, trainable.addRef());
    if (history.stream().mapToDouble(x -> x.fitness).min().orElse(1) > 1e-5) {
      if (!network.isFrozen()) {
        log.p("This training apply resulted in the following configuration:");
        log.eval(RefUtil.wrapInterface((UncheckedSupplier<String>) () -> {
          RefList<double[]> temp_18_0064 = network.state();
          assert temp_18_0064 != null;
          String temp_18_0063 = temp_18_0064.stream().map(RefArrays::toString).reduce((a, b) -> a + "\n" + b)
              .orElse("");
          temp_18_0064.freeRef();
          return temp_18_0063;
        }, network.addRef()));
      }
      if (0 < mask.length) {
        log.p("And regressed input:");
        log.eval(RefUtil.wrapInterface((UncheckedSupplier<String>) () -> {
          return RefArrays.stream(RefUtil.addRefs(data)).flatMap(x -> {
            RefStream<Tensor> temp_18_0014 = RefArrays.stream(RefUtil.addRefs(x));
            if (null != x)
              RefUtil.freeRefs(x);
            return temp_18_0014;
          }).limit(1).map(x -> {
            String temp_18_0015 = x.prettyPrint();
            x.freeRef();
            return temp_18_0015;
          }).reduce((a, b) -> a + "\n" + b).orElse("");
        }, RefUtil.addRefs(data)));
      }
      log.p("To produce the following output:");
      log.eval(RefUtil.wrapInterface((UncheckedSupplier<String>) () -> {
        Result[] array = ConstantResult.batchResultArray(pop(RefUtil.addRefs(data)));
        @Nullable
        Result eval = layer.eval(RefUtil.addRefs(array));
        for (@Nonnull
            Result result : array) {
          RefUtil.freeRef(result.getData());
        }
        RefUtil.freeRefs(array);
        assert eval != null;
        TensorList tensorList = eval.getData();
        eval.freeRef();
        String temp_18_0016 = tensorList.stream().limit(1).map(x -> {
          String temp_18_0017 = x.prettyPrint();
          x.freeRef();
          return temp_18_0017;
        }).reduce((a, b) -> a + "\n" + b).orElse("");
        tensorList.freeRef();
        return temp_18_0016;
      }, RefUtil.addRefs(data), layer.addRef()));
    } else {
      log.p("Training Converged");
    }
    trainable.freeRef();
    network.freeRef();
    layer.freeRef();
    RefUtil.freeRefs(data);
    return history;
  }

  public enum ResultType {
    Converged, NonConverged
  }

  public enum RandomizationMode {
    Permute {
      @Override
      public void shuffle(@Nonnull final Random random, @Nonnull final double[] buffer) {
        for (int i = 0; i < buffer.length; i++) {
          final int j = random.nextInt(buffer.length);
          final double v = buffer[i];
          buffer[i] = buffer[j];
          buffer[j] = v;
        }
      }
    },
    PermuteDuplicates {
      @Override
      public void shuffle(@Nonnull final Random random, @Nonnull final double[] buffer) {
        Permute.shuffle(random, buffer);
        for (int i = 0; i < buffer.length; i++) {
          buffer[i] = buffer[random.nextInt(buffer.length)];
        }
      }
    },
    Random {
      @Override
      public void shuffle(@Nonnull final Random random, @Nonnull final double[] buffer) {
        for (int i = 0; i < buffer.length; i++) {
          buffer[i] = 2 * (random.nextDouble() - 0.5);
        }
      }
    };

    public abstract void shuffle(Random random, double[] buffer);
  }

  public static class ComponentResult {
    final ProblemResult complete;
    final ProblemResult input;
    final ProblemResult model;

    public ComponentResult(final ProblemResult input, final ProblemResult model, final ProblemResult complete) {
      this.input = input;
      this.model = model;
      this.complete = complete;
    }

    @Nonnull
    @Override
    public String toString() {
      return RefString.format("{\"input\":%s, \"model\":%s, \"complete\":%s}", input, model, complete);
    }
  }

  public static class TestResult {
    final PlotCanvas iterPlot;
    final PlotCanvas timePlot;
    final ProblemResult value;

    public TestResult(final PlotCanvas iterPlot, final PlotCanvas timePlot, final ProblemResult value) {
      this.timePlot = timePlot;
      this.iterPlot = iterPlot;
      this.value = value;
    }
  }

  public static final class TrainingResult {
    final ResultType type;
    final double value;

    public TrainingResult(final ResultType type, final double value) {
      this.type = type;
      this.value = value;
    }

    @Nonnull
    @Override
    public String toString() {
      return RefString.format("{\"type\":\"%s\", value:%s}", type, value);
    }
  }

  public static class ProblemResult {
    @Nonnull
    final Map<CharSequence, TrainingResult> map;

    public ProblemResult() {
      this.map = new HashMap<>();
    }

    @Nonnull
    public void put(CharSequence key, TrainingResult result) {
      map.put(key, result);
    }

    @Nonnull
    @Override
    public String toString() {
      return RefUtil.get(map.entrySet().stream().map(e -> {
        String temp_18_0018 = RefString.format("\"%s\": %s", e.getKey(), e.getValue().toString());
        RefUtil.freeRef(e);
        return temp_18_0018;
      }).reduce((a, b) -> a + ", " + b));
    }
  }
}
