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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import smile.plot.PlotCanvas;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.swing.*;
import java.awt.*;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.function.BiFunction;
import java.util.function.IntFunction;

public abstract @RefAware
class TrainingTester extends ComponentTestBase<TrainingTester.ComponentResult> {
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

  public TrainingTester setBatches(final int batches) {
    this.batches = batches;
    return this.addRef();
  }

  public RandomizationMode getRandomizationMode() {
    return randomizationMode;
  }

  @Nonnull
  public TrainingTester setRandomizationMode(final RandomizationMode randomizationMode) {
    this.randomizationMode = randomizationMode;
    return this.addRef();
  }

  public boolean isThrowExceptions() {
    return throwExceptions;
  }

  @Nonnull
  public TrainingTester setThrowExceptions(boolean throwExceptions) {
    this.throwExceptions = throwExceptions;
    return this.addRef();
  }

  public boolean isVerbose() {
    return verbose;
  }

  @Nonnull
  public TrainingTester setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this.addRef();
  }

  public static TrainingMonitor getMonitor(@Nonnull final RefList<StepRecord> history) {
    try {
      return new TrainingMonitor() {
        @Override
        public void log(final String msg) {
          logger.info(msg);
        }

        @Override
        public void onStepComplete(@Nonnull final Step currentPoint) {
          history.add(new StepRecord(currentPoint.point.getMean(), currentPoint.time, currentPoint.iteration));
          currentPoint.freeRef();
        }
      };
    } finally {
      history.freeRef();
    }
  }

  public static Tensor[][] append(@Nonnull Tensor[][] left, Tensor[] right) {
    if (left.length != right.length) {
      IllegalArgumentException temp_18_0021 = new IllegalArgumentException(left.length + "!=" + right.length);
      if (null != left)
        ReferenceCounting.freeRefs(left);
      if (null != right)
        ReferenceCounting.freeRefs(right);
      throw temp_18_0021;
    }
    Tensor[][] temp_18_0020 = RefIntStream.range(0, left.length)
        .mapToObj(RefUtil.wrapInterface(
            (IntFunction<? extends Tensor[]>) i -> RefStream
                .concat(RefArrays.stream(Tensor.addRefs(left[i])),
                    RefStream.of(right[i].addRef()))
                .toArray(j -> new Tensor[j]),
            Tensor.addRefs(right), Tensor.addRefs(left)))
        .toArray(j -> new Tensor[j][]);
    if (null != right)
      ReferenceCounting.freeRefs(right);
    ReferenceCounting.freeRefs(left);
    return temp_18_0020;
  }

  public static Tensor[][] copy(@Nonnull Tensor[][] input_gd) {
    Tensor[][] temp_18_0022 = RefArrays
        .stream(Tensor.addRefs(input_gd)).map(t -> {
          Tensor[] temp_18_0001 = RefArrays
              .stream(Tensor.addRefs(t)).map(v -> {
                Tensor temp_18_0002 = v.copy();
                if (null != v)
                  v.freeRef();
                return temp_18_0002;
              }).toArray(i -> new Tensor[i]);
          if (null != t)
            ReferenceCounting.freeRefs(t);
          return temp_18_0001;
        }).toArray(i -> new Tensor[i][]);
    ReferenceCounting.freeRefs(input_gd);
    return temp_18_0022;
  }

  public static Tensor[][] pop(@Nonnull Tensor[][] data) {
    Tensor[][] temp_18_0023 = RefArrays
        .stream(Tensor.addRefs(data)).map(t -> {
          Tensor[] temp_18_0003 = RefArrays
              .stream(Tensor.addRefs(t)).limit(t.length - 1).toArray(i -> new Tensor[i]);
          if (null != t)
            ReferenceCounting.freeRefs(t);
          return temp_18_0003;
        }).toArray(i -> new Tensor[i][]);
    ReferenceCounting.freeRefs(data);
    return temp_18_0023;
  }

  public static @SuppressWarnings("unused")
  TrainingTester[] addRefs(TrainingTester[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(TrainingTester::addRef)
        .toArray((x) -> new TrainingTester[x]);
  }

  public static @SuppressWarnings("unused")
  TrainingTester[][] addRefs(TrainingTester[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(TrainingTester::addRefs)
        .toArray((x) -> new TrainingTester[x][]);
  }

  @Nonnull
  public ResultType getResultType(@Nonnull final RefList<StepRecord> lbfgsmin) {
    TrainingTester.ResultType temp_18_0024 = Math
        .abs(min(lbfgsmin == null ? null : lbfgsmin)) < 1e-9 ? ResultType.Converged : ResultType.NonConverged;
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
    final boolean testModel = !temp_18_0033.isEmpty();
    if (null != temp_18_0033)
      temp_18_0033.freeRef();
    RefList<double[]> temp_18_0034 = component.state();
    if (testModel && isZero(temp_18_0034.stream().flatMapToDouble(x1 -> RefArrays.stream(x1)))) {
      component.freeRef();
      ReferenceCounting.freeRefs(inputPrototype);
      throw new AssertionError("Weights are all zero?");
    }
    if (null != temp_18_0034)
      temp_18_0034.freeRef();
    if (isZero(RefArrays.stream(Tensor.addRefs(inputPrototype)).flatMapToDouble(x -> {
      RefDoubleStream temp_18_0004 = RefArrays.stream(x.getData());
      if (null != x)
        x.freeRef();
      return temp_18_0004;
    }))) {
      component.freeRef();
      ReferenceCounting.freeRefs(inputPrototype);
      throw new AssertionError("Inputs are all zero?");
    }
    @Nonnull final Random random = new Random();
    final boolean testInput = RefArrays.stream(Tensor.addRefs(inputPrototype))
        .anyMatch(x -> {
          boolean temp_18_0005 = x.length() > 0;
          if (null != x)
            x.freeRef();
          return temp_18_0005;
        });
    @Nullable
    TestResult inputLearning;
    if (testInput) {
      log.h2("Input Learning");
      inputLearning = testInputLearning(log, component == null ? null : component.addRef(), random,
          Tensor.addRefs(inputPrototype));
    } else {
      inputLearning = null;
    }
    @Nullable
    TestResult modelLearning;
    if (testModel) {
      log.h2("Model Learning");
      modelLearning = testModelLearning(log, component == null ? null : component.addRef(), random,
          Tensor.addRefs(inputPrototype));
    } else {
      modelLearning = null;
    }
    @Nullable
    TestResult completeLearning;
    if (testInput && testModel) {
      log.h2("Composite Learning");
      completeLearning = testCompleteLearning(log, component == null ? null : component.addRef(), random,
          Tensor.addRefs(inputPrototype));
    } else {
      completeLearning = null;
    }
    ReferenceCounting.freeRefs(inputPrototype);
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
    @Nonnull final Layer network_target = temp_18_0035.freeze();
    if (null != temp_18_0035)
      temp_18_0035.freeRef();
    final Tensor[][] input_target = shuffleCopy(random, Tensor.addRefs(inputPrototype));
    log.p(
        "In this apply, attempt to train a network to emulate a randomized network given an example input/output. The target state is:");
    log.eval(RefUtil
        .wrapInterface((UncheckedSupplier<String>) () -> {
          RefList<double[]> temp_18_0037 = network_target.state();
          String temp_18_0036 = temp_18_0037.stream().map(RefArrays::toString).reduce((a, b) -> a + "\n" + b)
              .orElse("");
          if (null != temp_18_0037)
            temp_18_0037.freeRef();
          return temp_18_0036;
        }, network_target == null ? null : network_target.addRef()));
    log.p("We simultaneously regress this target input:");
    log.eval(RefUtil
        .wrapInterface((UncheckedSupplier<String>) () -> {
          return RefArrays.stream(Tensor.addRefs(input_target)).flatMap(x -> {
            RefStream<Tensor> temp_18_0006 = RefArrays
                .stream(Tensor.addRefs(x));
            if (null != x)
              ReferenceCounting.freeRefs(x);
            return temp_18_0006;
          }).map(x -> {
            String temp_18_0007 = x.prettyPrint();
            if (null != x)
              x.freeRef();
            return temp_18_0007;
          }).reduce((a, b) -> a + "\n" + b).orElse("");
        }, Tensor.addRefs(input_target)));
    log.p("Which produces the following output:");
    Result[] inputs = ConstantResult.batchResultArray(Tensor.addRefs(input_target));
    if (null != input_target)
      ReferenceCounting.freeRefs(input_target);
    Result temp_18_0038 = network_target
        .eval(Result.addRefs(inputs));
    TensorList result = temp_18_0038.getData();
    if (null != temp_18_0038)
      temp_18_0038.freeRef();
    if (null != inputs)
      ReferenceCounting.freeRefs(inputs);
    network_target.freeRef();
    final Tensor[] output_target = result.stream().toArray(i -> new Tensor[i]);
    if (null != result)
      result.freeRef();
    log.eval(RefUtil
        .wrapInterface((UncheckedSupplier<String>) () -> {
          return RefStream.of(Tensor.addRefs(output_target)).map(x -> {
            String temp_18_0008 = x.prettyPrint();
            if (null != x)
              x.freeRef();
            return temp_18_0008;
          }).reduce((a, b) -> a + "\n" + b).orElse("");
        }, Tensor.addRefs(output_target)));
    //if (output_target.length != inputPrototype.length) return null;
    Tensor[][] trainingInput = append(
        shuffleCopy(random, Tensor.addRefs(inputPrototype)),
        Tensor.addRefs(output_target));
    if (null != output_target)
      ReferenceCounting.freeRefs(output_target);
    TrainingTester.TestResult temp_18_0009 = trainAll("Integrated Convergence", log,
        Tensor.addRefs(trainingInput), shuffle(random, component.copy()),
        buildMask(inputPrototype.length));
    ReferenceCounting.freeRefs(inputPrototype);
    component.freeRef();
    if (null != trainingInput)
      ReferenceCounting.freeRefs(trainingInput);
    return temp_18_0009;
  }

  public TestResult testInputLearning(@Nonnull final NotebookOutput log, @Nonnull final Layer component,
                                      final Random random, @Nonnull final Tensor[] inputPrototype) {
    Layer temp_18_0039 = shuffle(random, component.copy());
    @Nonnull final Layer network = temp_18_0039.freeze();
    if (null != temp_18_0039)
      temp_18_0039.freeRef();
    component.freeRef();
    final Tensor[][] input_target = shuffleCopy(random, Tensor.addRefs(inputPrototype));
    log.p("In this apply, we use a network to learn this target input, given it's pre-evaluated output:");
    log.eval(RefUtil
        .wrapInterface((UncheckedSupplier<String>) () -> {
          return RefArrays.stream(Tensor.addRefs(input_target)).flatMap(x -> {
            RefStream<Tensor> temp_18_0011 = RefArrays
                .stream(Tensor.addRefs(x));
            if (null != x)
              ReferenceCounting.freeRefs(x);
            return temp_18_0011;
          }).map(x -> {
            String temp_18_0012 = x.prettyPrint();
            if (null != x)
              x.freeRef();
            return temp_18_0012;
          }).reduce((a, b) -> a + "\n" + b).orElse("");
        }, Tensor.addRefs(input_target)));
    Result[] array = ConstantResult.batchResultArray(Tensor.addRefs(input_target));
    if (null != input_target)
      ReferenceCounting.freeRefs(input_target);
    @Nullable
    Result eval = network.eval(Result.addRefs(array));
    TensorList result = eval.getData();
    if (null != eval)
      eval.freeRef();
    final Tensor[] output_target = result.stream().toArray(i -> new Tensor[i]);
    if (null != result)
      result.freeRef();
    if (output_target.length != getBatches()) {
      logger.info(String.format("Meta layers not supported. %d != %d", output_target.length, getBatches()));
      network.freeRef();
      if (null != array)
        ReferenceCounting.freeRefs(array);
      if (null != output_target)
        ReferenceCounting.freeRefs(output_target);
      ReferenceCounting.freeRefs(inputPrototype);
      return null;
    }

    for (@Nonnull
        Result nnResult : array) {
      RefUtil.freeRef(nnResult.getData());
    }
    if (null != array)
      ReferenceCounting.freeRefs(array);
    //if (output_target.length != inputPrototype.length) return null;
    Tensor[][] trainingInput = append(
        shuffleCopy(random, Tensor.addRefs(inputPrototype)),
        Tensor.addRefs(output_target));
    if (null != output_target)
      ReferenceCounting.freeRefs(output_target);
    TrainingTester.TestResult temp_18_0010 = trainAll("Input Convergence", log,
        Tensor.addRefs(trainingInput), network == null ? null : network,
        buildMask(inputPrototype.length));
    ReferenceCounting.freeRefs(inputPrototype);
    if (null != trainingInput)
      ReferenceCounting.freeRefs(trainingInput);
    return temp_18_0010;
  }

  public TestResult testModelLearning(@Nonnull final NotebookOutput log, @Nonnull final Layer component,
                                      final Random random, final Tensor[] inputPrototype) {
    Layer temp_18_0040 = shuffle(random, component.copy());
    @Nonnull final Layer network_target = temp_18_0040.freeze();
    if (null != temp_18_0040)
      temp_18_0040.freeRef();
    final Tensor[][] input_target = shuffleCopy(random, Tensor.addRefs(inputPrototype));
    if (null != inputPrototype)
      ReferenceCounting.freeRefs(inputPrototype);
    log.p(
        "In this apply, attempt to train a network to emulate a randomized network given an example input/output. The target state is:");
    log.eval(RefUtil
        .wrapInterface((UncheckedSupplier<String>) () -> {
          RefList<double[]> temp_18_0042 = network_target.state();
          String temp_18_0041 = temp_18_0042.stream().map(RefArrays::toString).reduce((a, b) -> a + "\n" + b)
              .orElse("");
          if (null != temp_18_0042)
            temp_18_0042.freeRef();
          return temp_18_0041;
        }, network_target == null ? null : network_target.addRef()));
    Result[] array = ConstantResult.batchResultArray(Tensor.addRefs(input_target));
    Result eval = network_target.eval(Result.addRefs(array));
    if (null != array)
      ReferenceCounting.freeRefs(array);
    network_target.freeRef();
    TensorList result = eval.getData();
    if (null != eval)
      eval.freeRef();
    final Tensor[] output_target = result.stream().toArray(i -> new Tensor[i]);
    if (null != result)
      result.freeRef();
    if (output_target.length != input_target.length) {
      logger.info("Batch layers not supported");
      if (null != input_target)
        ReferenceCounting.freeRefs(input_target);
      if (null != output_target)
        ReferenceCounting.freeRefs(output_target);
      component.freeRef();
      return null;
    }
    Tensor[][] trainingInput = append(Tensor.addRefs(input_target),
        Tensor.addRefs(output_target));
    if (null != output_target)
      ReferenceCounting.freeRefs(output_target);
    if (null != input_target)
      ReferenceCounting.freeRefs(input_target);
    TrainingTester.TestResult temp_18_0013 = trainAll("Model Convergence", log,
        Tensor.addRefs(trainingInput), shuffle(random, component.copy()));
    component.freeRef();
    if (null != trainingInput)
      ReferenceCounting.freeRefs(trainingInput);
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
    {
      log.h3("Gradient Descent");
      final RefList<StepRecord> gd = train(log, this::trainGD, layer.copy(),
          copy(Tensor.addRefs(trainingInput)), mask);
      log.h3("Conjugate Gradient Descent");
      final RefList<StepRecord> cjgd = train(log, this::trainCjGD, layer.copy(),
          copy(Tensor.addRefs(trainingInput)), mask);
      log.h3("Limited-Memory BFGS");
      final RefList<StepRecord> lbfgs = train(log, this::trainLBFGS, layer.copy(),
          copy(Tensor.addRefs(trainingInput)), mask);
      @Nonnull final ProblemRun[] runs = {
          new ProblemRun("GD", gd == null ? null : gd.addRef(), Color.GRAY, ProblemRun.PlotType.Line),
          new ProblemRun("CjGD", cjgd == null ? null : cjgd.addRef(), Color.CYAN, ProblemRun.PlotType.Line),
          new ProblemRun("LBFGS", lbfgs == null ? null : lbfgs.addRef(), Color.GREEN, ProblemRun.PlotType.Line)};
      @Nonnull
      ProblemResult result = new ProblemResult();
      result.put("GD",
          new TrainingResult(getResultType(gd == null ? null : gd.addRef()), min(gd == null ? null : gd.addRef())));
      if (null != gd)
        gd.freeRef();
      result.put("CjGD", new TrainingResult(getResultType(cjgd == null ? null : cjgd.addRef()),
          min(cjgd == null ? null : cjgd.addRef())));
      if (null != cjgd)
        cjgd.freeRef();
      result.put("LBFGS", new TrainingResult(getResultType(lbfgs == null ? null : lbfgs.addRef()),
          min(lbfgs == null ? null : lbfgs.addRef())));
      if (null != lbfgs)
        lbfgs.freeRef();
      if (verbose) {
        final PlotCanvas iterPlot = log.eval(() -> {
          return TestUtil.compare(title + " vs Iteration", runs);
        });
        final PlotCanvas timePlot = log.eval(() -> {
          return TestUtil.compareTime(title + " vs Time", runs);
        });
        ReferenceCounting.freeRefs(trainingInput);
        layer.freeRef();
        return new TestResult(iterPlot, timePlot, result);
      } else {
        @Nullable final PlotCanvas iterPlot = TestUtil.compare(title + " vs Iteration", runs);
        @Nullable final PlotCanvas timePlot = TestUtil.compareTime(title + " vs Time", runs);
        ReferenceCounting.freeRefs(trainingInput);
        layer.freeRef();
        return new TestResult(iterPlot, timePlot, result);
      }
    }
  }

  @Nonnull
  public RefList<StepRecord> trainCjGD(@Nonnull final NotebookOutput log, final Trainable trainable) {
    log.p(
        "First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.");
    @Nonnull final RefList<StepRecord> history = new RefArrayList<>();
    @Nonnull final TrainingMonitor monitor = TrainingTester.getMonitor(history == null ? null : history.addRef());
    try {
      log.eval(RefUtil
          .wrapInterface((UncheckedSupplier<Double>) () -> {
            IterativeTrainer temp_18_0028 = new IterativeTrainer(
                trainable == null ? null : trainable.addRef());
            IterativeTrainer temp_18_0043 = temp_18_0028
                .setLineSearchFactory(label -> new QuadraticSearch());
            IterativeTrainer temp_18_0044 = temp_18_0043
                .setOrientation(new GradientDescent());
            IterativeTrainer temp_18_0045 = temp_18_0044.setMonitor(monitor);
            IterativeTrainer temp_18_0046 = temp_18_0045.setTimeout(30, TimeUnit.SECONDS);
            IterativeTrainer temp_18_0047 = temp_18_0046.setMaxIterations(250);
            IterativeTrainer temp_18_0048 = temp_18_0047.setTerminateThreshold(0);
            double temp_18_0027 = temp_18_0048.run();
            if (null != temp_18_0048)
              temp_18_0048.freeRef();
            if (null != temp_18_0047)
              temp_18_0047.freeRef();
            if (null != temp_18_0046)
              temp_18_0046.freeRef();
            if (null != temp_18_0045)
              temp_18_0045.freeRef();
            if (null != temp_18_0044)
              temp_18_0044.freeRef();
            if (null != temp_18_0043)
              temp_18_0043.freeRef();
            if (null != temp_18_0028)
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
  public RefList<StepRecord> trainGD(@Nonnull final NotebookOutput log, final Trainable trainable) {
    log.p("First, we train using basic gradient descent method apply weak line search conditions.");
    @Nonnull final RefList<StepRecord> history = new RefArrayList<>();
    @Nonnull final TrainingMonitor monitor = TrainingTester.getMonitor(history == null ? null : history.addRef());
    try {
      log.eval(RefUtil
          .wrapInterface((UncheckedSupplier<Double>) () -> {
            IterativeTrainer temp_18_0030 = new IterativeTrainer(
                trainable == null ? null : trainable.addRef());
            IterativeTrainer temp_18_0049 = temp_18_0030
                .setLineSearchFactory(label -> new ArmijoWolfeSearch());
            IterativeTrainer temp_18_0050 = temp_18_0049
                .setOrientation(new GradientDescent());
            IterativeTrainer temp_18_0051 = temp_18_0050.setMonitor(monitor);
            IterativeTrainer temp_18_0052 = temp_18_0051.setTimeout(30, TimeUnit.SECONDS);
            IterativeTrainer temp_18_0053 = temp_18_0052.setMaxIterations(250);
            IterativeTrainer temp_18_0054 = temp_18_0053.setTerminateThreshold(0);
            double temp_18_0029 = temp_18_0054.run();
            if (null != temp_18_0054)
              temp_18_0054.freeRef();
            if (null != temp_18_0053)
              temp_18_0053.freeRef();
            if (null != temp_18_0052)
              temp_18_0052.freeRef();
            if (null != temp_18_0051)
              temp_18_0051.freeRef();
            if (null != temp_18_0050)
              temp_18_0050.freeRef();
            if (null != temp_18_0049)
              temp_18_0049.freeRef();
            if (null != temp_18_0030)
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
  public RefList<StepRecord> trainLBFGS(@Nonnull final NotebookOutput log, final Trainable trainable) {
    log.p(
        "Next, we apply the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.");
    @Nonnull final RefList<StepRecord> history = new RefArrayList<>();
    @Nonnull final TrainingMonitor monitor = TrainingTester.getMonitor(history == null ? null : history.addRef());
    try {
      log.eval(RefUtil
          .wrapInterface((UncheckedSupplier<Double>) () -> {
            IterativeTrainer temp_18_0032 = new IterativeTrainer(
                trainable == null ? null : trainable.addRef());
            IterativeTrainer temp_18_0055 = temp_18_0032
                .setLineSearchFactory(label -> new ArmijoWolfeSearch());
            IterativeTrainer temp_18_0056 = temp_18_0055.setOrientation(new LBFGS());
            IterativeTrainer temp_18_0057 = temp_18_0056.setMonitor(monitor);
            IterativeTrainer temp_18_0058 = temp_18_0057.setTimeout(30, TimeUnit.SECONDS);
            IterativeTrainer temp_18_0059 = temp_18_0058.setIterationsPerSample(100);
            IterativeTrainer temp_18_0060 = temp_18_0059.setMaxIterations(250);
            IterativeTrainer temp_18_0061 = temp_18_0060.setTerminateThreshold(0);
            double temp_18_0031 = temp_18_0061.run();
            if (null != temp_18_0061)
              temp_18_0061.freeRef();
            if (null != temp_18_0060)
              temp_18_0060.freeRef();
            if (null != temp_18_0059)
              temp_18_0059.freeRef();
            if (null != temp_18_0058)
              temp_18_0058.freeRef();
            if (null != temp_18_0057)
              temp_18_0057.freeRef();
            if (null != temp_18_0056)
              temp_18_0056.freeRef();
            if (null != temp_18_0055)
              temp_18_0055.freeRef();
            if (null != temp_18_0032)
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
  }

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
    temp_18_0062.forEach(buffer -> {
      randomizationMode.shuffle(random, buffer);
    });
    if (null != temp_18_0062)
      temp_18_0062.freeRef();
    return testComponent;
  }

  private Tensor[][] shuffleCopy(final Random random, @Nonnull final Tensor... copy) {
    Tensor[][] temp_18_0026 = RefIntStream.range(0, getBatches())
        .mapToObj(RefUtil
            .wrapInterface((IntFunction<? extends Tensor[]>) i -> {
              return RefArrays.stream(Tensor.addRefs(copy)).map(tensor -> {
                @Nonnull final Tensor cpy = tensor.copy();
                if (null != tensor)
                  tensor.freeRef();
                randomizationMode.shuffle(random, cpy.getData());
                return cpy;
              }).toArray(j -> new Tensor[j]);
            }, Tensor.addRefs(copy)))
        .toArray(i -> new Tensor[i][]);
    ReferenceCounting.freeRefs(copy);
    return temp_18_0026;
  }

  private RefList<StepRecord> train(@Nonnull NotebookOutput log,
                                    @Nonnull BiFunction<NotebookOutput, Trainable, RefList<StepRecord>> opt, @Nonnull Layer layer,
                                    @Nonnull Tensor[][] data, @Nonnull boolean... mask) {
    {
      int inputs = data[0].length;
      @Nonnull final PipelineNetwork network = new PipelineNetwork(inputs);
      Layer lossLayer = lossLayer();
      assert null != lossLayer : getClass().toString();
      RefUtil
          .freeRef(network.add(lossLayer == null ? null : lossLayer.addRef(),
              network.add(layer == null ? null : layer.addRef(),
                  RefIntStream.range(0, inputs - 1).mapToObj(RefUtil.wrapInterface(
                      (IntFunction<? extends DAGNode>) i -> network
                          .getInput(i),
                      network == null ? null : network.addRef())).toArray(i -> new DAGNode[i])),
              network.getInput(inputs - 1)));
      if (null != lossLayer)
        lossLayer.freeRef();
      @Nonnull
      ArrayTrainable trainable = new ArrayTrainable(Tensor.addRefs(data),
          network == null ? null : network.addRef());
      if (0 < mask.length)
        trainable.setMask(mask);
      RefList<StepRecord> history;
      {
        history = opt.apply(log, trainable);
        if (history.stream().mapToDouble(x -> x.fitness).min().orElse(1) > 1e-5) {
          if (!network.isFrozen()) {
            log.p("This training apply resulted in the following configuration:");
            log.eval(RefUtil
                .wrapInterface((UncheckedSupplier<String>) () -> {
                  RefList<double[]> temp_18_0064 = network.state();
                  String temp_18_0063 = temp_18_0064.stream().map(RefArrays::toString)
                      .reduce((a, b) -> a + "\n" + b).orElse("");
                  if (null != temp_18_0064)
                    temp_18_0064.freeRef();
                  return temp_18_0063;
                }, network == null ? null : network.addRef()));
          }
          if (0 < mask.length) {
            log.p("And regressed input:");
            log.eval(RefUtil
                .wrapInterface((UncheckedSupplier<String>) () -> {
                  return RefArrays.stream(Tensor.addRefs(data)).flatMap(x -> {
                    RefStream<Tensor> temp_18_0014 = RefArrays
                        .stream(Tensor.addRefs(x));
                    if (null != x)
                      ReferenceCounting.freeRefs(x);
                    return temp_18_0014;
                  }).limit(1).map(x -> {
                    String temp_18_0015 = x.prettyPrint();
                    if (null != x)
                      x.freeRef();
                    return temp_18_0015;
                  }).reduce((a, b) -> a + "\n" + b).orElse("");
                }, Tensor.addRefs(data)));
          }
          log.p("To produce the following output:");
          log.eval(RefUtil
              .wrapInterface((UncheckedSupplier<String>) () -> {
                Result[] array = ConstantResult
                    .batchResultArray(pop(Tensor.addRefs(data)));
                @Nullable
                Result eval = layer.eval(Result.addRefs(array));
                for (@Nonnull
                    Result result : array) {
                  RefUtil.freeRef(result.getData());
                }
                if (null != array)
                  ReferenceCounting.freeRefs(array);
                TensorList tensorList = eval.getData();
                if (null != eval)
                  eval.freeRef();
                String temp_18_0016 = tensorList.stream().limit(1).map(x -> {
                  String temp_18_0017 = x.prettyPrint();
                  if (null != x)
                    x.freeRef();
                  return temp_18_0017;
                }).reduce((a, b) -> a + "\n" + b).orElse("");
                if (null != tensorList)
                  tensorList.freeRef();
                return temp_18_0016;
              }, Tensor.addRefs(data), layer == null ? null : layer.addRef()));
        } else {
          log.p("Training Converged");
        }
      }
      trainable.freeRef();
      network.freeRef();
      layer.freeRef();
      ReferenceCounting.freeRefs(data);
      return history;
    }
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

  public static @RefAware
  class ComponentResult {
    final ProblemResult complete;
    final ProblemResult input;
    final ProblemResult model;

    public ComponentResult(final ProblemResult input, final ProblemResult model, final ProblemResult complete) {
      this.input = input;
      this.model = model;
      this.complete = complete;
    }

    @Override
    public String toString() {
      return String.format("{\"input\":%s, \"model\":%s, \"complete\":%s}", input, model, complete);
    }
  }

  public static @RefAware
  class TestResult {
    final PlotCanvas iterPlot;
    final PlotCanvas timePlot;
    final ProblemResult value;

    public TestResult(final PlotCanvas iterPlot, final PlotCanvas timePlot, final ProblemResult value) {
      this.timePlot = timePlot;
      this.iterPlot = iterPlot;
      this.value = value;
    }
  }

  public static @RefAware
  final class TrainingResult {
    final ResultType type;
    final double value;

    public TrainingResult(final ResultType type, final double value) {
      this.type = type;
      this.value = value;
    }

    @Override
    public String toString() {
      return String.format("{\"type\":\"%s\", value:%s}", type, value);
    }
  }

  public static @RefAware
  class ProblemResult {
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
      return map.entrySet().stream().map(e -> {
        String temp_18_0018 = String.format("\"%s\": %s", e.getKey(), e.getValue().toString());
        if (null != e)
          RefUtil.freeRef(e);
        return temp_18_0018;
      }).reduce((a, b) -> a + ", " + b).get();
    }
  }
}
