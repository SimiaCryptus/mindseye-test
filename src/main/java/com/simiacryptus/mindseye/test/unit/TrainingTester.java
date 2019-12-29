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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import smile.plot.PlotCanvas;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.swing.*;
import java.awt.*;
import java.util.*;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.function.BiFunction;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

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

  public TrainingTester setBatches(final int batches) {
    this.batches = batches;
    return this;
  }

  public RandomizationMode getRandomizationMode() {
    return randomizationMode;
  }

  @Nonnull
  public TrainingTester setRandomizationMode(final RandomizationMode randomizationMode) {
    this.randomizationMode = randomizationMode;
    return this;
  }

  public boolean isThrowExceptions() {
    return throwExceptions;
  }

  @Nonnull
  public TrainingTester setThrowExceptions(boolean throwExceptions) {
    this.throwExceptions = throwExceptions;
    return this;
  }

  public boolean isVerbose() {
    return verbose;
  }

  @Nonnull
  public TrainingTester setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }

  public static TrainingMonitor getMonitor(@Nonnull final List<StepRecord> history) {
    return new TrainingMonitor() {
      @Override
      public void log(final String msg) {
        logger.info(msg);
      }

      @Override
      public void onStepComplete(@Nonnull final Step currentPoint) {
        history.add(new StepRecord(currentPoint.point.getMean(), currentPoint.time, currentPoint.iteration));
      }
    };
  }

  public static Tensor[][] append(@Nonnull Tensor[][] left, Tensor[] right) {
    if (left.length != right.length)
      throw new IllegalArgumentException(left.length + "!=" + right.length);
    return IntStream.range(0, left.length)
        .mapToObj(i -> Stream.concat(Arrays.stream(left[i]), Stream.of(right[i])).toArray(j -> new Tensor[j]))
        .toArray(j -> new Tensor[j][]);
  }

  public static Tensor[][] copy(@Nonnull Tensor[][] input_gd) {
    return Arrays.stream(input_gd).map(t -> Arrays.stream(t).map(v -> v.copy()).toArray(i -> new Tensor[i]))
        .toArray(i -> new Tensor[i][]);
  }

  public static Tensor[][] pop(@Nonnull Tensor[][] data) {
    return Arrays.stream(data).map(t -> Arrays.stream(t).limit(t.length - 1).toArray(i -> new Tensor[i]))
        .toArray(i -> new Tensor[i][]);
  }

  @Nonnull
  public ResultType getResultType(@Nonnull final List<StepRecord> lbfgsmin) {
    return Math.abs(min(lbfgsmin)) < 1e-9 ? ResultType.Converged : ResultType.NonConverged;
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

  public boolean isZero(@Nonnull final DoubleStream stream) {
    return isZero(stream, 1e-14);
  }

  public boolean isZero(@Nonnull final DoubleStream stream, double zeroTol) {
    final double[] array = stream.toArray();
    if (array.length == 0)
      return false;
    return Arrays.stream(array).map(x -> Math.abs(x)).sum() < zeroTol;
  }

  @Override
  public ComponentResult test(@Nonnull final NotebookOutput log, @Nonnull final Layer component,
                              @Nonnull final Tensor... inputPrototype) {
    printHeader(log);
    final boolean testModel = !component.state().isEmpty();
    if (testModel && isZero(component.state().stream().flatMapToDouble(x1 -> Arrays.stream(x1)))) {
      throw new AssertionError("Weights are all zero?");
    }
    if (isZero(Arrays.stream(inputPrototype).flatMapToDouble(x -> Arrays.stream(x.getData())))) {
      throw new AssertionError("Inputs are all zero?");
    }
    @Nonnull final Random random = new Random();
    final boolean testInput = Arrays.stream(inputPrototype).anyMatch(x -> x.length() > 0);
    @Nullable
    TestResult inputLearning;
    if (testInput) {
      log.h2("Input Learning");
      inputLearning = testInputLearning(log, component, random, inputPrototype);
    } else {
      inputLearning = null;
    }
    @Nullable
    TestResult modelLearning;
    if (testModel) {
      log.h2("Model Learning");
      modelLearning = testModelLearning(log, component, random, inputPrototype);
    } else {
      modelLearning = null;
    }
    @Nullable
    TestResult completeLearning;
    if (testInput && testModel) {
      log.h2("Composite Learning");
      completeLearning = testCompleteLearning(log, component, random, inputPrototype);
    } else {
      completeLearning = null;
    }
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
    @Nonnull final Layer network_target = shuffle(random, component.copy()).freeze();
    final Tensor[][] input_target = shuffleCopy(random, inputPrototype);
    log.p(
        "In this apply, attempt to train a network to emulate a randomized network given an example input/output. The target state is:");
    log.eval(() -> {
      return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
    });
    log.p("We simultaneously regress this target input:");
    log.eval(() -> {
      return Arrays.stream(input_target).flatMap(x -> Arrays.stream(x)).map(x -> x.prettyPrint())
          .reduce((a, b) -> a + "\n" + b).orElse("");
    });
    log.p("Which produces the following output:");
    Result[] inputs = ConstantResult.batchResultArray(input_target);
    TensorList result = network_target.eval(inputs).getData();
    final Tensor[] output_target = result.stream().toArray(i -> new Tensor[i]);
    log.eval(() -> {
      return Stream.of(output_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
    });
    //if (output_target.length != inputPrototype.length) return null;
    Tensor[][] trainingInput = append(shuffleCopy(random, inputPrototype), output_target);
    return trainAll("Integrated Convergence", log, trainingInput,
        shuffle(random, component.copy()), buildMask(inputPrototype.length));
  }

  public TestResult testInputLearning(@Nonnull final NotebookOutput log, @Nonnull final Layer component,
                                      final Random random, @Nonnull final Tensor[] inputPrototype) {
    @Nonnull final Layer network = shuffle(random, component.copy()).freeze();
    final Tensor[][] input_target = shuffleCopy(random, inputPrototype);
    log.p("In this apply, we use a network to learn this target input, given it's pre-evaluated output:");
    log.eval(() -> {
      return Arrays.stream(input_target).flatMap(x -> Arrays.stream(x)).map(x -> x.prettyPrint())
          .reduce((a, b) -> a + "\n" + b).orElse("");
    });
    Result[] array = ConstantResult.batchResultArray(input_target);
    @Nullable
    Result eval = network.eval(array);
    TensorList result = eval.getData();
    final Tensor[] output_target = result.stream().toArray(i -> new Tensor[i]);
    if (output_target.length != getBatches()) {
      logger.info(String.format("Meta layers not supported. %d != %d", output_target.length, getBatches()));
      return null;
    }

    for (@Nonnull
        Result nnResult : array) {
      nnResult.getData();
    }
    //if (output_target.length != inputPrototype.length) return null;
    Tensor[][] trainingInput = append(shuffleCopy(random, inputPrototype), output_target);
    return trainAll("Input Convergence", log, trainingInput, network, buildMask(inputPrototype.length));
  }

  public TestResult testModelLearning(@Nonnull final NotebookOutput log, @Nonnull final Layer component,
                                      final Random random, final Tensor[] inputPrototype) {
    @Nonnull final Layer network_target = shuffle(random, component.copy()).freeze();
    final Tensor[][] input_target = shuffleCopy(random, inputPrototype);
    log.p(
        "In this apply, attempt to train a network to emulate a randomized network given an example input/output. The target state is:");
    log.eval(() -> {
      return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
    });
    Result[] array = ConstantResult.batchResultArray(input_target);
    Result eval = network_target.eval(array);
    TensorList result = eval.getData();
    final Tensor[] output_target = result.stream().toArray(i -> new Tensor[i]);
    if (output_target.length != input_target.length) {
      logger.info("Batch layers not supported");
      return null;
    }
    Tensor[][] trainingInput = append(input_target, output_target);
    return trainAll("Model Convergence", log, trainingInput, shuffle(random, component.copy()));
  }

  public double min(@Nonnull List<StepRecord> history) {
    return history.stream().mapToDouble(x -> x.fitness).min().orElse(Double.NaN);
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
      final List<StepRecord> gd = train(log, this::trainGD, layer.copy(), copy(trainingInput), mask);
      log.h3("Conjugate Gradient Descent");
      final List<StepRecord> cjgd = train(log, this::trainCjGD, layer.copy(), copy(trainingInput), mask);
      log.h3("Limited-Memory BFGS");
      final List<StepRecord> lbfgs = train(log, this::trainLBFGS, layer.copy(), copy(trainingInput), mask);
      @Nonnull final ProblemRun[] runs = {new ProblemRun("GD", gd, Color.GRAY, ProblemRun.PlotType.Line),
          new ProblemRun("CjGD", cjgd, Color.CYAN, ProblemRun.PlotType.Line),
          new ProblemRun("LBFGS", lbfgs, Color.GREEN, ProblemRun.PlotType.Line)};
      @Nonnull
      ProblemResult result = new ProblemResult();
      result.put("GD", new TrainingResult(getResultType(gd), min(gd)));
      result.put("CjGD", new TrainingResult(getResultType(cjgd), min(cjgd)));
      result.put("LBFGS", new TrainingResult(getResultType(lbfgs), min(lbfgs)));
      if (verbose) {
        final PlotCanvas iterPlot = log.eval(() -> {
          return TestUtil.compare(title + " vs Iteration", runs);
        });
        final PlotCanvas timePlot = log.eval(() -> {
          return TestUtil.compareTime(title + " vs Time", runs);
        });
        return new TestResult(iterPlot, timePlot, result);
      } else {
        @Nullable final PlotCanvas iterPlot = TestUtil.compare(title + " vs Iteration", runs);
        @Nullable final PlotCanvas timePlot = TestUtil.compareTime(title + " vs Time", runs);
        return new TestResult(iterPlot, timePlot, result);
      }
    }
  }

  @Nonnull
  public List<StepRecord> trainCjGD(@Nonnull final NotebookOutput log, final Trainable trainable) {
    log.p(
        "First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.");
    @Nonnull final List<StepRecord> history = new ArrayList<>();
    @Nonnull final TrainingMonitor monitor = TrainingTester.getMonitor(history);
    try {
      log.eval(() -> {
        return new IterativeTrainer(trainable).setLineSearchFactory(label -> new QuadraticSearch())
            .setOrientation(new GradientDescent()).setMonitor(monitor).setTimeout(30, TimeUnit.SECONDS)
            .setMaxIterations(250).setTerminateThreshold(0).run();
      });
    } catch (Throwable e) {
      if (isThrowExceptions())
        throw new RuntimeException(e);
    }
    return history;
  }

  @Nonnull
  public List<StepRecord> trainGD(@Nonnull final NotebookOutput log, final Trainable trainable) {
    log.p("First, we train using basic gradient descent method apply weak line search conditions.");
    @Nonnull final List<StepRecord> history = new ArrayList<>();
    @Nonnull final TrainingMonitor monitor = TrainingTester.getMonitor(history);
    try {
      log.eval(() -> {
        return new IterativeTrainer(trainable).setLineSearchFactory(label -> new ArmijoWolfeSearch())
            .setOrientation(new GradientDescent()).setMonitor(monitor).setTimeout(30, TimeUnit.SECONDS)
            .setMaxIterations(250).setTerminateThreshold(0).run();
      });
    } catch (Throwable e) {
      if (isThrowExceptions())
        throw new RuntimeException(e);
    }
    return history;
  }

  @Nonnull
  public List<StepRecord> trainLBFGS(@Nonnull final NotebookOutput log, final Trainable trainable) {
    log.p(
        "Next, we apply the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.");
    @Nonnull final List<StepRecord> history = new ArrayList<>();
    @Nonnull final TrainingMonitor monitor = TrainingTester.getMonitor(history);
    try {
      log.eval(() -> {
        return new IterativeTrainer(trainable).setLineSearchFactory(label -> new ArmijoWolfeSearch())
            .setOrientation(new LBFGS()).setMonitor(monitor).setTimeout(30, TimeUnit.SECONDS)
            .setIterationsPerSample(100).setMaxIterations(250).setTerminateThreshold(0).run();
      });
    } catch (Throwable e) {
      if (isThrowExceptions())
        throw new RuntimeException(e);
    }
    return history;
  }

  @Nonnull
  @Override
  public String toString() {
    return "TrainingTester{" + "batches=" + batches + ", randomizationMode=" + randomizationMode + ", verbose="
        + verbose + ", throwExceptions=" + throwExceptions + '}';
  }

  protected void printHeader(@Nonnull NotebookOutput log) {
    log.h1("Training Characteristics");
  }

  protected abstract Layer lossLayer();

  @Nonnull
  private Layer shuffle(final Random random, @Nonnull final Layer testComponent) {
    testComponent.state().forEach(buffer -> {
      randomizationMode.shuffle(random, buffer);
    });
    return testComponent;
  }

  private Tensor[][] shuffleCopy(final Random random, @Nonnull final Tensor... copy) {
    return IntStream.range(0, getBatches()).mapToObj(i -> {
      return Arrays.stream(copy).map(tensor -> {
        @Nonnull final Tensor cpy = tensor.copy();
        randomizationMode.shuffle(random, cpy.getData());
        return cpy;
      }).toArray(j -> new Tensor[j]);
    }).toArray(i -> new Tensor[i][]);
  }

  private List<StepRecord> train(@Nonnull NotebookOutput log,
                                 @Nonnull BiFunction<NotebookOutput, Trainable, List<StepRecord>> opt, @Nonnull Layer layer,
                                 @Nonnull Tensor[][] data, @Nonnull boolean... mask) {
    {
      int inputs = data[0].length;
      @Nonnull final PipelineNetwork network = new PipelineNetwork(inputs);
      Layer lossLayer = lossLayer();
      assert null != lossLayer : getClass().toString();
      network.add(lossLayer, network.add(layer,
                IntStream.range(0, inputs - 1).mapToObj(i -> network.getInput(i)).toArray(i -> new DAGNode[i])), network.getInput(inputs - 1));
      @Nonnull
      ArrayTrainable trainable = new ArrayTrainable(data, network);
      if (0 < mask.length)
        trainable.setMask(mask);
      List<StepRecord> history;
      {
        history = opt.apply(log, trainable);
        if (history.stream().mapToDouble(x -> x.fitness).min().orElse(1) > 1e-5) {
          if (!network.isFrozen()) {
            log.p("This training apply resulted in the following configuration:");
            log.eval(() -> {
              return network.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
            });
          }
          if (0 < mask.length) {
            log.p("And regressed input:");
            log.eval(() -> {
              return Arrays.stream(data).flatMap(x -> Arrays.stream(x)).limit(1).map(x -> x.prettyPrint())
                  .reduce((a, b) -> a + "\n" + b).orElse("");
            });
          }
          log.p("To produce the following output:");
          log.eval(() -> {
            Result[] array = ConstantResult.batchResultArray(pop(data));
            @Nullable
            Result eval = layer.eval(array);
            for (@Nonnull
                Result result : array) {
              result.getData();
            }
            TensorList tensorList = eval.getData();
            return tensorList.stream().limit(1).map(x -> {
              return x.prettyPrint();
            }).reduce((a, b) -> a + "\n" + b).orElse("");
          });
        } else {
          log.p("Training Converged");
        }
      }
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

  public static class ComponentResult {
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

  public static class TrainingResult {
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

  public static class ProblemResult {
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
        return String.format("\"%s\": %s", e.getKey(), e.getValue().toString());
      }).reduce((a, b) -> a + ", " + b).get();
    }
  }
}
