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
import com.google.gson.JsonObject;
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
import com.simiacryptus.ref.lang.RefIgnore;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.Util;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import smile.plot.swing.PlotCanvas;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.swing.*;
import java.awt.*;
import java.util.List;
import java.util.*;
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
  public static TrainingMonitor getMonitor(@Nonnull final List<StepRecord> history) {
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
  }

  @Nonnull
  public static Tensor[][] append(@Nonnull Tensor[][] left, @Nonnull Tensor[] right) {
    if (left.length != right.length) {
      IllegalArgumentException temp_18_0021 = new IllegalArgumentException(left.length + "!=" + right.length);
      RefUtil.freeRef(left);
      RefUtil.freeRef(right);
      throw temp_18_0021;
    }
    return RefIntStream.range(0, left.length)
        .mapToObj(RefUtil.wrapInterface(i -> RefStream
            .concat(RefArrays.stream(RefUtil.addRef(left[i])), RefStream.of(right[i].addRef()))
            .toArray(Tensor[]::new), right, left))
        .toArray(Tensor[][]::new);
  }

  @Nonnull
  public static Tensor[][] copy(@Nonnull Tensor[][] input_gd) {
    return RefArrays.stream(input_gd).map(t -> {
      return RefArrays.stream(t).map(v -> {
        Tensor temp_18_0002 = v.copy();
        v.freeRef();
        return temp_18_0002;
      }).toArray(Tensor[]::new);
    }).toArray(Tensor[][]::new);
  }

  @Nonnull
  public static Tensor[][] pop(@Nonnull Tensor[][] data) {
    return RefArrays.stream(data).map(t -> {
      return RefArrays.stream(t).limit(t.length - 1).toArray(Tensor[]::new);
    }).toArray(Tensor[][]::new);
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
    return RefArrays.stream(array).map(Math::abs).sum() < zeroTol;
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
    if (testModel && isZero(temp_18_0034.stream().flatMapToDouble(RefArrays::stream))) {
      component.freeRef();
      RefUtil.freeRef(inputPrototype);
      temp_18_0034.freeRef();
      throw new AssertionError("Weights are all zero?");
    }
    temp_18_0034.freeRef();
    if (isZero(RefArrays.stream(RefUtil.addRef(inputPrototype)).flatMapToDouble(tensor -> {
      RefDoubleStream doubleStream = tensor.doubleStream();
      tensor.freeRef();
      return doubleStream;
    }))) {
      component.freeRef();
      RefUtil.freeRef(inputPrototype);
      throw new AssertionError("Inputs are all zero?");
    }
    @Nonnull final Random random = new Random();
    final boolean testInput = RefArrays.stream(RefUtil.addRef(inputPrototype)).anyMatch(x -> {
      boolean temp_18_0005 = x.length() > 0;
      x.freeRef();
      return temp_18_0005;
    });
    @Nullable
    TestResult inputLearning;
    if (testInput) {
      log.h2("Input Learning");
      inputLearning = testInputLearning(log, component.addRef(), random,
          RefUtil.addRef(inputPrototype));
    } else {
      inputLearning = null;
    }
    @Nullable
    TestResult modelLearning;
    if (testModel) {
      log.h2("Model Learning");
      modelLearning = testModelLearning(log, component.addRef(), random,
          RefUtil.addRef(inputPrototype));
    } else {
      modelLearning = null;
    }
    @Nullable
    TestResult completeLearning;
    if (testInput && testModel) {
      log.h2("Composite Learning");
      completeLearning = testCompleteLearning(log, component.addRef(), random,
          RefUtil.addRef(inputPrototype));
    } else {
      completeLearning = null;
    }
    RefUtil.freeRef(inputPrototype);
    component.freeRef();
    log.h2("Results");
    log.eval(() -> {
      return grid(inputLearning, modelLearning, completeLearning);
    });
    ComponentResult result = log.eval(() -> {
      return new ComponentResult(null == inputLearning ? null : inputLearning.value,
          null == modelLearning ? null : modelLearning.value, null == completeLearning ? null : completeLearning.value);
    });
    log.setMetadata("training_analysis", new GsonBuilder().create().fromJson(result.toString(), JsonObject.class));
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
    final Tensor[][] input_target = shuffleCopy(random, RefUtil.addRef(inputPrototype));
    log.p(
        "In this apply, attempt to train a network to emulate a randomized network given an example input/output. The target state is:");
    log.eval(RefUtil.wrapInterface((UncheckedSupplier<String>) () -> {
      RefList<double[]> temp_18_0037 = temp_18_0035.state();
      assert temp_18_0037 != null;
      String temp_18_0036 = temp_18_0037.stream().map(RefArrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
      temp_18_0037.freeRef();
      return temp_18_0036;
    }, temp_18_0035.addRef()));
    log.p("We simultaneously regress this target input:");
    log.eval(RefUtil.wrapInterface((UncheckedSupplier<String>) () -> {
      return RefArrays.stream(RefUtil.addRef(input_target)).flatMap(x -> {
        RefStream<Tensor> temp_18_0006 = RefArrays.stream(RefUtil.addRef(x));
        if (null != x)
          RefUtil.freeRef(x);
        return temp_18_0006;
      }).map(x -> {
        String temp_18_0007 = x.prettyPrint();
        x.freeRef();
        return temp_18_0007;
      }).reduce((a, b) -> a + "\n" + b).orElse("");
    }, RefUtil.addRef(input_target)));
    log.p("Which produces the following output:");
    Result[] inputs = ConstantResult.batchResultArray(RefUtil.addRef(input_target));
    RefUtil.freeRef(input_target);
    Result temp_18_0038 = temp_18_0035.eval(inputs);
    assert temp_18_0038 != null;
    TensorList result = Result.getData(temp_18_0038);
    temp_18_0035.freeRef();
    final Tensor[] output_target = result.stream().toArray(Tensor[]::new);
    result.freeRef();
    log.eval(RefUtil.wrapInterface((UncheckedSupplier<String>) () -> {
      return RefStream.of(RefUtil.addRef(output_target)).map(x -> {
        String temp_18_0008 = x.prettyPrint();
        x.freeRef();
        return temp_18_0008;
      }).reduce((a, b) -> a + "\n" + b).orElse("");
    }, RefUtil.addRef(output_target)));
    //if (output_target.length != inputPrototype.length) return null;
    int length = inputPrototype.length;
    Tensor[][] trainingInput = append(shuffleCopy(random, inputPrototype), output_target);
    TrainingTester.TestResult temp_18_0009 = trainAll("Integrated Convergence", log, trainingInput,
        shuffle(random, component.copy()), buildMask(length));
    component.freeRef();
    return temp_18_0009;
  }

  @Nullable
  public TestResult testInputLearning(@Nonnull final NotebookOutput log, @Nonnull final Layer component,
                                      final Random random, @Nonnull final Tensor[] inputPrototype) {
    Layer network = shuffle(random, component.copy());
    network.freeze();
    component.freeRef();
    final Tensor[][] input_target = shuffleCopy(random, RefUtil.addRef(inputPrototype));
    log.p("In this apply, we use a network to learn this target input, given it's pre-evaluated output:");
    log.eval(RefUtil.wrapInterface((UncheckedSupplier<String>) () -> {
      return RefArrays.stream(RefUtil.addRef(input_target)).flatMap(RefArrays::stream).map(x -> {
        try {
          return x.prettyPrint();
        } finally {
          x.freeRef();
        }
      }).reduce((a, b) -> a + "\n" + b).orElse("");
    }, RefUtil.addRef(input_target)));
    Result eval = network.eval(ConstantResult.batchResultArray(input_target));
    TensorList result = Result.getData(eval);
    int resultLength = result.length();
    if (resultLength != getBatches()) {
      logger.info(RefString.format("Meta layers not supported. %d != %d", resultLength, getBatches()));
      network.freeRef();
      RefUtil.freeRef(inputPrototype);
      result.freeRef();
      return null;
    }
    final Tensor[] output_target = result.stream().toArray(Tensor[]::new);
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
    Layer network_target = shuffle(random, component.copy());
    network_target.freeze();
    final Tensor[][] input_target = shuffleCopy(random, inputPrototype);
    log.p(
        "In this apply, attempt to train a network to emulate a randomized network given an example input/output. The target state is:");
    log.eval(RefUtil.wrapInterface((UncheckedSupplier<String>) () -> {
      RefList<double[]> temp_18_0042 = network_target.state();
      assert temp_18_0042 != null;
      String temp_18_0041 = temp_18_0042.stream().map(RefArrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
      temp_18_0042.freeRef();
      return temp_18_0041;
    }, network_target.addRef()));
    Result[] array = ConstantResult.batchResultArray(RefUtil.addRef(input_target));
    Result eval = network_target.eval(array);
    network_target.freeRef();
    assert eval != null;
    TensorList result = Result.getData(eval);
    final Tensor[] output_target = result.stream().toArray(Tensor[]::new);
    result.freeRef();
    if (output_target.length != input_target.length) {
      logger.info("Batch layers not supported");
      RefUtil.freeRef(input_target);
      RefUtil.freeRef(output_target);
      component.freeRef();
      return null;
    }
    Tensor[][] trainingInput = append(input_target, output_target);
    Layer copy = component.copy();
    component.freeRef();
    return trainAll("Model Convergence", log, trainingInput, shuffle(random, copy));
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
    log.h3("Gradient Descent");
    final List<StepRecord> gd = train(log, this::trainGD, layer.copy(), copy(RefUtil.addRef(trainingInput)), mask);
    log.h3("Conjugate Gradient Descent");
    final List<StepRecord> cjgd = train(log, this::trainCjGD, layer.copy(), copy(RefUtil.addRef(trainingInput)), mask);
    log.h3("Limited-Memory BFGS");
    final List<StepRecord> lbfgs = train(log, this::trainLBFGS, layer.copy(), copy(RefUtil.addRef(trainingInput)), mask);
    RefUtil.freeRef(trainingInput);
    layer.freeRef();
    @Nonnull final ProblemRun[] runs = {
        new ProblemRun("GD", gd, Color.GRAY, ProblemRun.PlotType.Line),
        new ProblemRun("CjGD", cjgd, Color.CYAN, ProblemRun.PlotType.Line),
        new ProblemRun("LBFGS", lbfgs, Color.GREEN, ProblemRun.PlotType.Line)
    };
    @Nonnull
    ProblemResult result = new ProblemResult();
    result.put("GD", getResult(min(gd)));
    result.put("CjGD", getResult(min(cjgd)));
    result.put("LBFGS", getResult(min(lbfgs)));
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

  @Nonnull
  public List<StepRecord> trainCjGD(@Nonnull final NotebookOutput log, @Nullable final Trainable trainable) {
    log.p(
        "First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.");
    @Nonnull final List<StepRecord> history = new ArrayList<>();
    try {
      log.eval(() -> {
        IterativeTrainer iterativeTrainer = new IterativeTrainer(trainable.addRef());
        try {
          iterativeTrainer.setLineSearchFactory(label -> new QuadraticSearch());
          iterativeTrainer.setOrientation(new GradientDescent());
          iterativeTrainer.setMonitor(TrainingTester.getMonitor(history));
          iterativeTrainer.setTimeout(30, TimeUnit.SECONDS);
          iterativeTrainer.setMaxIterations(250);
          iterativeTrainer.setTerminateThreshold(0);
          return iterativeTrainer.run();
        } finally {
          iterativeTrainer.freeRef();
        }
      });
    } catch (Throwable e) {
      if (isThrowExceptions())
        throw Util.throwException(e);
    } finally {
      trainable.freeRef();
    }
    return history;
  }

  @Nonnull
  public List<StepRecord> trainGD(@Nonnull final NotebookOutput log, @Nullable final Trainable trainable) {
    log.p("First, we train using basic gradient descent method apply weak line search conditions.");
    @Nonnull final List<StepRecord> history = new ArrayList<>();
    try {
      log.eval(() -> {
        IterativeTrainer iterativeTrainer = new IterativeTrainer(trainable.addRef());
        try {
          iterativeTrainer.setLineSearchFactory(label -> new ArmijoWolfeSearch());
          iterativeTrainer.setOrientation(new GradientDescent());
          iterativeTrainer.setMonitor(TrainingTester.getMonitor(history));
          iterativeTrainer.setTimeout(30, TimeUnit.SECONDS);
          iterativeTrainer.setMaxIterations(250);
          iterativeTrainer.setTerminateThreshold(0);
          return iterativeTrainer.run();
        } finally {
          iterativeTrainer.freeRef();
        }
      });
    } catch (Throwable e) {
      if (isThrowExceptions())
        throw Util.throwException(e);
    } finally {
      trainable.freeRef();
    }
    return history;
  }

  @Nonnull
  public List<StepRecord> trainLBFGS(@Nonnull final NotebookOutput log, @Nullable final Trainable trainable) {
    log.p(
        "Next, we apply the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.");
    @Nonnull final List<StepRecord> history = new ArrayList<>();
    try {
      log.eval(() -> {
        IterativeTrainer iterativeTrainer = new IterativeTrainer(trainable.addRef());
        try {
          iterativeTrainer.setLineSearchFactory(label -> new ArmijoWolfeSearch());
          iterativeTrainer.setOrientation(new LBFGS());
          iterativeTrainer.setMonitor(TrainingTester.getMonitor(history));
          iterativeTrainer.setTimeout(30, TimeUnit.SECONDS);
          iterativeTrainer.setIterationsPerSample(100);
          iterativeTrainer.setMaxIterations(250);
          iterativeTrainer.setTerminateThreshold(0);
          return iterativeTrainer.run();
        } finally {
          iterativeTrainer.freeRef();
        }
      });
    } catch (Throwable e) {
      if (isThrowExceptions())
        throw Util.throwException(e);
    } finally {
      trainable.freeRef();
    }
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

  private TrainingTester.TrainingResult getResult(double min) {
    return new TrainingResult(Math.abs(min) < 1e-9
        ? ResultType.Converged
        : ResultType.NonConverged, min);
  }

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
    return RefIntStream.range(0, getBatches())
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor[]>) i -> {
          return RefArrays.stream(RefUtil.addRef(copy)).map(tensor -> {
            @Nonnull final Tensor cpy = tensor.copy();
            tensor.freeRef();
            randomizationMode.shuffle(random, cpy.getData());
            return cpy;
          }).toArray(Tensor[]::new);
        }, copy)).toArray(Tensor[][]::new);
  }

  private List<StepRecord> train(@Nonnull NotebookOutput log,
                                 @Nonnull RefBiFunction<NotebookOutput, Trainable, List<StepRecord>> opt,
                                 @Nonnull Layer layer,
                                 @Nonnull Tensor[][] data, @Nonnull boolean... mask) {
    int inputs = data[0].length;
    @Nonnull final PipelineNetwork network = new PipelineNetwork(inputs);
    Layer lossLayer = lossLayer();
    assert null != lossLayer : getClass().toString();
    RefUtil.freeRef(network.add(lossLayer,
        network.add(layer.addRef(),
            RefIntStream.range(0, inputs - 1)
                .mapToObj(index -> network.getInput(index))
                .toArray(DAGNode[]::new)),
        network.getInput(inputs - 1)));
    @Nonnull
    ArrayTrainable trainable = new ArrayTrainable(RefUtil.addRef(data), network.addRef());
    if (0 < mask.length)
      trainable.setMask(mask);
    List<StepRecord> history = runOpt(log, opt, trainable);
    if (history.stream().mapToDouble(x -> x.fitness).min().orElse(1) > 1e-5) {
      if (!network.isFrozen()) {
        log.p("This training apply resulted in the following configuration:");
        log.eval(() -> {
          RefList<double[]> state = network.state();
          assert state != null;
          String description = state.stream().map(RefArrays::toString).reduce((a, b) -> a + "\n" + b)
              .orElse("");
          state.freeRef();
          return description;
        });
      }
      network.freeRef();
      if (0 < mask.length) {
        log.p("And regressed input:");
        log.eval(RefUtil.wrapInterface((UncheckedSupplier<String>) () -> {
          return RefArrays.stream(RefUtil.addRef(data)).flatMap(x -> {
            return RefArrays.stream(x);
          }).limit(1).map(x -> {
            String temp_18_0015 = x.prettyPrint();
            x.freeRef();
            return temp_18_0015;
          }).reduce((a, b) -> a + "\n" + b).orElse("");
        }, RefUtil.addRef(data)));
      }
      log.p("To produce the following output:");
      log.eval(RefUtil.wrapInterface((UncheckedSupplier<String>) () -> {
        Result[] array = ConstantResult.batchResultArray(pop(RefUtil.addRef(data)));
        @Nullable
        Result eval = layer.eval(array);
        assert eval != null;
        TensorList tensorList = Result.getData(eval);
        String temp_18_0016 = tensorList.stream().limit(1).map(x -> {
          String temp_18_0017 = x.prettyPrint();
          x.freeRef();
          return temp_18_0017;
        }).reduce((a, b) -> a + "\n" + b).orElse("");
        tensorList.freeRef();
        return temp_18_0016;
      }, data, layer));
    } else {
      log.p("Training Converged");
      RefUtil.freeRef(data);
      network.freeRef();
      layer.freeRef();
    }
    return history;
  }

  @RefIgnore
  private List<StepRecord> runOpt(@Nonnull NotebookOutput log, @Nonnull RefBiFunction<NotebookOutput, Trainable, List<StepRecord>> opt, ArrayTrainable trainable) {
    List<StepRecord> history = opt.apply(log, trainable);
    trainable.assertFreed();
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
      return RefString.format("{ \"type\": \"%s\", \"value\": %s }", type, value);
    }
  }

  public static class ProblemResult {
    @Nonnull
    final Map<CharSequence, TrainingResult> map;

    public ProblemResult() {
      this.map = new HashMap<>();
    }

    public void put(CharSequence key, TrainingResult result) {
      map.put(key, result);
    }

    @Nonnull
    @Override
    public String toString() {
      return "{ " + map.entrySet().stream().map(e ->
          String.format("\"%s\": %s", e.getKey(), e.getValue().toString())
      ).reduce((a, b) -> a + ", " + b).get() + " }";
    }
  }
}
