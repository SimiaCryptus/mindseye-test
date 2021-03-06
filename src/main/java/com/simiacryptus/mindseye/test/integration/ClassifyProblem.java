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

package com.simiacryptus.mindseye.test.integration;

import com.simiacryptus.lang.UncheckedSupplier;
import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.SampledArrayTrainable;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.SimpleLossNetwork;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.ValidatingTrainer;
import com.simiacryptus.mindseye.test.GraphVizNetworkInspector;
import com.simiacryptus.mindseye.test.StepRecord;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.TableOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.test.LabeledObject;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.function.ToDoubleFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * The type Classify problem.
 */
public abstract class ClassifyProblem implements Problem {

  private static final Logger logger = LoggerFactory.getLogger(ClassifyProblem.class);

  private static int modelNo = 0;
  private final int categories;
  private final ImageProblemData data;
  private final FwdNetworkFactory fwdFactory;
  private final List<StepRecord> history = new ArrayList<>();
  private final OptimizationStrategy optimizer;
  private final List<CharSequence> labels;
  private int batchSize = 10000;
  private int timeoutMinutes = 1;

  /**
   * Instantiates a new Classify problem.
   *
   * @param fwdFactory the fwd factory
   * @param optimizer  the optimizer
   * @param data       the data
   * @param categories the categories
   */
  public ClassifyProblem(final FwdNetworkFactory fwdFactory, final OptimizationStrategy optimizer,
                         final ImageProblemData data, final int categories) {
    this.fwdFactory = fwdFactory;
    this.optimizer = optimizer;
    this.data = data;
    this.categories = categories;
    try {
      this.labels = Stream.concat(this.data.trainingData(), this.data.validationData()).map(x -> {
        String label = x.label;
        x.freeRef();
        return label;
      }).distinct().sorted().collect(Collectors.toList());
    } catch (IOException e) {
      throw Util.throwException(e);
    }
  }

  /**
   * Gets batch size.
   *
   * @return the batch size
   */
  public int getBatchSize() {
    return batchSize;
  }

  /**
   * Sets batch size.
   *
   * @param batchSize the batch size
   * @return the batch size
   */
  @Nonnull
  public ClassifyProblem setBatchSize(int batchSize) {
    this.batchSize = batchSize;
    return this;
  }

  @Nonnull
  @Override
  public List<StepRecord> getHistory() {
    return history;
  }

  /**
   * Gets timeout minutes.
   *
   * @return the timeout minutes
   */
  public int getTimeoutMinutes() {
    return timeoutMinutes;
  }

  /**
   * Sets timeout minutes.
   *
   * @param timeoutMinutes the timeout minutes
   * @return the timeout minutes
   */
  @Nonnull
  public ClassifyProblem setTimeoutMinutes(final int timeoutMinutes) {
    this.timeoutMinutes = timeoutMinutes;
    return this;
  }

  /**
   * Get training data tensor [ ] [ ].
   *
   * @return the tensor [ ] [ ]
   */
  @Nonnull
  public Tensor[][] getTrainingData() {
    try {
      return data.trainingData().map(labeledObject -> {
        @Nonnull final Tensor categoryTensor = new Tensor(categories);
        final int category = parse(labeledObject.label);
        categoryTensor.set(category, 1);
        Tensor data = labeledObject.data;
        labeledObject.freeRef();
        return new Tensor[]{data, categoryTensor};
      }).toArray(Tensor[][]::new);
    } catch (@Nonnull final IOException e) {
      throw Util.throwException(e);
    }
  }

  /**
   * Parse int.
   *
   * @param label the label
   * @return the int
   */
  public int parse(final CharSequence label) {
    return this.labels.indexOf(label);
  }

  /**
   * Predict int [ ].
   *
   * @param network       the network
   * @param labeledObject the labeled object
   * @return the int [ ]
   */
  public int[] predict(@Nonnull final Layer network, @Nonnull final LabeledObject<Tensor> labeledObject) {
    Result result = network.eval(labeledObject.data.addRef());
    labeledObject.freeRef();
    assert result != null;
    TensorList tensorList = result.getData();
    Tensor tensor = tensorList.get(0);
    tensorList.freeRef();
    result.freeRef();
    network.freeRef();
    int[] prediction = IntStream.range(0, categories).mapToObj(x -> x).sorted(Comparator.comparingDouble(i -> -tensor.get(i)))
        .mapToInt(x -> x).toArray();
    tensor.freeRef();
    return prediction;
  }

  @Nonnull
  @Override
  public ClassifyProblem run(@Nonnull final NotebookOutput log) {
    @Nonnull final TrainingMonitor monitor = TestUtil.getMonitor(history);
    final Tensor[][] trainingData = getTrainingData();

    @Nonnull final DAGNetwork network = fwdFactory.imageToVector(log, categories);
    log.h3("Network Diagram");
    log.eval(RefUtil.wrapInterface((UncheckedSupplier<BufferedImage>) () -> {
      return Graphviz.fromGraph(GraphVizNetworkInspector.toGraphviz(network.addRef())).height(400)
          .width(600).render(Format.PNG).toImage();
    }, network.addRef()));

    log.h3("Training");
    @Nonnull final SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network.addRef(),
        lossLayer());
    TestUtil.instrumentPerformance(supervisedNetwork.addRef());
    int initialSampleSize = Math.max(trainingData.length / 5, Math.min(10, trainingData.length / 2));
    @Nonnull final ValidatingTrainer trainer = optimizer.train(log,
        new SampledArrayTrainable(RefUtil.addRef(trainingData),
            supervisedNetwork.addRef(), initialSampleSize, getBatchSize()),
        new ArrayTrainable(RefUtil.addRef(trainingData), supervisedNetwork.addRef(),
            getBatchSize()),
        monitor);
    RefUtil.freeRef(trainingData);
    log.run(RefUtil.wrapInterface(() -> {
      trainer.setTimeout(timeoutMinutes, TimeUnit.MINUTES);
      ValidatingTrainer temp_12_0006 = trainer.addRef();
      temp_12_0006.setMaxIterations(10000);
      ValidatingTrainer temp_12_0007 = temp_12_0006.addRef();
      temp_12_0007.run();
      temp_12_0007.freeRef();
      temp_12_0006.freeRef();
    }, trainer));
    if (!history.isEmpty()) {
      log.eval(() -> {
        return TestUtil.plot(history);
      });
      log.eval(() -> {
        return TestUtil.plotTime(history);
      });
    }

    @Nonnull
    String training_name = log.getFileName() + "_" + ClassifyProblem.modelNo++ + "_plot.png";
    try {
      BufferedImage image = Util.toImage(TestUtil.plot(history));
      if (null != image)
        ImageIO.write(image, "png", log.file(training_name));
    } catch (IOException e) {
      logger.warn("Error writing result images", e);
    }
    log.addMetadata("result_plot", new File(log.getResourceDir(), training_name).toString());

    TestUtil.extractPerformance(log, supervisedNetwork);
    @Nonnull final String modelName = "classification_model_" + ClassifyProblem.modelNo++ + ".json";
    log.addMetadata("result_model", modelName);
    log.p("Saved model as " + log.file(network.getJson().toString(), modelName, modelName));

    log.h3("Validation");
    log.p("If we apply our model against the entire validation dataset, we get this accuracy:");
    log.eval(RefUtil.wrapInterface((UncheckedSupplier<Double>) () -> {
      return data.validationData()
          .mapToDouble(RefUtil.wrapInterface(
              (ToDoubleFunction<? super LabeledObject<Tensor>>) labeledObject -> predict(
                  network.addRef(), labeledObject)[0] == parse(labeledObject.label) ? 1 : 0,
              network.addRef()))
          .average().getAsDouble() * 100;
    }, network.addRef()));

    log.p("Let's examine some incorrectly predicted results in more detail:");
    log.eval(RefUtil.wrapInterface((UncheckedSupplier<TableOutput>) () -> {
      try {
        @Nonnull final TableOutput table = new TableOutput();
        RefList<RefList<LabeledObject<Tensor>>> partitioned = RefLists.partition(data.validationData().collect(RefCollectors.toList()), 100);
        partitioned.stream().flatMap(RefUtil.wrapInterface(
            (Function<RefList<LabeledObject<Tensor>>, RefStream<LinkedHashMap<CharSequence, Object>>>) batch -> {
              @Nonnull
              TensorList batchIn = new TensorArray(batch.stream().map(x -> {
                Tensor data = x.data;
                x.freeRef();
                return data;
              }).toArray(Tensor[]::new));
              Result temp_12_0008 = network.eval(new ConstantResult(batchIn));
              assert temp_12_0008 != null;
              TensorList batchOut = Result.getData(temp_12_0008);
              return RefIntStream.range(0, batchOut.length())
                  .mapToObj(RefUtil.wrapInterface((IntFunction<? extends LinkedHashMap<CharSequence, Object>>) i -> {
                    Tensor tensor = batchOut.get(i);
                    LinkedHashMap<CharSequence, Object> row = toRow(log, batch.get(i), tensor.getData());
                    tensor.freeRef();
                    return row;
                  }, batchOut, batch));
            }, network.addRef())).filter(Objects::nonNull).limit(10).forEach(table::putRow);
        partitioned.freeRef();
        return table;
      } catch (@Nonnull final IOException e) {
        throw Util.throwException(e);
      }
    }, network));
    return this;
  }

  /**
   * To row linked hash map.
   *
   * @param log              the log
   * @param labeledObject    the labeled object
   * @param predictionSignal the prediction signal
   * @return the linked hash map
   */
  @Nullable
  public LinkedHashMap<CharSequence, Object> toRow(@Nonnull final NotebookOutput log,
                                                   @Nonnull final LabeledObject<Tensor> labeledObject, final double[] predictionSignal) {
    final int actualCategory = parse(labeledObject.label);
    final int[] predictionList = IntStream.range(0, categories).mapToObj(x -> x)
        .sorted(Comparator.comparing(i -> -predictionSignal[i])).mapToInt(x -> x).toArray();
    if (predictionList[0] == actualCategory) {
      labeledObject.freeRef();
      return null; // We will only examine mispredicted rows
    }
    @Nonnull final LinkedHashMap<CharSequence, Object> row = new LinkedHashMap<>();
    row.put("Image", log.png(labeledObject.data.toImage(), labeledObject.label));
    labeledObject.freeRef();
    row.put("Prediction",
        RefUtil.get(Arrays.stream(predictionList).limit(3)
            .mapToObj(i -> RefString.format("%d (%.1f%%)", i, 100.0 * predictionSignal[i]))
            .reduce((a, b) -> a + ", " + b)));
    return row;
  }

  /**
   * Loss layer layer.
   *
   * @return the layer
   */
  @Nonnull
  protected abstract Layer lossLayer();
}
