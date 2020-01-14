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
import com.simiacryptus.mindseye.eval.SampledTrainable;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.ValidatingTrainer;
import com.simiacryptus.mindseye.test.StepRecord;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.util.ImageUtil;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.TableOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefDoubleStream;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.data.ScalarStatistics;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import guru.nidi.graphviz.model.Graph;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;

public abstract class EncodingProblem implements Problem {

  private static int modelNo = 0;
  private final ImageProblemData data;
  private final List<StepRecord> history = new ArrayList<>();
  private final OptimizationStrategy optimizer;
  private final RevNetworkFactory revFactory;
  private int batchSize = 10000;
  private int features;
  private int timeoutMinutes = 1;
  private int trainingSize = 15000;

  public EncodingProblem(final RevNetworkFactory revFactory, final OptimizationStrategy optimizer,
                         final ImageProblemData data, final int features) {
    this.revFactory = revFactory;
    this.optimizer = optimizer;
    this.data = data;
    this.features = features;
  }

  public int getBatchSize() {
    return batchSize;
  }

  @Nonnull
  public EncodingProblem setBatchSize(final int batchSize) {
    this.batchSize = batchSize;
    return this;
  }

  public int getFeatures() {
    return features;
  }

  @Nonnull
  public EncodingProblem setFeatures(final int features) {
    this.features = features;
    return this;
  }

  @Nonnull
  @Override
  public List<StepRecord> getHistory() {
    return history;
  }

  public int getTimeoutMinutes() {
    return timeoutMinutes;
  }

  @Nonnull
  public EncodingProblem setTimeoutMinutes(final int timeoutMinutes) {
    this.timeoutMinutes = timeoutMinutes;
    return this;
  }

  public int getTrainingSize() {
    return trainingSize;
  }

  @Nonnull
  public EncodingProblem setTrainingSize(final int trainingSize) {
    this.trainingSize = trainingSize;
    return this;
  }

  public double random() {
    return 0.1 * (Math.random() - 0.5);
  }

  @Nonnull
  @Override
  public EncodingProblem run(@Nonnull final NotebookOutput log) {
    @Nonnull final TrainingMonitor monitor = TestUtil.getMonitor(history);
    Tensor[][] trainingData;
    try {
      trainingData = data.trainingData().map(labeledObject -> {
        Tensor temp_20_0003 = new Tensor(features);
        Tensor[] temp_20_0002 = new Tensor[]{temp_20_0003.set(this::random), labeledObject.data};
        temp_20_0003.freeRef();
        return temp_20_0002;
      }).toArray(i -> new Tensor[i][]);
    } catch (@Nonnull final IOException e) {
      throw new RuntimeException(e);
    }

    @Nonnull final DAGNetwork imageNetwork = revFactory.vectorToImage(log, features);
    log.h3("Network Diagram");
    log.eval(RefUtil.wrapInterface((UncheckedSupplier<BufferedImage>) () -> {
      return Graphviz.fromGraph((Graph) TestUtil.toGraph(imageNetwork.addRef()))
          .height(400).width(600).render(Format.PNG).toImage();
    }, imageNetwork.addRef()));

    final DAGNetwork trainingNetwork = trainingNetwork(imageNetwork.addRef());
    log.h3("Training");
    log.p("We start by training apply a very small population to improve initial convergence performance:");
    TestUtil.instrumentPerformance(trainingNetwork.addRef());
    @Nonnull final Tensor[][] primingData = RefArrays.copyOfRange(Tensor.addRefs(trainingData), 0, 1000);
    SampledArrayTrainable temp_20_0004 = new SampledArrayTrainable(Tensor.addRefs(primingData),
        trainingNetwork.addRef(), trainingSize, batchSize);
    SampledArrayTrainable temp_20_0007 = temp_20_0004.setMinSamples(trainingSize);
    @Nonnull final ValidatingTrainer preTrainer = optimizer.train(log, (SampledTrainable) temp_20_0007.setMask(true, false),
        new ArrayTrainable(Tensor.addRefs(primingData), trainingNetwork.addRef(),
            batchSize),
        monitor);
    temp_20_0007.freeRef();
    temp_20_0004.freeRef();
    ReferenceCounting.freeRefs(primingData);
    log.run(RefUtil.wrapInterface(() -> {
      ValidatingTrainer temp_20_0008 = preTrainer.setTimeout(timeoutMinutes / 2, TimeUnit.MINUTES);
      ValidatingTrainer temp_20_0009 = temp_20_0008.setMaxIterations(batchSize);
      temp_20_0009.run();
      temp_20_0009.freeRef();
      temp_20_0008.freeRef();
    }, preTrainer));
    TestUtil.extractPerformance(log, trainingNetwork.addRef());

    log.p("Then our main training phase:");
    TestUtil.instrumentPerformance(trainingNetwork.addRef());
    SampledArrayTrainable temp_20_0005 = new SampledArrayTrainable(Tensor.addRefs(trainingData),
        trainingNetwork.addRef(), trainingSize, batchSize);
    SampledArrayTrainable temp_20_0010 = temp_20_0005.setMinSamples(trainingSize);
    @Nonnull final ValidatingTrainer mainTrainer = optimizer.train(log, (SampledTrainable) temp_20_0010.setMask(true, false),
        new ArrayTrainable(Tensor.addRefs(trainingData), trainingNetwork.addRef(),
            batchSize),
        monitor);
    temp_20_0010.freeRef();
    temp_20_0005.freeRef();
    log.run(RefUtil.wrapInterface(() -> {
      ValidatingTrainer temp_20_0011 = mainTrainer.setTimeout(timeoutMinutes, TimeUnit.MINUTES);
      ValidatingTrainer temp_20_0012 = temp_20_0011.setMaxIterations(batchSize);
      temp_20_0012.run();
      temp_20_0012.freeRef();
      temp_20_0011.freeRef();
    }, mainTrainer));
    TestUtil.extractPerformance(log, trainingNetwork.addRef());

    if (!history.isEmpty()) {
      log.eval(() -> {
        return TestUtil.plot(history);
      });
      log.eval(() -> {
        return TestUtil.plotTime(history);
      });
    }

    try {
      @Nonnull
      String filename = log.getName() + EncodingProblem.modelNo++ + "_plot.png";
      ImageIO.write(Util.toImage(TestUtil.plot(history)), "png", log.file(filename));
      log.appendFrontMatterProperty("result_plot", filename, ";");
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

    //log.file()
    @Nonnull final String modelName = "encoding_model_" + EncodingProblem.modelNo++ + ".json";
    log.appendFrontMatterProperty("result_model", modelName, ";");
    log.p("Saved model as " + log.file(trainingNetwork.getJson().toString(), modelName, modelName));

    log.h3("Results");
    @Nonnull final PipelineNetwork testNetwork = new PipelineNetwork(2);
    RefUtil.freeRef(testNetwork.add(imageNetwork.addRef(), testNetwork.getInput(0)));
    log.eval(RefUtil.wrapInterface((UncheckedSupplier<TableOutput>) () -> {
      @Nonnull final TableOutput table = new TableOutput();
      RefArrays.stream(Tensor.addRefs(trainingData)).map(RefUtil
          .wrapInterface((Function<? super Tensor[], ? extends LinkedHashMap<CharSequence, Object>>) tensorArray -> {
            Result temp_20_0013 = testNetwork.eval(Tensor.addRefs(tensorArray));
            assert temp_20_0013 != null;
            TensorList temp_20_0014 = temp_20_0013.getData();
            @Nullable final Tensor predictionSignal = temp_20_0014.get(0);
            temp_20_0014.freeRef();
            temp_20_0013.freeRef();
            @Nonnull final LinkedHashMap<CharSequence, Object> row = new LinkedHashMap<>();
            row.put("Source", log.png(tensorArray[1].toImage(), ""));
            ReferenceCounting.freeRefs(tensorArray);
            row.put("Echo", log.png(predictionSignal.toImage(), ""));
            predictionSignal.freeRef();
            return row;
          }, testNetwork.addRef())).filter(x -> null != x).limit(10)
          .forEach(table::putRow);
      return table;
    }, testNetwork, Tensor.addRefs(trainingData)));

    log.p("Learned Model Statistics:");
    RefUtil.freeRef(log.eval(RefUtil.wrapInterface((UncheckedSupplier<Map<CharSequence, Object>>) () -> {
      @Nonnull final ScalarStatistics scalarStatistics = new ScalarStatistics();
      RefList<double[]> temp_20_0015 = trainingNetwork.state();
      assert temp_20_0015 != null;
      temp_20_0015.stream().flatMapToDouble(x -> Arrays.stream(x)).forEach(v -> scalarStatistics.add(v));
      temp_20_0015.freeRef();
      return scalarStatistics.getMetrics();
    }, trainingNetwork.addRef())));

    trainingNetwork.freeRef();
    log.p("Learned Representation Statistics:");
    RefUtil.freeRef(log.eval(RefUtil.wrapInterface((UncheckedSupplier<Map<CharSequence, Object>>) () -> {
      @Nonnull final ScalarStatistics scalarStatistics = new ScalarStatistics();
      RefArrays.stream(Tensor.addRefs(trainingData)).flatMapToDouble(row -> {
        RefDoubleStream temp_20_0001 = RefArrays.stream(row[0].getData());
        ReferenceCounting.freeRefs(row);
        return temp_20_0001;
      }).forEach(v -> scalarStatistics.add(v));
      return scalarStatistics.getMetrics();
    }, Tensor.addRefs(trainingData))));

    ReferenceCounting.freeRefs(trainingData);
    log.p("Some rendered unit vectors:");
    for (int featureNumber = 0; featureNumber < features; featureNumber++) {
      Tensor temp_20_0006 = new Tensor(features);
      @Nonnull final Tensor input = temp_20_0006.set(featureNumber, 1);
      temp_20_0006.freeRef();
      Result temp_20_0016 = imageNetwork.eval(input);
      assert temp_20_0016 != null;
      TensorList temp_20_0017 = temp_20_0016.getData();
      @Nullable final Tensor tensor = temp_20_0017.get(0);
      temp_20_0017.freeRef();
      temp_20_0016.freeRef();
      ImageUtil.renderToImages(tensor.addRef(), true).forEach(img -> {
        log.out(log.png(img, ""));
      });
      tensor.freeRef();
    }

    imageNetwork.freeRef();
    return this;
  }

  @Nonnull
  protected abstract DAGNetwork trainingNetwork(DAGNetwork imageNetwork);

}
