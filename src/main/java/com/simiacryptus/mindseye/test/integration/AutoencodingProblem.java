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
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.layers.StochasticComponent;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.ValidatingTrainer;
import com.simiacryptus.mindseye.test.StepRecord;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.TableOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.util.test.LabeledObject;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import guru.nidi.graphviz.model.Graph;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;

@SuppressWarnings("FieldCanBeLocal")
public abstract class AutoencodingProblem implements Problem {

  private static int modelNo = 0;

  private final int batchSize = 10000;
  private final ImageProblemData data;
  private final double dropout;
  private final int features;
  private final FwdNetworkFactory fwdFactory;
  @Nonnull
  private final List<StepRecord> history = new ArrayList<>();
  private final OptimizationStrategy optimizer;
  private final RevNetworkFactory revFactory;
  private int timeoutMinutes = 1;

  public AutoencodingProblem(final FwdNetworkFactory fwdFactory, final OptimizationStrategy optimizer,
                             final RevNetworkFactory revFactory, final ImageProblemData data, final int features, final double dropout) {
    this.fwdFactory = fwdFactory;
    this.optimizer = optimizer;
    this.revFactory = revFactory;
    this.data = data;
    this.features = features;
    this.dropout = dropout;
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
  public AutoencodingProblem setTimeoutMinutes(final int timeoutMinutes) {
    this.timeoutMinutes = timeoutMinutes;
    return this;
  }

  @Nonnull
  public Tensor[][] getTrainingData(final NotebookOutput log) {
    try {
      return data.trainingData().map(labeledObject -> {
        return new Tensor[]{labeledObject.data};
      }).toArray(i -> new Tensor[i][]);
    } catch (@Nonnull final IOException e) {
      throw new RuntimeException(e);
    }
  }

  public int parse(@Nonnull final String label) {
    return Integer.parseInt(label.replaceAll("[^\\d]", ""));
  }

  @Nonnull
  @Override
  public AutoencodingProblem run(@Nonnull final NotebookOutput log) {

    @Nonnull final DAGNetwork fwdNetwork = fwdFactory.imageToVector(log, features);
    @Nonnull final DAGNetwork revNetwork = revFactory.vectorToImage(log, features);

    @Nonnull final PipelineNetwork echoNetwork = new PipelineNetwork(1);
    RefUtil.freeRef(echoNetwork.add(fwdNetwork.addRef()));
    RefUtil.freeRef(echoNetwork.add(revNetwork.addRef()));

    @Nonnull final PipelineNetwork supervisedNetwork = new PipelineNetwork(1);
    RefUtil.freeRef(supervisedNetwork.add(fwdNetwork.addRef()));
    @Nonnull final StochasticComponent dropoutNoiseLayer = dropout(dropout);
    RefUtil.freeRef(supervisedNetwork.add(dropoutNoiseLayer));
    RefUtil.freeRef(supervisedNetwork.add(revNetwork.addRef()));
    RefUtil.freeRef(supervisedNetwork.add(lossLayer(), supervisedNetwork.getHead(), supervisedNetwork.getInput(0)));

    log.h3("Network Diagrams");
    log.eval(RefUtil.wrapInterface((UncheckedSupplier<BufferedImage>) () -> {
      return Graphviz.fromGraph((Graph) TestUtil.toGraph(fwdNetwork.addRef())).height(400)
          .width(600).render(Format.PNG).toImage();
    }, fwdNetwork.addRef()));
    log.eval(RefUtil.wrapInterface((UncheckedSupplier<BufferedImage>) () -> {
      return Graphviz.fromGraph((Graph) TestUtil.toGraph(revNetwork.addRef())).height(400)
          .width(600).render(Format.PNG).toImage();
    }, revNetwork.addRef()));
    log.eval(RefUtil.wrapInterface((UncheckedSupplier<BufferedImage>) () -> {
      return Graphviz.fromGraph((Graph) TestUtil.toGraph(supervisedNetwork.addRef()))
          .height(400).width(600).render(Format.PNG).toImage();
    }, supervisedNetwork.addRef()));

    @Nonnull final TrainingMonitor monitor = new TrainingMonitor() {
      @Nonnull
      final TrainingMonitor inner = TestUtil.getMonitor(history);

      @Override
      public void log(final String msg) {
        inner.log(msg);
      }

      @Override
      public void onStepComplete(@Nullable final Step currentPoint) {
        inner.onStepComplete(currentPoint == null ? null : currentPoint.addRef());
        if (null != currentPoint)
          currentPoint.freeRef();
      }
    };

    final Tensor[][] trainingData = getTrainingData(log);

    //MonitoredObject monitoringRoot = new MonitoredObject();
    //TestUtil.addMonitoring(supervisedNetwork, monitoringRoot);

    log.h3("Training");
    TestUtil.instrumentPerformance(supervisedNetwork.addRef());
    @Nonnull final ValidatingTrainer trainer = optimizer.train(log,
        new SampledArrayTrainable(RefUtil.addRefs(trainingData),
            supervisedNetwork.addRef(), trainingData.length / 2, batchSize),
        new ArrayTrainable(RefUtil.addRefs(trainingData), supervisedNetwork.addRef(),
            batchSize),
        monitor);
    RefUtil.freeRefs(trainingData);
    log.run(RefUtil.wrapInterface(() -> {
      trainer.setTimeout(timeoutMinutes, TimeUnit.MINUTES);
      ValidatingTrainer temp_21_0003 = trainer.addRef();
      temp_21_0003.setMaxIterations(10000);
      ValidatingTrainer temp_21_0004 = temp_21_0003.addRef();
      temp_21_0004.run();
      temp_21_0004.freeRef();
      temp_21_0003.freeRef();
    }, trainer));
    if (!history.isEmpty()) {
      log.eval(() -> {
        return TestUtil.plot(history);
      });
      log.eval(() -> {
        return TestUtil.plotTime(history);
      });
    }
    TestUtil.extractPerformance(log, supervisedNetwork);

    {
      @Nonnull final String modelName = "encoder_model" + AutoencodingProblem.modelNo++ + ".json";
      log.p("Saved model as " + log.file(fwdNetwork.getJson().toString(), modelName, modelName));
    }

    fwdNetwork.freeRef();
    @Nonnull final String modelName = "decoder_model" + AutoencodingProblem.modelNo++ + ".json";
    log.p("Saved model as " + log.file(revNetwork.getJson().toString(), modelName, modelName));

    //    log.h3("Metrics");
    //    log.run(() -> {
    //      return TestUtil.toFormattedJson(monitoringRoot.getMetrics());
    //    });

    log.h3("Validation");

    log.p("Here are some re-encoded examples:");
    log.eval(RefUtil.wrapInterface((UncheckedSupplier<TableOutput>) () -> {
      @Nonnull final TableOutput table = new TableOutput();
      data.validationData().map(RefUtil.wrapInterface(
          (Function<? super LabeledObject<Tensor>, ? extends LinkedHashMap<CharSequence, Object>>) labeledObject -> {
            Result temp_21_0006 = echoNetwork.eval(labeledObject.data.addRef());
            assert temp_21_0006 != null;
            TensorList data = temp_21_0006.getData();
            Tensor tensor = data.get(0);
            LinkedHashMap<CharSequence, Object> temp_21_0005 = toRow(log, labeledObject, tensor.getData());
            tensor.freeRef();
            data.freeRef();
            temp_21_0006.freeRef();
            return temp_21_0005;
          }, echoNetwork.addRef())).filter(x -> null != x).limit(10)
          .forEach(table::putRow);
      return table;
    }, echoNetwork));

    log.p("Some rendered unit vectors:");
    for (int featureNumber = 0; featureNumber < features; featureNumber++) {
      Tensor temp_21_0001 = new Tensor(features);
      temp_21_0001.set(featureNumber, 1);
      @Nonnull final Tensor input = temp_21_0001.addRef();
      temp_21_0001.freeRef();
      Result temp_21_0007 = revNetwork.eval(input);
      assert temp_21_0007 != null;
      TensorList temp_21_0008 = temp_21_0007.getData();
      @Nullable final Tensor tensor = temp_21_0008.get(0);
      temp_21_0008.freeRef();
      temp_21_0007.freeRef();
      log.out(log.png(tensor.toImage(), ""));
      tensor.freeRef();
    }
    revNetwork.freeRef();
    return this;
  }

  @Nonnull
  public LinkedHashMap<CharSequence, Object> toRow(@Nonnull final NotebookOutput log,
                                                   @Nonnull final LabeledObject<Tensor> labeledObject, final double[] predictionSignal) {
    @Nonnull final LinkedHashMap<CharSequence, Object> row = new LinkedHashMap<>();
    row.put("Image", log.png(labeledObject.data.toImage(), labeledObject.label));
    Tensor temp_21_0002 = new Tensor(predictionSignal, labeledObject.data.getDimensions());
    row.put("Echo", log.png(temp_21_0002.toImage(), labeledObject.label));
    temp_21_0002.freeRef();
    return row;
  }

  @Nonnull
  protected abstract Layer lossLayer();

  @Nonnull
  protected abstract StochasticComponent dropout(double dropout);
}
