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

package com.simiacryptus.mindseye.test;

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.MonitoringWrapperLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.util.ImageUtil;
import com.simiacryptus.notebook.FileHTTPD;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.data.PercentileStatistics;
import com.simiacryptus.util.io.GifSequenceWriter;
import guru.nidi.graphviz.attribute.Label;
import guru.nidi.graphviz.attribute.RankDir;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import guru.nidi.graphviz.model.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import smile.plot.PlotCanvas;
import smile.plot.ScatterPlot;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.*;
import java.util.function.*;
import java.util.stream.Collectors;

public class TestUtil {
  public static final URI S3_ROOT = URI.create("https://s3-us-west-2.amazonaws.com/simiacryptus/");
  private static final Logger logger = LoggerFactory.getLogger(TestUtil.class);

  public static Map<String, List<String>> getStackInfo() {
    return Thread.getAllStackTraces().entrySet().stream().collect(Collectors.toMap(entry -> {
      Thread key = entry.getKey();
      return RefString.format("%s@%d", key.getName(), key.getId());
    }, entry -> Arrays.stream(entry.getValue()).map(StackTraceElement::toString).collect(Collectors.toList())));
  }

  @Nullable
  public static PlotCanvas compare(final String title, @Nonnull final ProblemRun... trials) {
    try {
      final DoubleSummaryStatistics xStatistics = RefArrays.stream(trials)
          .flatMapToDouble(x -> x.history.stream().mapToDouble(step -> step.iteration)).filter(Double::isFinite)
          .summaryStatistics();
      final DoubleSummaryStatistics yStatistics = RefArrays.stream(trials)
          .flatMapToDouble(
              x -> x.history.stream().filter(y -> y.fitness > 0).mapToDouble(step -> Math.log10(step.fitness)))
          .filter(Double::isFinite).summaryStatistics();
      if (xStatistics.getCount() == 0) {
        logger.info("No Data");
        return null;
      }
      @Nonnull final double[] lowerBound = {xStatistics.getMin(),
          yStatistics.getCount() < 2 ? 0 : yStatistics.getMin()};
      @Nonnull final double[] upperBound = {xStatistics.getMax(),
          yStatistics.getCount() < 2 ? 1 : yStatistics.getMax()};
      @Nonnull final PlotCanvas canvas = new PlotCanvas(lowerBound, upperBound);
      canvas.setTitle(title);
      canvas.setAxisLabels("Iteration", "log10(Fitness)");
      canvas.setSize(600, 400);
      final RefList<ProblemRun> filtered = RefArrays.stream(trials).filter(x -> !x.history.isEmpty())
          .collect(RefCollectors.toList());
      if (filtered.isEmpty()) {
        logger.info("No Data");
        filtered.freeRef();
        return null;
      }
      DoubleSummaryStatistics valueStatistics = filtered.stream().flatMap(x -> x.history.stream())
          .mapToDouble(x -> x.fitness).filter(x -> x > 0).summaryStatistics();
      logger.info(RefString.format("Plotting range=%s, %s; valueStats=%s", RefArrays.toString(lowerBound),
          RefArrays.toString(upperBound), valueStatistics));
      filtered.forEach(trial -> {
        final double[][] pts = trial.history.stream()
            .map(step -> new double[]{step.iteration, Math.log10(Math.max(step.fitness, valueStatistics.getMin()))})
            .filter(x -> RefArrays.stream(x).allMatch(Double::isFinite)).toArray(double[][]::new);
        if (pts.length > 1) {
          logger.info(RefString.format("Plotting %s points for %s", pts.length, trial.name));
          canvas.add(trial.plot(pts));
        } else {
          logger.info(RefString.format("Only %s points for %s", pts.length, trial.name));
        }
      });
      filtered.freeRef();
      return canvas;
    } catch (@Nonnull final Exception e) {
      e.printStackTrace(RefSystem.out);
      return null;
    }
  }

  @Nullable
  public static PlotCanvas compareTime(final String title, @Nonnull final ProblemRun... trials) {
    try {
      final DoubleSummaryStatistics[] xStatistics = RefArrays.stream(trials)
          .map(x -> x.history.stream().mapToDouble(step -> step.epochTime).filter(Double::isFinite).summaryStatistics())
          .toArray(i -> new DoubleSummaryStatistics[i]);
      final double totalTime = RefArrays.stream(xStatistics).mapToDouble(x -> x.getMax() - x.getMin()).max()
          .getAsDouble();
      final DoubleSummaryStatistics yStatistics = RefArrays.stream(trials)
          .flatMapToDouble(
              x -> x.history.stream().filter(y -> y.fitness > 0).mapToDouble(step -> Math.log10(step.fitness)))
          .filter(Double::isFinite).summaryStatistics();
      if (yStatistics.getCount() == 0) {
        logger.info("No Data");
        return null;
      }
      @Nonnull final double[] lowerBound = {0, yStatistics.getMin()};
      @Nonnull final double[] upperBound = {totalTime / 1000.0, yStatistics.getCount() == 1 ? 0 : yStatistics.getMax()};
      @Nonnull final PlotCanvas canvas = new PlotCanvas(lowerBound, upperBound);
      canvas.setTitle(title);
      canvas.setAxisLabels("Time", "log10(Fitness)");
      canvas.setSize(600, 400);
      final RefList<ProblemRun> filtered = RefArrays.stream(trials).filter(x -> !x.history.isEmpty())
          .collect(RefCollectors.toList());
      if (filtered.isEmpty()) {
        logger.info("No Data");
        filtered.freeRef();
        return null;
      }
      DoubleSummaryStatistics valueStatistics = filtered.stream().flatMap(x -> x.history.stream())
          .mapToDouble(x -> x.fitness).filter(x -> x > 0).summaryStatistics();
      logger.info(RefString.format("Plotting range=%s, %s; valueStats=%s", RefArrays.toString(lowerBound),
          RefArrays.toString(upperBound), valueStatistics));
      for (int t = 0; t < filtered.size(); t++) {
        final ProblemRun trial = filtered.get(t);
        final DoubleSummaryStatistics trialStats = xStatistics[t];
        final double[][] pts = trial.history.stream().map(step -> {
          return new double[]{(step.epochTime - trialStats.getMin()) / 1000.0,
              Math.log10(Math.max(step.fitness, valueStatistics.getMin()))};
        }).filter(x -> RefArrays.stream(x).allMatch(Double::isFinite)).toArray(i -> new double[i][]);
        if (pts.length > 1) {
          logger.info(RefString.format("Plotting %s points for %s", pts.length, trial.name));
          canvas.add(trial.plot(pts));
        } else {
          logger.info(RefString.format("Only %s points for %s", pts.length, trial.name));
        }
      }
      filtered.freeRef();
      return canvas;
    } catch (@Nonnull final Exception e) {
      e.printStackTrace(RefSystem.out);
      return null;
    }
  }

  public static void extractPerformance(@Nonnull final NotebookOutput log, @Nonnull final DAGNetwork network) {
    log.p("Per-key Performance Metrics:");
    log.run(RefUtil.wrapInterface(() -> {
      @Nonnull final RefMap<CharSequence, MonitoringWrapperLayer> metrics = new RefHashMap<>();
      network.visitNodes(RefUtil.wrapInterface(node -> {
        Layer nodeLayer = node.getLayer();
        if (nodeLayer instanceof MonitoringWrapperLayer) {
          @Nullable final MonitoringWrapperLayer layer = (MonitoringWrapperLayer) nodeLayer.addRef();
          Layer inner = layer.getInner();
          assert inner != null;
          String str = inner.toString();
          str += " class=" + inner.getClass().getName();
          inner.freeRef();
          RefUtil.freeRef(metrics.put(str, layer.addRef()));
          layer.freeRef();
        }
        assert nodeLayer != null;
        nodeLayer.freeRef();
        node.freeRef();
      }, metrics.addRef()));
      RefSet<Map.Entry<CharSequence, MonitoringWrapperLayer>> temp_13_0018 = metrics.entrySet();
      TestUtil.logger.info("Performance: \n\t" + RefUtil.get(temp_13_0018.stream().sorted(RefComparator.comparing(x -> {
        MonitoringWrapperLayer temp_13_0019 = x.getValue();
        double temp_13_0002 = -temp_13_0019.getForwardPerformance().getMean();
        temp_13_0019.freeRef();
        RefUtil.freeRef(x);
        return temp_13_0002;
      })).map(e -> {
        MonitoringWrapperLayer temp_13_0020 = e.getValue();
        @Nonnull final PercentileStatistics performanceF = temp_13_0020.getForwardPerformance();
        temp_13_0020.freeRef();
        MonitoringWrapperLayer temp_13_0021 = e.getValue();
        @Nonnull final PercentileStatistics performanceB = temp_13_0021.getBackwardPerformance();
        temp_13_0021.freeRef();
        String temp_13_0003 = RefString.format("%.6fs +- %.6fs (%d) <- %s", performanceF.getMean(),
            performanceF.getStdDev(), performanceF.getCount(), e.getKey())
            + (performanceB.getCount() == 0 ? ""
            : RefString.format("%n\tBack: %.6fs +- %.6fs (%s)", performanceB.getMean(), performanceB.getStdDev(),
            performanceB.getCount()));
        RefUtil.freeRef(e);
        return temp_13_0003;
      }).reduce((a, b) -> a + "\n\t" + b)));
      temp_13_0018.freeRef();
      metrics.freeRef();
    }, network.addRef()));
    removeInstrumentation(network);
  }

  public static void removeInstrumentation(@Nonnull final DAGNetwork network) {
    network.visitNodes(node -> {
      Layer nodeLayer = node.getLayer();
      if (nodeLayer instanceof MonitoringWrapperLayer) {
        MonitoringWrapperLayer temp_13_0022 = node.<MonitoringWrapperLayer>getLayer();
        Layer layer = temp_13_0022.getInner();
        temp_13_0022.freeRef();
        node.setLayer(layer == null ? null : layer.addRef());
        if (null != layer)
          layer.freeRef();
      }
      if (null != nodeLayer)
        nodeLayer.freeRef();
      node.freeRef();
    });
    network.freeRef();
  }

  @Nonnull
  public static RefMap<CharSequence, Object> samplePerformance(@Nonnull final DAGNetwork network) {
    @Nonnull final RefMap<CharSequence, Object> metrics = new RefHashMap<>();
    network.visitLayers(RefUtil.wrapInterface(layer -> {
      if (layer instanceof MonitoringWrapperLayer) {
        MonitoringWrapperLayer monitoringWrapperLayer = (MonitoringWrapperLayer) layer;
        Layer inner = monitoringWrapperLayer.getInner();
        assert inner != null;
        String str = inner.toString();
        str += " class=" + inner.getClass().getName();
        inner.freeRef();
        RefHashMap<CharSequence, Object> row = new RefHashMap<>();
        row.put("fwd", monitoringWrapperLayer.getForwardPerformance().getMetrics());
        row.put("rev", monitoringWrapperLayer.getBackwardPerformance().getMetrics());
        monitoringWrapperLayer.freeRef();
        metrics.put(str, RefUtil.addRef(row));
        row.freeRef();
      }
      if (null != layer)
        layer.freeRef();
    }, metrics.addRef()));
    network.freeRef();
    return metrics;
  }

  @Nonnull
  public static TrainingMonitor getMonitor(@Nonnull final List<StepRecord> history) {
    return getMonitor(history, null);
  }

  @Nonnull
  public static TrainingMonitor getMonitor(@Nonnull final List<StepRecord> history, @Nullable final Layer network) {
    if (null != network)
      network.freeRef();
    return new TrainingMonitor() {
      @Override
      public void clear() {
        super.clear();
      }

      @Override
      public void log(final String msg) {
        logger.info(msg);
        super.log(msg);
      }

      @Override
      public void onStepComplete(@Nonnull final Step currentPoint) {
        assert currentPoint.point != null;
        history.add(new StepRecord(currentPoint.point.getMean(), currentPoint.time, currentPoint.iteration));
        super.onStepComplete(currentPoint);
      }
    };
  }

  public static void instrumentPerformance(@Nonnull final DAGNetwork network) {
    network.visitNodes(node -> {
      Layer layer = node.getLayer();
      if (layer instanceof MonitoringWrapperLayer) {
        ((MonitoringWrapperLayer) layer).shouldRecordSignalMetrics(false);
        RefUtil.freeRef(((MonitoringWrapperLayer) layer).addRef());
      } else {
        MonitoringWrapperLayer temp_13_0015 = new MonitoringWrapperLayer(layer == null ? null : layer.addRef());
        temp_13_0015.shouldRecordSignalMetrics(false);
        @Nonnull
        MonitoringWrapperLayer monitoringWrapperLayer = temp_13_0015.addRef();
        temp_13_0015.freeRef();
        node.setLayer(monitoringWrapperLayer);
      }
      if (null != layer)
        layer.freeRef();
      node.freeRef();
    });
    network.freeRef();
  }

  @Nullable
  public static JPanel plot(@Nonnull final List<StepRecord> history) {
    try {
      final DoubleSummaryStatistics valueStats = history.stream().mapToDouble(x -> x.fitness).summaryStatistics();
      double min = valueStats.getMin();
      if (0 < min) {
        double[][] data = history.stream()
            .map(step -> new double[]{step.iteration, Math.log10(Math.max(min, step.fitness))})
            .filter(x -> Arrays.stream(x).allMatch(Double::isFinite)).toArray(i -> new double[i][]);
        if (Arrays.stream(data).mapToInt(x -> x.length).sum() == 0)
          return null;
        @Nonnull final PlotCanvas plot = ScatterPlot.plot(data);
        plot.setTitle("Convergence Plot");
        plot.setAxisLabels("Iteration", "log10(Fitness)");
        plot.setSize(600, 400);
        return plot;
      } else {
        double[][] data = history.stream().map(step -> new double[]{step.iteration, step.fitness})
            .filter(x -> Arrays.stream(x).allMatch(Double::isFinite)).toArray(i -> new double[i][]);
        if (Arrays.stream(data).mapToInt(x -> x.length).sum() == 0)
          return null;
        @Nonnull final PlotCanvas plot = ScatterPlot.plot(data);
        plot.setTitle("Convergence Plot");
        plot.setAxisLabels("Iteration", "Fitness");
        plot.setSize(600, 400);
        return plot;
      }
    } catch (@Nonnull final Exception e) {
      logger.warn("Error plotting", e);
      return null;
    }
  }

  @Nullable
  public static PlotCanvas plotTime(@Nonnull final List<StepRecord> history) {
    try {
      final LongSummaryStatistics timeStats = history.stream().mapToLong(x -> x.epochTime).summaryStatistics();
      final DoubleSummaryStatistics valueStats = history.stream().mapToDouble(x -> x.fitness).filter(x -> x > 0)
          .summaryStatistics();
      @Nonnull final PlotCanvas plot = ScatterPlot.plot(history.stream()
          .map(step -> new double[]{(step.epochTime - timeStats.getMin()) / 1000.0,
              Math.log10(Math.max(valueStats.getMin(), step.fitness))})
          .filter(x -> Arrays.stream(x).allMatch(Double::isFinite)).toArray(i -> new double[i][]));
      plot.setTitle("Convergence Plot");
      plot.setAxisLabels("Time", "log10(Fitness)");
      plot.setSize(600, 400);
      return plot;
    } catch (@Nonnull final Exception e) {
      e.printStackTrace(RefSystem.out);
      return null;
    }
  }

  @Nonnull
  public static Object toGraph(@Nonnull final DAGNetwork network) {
    Object temp_13_0011 = toGraph(network, TestUtil::getName);
    return temp_13_0011;
  }

  public static void graph(@Nonnull final NotebookOutput log, @Nonnull final DAGNetwork network) {
    Graphviz graphviz = Graphviz.fromGraph((Graph) toGraph(network, node -> {
      Layer layer = node.getLayer();
      if (null != layer) {
        String name = layer.getName();
        assert name != null;
        if (name.endsWith("Layer")) {
          node.freeRef();
          layer.freeRef();
          return name.substring(0, name.length() - 5);
        } else {
          node.freeRef();
          layer.freeRef();
          return name;
        }
      } else {
        DAGNetwork temp_13_0023 = node.getNetwork();
        assert temp_13_0023 != null;
        String temp_13_0004 = "Input " + temp_13_0023.inputHandles.indexOf(node.getId());
        temp_13_0023.freeRef();
        node.freeRef();
        return temp_13_0004;
      }
    }));
    log.out("\n" + log.png(graphviz.height(400).width(600).render(Format.PNG).toImage(), "Configuration Graph") + "\n");
    log.out(
        "\n" + log.svg(graphviz.height(400).width(600).render(Format.SVG_STANDALONE).toString(), "Configuration Graph")
            + "\n");
  }

  @Nonnull
  public static Object toGraph(@Nonnull final DAGNetwork network, @Nonnull Function<DAGNode, String> fn) {
    final RefList<DAGNode> nodes = network.getNodes();
    network.freeRef();
    final RefMap<UUID, MutableNode> graphNodes = nodes.stream().collect(RefCollectors.toMap(node -> {
      UUID temp_13_0005 = node.getId();
      node.freeRef();
      return temp_13_0005;
    }, node -> {
      String name = fn.apply(node);
      MutableNode temp_13_0006 = Factory.mutNode(Label.html(name + "<!-- " + node.getId().toString() + " -->"));
      node.freeRef();
      return temp_13_0006;
    }));
    final RefStream<UUID[]> stream = nodes.stream().flatMap(to -> {
      RefStream<UUID[]> temp_13_0007 = RefArrays.stream(to.getInputs())
          .map(RefUtil.wrapInterface((Function<? super DAGNode, ? extends UUID[]>) from -> {
            UUID[] temp_13_0008 = new UUID[]{from.getId(), to.getId()};
            from.freeRef();
            return temp_13_0008;
          }, to.addRef()));
      if (null != to)
        to.freeRef();
      return temp_13_0007;
    });
    final RefMap<UUID, RefList<UUID>> idMap = stream
        .collect(RefCollectors.groupingBy(x -> x[0], RefCollectors.mapping(x -> x[1], RefCollectors.toList())));
    nodes.forEach(RefUtil.wrapInterface((Consumer<? super DAGNode>) to -> {
      RefList<UUID> temp_13_0024 = idMap.getOrDefault(to.getId(), RefArrays.asList());
      assert temp_13_0024 != null;
      graphNodes.get(to.getId())
          .addLink(temp_13_0024.stream().map(RefUtil.wrapInterface((Function<? super UUID, ? extends Link>) from -> {
            return Link.to(graphNodes.get(from));
          }, graphNodes.addRef())).<LinkTarget>toArray(i -> new LinkTarget[i]));
      temp_13_0024.freeRef();
      to.freeRef();
    }, graphNodes == null ? null : graphNodes.addRef(), idMap == null ? null : idMap.addRef()));
    if (null != idMap)
      idMap.freeRef();
    nodes.freeRef();
    assert graphNodes != null;
    RefCollection<MutableNode> temp_13_0025 = graphNodes.values();
    final LinkSource[] nodeArray = temp_13_0025.stream().map(x -> (LinkSource) x).toArray(i -> new LinkSource[i]);
    temp_13_0025.freeRef();
    graphNodes.freeRef();
    return Factory.graph().with(nodeArray).graphAttr().with(RankDir.TOP_TO_BOTTOM).directed();
  }

  @Nonnull
  public static String getName(@Nonnull DAGNode node) {
    String name;
    @Nullable final Layer layer = node.getLayer();
    if (null == layer) {
      name = node.getId().toString();
    } else {
      final Class<? extends Layer> layerClass = layer.getClass();
      name = layerClass.getSimpleName() + "\n" + layer.getId();
    }
    node.freeRef();
    if (null != layer)
      layer.freeRef();
    return name;
  }

  @Nonnull
  public static RefIntStream shuffle(@Nonnull RefIntStream stream) {
    // http://primes.utm.edu/lists/small/10000.txt
    long coprimeA = 41387;
    long coprimeB = 9967;
    long ringSize = coprimeA * coprimeB - 1;
    @Nonnull
    IntToLongFunction fn = x -> (x * coprimeA * coprimeA) % ringSize;
    @Nonnull
    LongToIntFunction inv = x -> (int) ((x * coprimeB * coprimeB) % ringSize);
    @Nonnull
    IntUnaryOperator conditions = x -> {
      assert x < ringSize;
      assert x >= 0;
      return x;
    };
    return stream.map(conditions).mapToLong(fn).sorted().mapToInt(inv);
  }

  @Nonnull
  public static <T> Supplier<T> orElse(@Nonnull Supplier<T>... suppliers) {
    return () -> {
      for (@Nonnull
          Supplier<T> supplier : suppliers) {
        T t = supplier.get();
        if (null != t)
          return t;
      }
      return null;
    };
  }

  @Nonnull
  public static CharSequence animatedGif(@Nonnull final NotebookOutput log, @Nonnull final BufferedImage... images) {
    return animatedGif(log, 15000, images);
  }

  @Nonnull
  public static <K, V> RefMap<K, V> buildMap(@Nonnull RefConsumer<RefMap<K, V>> configure) {
    RefMap<K, V> map = new RefHashMap<>();
    configure.accept(map.addRef());
    return map;
  }

  @Nonnull
  public static Supplier<RefDoubleStream> geometricStream(final double start, final double end, final int steps) {
    double step = Math.pow(end / start, 1.0 / (steps - 1));
    return () -> RefDoubleStream.iterate(start, x -> x * step).limit(steps);
  }

  @Nonnull
  public static <T> RefList<T> shuffle(@Nullable final RefList<T> list) {
    RefArrayList<T> copy = new RefArrayList<>(list == null ? null : list.addRef());
    if (null != list)
      list.freeRef();
    RefCollections.shuffle(copy.addRef());
    return copy;
  }

  public static void addGlobalHandlers(@Nullable final FileHTTPD httpd) {
    if (null != httpd) {
      httpd.addGET("threads.json", "text/json", out -> {
        try {
          Map<String, List<String>> temp_13_0026 = getStackInfo();
          JsonUtil.getMapper().writer().writeValue(out, temp_13_0026);
          //JsonUtil.MAPPER.writer().writeValue(out, new HashMap<>());
          out.close();
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      });
    }
  }

  @Nonnull
  public static Tensor sum(@Nonnull RefCollection<Tensor> tensorStream) {
    Tensor temp_13_0012 = RefUtil.get(tensorStream.stream().reduce((a, b) -> {
      Tensor temp_13_0009 = a.addAndFree(b == null ? null : b.addRef());
      if (null != b)
        b.freeRef();
      a.freeRef();
      return temp_13_0009;
    }));
    tensorStream.freeRef();
    return temp_13_0012;
  }

  @Nonnull
  public static Tensor sum(@Nonnull RefStream<Tensor> tensorStream) {
    return RefUtil.get(tensorStream.reduce((a, b) -> {
      Tensor temp_13_0010 = a.addAndFree(b == null ? null : b.addRef());
      if (null != b)
        b.freeRef();
      a.freeRef();
      return temp_13_0010;
    }));
  }

  @Nonnull
  public static Tensor avg(@Nonnull RefCollection<? extends Tensor> values) {
    Tensor temp_13_0027 = sum(values.stream().map(x -> {
      return x;
    }));
    temp_13_0027.scaleInPlace(1.0 / values.size());
    Tensor temp_13_0013 = temp_13_0027.addRef();
    temp_13_0027.freeRef();
    values.freeRef();
    return temp_13_0013;
  }

  @Nonnull
  public static CharSequence render(@Nonnull final NotebookOutput log, @Nonnull final Tensor tensor,
                                    final boolean normalize) {
    String temp_13_0014 = RefUtil.get(ImageUtil.renderToImages(tensor, normalize).map(image -> {
      return log.png(image, "");
    }).reduce((a, b) -> a + b));
    return temp_13_0014;
  }

  @Nonnull
  public static CharSequence animatedGif(@Nonnull final NotebookOutput log, final int loopTimeMs,
                                         @Nonnull final BufferedImage... images) {
    try {
      @Nonnull
      String filename = UUID.randomUUID().toString() + ".gif";
      @Nonnull
      File file = new File(log.getResourceDir(), filename);
      GifSequenceWriter.write(file, loopTimeMs / images.length, true, images);
      return RefString.format("<img src=\"etc/%s\" />", filename);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

}
