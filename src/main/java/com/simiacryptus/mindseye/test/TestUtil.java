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
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.data.PercentileStatistics;
import com.simiacryptus.util.io.GifSequenceWriter;
import guru.nidi.graphviz.attribute.Label;
import guru.nidi.graphviz.attribute.RankDir;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import guru.nidi.graphviz.model.*;
import org.jetbrains.annotations.NotNull;
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
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class TestUtil {
  public static final URI S3_ROOT = URI.create("https://s3-us-west-2.amazonaws.com/simiacryptus/");
  private static final Logger logger = LoggerFactory.getLogger(TestUtil.class);

  public static Map<String, List<String>> getStackInfo() {
    return Thread.getAllStackTraces().entrySet().stream().collect(Collectors.toMap(entry -> {
      Thread key = entry.getKey();
      return String.format("%s@%d", key.getName(), key.getId());
    }, entry -> {
      return Arrays.stream(entry.getValue()).map(StackTraceElement::toString).collect(Collectors.toList());
    }));
  }

  public static PlotCanvas compare(final String title, @Nonnull final ProblemRun... trials) {
    try {
      final DoubleSummaryStatistics xStatistics = Arrays.stream(trials)
          .flatMapToDouble(x -> x.history.stream().mapToDouble(step -> step.iteration)).filter(Double::isFinite)
          .summaryStatistics();
      final DoubleSummaryStatistics yStatistics = Arrays.stream(trials)
          .flatMapToDouble(
              x -> x.history.stream().filter(y -> y.fitness > 0).mapToDouble(step -> Math.log10(step.fitness)))
          .filter(Double::isFinite).summaryStatistics();
      if (xStatistics.getCount() == 0) {
        logger.info("No Data");
        return null;
      }
      @Nonnull final double[] lowerBound = {xStatistics.getCount() == 0 ? 0 : xStatistics.getMin(),
          yStatistics.getCount() < 2 ? 0 : yStatistics.getMin()};
      @Nonnull final double[] upperBound = {xStatistics.getCount() == 0 ? 1 : xStatistics.getMax(),
          yStatistics.getCount() < 2 ? 1 : yStatistics.getMax()};
      @Nonnull final PlotCanvas canvas = new PlotCanvas(lowerBound, upperBound);
      canvas.setTitle(title);
      canvas.setAxisLabels("Iteration", "log10(Fitness)");
      canvas.setSize(600, 400);
      final List<ProblemRun> filtered = Arrays.stream(trials).filter(x -> !x.history.isEmpty())
          .collect(Collectors.toList());
      if (filtered.isEmpty()) {
        logger.info("No Data");
        return null;
      }
      DoubleSummaryStatistics valueStatistics = filtered.stream().flatMap(x -> x.history.stream())
          .mapToDouble(x -> x.fitness).filter(x -> x > 0).summaryStatistics();
      logger.info(String.format("Plotting range=%s, %s; valueStats=%s", Arrays.toString(lowerBound),
          Arrays.toString(upperBound), valueStatistics));
      for (@Nonnull final ProblemRun trial : filtered) {
        final double[][] pts = trial.history.stream()
            .map(step -> new double[]{step.iteration, Math.log10(Math.max(step.fitness, valueStatistics.getMin()))})
            .filter(x -> Arrays.stream(x).allMatch(Double::isFinite)).toArray(i -> new double[i][]);
        if (pts.length > 1) {
          logger.info(String.format("Plotting %s points for %s", pts.length, trial.name));
          canvas.add(trial.plot(pts));
        } else {
          logger.info(String.format("Only %s points for %s", pts.length, trial.name));
        }
      }
      return canvas;
    } catch (@Nonnull final Exception e) {
      e.printStackTrace(System.out);
      return null;
    }
  }

  public static PlotCanvas compareTime(final String title, @Nonnull final ProblemRun... trials) {
    try {
      final DoubleSummaryStatistics[] xStatistics = Arrays.stream(trials)
          .map(x -> x.history.stream().mapToDouble(step -> step.epochTime).filter(Double::isFinite).summaryStatistics())
          .toArray(i -> new DoubleSummaryStatistics[i]);
      final double totalTime = Arrays.stream(xStatistics).mapToDouble(x -> x.getMax() - x.getMin()).max().getAsDouble();
      final DoubleSummaryStatistics yStatistics = Arrays.stream(trials)
          .flatMapToDouble(
              x -> x.history.stream().filter(y -> y.fitness > 0).mapToDouble(step -> Math.log10(step.fitness)))
          .filter(Double::isFinite).summaryStatistics();
      if (yStatistics.getCount() == 0) {
        logger.info("No Data");
        return null;
      }
      @Nonnull final double[] lowerBound = {0, yStatistics.getCount() == 0 ? 0 : yStatistics.getMin()};
      @Nonnull final double[] upperBound = {totalTime / 1000.0, yStatistics.getCount() == 1 ? 0 : yStatistics.getMax()};
      @Nonnull final PlotCanvas canvas = new PlotCanvas(lowerBound, upperBound);
      canvas.setTitle(title);
      canvas.setAxisLabels("Time", "log10(Fitness)");
      canvas.setSize(600, 400);
      final List<ProblemRun> filtered = Arrays.stream(trials).filter(x -> !x.history.isEmpty())
          .collect(Collectors.toList());
      if (filtered.isEmpty()) {
        logger.info("No Data");
        return null;
      }
      DoubleSummaryStatistics valueStatistics = filtered.stream().flatMap(x -> x.history.stream())
          .mapToDouble(x -> x.fitness).filter(x -> x > 0).summaryStatistics();
      logger.info(String.format("Plotting range=%s, %s; valueStats=%s", Arrays.toString(lowerBound),
          Arrays.toString(upperBound), valueStatistics));
      for (int t = 0; t < filtered.size(); t++) {
        final ProblemRun trial = filtered.get(t);
        final DoubleSummaryStatistics trialStats = xStatistics[t];
        final double[][] pts = trial.history.stream().map(step -> {
          return new double[]{(step.epochTime - trialStats.getMin()) / 1000.0,
              Math.log10(Math.max(step.fitness, valueStatistics.getMin()))};
        }).filter(x -> Arrays.stream(x).allMatch(Double::isFinite)).toArray(i -> new double[i][]);
        if (pts.length > 1) {
          logger.info(String.format("Plotting %s points for %s", pts.length, trial.name));
          canvas.add(trial.plot(pts));
        } else {
          logger.info(String.format("Only %s points for %s", pts.length, trial.name));
        }
      }
      return canvas;
    } catch (@Nonnull final Exception e) {
      e.printStackTrace(System.out);
      return null;
    }
  }

  public static void extractPerformance(@Nonnull final NotebookOutput log, @Nonnull final DAGNetwork network) {
    log.p("Per-key Performance Metrics:");
    log.run(() -> {
      @Nonnull final Map<CharSequence, MonitoringWrapperLayer> metrics = new HashMap<>();
      network.visitNodes(node -> {
        if (node.getLayer() instanceof MonitoringWrapperLayer) {
          @Nullable final MonitoringWrapperLayer layer = node.getLayer();
          Layer inner = layer.getInner();
          String str = inner.toString();
          str += " class=" + inner.getClass().getName();
          //          if(inner instanceof MultiPrecision<?>) {
          //            str += "; precision=" + ((MultiPrecision) inner).getPrecision().name();
          //          }
          metrics.put(str, layer);
        }
      });
      TestUtil.logger.info("Performance: \n\t" + metrics.entrySet().stream()
          .sorted(Comparator.comparing(x -> -x.getValue().getForwardPerformance().getMean())).map(e -> {
            @Nonnull final PercentileStatistics performanceF = e.getValue().getForwardPerformance();
            @Nonnull final PercentileStatistics performanceB = e.getValue().getBackwardPerformance();
            return String.format("%.6fs +- %.6fs (%d) <- %s", performanceF.getMean(), performanceF.getStdDev(),
                performanceF.getCount(), e.getKey())
                + (performanceB.getCount() == 0 ? ""
                : String.format("%n\tBack: %.6fs +- %.6fs (%s)", performanceB.getMean(), performanceB.getStdDev(),
                performanceB.getCount()));
          }).reduce((a, b) -> a + "\n\t" + b).get());
    });
    removeInstrumentation(network);
  }

  public static void removeInstrumentation(@Nonnull final DAGNetwork network) {
    network.visitNodes(node -> {
      if (node.getLayer() instanceof MonitoringWrapperLayer) {
        Layer layer = node.<MonitoringWrapperLayer>getLayer().getInner();
        node.setLayer(layer);
      }
    });
  }

  public static Map<CharSequence, Object> samplePerformance(@Nonnull final DAGNetwork network) {
    @Nonnull final Map<CharSequence, Object> metrics = new HashMap<>();
    network.visitLayers(layer -> {
      if (layer instanceof MonitoringWrapperLayer) {
        MonitoringWrapperLayer monitoringWrapperLayer = (MonitoringWrapperLayer) layer;
        Layer inner = monitoringWrapperLayer.getInner();
        String str = inner.toString();
        str += " class=" + inner.getClass().getName();
        HashMap<CharSequence, Object> row = new HashMap<>();
        row.put("fwd", monitoringWrapperLayer.getForwardPerformance().getMetrics());
        row.put("rev", monitoringWrapperLayer.getBackwardPerformance().getMetrics());
        metrics.put(str, row);
      }
    });
    return metrics;
  }

  public static TrainingMonitor getMonitor(@Nonnull final List<StepRecord> history) {
    return getMonitor(history, null);
  }

  public static TrainingMonitor getMonitor(@Nonnull final List<StepRecord> history, final Layer network) {
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
      } else {
        @Nonnull
        MonitoringWrapperLayer monitoringWrapperLayer = new MonitoringWrapperLayer(layer)
            .shouldRecordSignalMetrics(false);
        node.setLayer(monitoringWrapperLayer);
      }
    });
  }

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
      e.printStackTrace(System.out);
      return null;
    }
  }

  public static Object toGraph(@Nonnull final DAGNetwork network) {
    return toGraph(network, TestUtil::getName);
  }

  public static void graph(@Nonnull final NotebookOutput log, @Nonnull final DAGNetwork network) {
    Graphviz graphviz = Graphviz.fromGraph((Graph) toGraph(network, node -> {
      Layer layer = node.getLayer();
      if (null != layer) {
        String name = layer.getName();
        if (name.endsWith("Layer"))
          return name.substring(0, name.length() - 5);
        else
          return name;
      } else {
        return "Input " + node.getNetwork().inputHandles.indexOf(node.getId());
      }
    }));
    log.out("\n" + log.png(graphviz.height(400).width(600).render(Format.PNG).toImage(), "Configuration Graph") + "\n");
    log.out(
        "\n" + log.svg(graphviz.height(400).width(600).render(Format.SVG_STANDALONE).toString(), "Configuration Graph")
            + "\n");
  }

  public static Object toGraph(@Nonnull final DAGNetwork network, Function<DAGNode, String> fn) {
    final List<DAGNode> nodes = network.getNodes();
    final Map<UUID, MutableNode> graphNodes = nodes.stream().collect(Collectors.toMap(node -> node.getId(), node -> {
      String name = fn.apply(node);
      return Factory.mutNode(Label.html(name + "<!-- " + node.getId().toString() + " -->"));
    }));
    final Stream<UUID[]> stream = nodes.stream().flatMap(to -> {
      return Arrays.stream(to.getInputs()).map(from -> {
        return new UUID[]{from.getId(), to.getId()};
      });
    });
    final Map<UUID, List<UUID>> idMap = stream
        .collect(Collectors.groupingBy(x -> x[0], Collectors.mapping(x -> x[1], Collectors.toList())));
    nodes.forEach(to -> {
      graphNodes.get(to.getId()).addLink(idMap.getOrDefault(to.getId(), Arrays.asList()).stream().map(from -> {
        return Link.to(graphNodes.get(from));
      }).<LinkTarget>toArray(i -> new LinkTarget[i]));
    });
    final LinkSource[] nodeArray = graphNodes.values().stream().map(x -> (LinkSource) x)
        .toArray(i -> new LinkSource[i]);
    return Factory.graph().with(nodeArray).graphAttr().with(RankDir.TOP_TO_BOTTOM).directed();
  }

  @NotNull
  public static String getName(DAGNode node) {
    String name;
    @Nullable final Layer layer = node.getLayer();
    if (null == layer) {
      name = node.getId().toString();
    } else {
      final Class<? extends Layer> layerClass = layer.getClass();
      name = layerClass.getSimpleName() + "\n" + layer.getId();
    }
    return name;
  }

  public static IntStream shuffle(@Nonnull IntStream stream) {
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

  public static CharSequence animatedGif(@Nonnull final NotebookOutput log, @Nonnull final BufferedImage... images) {
    return animatedGif(log, 15000, images);
  }

  @Nonnull
  public static <K, V> Map<K, V> buildMap(Consumer<Map<K, V>> configure) {
    Map<K, V> map = new HashMap<>();
    configure.accept(map);
    return map;
  }

  @Nonnull
  public static Supplier<DoubleStream> geometricStream(final double start, final double end, final int steps) {
    double step = Math.pow(end / start, 1.0 / (steps - 1));
    return () -> DoubleStream.iterate(start, x -> x * step).limit(steps);
  }

  public static <T> List<T> shuffle(final List<T> list) {
    ArrayList<T> copy = new ArrayList<>(list);
    Collections.shuffle(copy);
    return copy;
  }

  public static void addGlobalHandlers(final FileHTTPD httpd) {
    if (null != httpd) {
      httpd.addGET("threads.json", "text/json", out -> {
        try {
          JsonUtil.getMapper().writer().writeValue(out, getStackInfo());
          //JsonUtil.MAPPER.writer().writeValue(out, new HashMap<>());
          out.close();
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      });
    }
  }

  @NotNull
  public static Tensor sum(Collection<Tensor> tensorStream) {
    return tensorStream.stream().reduce((a, b) -> {
      return a.addAndFree(b);
    }).get();
  }

  @NotNull
  public static Tensor sum(Stream<Tensor> tensorStream) {
    return tensorStream.reduce((a, b) -> {
      return a.addAndFree(b);
    }).get();
  }

  @NotNull
  public static Tensor avg(Collection<? extends Tensor> values) {
    return sum(values.stream().map(x -> {
      return x;
    })).scaleInPlace(1.0 / values.size());
  }

  public static CharSequence render(@Nonnull final NotebookOutput log, @Nonnull final Tensor tensor,
                                    final boolean normalize) {
    return ImageUtil.renderToImages(tensor, normalize).map(image -> {
      return log.png(image, "");
    }).reduce((a, b) -> a + b).get();
  }

  public static CharSequence animatedGif(@Nonnull final NotebookOutput log, final int loopTimeMs,
                                         @Nonnull final BufferedImage... images) {
    try {
      @Nonnull
      String filename = UUID.randomUUID().toString() + ".gif";
      @Nonnull
      File file = new File(log.getResourceDir(), filename);
      GifSequenceWriter.write(file, loopTimeMs / images.length, true, images);
      return String.format("<img src=\"etc/%s\" />", filename);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

}
