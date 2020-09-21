/*
 * Copyright (c) 2020 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.test;

import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.SerialPrecision;
import com.simiacryptus.mindseye.layers.WrapperLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefIgnore;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.*;
import org.apache.commons.io.FileUtils;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class MermaidGrapher {
  private final NotebookOutput log;
  private final boolean writeZip;

  public MermaidGrapher(NotebookOutput log, boolean writeZip) {
    this.log = log;
    this.writeZip = writeZip;
  }

  public static void writeZip(NotebookOutput log, Layer layer) {
    File file = new File(log.getResourceDir(), log.getFileName() + "_" + layer.getId() + ".zip");
    layer.writeZip(file, SerialPrecision.Double);
    layer.freeRef();
    log.p(log.link(file, RefString.format("Layer Zip (%.3fMiB bytes)", file.length() * 1.0 / 0x100000)));
  }

  public static void writeJson(NotebookOutput log, Layer layer) {
    File file = new File(log.getResourceDir(), log.getFileName() + "_" + layer.getId() + ".json");
    try {
      FileUtils.writeStringToFile(file, compactJson(layer), "UTF-8");
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    log.p(log.link(file, RefString.format("Layer Json (%.3fMiB bytes)", file.length() * 1.0 / 0x100000)));
  }

  @NotNull
  public static String getLabel(@Nonnull @RefIgnore DAGNetwork network, DAGNode node) {
    Layer layer = node.getLayer();
    UUID id = node.getId();
    node.freeRef();
    if (null != layer) {
      String name = layer.getName();
      layer.freeRef();
      assert name != null;
      if (name.endsWith("Layer")) {
        return name.substring(0, name.length() - 5);
      } else {
        return name;
      }
    } else {
      assert network != null;
      return "Input " + network.inputHandles.indexOf(id);
    }
  }

  public static Map<UUID, List<UUID>> connections(RefList<DAGNode> nodes) {
    Map<UUID, List<UUID>> collect = Arrays.stream(nodes.stream().flatMap(to -> {
      UUID toId = to.getId();
      @Nonnull DAGNode[] inputs = to.getInputs();
      to.freeRef();
      return RefArrays.stream(inputs).map(from -> new UUID[]{getUuid(from), toId});
    }).toArray(UUID[][]::new)).collect(Collectors.groupingBy(x -> x[0], Collectors.mapping(x -> x[1], Collectors.toList())));
    nodes.freeRef();
    return collect;
  }

  public static UUID getUuid(DAGNode node) {
    UUID id = node.getId();
    node.freeRef();
    return id;
  }

  private static String compactJson(Layer layer) {
    if (layer instanceof WrapperLayer) {
      Layer copy = layer.copy();
      layer.freeRef();
      layer = copy;
      ((WrapperLayer) layer).setInner(null);
    }
    JsonElement compactJson = layer.getJson(new HashMap<>(), SerialPrecision.Double);
    layer.freeRef();
    return new GsonBuilder().setPrettyPrinting().create().toJson(compactJson);
  }

  @NotNull
  private static String getLabel(@Nonnull @RefIgnore Layer layer) {
    String name = layer.getName();
    assert name != null;
    if (name.endsWith("Layer")) {
      return name.substring(0, name.length() - 5);
    } else {
      return name;
    }
  }

  private static String idString(@RefIgnore Layer l) {
    return l.getId().toString();
  }

  @NotNull
  public String toString(UUID toId) {
    return toId.toString().replaceAll("-", "");
  }

  public void mermaid(DAGNetwork network) {
    log.subreport("Network Diagrams for " + network.getName(), (NotebookOutput sublog) -> {
      List<UUID> logged = new ArrayList<>();
      RefHashMap<UUID, Layer> found = new RefHashMap<>();
      UUID networkId = network.getId();
      RefUtil.freeRef(found.put(networkId, network.addRef()));
      network.visitNodes(dagNode -> {
        Layer layer = dagNode.getLayer();
        dagNode.freeRef();
        if (null != layer) RefUtil.freeRef(found.put(layer.getId(), layer));
        else layer.freeRef();
      });
      RefHashSet<Layer> values = found.values();
      found.freeRef();
      values.stream()
          .collect(Collectors.groupingBy(Layer::getClass, Collectors.toList()))
          .entrySet().stream().sorted(Comparator.comparingDouble(x -> {
        Class<? extends Layer> key = x.getKey();
        RefUtil.freeRef(x);
        if (network.getClass().equals(key)) return -1.0;
        if (DAGNetwork.class.isAssignableFrom(key)) return 0.0;
        if (WrapperLayer.class.isAssignableFrom(key)) return 0.5;
        return 1.0;
      })).forEach(e -> {
        String type = e.getKey().getSimpleName();
        sublog.h1(type);
        out(sublog, logged, networkId, e.getValue());
        RefUtil.freeRef(e);
      });
      values.freeRef();
      return null;
    });
    network.freeRef();
  }

  private void out(NotebookOutput sublog, List<UUID> logged, UUID networkId, List<Layer> value) {
    value.stream()
        .filter((@RefIgnore Layer x) -> x.getId().equals(networkId))
        .sorted(Comparator.comparing(Layer::getName).thenComparing(MermaidGrapher::idString))
        .forEach(layer -> out(sublog, logged, layer, writeZip));
    value.stream()
        .filter((@RefIgnore Layer x) -> !x.getId().equals(networkId))
        .sorted(Comparator.comparing(Layer::getName).thenComparing(MermaidGrapher::idString))
        .forEach(layer -> out(sublog, logged, layer, writeZip));
    RefUtil.freeRef(value);
  }

  protected String describeNode(MermaidNode mermaidNode) {
    String name = nodeHtml((MermaidNode) mermaidNode.addRef());
    name = "\"" + name.replaceAll("\\\"", "\\\\\\\"") + "\"";
    return nodeID(mermaidNode) + "(" + name + ")";
  }

  @NotNull
  protected String nodeID(MermaidNode mermaidNode) {
    String str = toString(mermaidNode.id);
    mermaidNode.freeRef();
    return str;
  }

  protected String nodeHtml(MermaidNode mermaidNode) {
    String name = mermaidNode.name;
    if (null != mermaidNode.layer) name += " <br/> " + mermaidNode.layer.getId();
    if (mermaidNode.layer != null) {
      name = "<a href='#" + toString(mermaidNode.layer.getId()) + "'>" + name + "</a>";
    }
    mermaidNode.freeRef();
    return name;
  }

  private void out(NotebookOutput log, List<UUID> logged, Layer layer, boolean writeZip) {
    UUID id = layer.getId();
    String label = getLabel(layer);
    log.out(String.format("<a id=\"%s\"></a>", toString(id)));
    log.h2(label);
    log.p("ID: " + id.toString());
    log.p("Class: " + layer.getClass().getName());
    if (writeZip) writeZip(log, layer.addRef());
    add(log, layer);
    logged.add(id);
  }

  private void add(NotebookOutput log, Layer layer) {
    if (layer instanceof DAGNetwork) {
      writeJson(log, layer.addRef());
      String src = toMermaid((DAGNetwork) layer);
      if (Arrays.stream(src.split("\n")).map(String::trim).filter(x -> !x.isEmpty()).count() > 1) {
        log.mermaid(src);
      } else {
        log.p("Trivial Graph");
      }
    } else if (layer instanceof WrapperLayer) {
      log.p("Wrapper Type: " + layer.getClass().getSimpleName());
      //writeJson(log, layer);
      Layer inner = ((WrapperLayer) layer).getInner();
      log.out("\n\n```json\n" + compactJson(layer) + "\n```\n\n");
      add(log, inner);
    } else {
      writeJson(log, layer);
      //log.out("\n\n```json\n" + compactJson(layer) + "\n```\n\n");
    }
  }

  @NotNull
  private String toMermaid(@Nonnull DAGNetwork network) {
    RefFunction<DAGNode, String> fn = RefUtil.wrapInterface(node -> getLabel(network, node), network.addRef());
    return toMermaid(network, fn);
  }

  @Nonnull
  private String toMermaid(@Nonnull final DAGNetwork network, @RefAware @Nonnull RefFunction<DAGNode, String> fn) {
    final RefList<DAGNode> nodes = network.getNodes();
    network.freeRef();
    final RefMap<UUID, MermaidNode> graphNodes = nodes.stream().collect(RefCollectors.toMap(
        node -> getUuid(node),
        node -> new MermaidNode(fn.apply(node), node.getId(), node.getLayer())
    ));
    RefUtil.freeRef(fn);
    final Map<UUID, List<UUID>> connections = connections(nodes.addRef());

    try {
      assert graphNodes != null;
      return "graph TB\n\t" + nodes.stream().flatMap(from -> {
        UUID fromId = from.getId();
        List<UUID> toList = connections.computeIfAbsent(fromId, a1 -> Arrays.asList());
        try {
          if (toList != null) {
            MermaidNode fromNode = graphNodes.get(fromId);
            String describe = fromNode.describe();
            fromNode.freeRef();
            return toList.stream().map(toId -> {
              MermaidNode toNode = graphNodes.get(toId);
              String str = String.format("%s-->%s\n", describe, toNode.describe());
              toNode.freeRef();
              return str;
            });
          } else {
            return Stream.empty();
          }
        } finally {
          from.freeRef();
        }
      }).map(String::trim).filter(x -> !x.isEmpty())
          .reduce((a, b) -> a + "\n\t" + b).orElse("");
    } finally {
      nodes.freeRef();
      graphNodes.freeRef();
    }
  }

  protected final class MermaidNode extends ReferenceCountingBase {
    public final String name;
    public final UUID id;
    public final AtomicBoolean described = new AtomicBoolean(false);
    public final Layer layer;

    protected MermaidNode(String name, UUID id, Layer layer) {
      this.name = name;
      this.id = id;
      this.layer = layer;
    }

    @NotNull
    public String describe() {
      return this.described.getAndSet(true) ? nodeID((MermaidNode) this.addRef()) : describeNode((MermaidNode) this.addRef());
    }

    @Override
    protected void _free() {
      if (null != layer) layer.freeRef();
      super._free();
    }
  }

}
