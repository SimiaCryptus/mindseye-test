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

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.*;
import guru.nidi.graphviz.attribute.Label;
import guru.nidi.graphviz.attribute.Rank;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import guru.nidi.graphviz.model.*;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;

public final class GraphVizNetworkInspector {
  /**
   * To graph object.
   *
   * @param network the network
   * @return the object
   */
  @Nonnull
  public static Graph toGraphviz(@Nonnull final DAGNetwork network) {
    return toGraphviz(network, GraphVizNetworkInspector::getName);
  }

  /**
   * Graph.
   *
   * @param log     the log
   * @param network the network
   */
  public static void graph(@Nonnull final NotebookOutput log, @Nonnull final DAGNetwork network) {
    graph(log, toGraphviz(network, RefUtil.wrapInterface(node -> MermaidGrapher.getLabel(network, node), network.addRef())));
  }

  public static void graph(@Nonnull NotebookOutput log, Graph graph) {
    String svgSrc = Graphviz.fromGraph(graph).height(400).width(600).render(Format.SVG_STANDALONE).toString();
    log.out("\n" + log.svg(svgSrc, "Configuration Graph") + "\n");
  }

  /**
   * To graph object.
   *
   * @param network the network
   * @param fn      the fn
   * @return the object
   */
  @Nonnull
  public static Graph toGraphviz(@Nonnull final DAGNetwork network, @Nonnull @RefAware RefFunction<DAGNode, String> fn) {
    final RefList<DAGNode> nodes = network.getNodes();
    network.freeRef();
    final RefMap<UUID, MutableNode> graphNodes = nodes.stream().collect(RefCollectors.toMap(MermaidGrapher::getUuid, node -> {
      UUID id = node.getId();
      String name = fn.apply(node);
      return Factory.mutNode(Label.html(name + "<!-- " + id.toString() + " -->"));
    }));
    RefUtil.freeRef(fn);
    final Map<UUID, List<UUID>> connections = MermaidGrapher.connections(nodes.addRef());

    try {
      nodes.forEach(to -> {
        UUID toId = to.getId();
        List<UUID> fromList = connections.computeIfAbsent(toId, Arrays::asList);
        if (fromList != null) {
          MutableNode toNode = graphNodes.get(toId);
          toNode.addLink(
              fromList.stream().map(from -> {
                MutableNode fromNode = graphNodes.get(from);
                return Link.to(fromNode);
              }).<LinkTarget>toArray(LinkTarget[]::new));
        }
        to.freeRef();
      });
      assert graphNodes != null;
      RefCollection<MutableNode> values = graphNodes.values();
      final LinkSource[] nodeArray = values.stream().map(x -> (LinkSource) x).toArray(LinkSource[]::new);
      values.freeRef();
      return Factory.graph().with(nodeArray).graphAttr().with(Rank.dir(Rank.RankDir.TOP_TO_BOTTOM)).directed();
    } finally {
      nodes.freeRef();
    }
  }

  /**
   * Gets name.
   *
   * @param node the node
   * @return the name
   */
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
}
