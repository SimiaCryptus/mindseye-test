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
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.test.SimpleEval;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefHashMap;
import com.simiacryptus.ref.wrappers.RefString;
import com.simiacryptus.util.data.DoubleStatistics;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.function.BiConsumer;

public class ReferenceIO extends ComponentTestBase<ToleranceStatistics> {
  @Nullable
  final RefHashMap<Tensor[], Tensor> referenceIO;

  public ReferenceIO(@Nullable final RefHashMap<Tensor[], Tensor> referenceIO) {
    RefHashMap<Tensor[], Tensor> temp_05_0001 = RefUtil.addRef(referenceIO);
    this.referenceIO = RefUtil.addRef(temp_05_0001);
    if (null != temp_05_0001)
      temp_05_0001.freeRef();
    if (null != referenceIO)
      referenceIO.freeRef();
  }


  @Nullable
  @Override
  public ToleranceStatistics test(@Nonnull final NotebookOutput log, @Nonnull final Layer layer,
                                  @Nonnull final Tensor... inputPrototype) {
    assert referenceIO != null;
    if (!referenceIO.isEmpty()) {
      log.h1("Reference Input/Output Pairs");
      log.p("Display pre-setBytes input/output example pairs:");
      referenceIO.forEach(RefUtil.wrapInterface((BiConsumer<? super Tensor[], ? super Tensor>) (input, output) -> {
        log.eval(RefUtil.wrapInterface((UncheckedSupplier<String>) () -> {
          @Nonnull final SimpleEval eval = SimpleEval.run(layer.addRef(), RefUtil.addRefs(input));
          Tensor evalOutput = eval.getOutput();
          Tensor temp_05_0008 = output.scale(-1);
          Tensor difference = temp_05_0008.addAndFree(evalOutput == null ? null : evalOutput.addRef());
          temp_05_0008.freeRef();
          @Nonnull final DoubleStatistics error = new DoubleStatistics().accept(difference.getData());
          difference.freeRef();
          assert evalOutput != null;
          String temp_05_0002 = RefString.format(
              "--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s\n%s\nError: %s\n--------------------\nDerivative: \n%s",
              RefUtil.get(RefArrays.stream(RefUtil.addRefs(input)).map(t -> {
                String temp_05_0003 = RefArrays.toString(t.getDimensions()) + "\n" + t.prettyPrint();
                t.freeRef();
                return temp_05_0003;
              }).reduce((a, b) -> a + ",\n" + b)), RefArrays.toString(evalOutput.getDimensions()),
              evalOutput.prettyPrint(), error, RefUtil.get(RefArrays.stream(eval.getDerivative()).map(t -> {
                String temp_05_0004 = t.prettyPrint();
                t.freeRef();
                return temp_05_0004;
              }).reduce((a, b) -> a + ",\n" + b)));
          evalOutput.freeRef();
          eval.freeRef();
          return temp_05_0002;
        }, RefUtil.addRefs(input), layer.addRef(), output == null ? null : output.addRef()));
        if (null != output)
          output.freeRef();
        if (null != input)
          ReferenceCounting.freeRefs(input);
      }, layer.addRef()));
    } else {
      log.h1("Example Input/Output Pair");
      log.p("Display input/output pairs from random executions:");
      log.eval(RefUtil.wrapInterface((UncheckedSupplier<String>) () -> {
        @Nonnull final SimpleEval eval = SimpleEval.run(layer.addRef(), RefUtil.addRefs(inputPrototype));
        Tensor evalOutput = eval.getOutput();
        assert evalOutput != null;
        String temp_05_0005 = RefString.format(
            "--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s\n%s\n--------------------\nDerivative: \n%s",
            RefArrays.stream(RefUtil.addRefs(inputPrototype)).map(t -> {
              String temp_05_0006 = t.prettyPrint();
              t.freeRef();
              return temp_05_0006;
            }).reduce((a, b) -> a + ",\n" + b).orElse(""), RefArrays.toString(evalOutput.getDimensions()),
            evalOutput.prettyPrint(), RefArrays.stream(eval.getDerivative()).map(t -> {
              String temp_05_0007 = t.prettyPrint();
              t.freeRef();
              return temp_05_0007;
            }).reduce((a, b) -> a + ",\n" + b).orElse(""));
        evalOutput.freeRef();
        eval.freeRef();
        return temp_05_0005;
      }, RefUtil.addRefs(inputPrototype), layer.addRef()));
    }
    ReferenceCounting.freeRefs(inputPrototype);
    layer.freeRef();
    return null;
  }

  @Nonnull
  @Override
  public String toString() {
    return "ReferenceIO{" + "referenceIO=" + referenceIO + '}';
  }

  public void _free() {
    if (null != referenceIO)
      referenceIO.freeRef();
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ReferenceIO addRef() {
    return (ReferenceIO) super.addRef();
  }
}
