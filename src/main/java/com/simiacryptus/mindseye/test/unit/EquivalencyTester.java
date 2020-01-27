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
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.test.SimpleEval;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefString;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.function.IntFunction;

public class EquivalencyTester extends ComponentTestBase<ToleranceStatistics> {
  private static final Logger log = LoggerFactory.getLogger(EquivalencyTester.class);

  @Nullable
  private final Layer reference;
  private final double tolerance;

  public EquivalencyTester(final double tolerance, @Nullable final Layer referenceLayer) {
    this.tolerance = tolerance;
    Layer temp_08_0001 = referenceLayer == null ? null : referenceLayer.addRef();
    this.reference = temp_08_0001 == null ? null : temp_08_0001.addRef();
    if (null != temp_08_0001)
      temp_08_0001.freeRef();
    if (null != referenceLayer)
      referenceLayer.freeRef();
  }

  @Nonnull
  public ToleranceStatistics test(@Nullable final Layer subject, @Nonnull final Tensor[] inputPrototype) {
    if (null == reference || null == subject) {
      if (null != subject)
        subject.freeRef();
      RefUtil.freeRefs(inputPrototype);
      return new ToleranceStatistics();
    }
    reference.assertAlive();
    SimpleEval temp_08_0004 = SimpleEval.run(subject.addRef(), RefUtil.addRefs(inputPrototype));
    final Tensor subjectOutput = temp_08_0004.getOutput();
    temp_08_0004.freeRef();
    subject.freeRef();
    SimpleEval temp_08_0005 = SimpleEval.run(reference.addRef(), false,
        RefUtil.addRefs(inputPrototype));
    final Tensor referenceOutput = temp_08_0005.getOutput();
    temp_08_0005.freeRef();
    log.info(RefString.format("Inputs: %s", RefUtil.get(RefArrays.stream(RefUtil.addRefs(inputPrototype)).map(t -> {
      String temp_08_0002 = t.prettyPrint();
      t.freeRef();
      return temp_08_0002;
    }).reduce((a, b) -> a + ",\n" + b))));
    assert subjectOutput != null;
    log.info(RefString.format("Subject Output: %s", subjectOutput.prettyPrint()));
    assert referenceOutput != null;
    log.info(RefString.format("Reference Output: %s", referenceOutput.prettyPrint()));
    @Nonnull
    Tensor error = subjectOutput.minus(referenceOutput.addRef());
    log.info(RefString.format("Error: %s", error.prettyPrint()));
    @Nonnull final ToleranceStatistics result = RefUtil.get(RefIntStream.range(0, subjectOutput.length())
        .mapToObj(RefUtil.wrapInterface((IntFunction<ToleranceStatistics>) i1 -> {
              return new ToleranceStatistics().accumulate(subjectOutput.getData()[i1], referenceOutput.getData()[i1]);
            }, subjectOutput.addRef(),
            referenceOutput.addRef()))
        .reduce((a, b) -> a.combine(b)));
    log.info(RefString.format("Accuracy:"));
    log.info(RefString.format("absoluteTol: %s", result.absoluteTol.toString()));
    log.info(RefString.format("relativeTol: %s", result.relativeTol.toString()));
    if (!(result.absoluteTol.getMax() < tolerance)) {
      subjectOutput.freeRef();
      referenceOutput.freeRef();
      error.freeRef();
      RefUtil.freeRefs(inputPrototype);
      throw new AssertionError(result.toString());
    }
    subjectOutput.freeRef();
    referenceOutput.freeRef();
    error.freeRef();
    RefUtil.freeRefs(inputPrototype);
    return result;
  }

  @Override
  public ToleranceStatistics test(@Nonnull final NotebookOutput output, @Nonnull final Layer subject,
                                  @Nonnull final Tensor... inputPrototype) {
    output.h1("Reference Implementation");
    output.p("This key is an alternate implementation which is expected to behave the same as the following key:");
    output.run(() -> {
      assert reference != null;
      log.info(new GsonBuilder().setPrettyPrinting().create().toJson(reference.getJson()));
    });
    output.run(RefUtil.wrapInterface(() -> {
      log.info(new GsonBuilder().setPrettyPrinting().create().toJson(subject.getJson()));
    }, subject.addRef()));
    output.p("We measureStyle the agreement between the two layers in a random execution:");
    ToleranceStatistics temp_08_0003 = output.eval(RefUtil.wrapInterface(() -> {
      return test(subject.addRef(), RefUtil.addRefs(inputPrototype));
    }, RefUtil.addRefs(inputPrototype), subject.addRef()));
    RefUtil.freeRefs(inputPrototype);
    subject.freeRef();
    return temp_08_0003;
  }

  @Nonnull
  @Override
  public String toString() {
    return "EquivalencyTester{" + "reference=" + reference + ", tolerance=" + tolerance + '}';
  }

  public void _free() {
    if (null != reference)
      reference.freeRef();
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  EquivalencyTester addRef() {
    return (EquivalencyTester) super.addRef();
  }
}
