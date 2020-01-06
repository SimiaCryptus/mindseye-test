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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefString;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.function.IntFunction;

public @RefAware
class EquivalencyTester extends ComponentTestBase<ToleranceStatistics> {
  private static final Logger log = LoggerFactory.getLogger(EquivalencyTester.class);

  private final Layer reference;
  private final double tolerance;

  public EquivalencyTester(final double tolerance, final Layer referenceLayer) {
    this.tolerance = tolerance;
    {
      Layer temp_08_0001 = referenceLayer == null ? null : referenceLayer.addRef();
      this.reference = temp_08_0001 == null ? null : temp_08_0001.addRef();
      if (null != temp_08_0001)
        temp_08_0001.freeRef();
    }
    if (null != referenceLayer)
      referenceLayer.freeRef();
  }

  public static @SuppressWarnings("unused")
  EquivalencyTester[] addRefs(EquivalencyTester[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(EquivalencyTester::addRef)
        .toArray((x) -> new EquivalencyTester[x]);
  }

  public static @SuppressWarnings("unused")
  EquivalencyTester[][] addRefs(EquivalencyTester[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(EquivalencyTester::addRefs)
        .toArray((x) -> new EquivalencyTester[x][]);
  }

  public ToleranceStatistics test(@Nullable final Layer subject, @Nonnull final Tensor[] inputPrototype) {
    if (null == reference || null == subject) {
      if (null != subject)
        subject.freeRef();
      ReferenceCounting.freeRefs(inputPrototype);
      return new ToleranceStatistics();
    }
    reference.assertAlive();
    SimpleEval temp_08_0004 = SimpleEval.run(subject == null ? null : subject.addRef(),
        Tensor.addRefs(inputPrototype));
    final Tensor subjectOutput = temp_08_0004.getOutput();
    if (null != temp_08_0004)
      temp_08_0004.freeRef();
    if (null != subject)
      subject.freeRef();
    SimpleEval temp_08_0005 = SimpleEval.run(
        reference == null ? null : reference.addRef(), false,
        Tensor.addRefs(inputPrototype));
    final Tensor referenceOutput = temp_08_0005.getOutput();
    if (null != temp_08_0005)
      temp_08_0005.freeRef();
    @Nonnull
    Tensor error = null;
    {
      log.info(RefString.format("Inputs: %s",
          RefArrays.stream(Tensor.addRefs(inputPrototype)).map(t -> {
            String temp_08_0002 = t.prettyPrint();
            if (null != t)
              t.freeRef();
            return temp_08_0002;
          }).reduce((a, b) -> a + ",\n" + b).get()));
      log.info(RefString.format("Subject Output: %s", subjectOutput.prettyPrint()));
      log.info(RefString.format("Reference Output: %s", referenceOutput.prettyPrint()));
      error = subjectOutput.minus(referenceOutput == null ? null : referenceOutput.addRef());
      log.info(RefString.format("Error: %s", error.prettyPrint()));
      @Nonnull final ToleranceStatistics result = RefIntStream.range(0, subjectOutput.length())
          .mapToObj(RefUtil.wrapInterface(
              (IntFunction<? extends ToleranceStatistics>) i1 -> {
                return new ToleranceStatistics().accumulate(subjectOutput.getData()[i1], referenceOutput.getData()[i1]);
              }, subjectOutput == null ? null : subjectOutput.addRef(),
              referenceOutput == null ? null : referenceOutput.addRef()))
          .reduce((a, b) -> a.combine(b)).get();
      log.info(RefString.format("Accuracy:"));
      log.info(RefString.format("absoluteTol: %s", result.absoluteTol.toString()));
      log.info(RefString.format("relativeTol: %s", result.relativeTol.toString()));
      if (!(result.absoluteTol.getMax() < tolerance)) {
        if (null != subjectOutput)
          subjectOutput.freeRef();
        if (null != referenceOutput)
          referenceOutput.freeRef();
        error.freeRef();
        ReferenceCounting.freeRefs(inputPrototype);
        throw new AssertionError(result.toString());
      }
      if (null != subjectOutput)
        subjectOutput.freeRef();
      if (null != referenceOutput)
        referenceOutput.freeRef();
      error.freeRef();
      ReferenceCounting.freeRefs(inputPrototype);
      return result;
    }
  }

  @Override
  public ToleranceStatistics test(@Nonnull final NotebookOutput output, final Layer subject,
                                  @Nonnull final Tensor... inputPrototype) {
    output.h1("Reference Implementation");
    output.p("This key is an alternate implementation which is expected to behave the same as the following key:");
    output.run(() -> {
      log.info(new GsonBuilder().setPrettyPrinting().create().toJson(reference.getJson()));
    });
    output.run(RefUtil.wrapInterface(() -> {
      log.info(new GsonBuilder().setPrettyPrinting().create().toJson(subject.getJson()));
    }, subject == null ? null : subject.addRef()));
    output.p("We measureStyle the agreement between the two layers in a random execution:");
    ToleranceStatistics temp_08_0003 = output
        .eval(RefUtil.wrapInterface(
            () -> {
              return test(subject == null ? null : subject.addRef(),
                  Tensor.addRefs(inputPrototype));
            }, Tensor.addRefs(inputPrototype),
            subject == null ? null : subject.addRef()));
    ReferenceCounting.freeRefs(inputPrototype);
    if (null != subject)
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

  public @Override
  @SuppressWarnings("unused")
  EquivalencyTester addRef() {
    return (EquivalencyTester) super.addRef();
  }
}
