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

import com.simiacryptus.notebook.MarkdownNotebookOutput;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.RefConsumer;
import com.simiacryptus.ref.wrappers.RefString;
import com.simiacryptus.ref.wrappers.RefSystem;
import com.simiacryptus.util.CodeUtil;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.test.SysOutInterceptor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.Date;
import java.util.UUID;

public abstract class NotebookReportBase extends ReferenceCountingBase {

  protected static final Logger logger = LoggerFactory.getLogger(NotebookReportBase.class);

  static {
    SysOutInterceptor.INSTANCE.init();
  }

  @Nonnull
  protected String reportingFolder = "reports/_reports";

  public Class<? extends NotebookReportBase> getReportClass() {
    return getClass();
  }

  @Nonnull
  public abstract ReportType getReportType();

  protected abstract Class<?> getTargetClass();

  @Nullable
  public static CharSequence printHeader(@Nonnull NotebookOutput log, @Nullable Class<?> networkClass,
                                         final CharSequence prefix) {
    if (null == networkClass)
      return null;
    @Nullable
    String javadoc = CodeUtil.getJavadoc(networkClass);
    log.setFrontMatterProperty(prefix + "_class_short", networkClass.getSimpleName());
    log.setFrontMatterProperty(prefix + "_class_full", networkClass.getCanonicalName());
    assert javadoc != null;
    log.setFrontMatterProperty(prefix + "_class_doc", javadoc.replaceAll("\n", ""));
    return javadoc;
  }

  @Nonnull
  public static File getTestReportLocation(@Nonnull final Class<?> sourceClass, String reportingFolder,
                                           @Nonnull final CharSequence... suffix) {
    final StackTraceElement callingFrame = Thread.currentThread().getStackTrace()[2];
    final CharSequence methodName = callingFrame.getMethodName();
    final String className = sourceClass.getCanonicalName();
    String classFilename = className.replaceAll("\\.", "/").replaceAll("\\$", "/");
    @Nonnull
    File path = new File(Util.mkString(File.separator, reportingFolder, classFilename));
    for (int i = 0; i < suffix.length - 1; i++)
      path = new File(path, suffix[i].toString());
    String testName = suffix.length == 0 ? String.valueOf(methodName) : suffix[suffix.length - 1].toString();
    File parent = path;
    //parent = new File(path, new SimpleDateFormat("yyyy-MM-dd_HHmmss").format(new Date()));
    path = new File(parent, testName + ".md");
    path.getParentFile().mkdirs();
    logger.info(RefString.format("Output Location: %s", path.getAbsoluteFile()));
    return path;
  }

  public static void withRefLeakMonitor(@Nonnull NotebookOutput log, @Nonnull RefConsumer<NotebookOutput> fn) {
    try (
        CodeUtil.LogInterception refLeakLog = CodeUtil.intercept(log, ReferenceCountingBase.class.getCanonicalName())) {
      fn.accept(log);
      RefSystem.gc();
      if (refLeakLog.counter.get() != 0)
        throw new AssertionError(RefString.format("RefLeak logged %d bytes", refLeakLog.counter.get()));
    } catch (RuntimeException e) {
      throw e;
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  public void printHeader(@Nonnull NotebookOutput log) {
    log.setFrontMatterProperty("created_on", new Date().toString());
    log.setFrontMatterProperty("report_type", getReportType().name());
    @Nullable
    CharSequence targetJavadoc = printHeader(log, getTargetClass(), "network");
    @Nullable
    CharSequence reportJavadoc = printHeader(log, getReportClass(), "report");
    //    log.p("__Target Description:__ " + StringEscapeUtils.escapeHtml4(targetJavadoc));
    //    log.p("__Report Description:__ " + StringEscapeUtils.escapeHtml4(reportJavadoc));
    log.p("__Target Description:__ " + targetJavadoc);
    log.p("__Report Description:__ " + reportJavadoc);
  }

  public void run(@Nonnull RefConsumer<NotebookOutput> fn, @Nonnull CharSequence... logPath) {
    try (@Nonnull
         NotebookOutput log = getLog(logPath)) {
      withRefLeakMonitor(log, NotebookOutput.concat(this::printHeader, MarkdownNotebookOutput.wrapFrontmatter(fn)));
    } catch (RuntimeException e) {
      throw e;
    } catch (Throwable e) {
      throw new RuntimeException(e);
    }
  }

  @Nonnull
  public NotebookOutput getLog(@Nullable CharSequence... logPath) {
    if (null == logPath || logPath.length == 0)
      logPath = new String[]{getClass().getSimpleName()};
    final File path = getTestReportLocation(getTargetClass(), reportingFolder, logPath);
    try {
      StackTraceElement callingFrame = Thread.currentThread().getStackTrace()[3];
      String methodName = callingFrame.getMethodName() + "_" + UUID.randomUUID().toString();
      path.getParentFile().mkdirs();
      return new MarkdownNotebookOutput(new File(path, methodName), true) {
        @Nullable
        @Override
        public File writeZip(File root, String baseName) {
          return null;
        }
      };
    } catch (FileNotFoundException e) {
      throw new RuntimeException(e);
    }
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  NotebookReportBase addRef() {
    return (NotebookReportBase) super.addRef();
  }

  public enum ReportType {
    Applications, Components, Models, Data, Optimizers, Experiments
  }

}
