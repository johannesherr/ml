package ml;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Random;

import com.google.common.io.ByteStreams;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

public class SVM {

	/**
	 * Tasks:
	 * - Batched Training
	 * - Regularization
	 */

	private static final Random rnd = new Random(2423L);
//	private static final Random rnd = new Random();
	private final double[][] w;

	public SVM(int inDim, int outDim) {
		w = new double[inDim + 1][outDim];
		for (double[] row : w) 
			for (int i = 0; i < row.length; i++)
				row[i] = rnd.nextDouble();
	}

	private static void display(final List<double[]> data, SVM svm) throws IOException {
		HttpServer server = HttpServer.create(new InetSocketAddress("localhost", 6006), -1);
		server.start();
		String ctx = "/ml";
		server.createContext(ctx, new HttpHandler() {
			@Override
			public void handle(HttpExchange httpExchange) throws IOException {
				URI uri = httpExchange.getRequestURI();
				Path path = Paths.get(URI.create(ctx).relativize(uri).toString());

				if (path.toString().equals("samples")) {
					httpExchange.sendResponseHeaders(200, 0);

					try (OutputStreamWriter writer = new OutputStreamWriter(httpExchange.getResponseBody(), StandardCharsets.UTF_8)) {
						writer.append("{\"boundaries\": [");
						double[][] boundaries = getBoundaries(svm);
						for (int i = 0; i < boundaries.length; i++) {
							double[] boundary = boundaries[i];
							writer.append(String.format(Locale.US, "[ %f, %f ]", boundary[0], boundary[1]));
							if (i + 1 < boundaries.length)
								writer.append(",\n");
						}
						writer.append(" ],");
						writer.append(" \"samples\": [");
						for (Iterator<double[]> iterator = data.iterator(); iterator.hasNext(); ) {
							double[] doubles = iterator.next();
							writer.append(String.format(Locale.US, "{\"x\": %f, \"y\": %f, \"type\": %.0f}", doubles[0], doubles[1], doubles[2]));
							if (iterator.hasNext())
								writer.append(",\n");
						}
						writer.append("]}");
					}

				} else if (Files.exists(path)) {
					httpExchange.sendResponseHeaders(200, 0);
					try (InputStream inputStream = Files.newInputStream(path)) {
						ByteStreams.copy(inputStream, httpExchange.getResponseBody());
					}

				} else {
					httpExchange.sendResponseHeaders(404, 0);
				}

				httpExchange.getResponseBody().close();
				httpExchange.close();
			}
		});
	}

	// 12:02-
	public static void main(String[] args) throws IOException {
		List<double[]> data = createData();
		int sz = data.size();
		int trainEndIdx = (int) (sz - sz * 0.1);
		List<double[]> trainingData = data.subList(0, trainEndIdx);
		List<double[]> testData = data.subList(trainEndIdx, sz);

		SVM svm = new SVM(2, 3);
		svm.train(trainingData);
		double trainError = svm.test(testData);
		System.out.printf("testError = %s%n", trainError);

		getBoundaries(svm);
		svm.pp(svm.w);

		display(data, svm);
	}

	private static double[][] getBoundaries(SVM svm) {
		double[][] boundaries = new double[svm.w.length][2];
		for (int i = 0; i < svm.w.length; i++) {
			double m = -svm.w[i][0] / svm.w[i][1];
			double t = - svm.w[i][2] / svm.w[i][1];
			boundaries[i][0] = m;
			boundaries[i][1] = t;
		}
		return boundaries;
	}

	private static List<double[]>  createData() {
		double radius = 1;
		double[][] centers = {{3,3}, {-3,-3}, {-2,4}};

		List<double[]> data = new ArrayList<>();
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 200; j++) {
				data.add(new double[]{
						(rnd.nextGaussian() - 0.5) * radius + centers[i][0],
						(rnd.nextGaussian() - 0.5) * radius + centers[i][1],
						i
				});
			}
		}
		Collections.shuffle(data, rnd);
		return data;
	}

	private double test(List<double[]> testData) {
		double[][] samples = transpose(toMatrix(testData));
		double[][] activations = transpose(apply(samples));

		int wrong = 0;
		for (int i = 0; i < activations.length; i++) {
			if (maxIdx(activations[i]) != testData.get(i)[2]) {
				wrong++;
			}
		}
		return ((double) wrong) / testData.size() * 100;
	}

	private double[][] transpose(double[][] m) {
		double[][] t = new double[m[0].length][m.length];
		for (int i = 0; i < m.length; i++)
			for (int j = 0; j < m[0].length; j++)
				t[j][i] = m[i][j];
		return t;
	}

	private double[][] toMatrix(List<double[]> data) {
		double[][] m = new double[data.size()][data.get(0).length];
		for (int i = 0; i < m.length; i++) {
			System.arraycopy(data.get(i), 0, m[i], 0, 2);
			m[i][2] = 1; // bias
		}
		return m;
	}

	private int maxIdx(double[] vals) {
		if (vals.length == 0) throw new IllegalArgumentException();

		int maxI = 0;
		for (int i = 0; i < vals.length; i++)
			if (vals[i] > vals[maxI]) maxI = i;
		return maxI;
	}

	private double[][] apply(double[][] samples) {
		return mm(w, samples);
	}

	private double[][] mm(double[][] a, double[][] b) {
		if (a[0].length != b.length) throw new AssertionError();

		double[][] ret = new double[a.length][b[0].length];
		for (int i = 0; i < a.length; i++) {
			for (int k = 0; k < b[0].length; k++) {
				for (int j = 0; j < a[0].length; j++) {
					ret[i][k] += a[i][j] * b[j][k];
				}
			}
		}

		return ret;
	}

	private void train(List<double[]> labeledData) {
		double[][] inputData = transpose(toMatrix(labeledData));

		for (int i = 0; i < 1000; i++) {
			trainEpoch(labeledData, inputData);
			System.out.printf("epoch = %s; train-error: %,.2f%n", i, test(labeledData));
		}
	}

	private void trainEpoch(List<double[]> labeledData, double[][] inputData) {
		double[][] activations = apply(inputData);

		double[][] gradient = new double[activations.length][activations[0].length];
		for (int i = 0; i < labeledData.size(); i++) {
			double[] sample = labeledData.get(i);
			for (int j = 0; j < activations.length; j++) {
				if (j != sample[2]) {
					if (activations[j][i] + 1 - activations[(int) sample[2]][i] > 0) {
						gradient[j][i] = 1;
						gradient[(int) sample[2]][i] += -1;
					}
				}
			}
		}

		double[][] wGradient = mm(gradient, transpose(inputData));

		for (double[] row : wGradient) {
			for (int i = 0; i < row.length; i++) {
				row[i] /= labeledData.size();
				row[i] *= 1e-3;
				row[i] *= -1;
			}
		}

		plus(w, wGradient);
	}

	private void pp(double[][] m) {
		dim(m);
		for (double[] row : m) {
			System.out.println(Arrays.toString(row));
		}
		System.out.println();
	}

	private void dim(double[][] m) {
		System.out.printf("%d x %d%n", m.length, m[0].length);
	}

	private void plus(double[][] a, double[][] b) {
		if (a.length != b.length || a[0].length != b[0].length) throw new AssertionError();

		for (int i = 0; i < a.length; i++)
			for (int j = 0; j < a[0].length; j++)
				a[i][j] += b[i][j];
	}

}
