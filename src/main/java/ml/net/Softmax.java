package ml.net;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class Softmax implements Classifier {

	/**
	 * - subtract max(exp(..))
	 * - regularization
	 * - drop out
	 */

	public static final double LEARNING_RATE = 0.0004;
	private final Random rnd;
	private final double[][] weights1;
	private final double[][] weights2;

	public static void main(String[] args) {
		Random rnd = new Random();
		CreateData createData = new CreateData(rnd).spiral();
		List<Sample> train = createData.getTrain();
		List<Sample> test = createData.getTest();

		long seed = rnd.nextLong();
		System.out.println("with relu");
		trial(new ArrayList<>(train), test, new Softmax(seed));
	}

	private static void trial(List<Sample> train, List<Sample> test, Softmax softmax1) {
		Visualise.draw(softmax1, train);

		softmax1.train(train);
		double testError = softmax1.error(test);
		System.out.printf("Test Fehler: %,5.2f %%\n", testError);
		System.out.println();

		Visualise.draw(softmax1, train);
	}

	public Softmax(long seed) {
		this.rnd = new Random(seed);
		weights1 = initialize(new double[50][3]);
		weights2 = initialize(new double[2][weights1.length + 1]);
	}

	private double[][] initialize(double[][] w) {
		for (int i = 0; i < w.length; i++) {
			for (int j = 0; j < w[0].length; j++) {
				w[i][j] = rnd.nextGaussian() * 0.01;
				if (j == w[0].length - 1) w[i][j] = 0; // bias weight = 0
			}
		}
		return w;
	}

	private void train(List<Sample> train) {
		for (int epoch = 0; epoch < 9000; epoch++) {
			double trainError = error(train);
			if (epoch % 50 == 0)
				System.out.printf("trainError [%4d] = %,5.2f %%\n", epoch, trainError);

			Collections.shuffle(train, rnd);
			List<Sample> samples = train.subList(0, 40);

			double[][] input = toInputMatrix(samples);
			double[][] activations1 = addBias(relum(mm(weights1, input)));
			double[][] activations2 = mm(weights2, activations1);

			// hidden to output layer
			double[][] grad = getErrorGrad(samples, activations2);
			double[][] transpose2 = transpose(activations1);
			double[][] grad2 = mm(grad, transpose2);
			double[][] update = mm(-LEARNING_RATE * 1 / samples.size(), grad2);
			add(weights2, update);

			double[][] grad1 = onlyActivated(activations1, mm(transpose(weights2), grad));

			double[][] grad0 = removeBias(mm(grad1, transpose(input)));

			double[][] update0 = mm(-LEARNING_RATE * 1 / samples.size(), grad0);
			add(weights1, update0);
		}
	}

	private double[][] onlyActivated(double[][] activations1, double[][] grad) {
		if (activations1.length != grad.length || activations1[0].length != grad[0].length)
			throw new AssertionError();

		double[][] copy = new double[grad.length][grad[0].length];
		for (int i = 0; i < grad.length; i++) {
			for (int j = 0; j < grad[0].length; j++) {
				if (activations1[i][j] > 0) {
					copy[i][j] = grad[i][j];
				}
			}
		}
		return copy;
	}

	private double[][] relum(double[][] m) {
		double[][] copy = new double[m.length][m[0].length];
		for (int i = 0; i < m.length; i++) {
			for (int j = 0; j < m[0].length; j++) {
				copy[i][j] = Math.max(0, m[i][j]);
			}
		}
		return copy;
	}

	private double[][] removeBias(double[][] m) {
		double[][] copy = new double[m.length - 1][m[0].length];
		for (int i = 0; i < m.length - 1; i++) {
			for (int j = 0; j < m[0].length; j++) {
				copy[i][j] = m[i][j];
			}
		}
		return copy;
	}

	private double[][] addBias(double[][] m) {
		double[][] copy = new double[m.length + 1][m[0].length];
		for (int i = 0; i < m.length; i++)
			for (int j = 0; j < m[0].length; j++)
				copy[i][j] = m[i][j];
		for (int i = 0; i < m[0].length; i++)
			copy[m.length][i] = 1;
		return copy;
	}

	private double[][] getErrorGrad(List<Sample> samples, double[][] act) {
		double[][] grad = new double[act.length][act[0].length];
		for (int i = 0; i < act[0].length; i++) {
			double max = Double.NEGATIVE_INFINITY;
			for (double[] activation : act) max = Math.max(max, activation[i]);
			int sum = 0;
			for (double[] activation : act) sum += Math.exp(activation[i] - max);

			for (int j = 0; j < act.length; j++) {
				grad[j][i] = Math.exp(act[j][i] - max) / sum;
				if (j == samples.get(i).getLabel()) grad[j][i] -= 1;
			}
		}
		return grad;
	}

	private double error(List<Sample> samples) {
		int[] chosen = classify(samples);

		int correct = 0;
		for (int i = 0; i < chosen.length; i++) {
			if (samples.get(i).getLabel() == chosen[i]) correct++;
		}

		return (samples.size() - correct) / ((double) samples.size()) * 100;
	}

	private int[] classify(List<Sample> samples) {
		return classify(samples, false);
	}

	@Override
	public int[] classify(List<Sample> samples, boolean dbg) {
		double[][] doubles = toInputMatrix(samples);
		double[][] act1 = addBias(relum(mm(weights1, doubles)));
		double[][] act2 = mm(weights2, act1);

//		if (dbg) pp(act1);

		int[] chosen = new int[samples.size()];
		for (int i = 0; i < chosen.length; i++) {
			chosen[i] = (act2[0][i] > act2[1][i]) ? 0 : 1;
		}
		return chosen;
	}

	private static void pp(double[][] doubles) {
		for (double[] row : doubles) {
			System.out.println(Arrays.toString(row));
		}
	}

	private void add(double[][] a, double[][] b) {
		if (a.length != b.length || a[0].length != b[0].length)
			throw new AssertionError(String.format("a: %s, b: %s\n", dims(a), dims(b)));

		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[0].length; j++) {
				a[i][j] += b[i][j];
			}
		}
	}

	private String dims(double[][] a) {
		return String.format("(%d x %d)", a.length, a[0].length);
	}

	private double[][] transpose(double[][] input) {
		double[][] transpose = new double[input[0].length][input.length];
		for (int i = 0; i < input.length; i++) {
			for (int j = 0; j < input[0].length; j++) {
				transpose[j][i] = input[i][j];
			}
		}
		return transpose;
	}

	private double[][] mm(double s, double[][] b) {
		double[][] copy = new double[b.length][b[0].length];
		for (int i = 0; i < b.length; i++) {
			for (int j = 0; j < b[0].length; j++) {
				copy[i][j] = b[i][j] * s;
			}
		}
		return copy;
	}

	private double[][] mm(double[][] a, double[][] b) {
		if (a[0].length != b.length) throw new AssertionError();

		double[][] ret = new double[a.length][b[0].length];
		for (int i = 0; i < ret.length; i++) {
			for (int j = 0; j < ret[0].length; j++) {
				for (int k = 0; k < a[0].length; k++) {
					ret[i][j] += a[i][k] * b[k][j];
				}
			}
		}
		return ret;
	}

	private double[][] toInputMatrix(List<Sample> samples) {
		double[][] matrix = new double[samples.get(0).getData().length + 1][samples.size()];
		for (int i = 0; i < samples.size(); i++) {
			double[] data = samples.get(i).getData();
			for (int j = 0; j < data.length; j++) {
				matrix[j][i] = data[j];
			}
			matrix[data.length][i] = 1;
		}
		return matrix;
	}

}
